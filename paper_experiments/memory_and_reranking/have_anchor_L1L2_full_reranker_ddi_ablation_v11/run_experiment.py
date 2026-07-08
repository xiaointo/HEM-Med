#!/usr/bin/env python3
"""Run L1/L2-memory based DeepSeek medication recommendation experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from pathlib import Path
from threading import Lock
from typing import Any


DEEPSEEK_API_KEY = ""  # TODO: 在这里填写 DeepSeek API key，或设置环境变量 DEEPSEEK_API_KEY
DEEPSEEK_API_KEY = DEEPSEEK_API_KEY or os.getenv("DEEPSEEK_API_KEY", "")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

SYSTEM_PROMPT = """You are a cautious ICU medication recommendation assistant for research.

You will receive a de-identified ICU patient context and retrieved external memory from L1 and L2.

L1 contains cross-hospital empirical anchor-medication patterns.
L2 contains hospital-specific residual or preference patterns.
These memories are observational evidence, not confirmed prescription intent.

Your task is to recommend a set of medications for this ICU case.

Important rules:
1. Do not claim confirmed prescription intent.
2. Do not treat observational frequency as causal evidence.
3. Prefer medications supported by patient context and retrieved L1/L2 evidence.
4. L2 hospital_suppressed evidence means the current hospital uses the medication less than global reference; be cautious with such drugs.
5. Always inspect support counts, not only frequencies. High frequency with small support is weak evidence.
6. Avoid recommending duplicate medications.
7. Use retrieved DDI knowledge as a safety penalty, not an absolute contraindication. Do not drop strongly supported common ICU medications solely because of a DDI pair.
8. Prefer a concise medication set; prioritize the reranked candidate order unless patient context clearly argues otherwise.
9. Output valid JSON only.
10. Do not include explanations outside JSON.
"""

TEXT_KEYS = ["text", "anchor_string", "value", "name", "label", "diagnosis", "procedure", "lab", "item"]

CHECKPOINT_LOCK = Lock()
FILE_WRITE_LOCK = Lock()
CHECKPOINT_RETRY_SLEEP_SECONDS = [1, 2, 5, 5, 5]


class ApiFailure(RuntimeError):
    """DeepSeek API/network failure after retries."""


class ParseFailure(RuntimeError):
    """DeepSeek returned content that could not be parsed as JSON after retries."""


class CheckpointFailure(RuntimeError):
    """Checkpoint write failed after retries."""



def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", required=True, type=Path)
    parser.add_argument("--l1-file", required=True, type=Path)
    parser.add_argument("--l2-file", required=True, type=Path)
    parser.add_argument("--ddi-file", default=None, type=Path, help="Backward-compatible DDI file used for both RAG and metric when specific files are omitted.")
    parser.add_argument("--ddi-rag-file", default=None, type=Path, help="DDI file with optional interaction descriptions for prompt RAG.")
    parser.add_argument("--ddi-metric-file", default=None, type=Path, help="DDI pair-set file for DDI-rate metric.")
    parser.add_argument("--vocabulary-file", required=True, type=Path)
    parser.add_argument("--hospital-prior-file", default=None, type=Path)
    parser.add_argument("--global-prior-file", default=None, type=Path)
    parser.add_argument("--anchor-prior-file", default=None, type=Path)
    parser.add_argument("--hospital-anchor-prior-file", default=None, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-cases", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-l1-entries", type=int, default=30)
    parser.add_argument("--max-l2-entries", type=int, default=30)
    parser.add_argument("--max-candidate-meds", type=int, default=40)
    parser.add_argument("--max-prior-meds", type=int, default=20)
    parser.add_argument("--max-fuzzy-anchors", type=int, default=40)
    parser.add_argument("--ddi-candidate-penalty", type=float, default=0.08)
    parser.add_argument("--generic-icu-penalty", type=float, default=0.18)
    parser.add_argument("--generic-icu-prior-penalty", type=float, default=0.35)
    parser.add_argument("--missed-common-boost", type=float, default=0.16)
    parser.add_argument("--adaptive-gap-threshold", type=float, default=0.08)
    parser.add_argument("--adaptive-min-k", type=int, default=5)
    parser.add_argument("--adaptive-max-k", type=int, default=7)
    parser.add_argument("--rerank-top-n", type=int, default=12)
    parser.add_argument("--default-top-k", type=int, default=5)
    parser.add_argument("--max-replacements", type=int, default=2)
    parser.add_argument("--max-recommendations", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", type=parse_bool, default=False)
    parser.add_argument("--allow-llm-free-recommendation", type=parse_bool, default=False)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_text(value: Any) -> str:
    return clean_text(value).lower()


def canonical_drug_display(value: Any) -> str:
    text = clean_text(value)
    norm = normalize_text(text)
    if norm == "albuterol":
        return "Albuterol"
    if norm in {"insulin aspart", "insulin glargine", "insulin regular", "regular insulin", "insulin lispro"} or norm.startswith("insulin "):
        return "insulin"
    if norm in {"glucose", "dextrose", "dextrose, unspecified form"}:
        return "Dextrose, unspecified form"
    return text


def normalize_drug_name(value: Any) -> str:
    return normalize_text(canonical_drug_display(value))




GENERIC_ICU_MEDICATIONS = {
    normalize_drug_name("Albuterol"),
    normalize_drug_name("insulin"),
    normalize_drug_name("Vancomycin"),
    normalize_drug_name("Ipratropium"),
    normalize_drug_name("Dextrose, unspecified form"),
}

MISSED_COMMON_MEDICATIONS = {
    normalize_drug_name("Enoxaparin"),
    normalize_drug_name("Acetylsalicylic acid"),
    normalize_drug_name("Heparin"),
    normalize_drug_name("Norepinephrine"),
    normalize_drug_name("Midazolam"),
}

PRIOR_ONLY_SOURCES = {"hospital_prior", "global_prior"}

def description_mentions_drugs(description: str, drug_a: str, drug_b: str) -> bool:
    text = normalize_text(description)
    a = normalize_drug_name(drug_a)
    b = normalize_drug_name(drug_b)
    return bool(text and a and b and a in text and b in text)


def extract_item_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for key in TEXT_KEYS:
            if key in x and x[key]:
                return str(x[key])
    return str(x)


def extract_med_name(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return extract_med_name(x[0]) if x else ""
    if isinstance(x, dict):
        for key in ["medication", "drug", "drug_name", "med", "name", "label", "item"]:
            if key in x and x[key]:
                return str(x[key])
    return str(x)


def anchor_keys_for_case(case: dict[str, Any]) -> list[str]:
    keys = []
    for prefix, field in (("diag", "dx"), ("proc", "px"), ("lab", "lab")):
        for item in case.get(field) or []:
            text = normalize_text(extract_item_text(item))
            if text:
                keys.append(f"{prefix}:{text}")
    return list(dict.fromkeys(keys))



def anchor_text(anchor_key: str) -> str:
    return anchor_key.split(":", 1)[1] if ":" in anchor_key else anchor_key


def anchor_prefix_from_key(anchor_key: str) -> str:
    return anchor_key.split(":", 1)[0] if ":" in anchor_key else ""


def last_anchor_segment(text: str) -> str:
    parts = [clean_text(part) for part in text.split("|") if clean_text(part)]
    return normalize_text(parts[-1] if parts else text)


def build_memory_anchor_index(l1: dict[str, Any], l2: dict[str, Any]) -> dict[str, set[str]]:
    anchors = set(l1)
    for hospital_data in l2.values():
        anchors.update(hospital_data.keys())
    by_prefix: dict[str, set[str]] = defaultdict(set)
    for key in anchors:
        by_prefix[anchor_prefix_from_key(key)].add(key)
    return by_prefix


def fuzzy_anchor_keys_for_case(
    case: dict[str, Any],
    memory_anchor_index: dict[str, set[str]],
    max_fuzzy_anchors: int,
) -> list[dict[str, Any]]:
    queries = anchor_keys_for_case(case)
    scored: dict[str, dict[str, Any]] = {}
    for query in queries:
        prefix = anchor_prefix_from_key(query)
        qtext = anchor_text(query)
        qlast = last_anchor_segment(qtext)
        candidates = memory_anchor_index.get(prefix, set())
        for mem_key in candidates:
            mtext = anchor_text(mem_key)
            mlast = last_anchor_segment(mtext)
            score = 0
            match_type = ""
            if mem_key == query:
                score = 100
                match_type = "exact"
            elif qtext.endswith("|" + mtext) or mtext.endswith("|" + qtext):
                score = 80
                match_type = "hierarchical_suffix"
            elif qlast and mlast and qlast == mlast:
                score = 70
                match_type = "last_segment"
            elif len(mtext) >= 4 and mtext in qtext:
                score = 55
                match_type = "memory_text_in_case_anchor"
            elif len(qtext) >= 4 and qtext in mtext:
                score = 50
                match_type = "case_anchor_in_memory_text"
            if score <= 0:
                continue
            previous = scored.get(mem_key)
            if previous is None or score > previous["score"]:
                scored[mem_key] = {
                    "anchor_key": mem_key,
                    "query_anchor_key": query,
                    "match_type": match_type,
                    "score": score,
                }
    rows = sorted(scored.values(), key=lambda x: (-x["score"], x["anchor_key"]))
    return rows[:max_fuzzy_anchors]


def load_prior_rows(path: Path | None) -> Any:
    if not path or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def prior_med_rows(prior: Any, hospital_id_value: str | None = None, max_rows: int = 20) -> list[dict[str, Any]]:
    if not prior:
        return []
    if hospital_id_value is not None:
        rows = ((prior.get("hospitals") or {}).get(hospital_id_value) or [])
    else:
        rows = prior.get("medications") or []
    output = []
    for row in rows[:max_rows]:
        med = clean_text(row.get("medication"))
        if med:
            output.append(row)
    return output

def case_id(case: dict[str, Any]) -> str:
    return clean_text(case.get("id") or case.get("patient_id") or case.get("stay_id") or case.get("stayid"))


def hospital_id(case: dict[str, Any]) -> str:
    return clean_text(case.get("hid") or case.get("hospital_id"))


def true_meds(case: dict[str, Any]) -> list[str]:
    meds = []
    seen = set()
    for item in case.get("med") or []:
        med = clean_text(extract_med_name(item))
        norm = normalize_drug_name(med)
        if med and norm not in seen:
            seen.add(norm)
            meds.append(med)
    return meds


def read_jsonl_cases(path: Path, offset: int, max_cases: int) -> list[dict[str, Any]]:
    selected = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if idx < offset:
                continue
            if len(selected) >= max_cases:
                break
            line = line.strip()
            if line:
                selected.append(json.loads(line))
    return selected


def sort_l1(entry: dict[str, Any]) -> tuple[int, float, float]:
    ge = (entry.get("statistical_data") or {}).get("global_evidence") or {}
    return (
        int(ge.get("global_context_drug_support_count", 0)),
        float(ge.get("global_context_drug_frequency", 0.0)),
        float(ge.get("context_drug_hospital_frequency", 0.0)),
    )


def l2_priority(entry: dict[str, Any]) -> int:
    order = {
        "hospital_amplified": 0,
        "hospital_specific": 0,
        "l2_residual_not_in_l1": 1,
        "hospital_suppressed": 2,
    }
    return order.get(entry.get("residual_type"), 3)


def sort_l2(entry: dict[str, Any]) -> tuple[int, int, float]:
    le = entry.get("statistical_evidence") or {}
    return (
        -l2_priority(entry),
        int(le.get("local_context_drug_support_count", 0)),
        float(le.get("local_context_drug_frequency", 0.0)),
    )


def retrieve_memory(
    case: dict[str, Any],
    l1: dict[str, Any],
    l2: dict[str, Any],
    max_l1: int,
    max_l2: int,
    memory_anchor_index: dict[str, set[str]],
    max_fuzzy_anchors: int,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    anchor_matches = fuzzy_anchor_keys_for_case(case, memory_anchor_index, max_fuzzy_anchors)
    anchors = [row["anchor_key"] for row in anchor_matches]
    match_by_anchor = {row["anchor_key"]: row for row in anchor_matches}
    l1_hits = []
    l2_hits = []
    hid = hospital_id(case)

    for anchor_key in anchors:
        match = match_by_anchor.get(anchor_key, {})
        match_score = int(match.get("score", 0))
        for med, entry in (l1.get(anchor_key) or {}).items():
            l1_hits.append(
                {
                    "source": "L1",
                    "anchor_key": anchor_key,
                    "query_anchor_key": match.get("query_anchor_key", anchor_key),
                    "anchor_match_type": match.get("match_type", "exact"),
                    "anchor_match_score": match_score,
                    "medication": med,
                    "medication_function": entry.get("medication_function", ""),
                    "statistical_data": entry.get("statistical_data", {}),
                    "attribution_hypothesis": entry.get("attribution_hypothesis", ""),
                    "extra_summaries": entry.get("extra_summaries", {}),
                }
            )
        for med, entry in ((l2.get(hid) or {}).get(anchor_key) or {}).items():
            l2_hits.append(
                {
                    "source": "L2",
                    "hospital_id": hid,
                    "anchor_key": anchor_key,
                    "query_anchor_key": match.get("query_anchor_key", anchor_key),
                    "anchor_match_type": match.get("match_type", "exact"),
                    "anchor_match_score": match_score,
                    "medication": med,
                    "medication_function": entry.get("medication_function", ""),
                    "statistical_evidence": entry.get("statistical_evidence", {}),
                    "global_reference": entry.get("global_reference", {}),
                    "residual_type": entry.get("residual_type", ""),
                    "l1_membership": entry.get("l1_membership", False),
                    "attribution_hypothesis": entry.get("attribution_hypothesis", ""),
                    "extra_summaries": entry.get("extra_summaries", {}),
                }
            )

    l1_hits.sort(key=lambda entry: (entry.get("anchor_match_score", 0), *sort_l1(entry)), reverse=True)
    l2_hits.sort(key=lambda entry: (entry.get("anchor_match_score", 0), *sort_l2(entry)), reverse=True)
    return anchors, l1_hits[:max_l1], l2_hits[:max_l2], anchor_matches



def anchor_prior_rows(prior: Any, anchor_matches: list[dict[str, Any]], max_rows: int = 20) -> list[dict[str, Any]]:
    if not prior:
        return []
    anchors = prior.get("anchors") or {}
    output = []
    seen = set()
    for match in anchor_matches:
        weight = float(match.get("score", 0)) / 100.0
        anchor_key = match.get("anchor_key")
        for row in (anchors.get(anchor_key) or [])[:max_rows]:
            med = clean_text(row.get("medication"))
            norm = normalize_drug_name(med)
            if not norm or (anchor_key, norm) in seen:
                continue
            seen.add((anchor_key, norm))
            copied = dict(row)
            copied["anchor_key"] = anchor_key
            copied["anchor_match_score"] = match.get("score", 0)
            copied["anchor_weight"] = weight
            output.append(copied)
    return output


def hospital_anchor_prior_rows(prior: Any, hospital_id_value: str, anchor_matches: list[dict[str, Any]], max_rows: int = 20) -> list[dict[str, Any]]:
    if not prior:
        return []
    hospital_anchors = prior.get("hospital_anchors") or {}
    output = []
    seen = set()
    for match in anchor_matches:
        anchor_key = match.get("anchor_key")
        combo_key = f"{hospital_id_value}::{anchor_key}"
        weight = float(match.get("score", 0)) / 100.0
        for row in (hospital_anchors.get(combo_key) or [])[:max_rows]:
            med = clean_text(row.get("medication"))
            norm = normalize_drug_name(med)
            if not norm or (combo_key, norm) in seen:
                continue
            seen.add((combo_key, norm))
            copied = dict(row)
            copied["anchor_key"] = anchor_key
            copied["anchor_match_score"] = match.get("score", 0)
            copied["anchor_weight"] = weight
            output.append(copied)
    return output

def candidate_medications(
    l1_hits: list[dict[str, Any]],
    l2_hits: list[dict[str, Any]],
    hospital_prior_rows: list[dict[str, Any]],
    global_prior_rows: list[dict[str, Any]],
    anchor_prior_items: list[dict[str, Any]],
    hospital_anchor_prior_items: list[dict[str, Any]],
    ddi_pairs: set[tuple[str, str]] | None,
    max_meds: int,
    ddi_penalty: float,
    generic_icu_penalty: float,
    generic_icu_prior_penalty: float,
    missed_common_boost: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    by_norm: dict[str, dict[str, Any]] = {}

    def add_score(med: str, score: float, source: str, detail: dict[str, Any]) -> None:
        norm = normalize_drug_name(med)
        if not norm:
            return
        display = canonical_drug_display(med)
        item = by_norm.setdefault(norm, {"medication": display, "score": 0.0, "sources": []})
        adjusted_score = float(score)
        if norm in GENERIC_ICU_MEDICATIONS and source in PRIOR_ONLY_SOURCES:
            adjusted_score *= max(0.0, 1.0 - generic_icu_prior_penalty)
        item["score"] += adjusted_score
        source_detail = {"source": source, **detail}
        if adjusted_score != float(score):
            source_detail["pre_calibration_score"] = round(float(score), 6)
            source_detail["prior_generic_icu_penalty"] = round(generic_icu_prior_penalty, 6)
        item["sources"].append(source_detail)

    for entry in l2_hits:
        residual = entry.get("residual_type")
        le = entry.get("statistical_evidence") or {}
        support = int(le.get("local_context_drug_support_count", 0))
        freq = float(le.get("local_context_drug_frequency", 0.0))
        anchor_key = entry.get("anchor_key", "")
        prefix_bonus = 1.15 if anchor_key.startswith(("diag:", "proc:")) else 0.82
        match_multiplier = prefix_bonus * (0.4 + float(entry.get("anchor_match_score", 100)) / 100.0)
        base = {"hospital_amplified": 5.0, "hospital_specific": 4.4, "l2_residual_not_in_l1": 3.5, "hospital_suppressed": 0.25}.get(residual, 1.5)
        score = match_multiplier * (base + min(support / 10.0, 2.0) + freq)
        add_score(entry["medication"], score, "L2", {"anchor_key": entry.get("anchor_key"), "residual_type": residual, "support": support, "frequency": round(freq, 6)})

    for entry in l1_hits:
        ge = ((entry.get("statistical_data") or {}).get("global_evidence") or {})
        support = int(ge.get("global_context_drug_support_count", 0))
        freq = float(ge.get("global_context_drug_frequency", 0.0))
        anchor_key = entry.get("anchor_key", "")
        prefix_bonus = 1.10 if anchor_key.startswith(("diag:", "proc:")) else 0.78
        match_multiplier = prefix_bonus * (0.35 + float(entry.get("anchor_match_score", 100)) / 100.0)
        score = match_multiplier * (2.1 + min(support / 1000.0, 1.8) + freq)
        add_score(entry["medication"], score, "L1", {"anchor_key": entry.get("anchor_key"), "support": support, "frequency": round(freq, 6)})

    for row in hospital_anchor_prior_items:
        support = int(row.get("support_count", 0))
        freq = float(row.get("frequency", 0.0))
        weight = float(row.get("anchor_weight", 1.0))
        add_score(row.get("medication", ""), weight * (5.8 + min(support / 100.0, 2.0) + 2.0 * freq), "hospital_anchor_prior", {"anchor_key": row.get("anchor_key"), "support": support, "frequency": round(freq, 6)})

    for row in anchor_prior_items:
        support = int(row.get("support_count", 0))
        freq = float(row.get("frequency", 0.0))
        weight = float(row.get("anchor_weight", 1.0))
        add_score(row.get("medication", ""), weight * (4.4 + min(support / 500.0, 1.8) + 1.5 * freq), "anchor_prior", {"anchor_key": row.get("anchor_key"), "support": support, "frequency": round(freq, 6)})

    for row in hospital_prior_rows:
        support = int(row.get("support_count", 0))
        freq = float(row.get("frequency", 0.0))
        add_score(row.get("medication", ""), 1.4 + min(support / 1200.0, 1.0) + 0.8 * freq, "hospital_prior", {"support": support, "frequency": round(freq, 6)})

    for row in global_prior_rows:
        support = int(row.get("support_count", 0))
        freq = float(row.get("frequency", 0.0))
        add_score(row.get("medication", ""), 0.45 + min(support / 20000.0, 0.8) + 0.4 * freq, "global_prior", {"support": support, "frequency": round(freq, 6)})

    for item in by_norm.values():
        norm = normalize_drug_name(item["medication"])
        source_names = {src.get("source") for src in item.get("sources", [])}
        item["pre_calibration_score"] = round(float(item["score"]), 6)
        calibration_notes = []
        if norm in GENERIC_ICU_MEDICATIONS:
            strong_sources = source_names - PRIOR_ONLY_SOURCES
            factor = max(0.0, 1.0 - generic_icu_penalty)
            if strong_sources:
                factor = max(factor, 1.0 - generic_icu_penalty / 2.0)
            item["score"] *= factor
            calibration_notes.append({"type": "generic_icu_penalty", "factor": round(factor, 6), "strong_sources": sorted(strong_sources)})
        if norm in MISSED_COMMON_MEDICATIONS:
            factor = 1.0 + missed_common_boost
            item["score"] *= factor
            calibration_notes.append({"type": "missed_common_boost", "factor": round(factor, 6)})
        if calibration_notes:
            item["calibration_notes"] = calibration_notes

    ranked = sorted(by_norm.values(), key=lambda x: (-x["score"], normalize_drug_name(x["medication"])))
    selected = []
    selected_norms = []
    for item in ranked:
        norm = normalize_drug_name(item["medication"])
        ddi_hits = 0
        if ddi_pairs is not None:
            for prev in selected_norms:
                if tuple(sorted((norm, prev))) in ddi_pairs:
                    ddi_hits += 1
        item["raw_score"] = round(item["score"], 6)
        item["ddi_candidate_penalty_hits"] = ddi_hits
        item["score"] = round(item["score"] - ddi_penalty * ddi_hits, 6)
        selected.append(item)
        selected_norms.append(norm)
    selected.sort(key=lambda x: (-x["score"], normalize_drug_name(x["medication"])))
    selected = selected[:max_meds]
    return [item["medication"] for item in selected], selected


def build_user_prompt(
    case: dict[str, Any],
    l1_hits: list[dict[str, Any]],
    l2_hits: list[dict[str, Any]],
    candidates: list[str],
    candidate_details: list[dict[str, Any]],
    max_recommendations: int,
    rerank_top_n: int,
    default_top_k: int,
    max_replacements: int,
    ddi_pairs: set[tuple[str, str]] | None,
    ddi_descriptions: dict[tuple[str, str], str] | None,
) -> dict[str, Any]:
    prompt_candidates = candidates[:rerank_top_n]
    prompt_norms = {normalize_drug_name(med) for med in prompt_candidates}
    prompt_details = [item for item in candidate_details if normalize_drug_name(item.get("medication")) in prompt_norms][:rerank_top_n]
    default_recommendations = prompt_candidates[:default_top_k]
    candidate_by_norm = {normalize_drug_name(med): med for med in prompt_candidates}
    known_candidate_ddis = []
    if ddi_pairs is not None:
        for left, right in combinations(sorted(candidate_by_norm), 2):
            key = tuple(sorted((left, right)))
            if key in ddi_pairs:
                description = (ddi_descriptions or {}).get(key, "")
                item = {
                    "drug_1": candidate_by_norm[left],
                    "drug_2": candidate_by_norm[right],
                }
                if description:
                    item["interaction_description"] = description
                else:
                    item["interaction_description"] = "Known DDI pair in the retrieved DDI set; source description omitted because it did not explicitly name both normalized candidate drugs."
                known_candidate_ddis.append(item)
    known_candidate_ddis = known_candidate_ddis[:100]
    return {
        "task": "Rerank a concise ICU medication set using retrieved have-anchor L1/L2 memory and DDI RAG safety evidence.",
        "patient": {
            "patient_id": case_id(case),
            "hospital_id": hospital_id(case),
            "age": case.get("age"),
            "sex": case.get("sex"),
            "dx": case.get("dx", []),
            "px": case.get("px", []),
            "lab": case.get("lab", []),
        },
        "retrieved_memory": {"l1_global_memory": l1_hits, "l2_hospital_memory": l2_hits},
        "candidate_medications": prompt_candidates,
        "default_recommendations_top5": default_recommendations,
        "candidate_ranking_details": prompt_details,
        "known_ddi_knowledge_among_candidates": known_candidate_ddis,
        "recommendation_constraints": {
            "max_recommendations": max_recommendations,
            "prefer_candidate_medications": True,
            "candidate_scope": f"Only recommend medications from candidate_medications top {rerank_top_n}.",
            "default_top5_policy": "Start from default_recommendations_top5. Keep them unless a replacement is clearly better supported by patient context.",
            "replacement_limit": f"You may replace at most {max_replacements} of the default top {default_top_k} medications. Every replacement must state a short reason in hypothesis.",
            "avoid_duplicates": True,
            "avoid_weak_l2_noise": "L2 has already been filtered to strong evidence; still prefer higher support counts and avoid low-support entries.",
            "ddi_awareness": "Use known DDI knowledge as a penalty/risk signal, not as an absolute ban. Strongly supported common ICU medications may still be recommended when context and memory support them.",
            "rerank_instruction": "This is rerank-only, not free generation. The candidate order already combines exact/fuzzy anchor memory, hospital-anchor prior, anchor prior, hospital prior, global prior, and a small DDI penalty. Prefer top-ranked candidates and keep the default top5 unless there is strong evidence to replace.",
        },
        "required_output_schema": {
            "patient_id": "string",
            "hospital_id": "string",
            "recommended_medications": [
                {
                    "medication": "string",
                    "confidence": "number from 0 to 1",
                    "hypothesis": "Explain why this medication is recommended based on patient context, retrieved L1/L2 support, and any DDI safety tradeoff.",
                }
            ],
        },
    }


def load_ddi_knowledge(
    path: Path | None, warnings: list[str]
) -> tuple[set[tuple[str, str]] | None, dict[tuple[str, str], str], str | None]:
    candidates = []
    if path:
        candidates.append(path)
    base = Path("outputs/ddi_cleaned_eicu_space")
    candidates.extend([
        base / "eicu_space_ddi_pairs_simple.csv",
        base / "eicu_space_ddi_pair_set.csv",
        base / "eicu_space_ddi_pairs.csv",
    ])
    for candidate in candidates:
        if candidate and candidate.exists():
            pairs = set()
            descriptions: dict[tuple[str, str], str] = {}
            with candidate.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    a = row.get("drug_name_1") or row.get("eicu_drug_name_1")
                    b = row.get("drug_name_2") or row.get("eicu_drug_name_2")
                    if not a or not b:
                        continue
                    key = tuple(sorted((normalize_drug_name(a), normalize_drug_name(b))))
                    pairs.add(key)
                    desc = clean_text(row.get("interaction_description") or row.get("description") or row.get("ddi_description"))
                    if desc and key not in descriptions and description_mentions_drugs(desc, a, b):
                        descriptions[key] = desc
            return pairs, descriptions, str(candidate)
    warnings.append("No DDI file found; DDI rate is null.")
    return None, {}, None


def connect_checkpoint(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn


def init_checkpoint(path: Path) -> None:
    with CHECKPOINT_LOCK:
        with connect_checkpoint(path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS case_status (
                    case_key TEXT PRIMARY KEY,
                    patient_id TEXT,
                    hospital_id TEXT,
                    status TEXT,
                    attempt_count INTEGER,
                    error_message TEXT,
                    updated_at TEXT
                )
                """
            )


def update_checkpoint(
    path: Path,
    case_key: str,
    pid: str,
    hid: str,
    status: str,
    attempts: int,
    error: str = "",
) -> None:
    last_error = ""
    for retry_index, sleep_seconds in enumerate([0, *CHECKPOINT_RETRY_SLEEP_SECONDS]):
        if sleep_seconds:
            time.sleep(sleep_seconds)
        try:
            with CHECKPOINT_LOCK:
                with connect_checkpoint(path) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO case_status
                        (case_key, patient_id, hospital_id, status, attempt_count, error_message, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (case_key, pid, hid, status, attempts, error, datetime.utcnow().isoformat()),
                    )
            return
        except sqlite3.OperationalError as exc:
            last_error = str(exc)
            if "database is locked" not in last_error.lower() or retry_index >= len(CHECKPOINT_RETRY_SLEEP_SECONDS):
                break
    raise CheckpointFailure(f"checkpoint write failed after retries: {last_error}")


def checkpoint_successes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with CHECKPOINT_LOCK:
        with connect_checkpoint(path) as conn:
            return {row[0] for row in conn.execute("SELECT case_key FROM case_status WHERE status='success'")}


def call_deepseek(user_prompt: dict[str, Any], timeout: int, max_retries: int) -> tuple[dict[str, Any], dict[str, Any]]:
    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    last_error = ""
    last_error_kind = "api"
    for retry in range(max_retries + 1):
        for use_response_format in (True, False):
            payload = {
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
                ],
                "temperature": 0.1,
                "max_tokens": 4096,
            }
            if use_response_format:
                payload["response_format"] = {"type": "json_object"}
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                content = raw["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                return raw, parsed
            except urllib.error.HTTPError as exc:
                last_error_kind = "api"
                last_error = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='ignore')[:500]}"
                if exc.code == 400 and use_response_format:
                    continue
                if exc.code not in {429, 500, 502, 503, 504}:
                    raise ApiFailure(last_error)
            except json.JSONDecodeError as exc:
                last_error_kind = "parse"
                last_error = str(exc)
                if use_response_format:
                    continue
            except Exception as exc:
                last_error_kind = "api"
                last_error = str(exc)
                if use_response_format:
                    continue
            break
        if retry < max_retries:
            time.sleep(min(60, (2**retry) * 5))
    if last_error_kind == "parse":
        raise ParseFailure(last_error)
    raise ApiFailure(last_error)

def parse_recommended(
    parsed: dict[str, Any], max_recommendations: int
) -> tuple[list[str], dict[str, float]]:
    meds = []
    scores: dict[str, float] = {}
    seen = set()
    for item in parsed.get("recommended_medications") or []:
        med = item.get("medication") if isinstance(item, dict) else item
        confidence = item.get("confidence", 0.5) if isinstance(item, dict) else 0.5
        med = clean_text(med)
        norm = normalize_drug_name(med)
        try:
            score = min(1.0, max(0.0, float(confidence)))
        except (TypeError, ValueError):
            score = 0.5
        if med and norm not in seen:
            seen.add(norm)
            meds.append(med)
            scores[norm] = score
        elif med:
            scores[norm] = max(scores.get(norm, 0.0), score)
        if len(meds) >= max_recommendations:
            break
    return meds, scores



def enforce_rerank_only(
    raw_pred: list[str],
    raw_scores: dict[str, float],
    candidates: list[str],
    default_top_k: int,
    rerank_top_n: int,
    max_replacements: int,
    max_recommendations: int,
) -> tuple[list[str], dict[str, float]]:
    allowed = candidates[:rerank_top_n]
    allowed_norms = {normalize_drug_name(x) for x in allowed}
    selected = list(candidates[:default_top_k])
    selected_norms = {normalize_drug_name(x) for x in selected}
    replacements = 0
    for med in raw_pred:
        norm = normalize_drug_name(med)
        if norm not in allowed_norms or norm in selected_norms:
            continue
        if replacements >= max_replacements:
            continue
        remove_index = None
        raw_norms = {normalize_drug_name(x) for x in raw_pred}
        for idx in range(len(selected) - 1, -1, -1):
            if normalize_drug_name(selected[idx]) not in raw_norms:
                remove_index = idx
                break
        if remove_index is None and selected:
            remove_index = len(selected) - 1
        if remove_index is not None:
            selected.pop(remove_index)
        selected.append(canonical_drug_display(med))
        selected_norms = {normalize_drug_name(x) for x in selected}
        replacements += 1
    rank = {normalize_drug_name(med): i for i, med in enumerate(allowed)}
    selected = sorted(selected, key=lambda x: rank.get(normalize_drug_name(x), 999))[:max_recommendations]
    scores = {}
    for i, med in enumerate(selected):
        norm = normalize_drug_name(med)
        scores[norm] = max(float(raw_scores.get(norm, 0.0)), (len(selected) - i) / max(len(selected), 1))
    return selected, scores

def average_precision(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if not labels or positives == 0:
        return None
    grouped: dict[float, list[int]] = {}
    for label, score in zip(labels, scores):
        grouped.setdefault(float(score), []).append(int(label))
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    ap = 0.0
    for score in sorted(grouped, reverse=True):
        group = grouped[score]
        true_positives += sum(group)
        false_positives += len(group) - sum(group)
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return ap


def metric_for_case(
    true_list: list[str],
    pred_list: list[str],
    predicted_scores: dict[str, float],
    ddi_pairs: set[tuple[str, str]] | None,
    vocabulary: set[str],
) -> dict[str, Any]:
    true_set = {normalize_drug_name(x) for x in true_list if normalize_drug_name(x)}
    pred_set = {normalize_drug_name(x) for x in pred_list if normalize_drug_name(x)}
    inter = true_set & pred_set
    union = true_set | pred_set
    precision = len(inter) / len(pred_set) if pred_set else 0.0
    recall = len(inter) / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    jaccard = len(inter) / len(union) if union else 0.0
    ddi_count = 0
    total_pairs = 0
    if ddi_pairs is not None:
        for a, b in combinations(sorted(pred_set), 2):
            total_pairs += 1
            if tuple(sorted((a, b))) in ddi_pairs:
                ddi_count += 1
    case_ddi_rate = (ddi_count / total_pairs) if total_pairs else 0.0
    scoring_vocabulary = sorted(vocabulary | true_set | pred_set)
    labels = [1 if medication in true_set else 0 for medication in scoring_vocabulary]
    scores = [float(predicted_scores.get(medication, 0.0)) for medication in scoring_vocabulary]
    case_prauc = average_precision(labels, scores)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "prauc": case_prauc,
        "exact_match": pred_set == true_set,
        "ddi_pair_count": ddi_count,
        "total_pred_pairs": total_pairs,
        "case_ddi_rate": case_ddi_rate,
        "safety_adjusted_jaccard": jaccard * (1.0 - case_ddi_rate),
    }


def micro_prauc(
    prediction_rows: list[dict[str, Any]], vocabulary: set[str]
) -> float | None:
    labels: list[int] = []
    scores: list[float] = []
    for row in prediction_rows:
        true_set = {
            normalize_drug_name(medication)
            for medication in row["true_medications"]
            if normalize_drug_name(medication)
        }
        predicted_scores = row.get("predicted_scores") or {}
        scoring_vocabulary = sorted(
            vocabulary | true_set | set(predicted_scores)
        )
        labels.extend(1 if medication in true_set else 0 for medication in scoring_vocabulary)
        scores.extend(float(predicted_scores.get(medication, 0.0)) for medication in scoring_vocabulary)
    return average_precision(labels, scores)


def write_jsonl_line(path: Path, payload: dict[str, Any], lock: Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")



def metric_summary(
    prediction_items: list[dict[str, Any]],
    vocabulary: set[str],
    ddi_pairs: set[tuple[str, str]] | None,
) -> dict[str, Any]:
    rows = [metric_for_case(item["true_medications"], item["pred"], item["scores"], ddi_pairs, vocabulary) for item in prediction_items]
    prediction_rows = [{"true_medications": item["true_medications"], "predicted_scores": item["scores"]} for item in prediction_items]

    def local_mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 6) if values else None

    total_ddi = sum(row["ddi_pair_count"] for row in rows)
    total_pairs = sum(row["total_pred_pairs"] for row in rows)
    jaccard = local_mean([row["jaccard"] for row in rows])
    ddi_rate = round(total_ddi / total_pairs, 6) if total_pairs and ddi_pairs is not None else None
    return {
        "precision": local_mean([row["precision"] for row in rows]),
        "recall": local_mean([row["recall"] for row in rows]),
        "f1": local_mean([row["f1"] for row in rows]),
        "jaccard": jaccard,
        "prauc_micro": round(micro_prauc(prediction_rows, vocabulary), 6),
        "prauc_mean_case": local_mean([row["prauc"] for row in rows if row["prauc"] is not None]),
        "ddi_rate": ddi_rate,
        "safety_adjusted_jaccard": round(jaccard * (1 - ddi_rate), 6) if jaccard is not None and ddi_rate is not None else None,
    }


def adaptive_topk_prediction(
    candidate_details: list[dict[str, Any]],
    min_k: int,
    max_k: int,
    gap_threshold: float,
) -> tuple[list[str], dict[str, float]]:
    if not candidate_details:
        return [], {}
    max_k = max(min_k, max_k)
    max_k = min(max_k, len(candidate_details))
    min_k = min(min_k, max_k)
    selected_count = min_k
    for idx in range(min_k, max_k):
        prev_score = float(candidate_details[idx - 1].get("score", 0.0))
        next_score = float(candidate_details[idx].get("score", 0.0))
        if prev_score <= 0:
            break
        relative_gap = (prev_score - next_score) / prev_score
        if relative_gap <= gap_threshold:
            selected_count = idx + 1
        else:
            break
    pred = [item["medication"] for item in candidate_details[:selected_count]]
    scores = {normalize_drug_name(med): (selected_count - idx) / selected_count for idx, med in enumerate(pred)}
    return pred, scores


def deterministic_baseline_metrics(
    selected_items: list[dict[str, Any]],
    vocabulary: set[str],
    ddi_pairs: set[tuple[str, str]] | None,
    ks: list[int],
    adaptive_min_k: int,
    adaptive_max_k: int,
    adaptive_gap_threshold: float,
) -> dict[str, Any]:
    output = {}
    for k in ks:
        prediction_items = []
        for item in selected_items:
            pred = item["candidate_medications"][:k]
            scores = {normalize_drug_name(med): (k - idx) / k for idx, med in enumerate(pred)}
            prediction_items.append({"true_medications": item["true_medications"], "pred": pred, "scores": scores})
        output[f"deterministic_top{k}"] = metric_summary(prediction_items, vocabulary, ddi_pairs)

    adaptive_items = []
    adaptive_sizes = []
    for item in selected_items:
        pred, scores = adaptive_topk_prediction(
            item.get("candidate_details", []), adaptive_min_k, adaptive_max_k, adaptive_gap_threshold
        )
        adaptive_sizes.append(len(pred))
        adaptive_items.append({"true_medications": item["true_medications"], "pred": pred, "scores": scores})
    adaptive_metrics = metric_summary(adaptive_items, vocabulary, ddi_pairs)
    adaptive_metrics["avg_pred_med_count"] = round(sum(adaptive_sizes) / len(adaptive_sizes), 6) if adaptive_sizes else None
    adaptive_metrics["min_pred_med_count"] = min(adaptive_sizes) if adaptive_sizes else None
    adaptive_metrics["max_pred_med_count"] = max(adaptive_sizes) if adaptive_sizes else None
    adaptive_metrics["min_k"] = adaptive_min_k
    adaptive_metrics["max_k"] = adaptive_max_k
    adaptive_metrics["gap_threshold"] = adaptive_gap_threshold
    output["deterministic_adaptive_topk"] = adaptive_metrics
    return output


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    used_api = bool(DEEPSEEK_API_KEY) and not args.dry_run
    if not DEEPSEEK_API_KEY:
        warnings.append("DEEPSEEK_API_KEY is empty; prompts were generated but API was not called.")
    if args.dry_run:
        warnings.append("dry_run=true; prompts were generated but API was not called.")

    l1 = json.loads(args.l1_file.read_text(encoding="utf-8"))
    l2 = json.loads(args.l2_file.read_text(encoding="utf-8"))
    memory_anchor_index = build_memory_anchor_index(l1, l2)
    hospital_prior = load_prior_rows(args.hospital_prior_file)
    global_prior = load_prior_rows(args.global_prior_file)
    anchor_prior = load_prior_rows(args.anchor_prior_file)
    hospital_anchor_prior = load_prior_rows(args.hospital_anchor_prior_file)
    vocabulary_payload = json.loads(args.vocabulary_file.read_text(encoding="utf-8"))
    vocabulary_values = (
        vocabulary_payload.get("medications", [])
        if isinstance(vocabulary_payload, dict)
        else vocabulary_payload
    )
    vocabulary = {
        normalize_drug_name(medication)
        for medication in vocabulary_values
        if normalize_drug_name(medication)
    }
    ddi_rag_path = args.ddi_rag_file or args.ddi_file
    ddi_metric_path = args.ddi_metric_file or args.ddi_file or args.ddi_rag_file
    ddi_rag_pairs, ddi_descriptions, ddi_rag_file_used = load_ddi_knowledge(ddi_rag_path, warnings)
    ddi_metric_pairs, _ddi_metric_descriptions, ddi_metric_file_used = load_ddi_knowledge(ddi_metric_path, warnings)
    cases = read_jsonl_cases(args.test_file, args.offset, args.max_cases)
    checkpoint = args.output_dir / "recommendation_checkpoint.sqlite"
    init_checkpoint(checkpoint)

    output_paths = {
        "selected": args.output_dir / "selected_test_cases.jsonl",
        "retrieved": args.output_dir / "retrieved_l1_l2_memory.jsonl",
        "prompts": args.output_dir / "deepseek_recommendation_prompts.jsonl",
        "raw": args.output_dir / "deepseek_recommendation_raw_responses.jsonl",
        "predictions": args.output_dir / "deepseek_recommendation_predictions.jsonl",
        "failures": args.output_dir / "deepseek_recommendation_failures.jsonl",
    }
    for name, path in output_paths.items():
        preserve_for_resume = (
            args.resume and used_api and name in {"raw", "predictions", "failures"}
            and path.exists()
        )
        if not preserve_for_resume:
            path.write_text("", encoding="utf-8")

    selected_items = []
    for case in cases:
        pid, hid = case_id(case), hospital_id(case)
        anchors, l1_hits, l2_hits, anchor_matches = retrieve_memory(
            case, l1, l2, args.max_l1_entries, args.max_l2_entries, memory_anchor_index, args.max_fuzzy_anchors
        )
        hospital_rows = prior_med_rows(hospital_prior, hid, args.max_prior_meds)
        global_rows = prior_med_rows(global_prior, None, args.max_prior_meds)
        anchor_rows = anchor_prior_rows(anchor_prior, anchor_matches, args.max_prior_meds)
        hospital_anchor_rows = hospital_anchor_prior_rows(hospital_anchor_prior, hid, anchor_matches, args.max_prior_meds)
        candidates, candidate_details = candidate_medications(
            l1_hits, l2_hits, hospital_rows, global_rows, anchor_rows, hospital_anchor_rows, ddi_metric_pairs, args.max_candidate_meds, args.ddi_candidate_penalty,
            args.generic_icu_penalty, args.generic_icu_prior_penalty, args.missed_common_boost
        )
        prompt = build_user_prompt(
            case, l1_hits, l2_hits, candidates, candidate_details, args.max_recommendations, args.rerank_top_n, args.default_top_k, args.max_replacements, ddi_rag_pairs, ddi_descriptions
        )
        item = {
            "case_key": f"{pid}|{hid}",
            "patient_id": pid,
            "hospital_id": hid,
            "case": case,
            "true_medications": true_meds(case),
            "retrieved_anchor_keys": anchors,
            "l1_hits": l1_hits,
            "l2_hits": l2_hits,
            "candidate_medications": candidates,
            "candidate_details": candidate_details,
            "anchor_matches": anchor_matches,
            "prompt": prompt,
        }
        selected_items.append(item)
        write_jsonl_line(output_paths["selected"], {"patient_id": pid, "hospital_id": hid, "case": case}, FILE_WRITE_LOCK)
        write_jsonl_line(
            output_paths["retrieved"],
            {
                "patient_id": pid,
                "hospital_id": hid,
                "retrieved_anchor_keys": anchors,
                "num_l1_hits": len(l1_hits),
                "num_l2_hits": len(l2_hits),
                "l1_hits": l1_hits,
                "l2_hits": l2_hits,
                "candidate_medications": candidates,
                "candidate_details": candidate_details,
                "anchor_matches": anchor_matches,
            },
            FILE_WRITE_LOCK,
        )
        write_jsonl_line(
            output_paths["prompts"],
            {"patient_id": pid, "hospital_id": hid, "system_prompt": SYSTEM_PROMPT, "user_prompt": prompt},
            FILE_WRITE_LOCK,
        )
        if not candidates and not args.allow_llm_free_recommendation:
            warnings.append(f"No L1/L2 hits for case {pid}|{hid}; empty recommendation will be used if API is skipped.")
        if not used_api:
            update_checkpoint(checkpoint, f"{pid}|{hid}", pid, hid, "prompt_generated", 0)

    baseline_metrics = deterministic_baseline_metrics(
        selected_items, vocabulary, ddi_metric_pairs, [5, 6],
        args.adaptive_min_k, args.adaptive_max_k, args.adaptive_gap_threshold
    )
    (args.output_dir / "deterministic_baseline_metrics.json").write_text(
        json.dumps(baseline_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    prediction_rows = []
    if args.resume and used_api and output_paths["predictions"].exists():
        with output_paths["predictions"].open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    prediction_rows.append(json.loads(line))
    failures = []
    failed_case_keys = set()
    failure_counts = {
        "api_failure": 0,
        "parse_failure": 0,
        "checkpoint_failure": 0,
        "infrastructure_failure": 0,
    }
    lock = FILE_WRITE_LOCK

    def run_one(item: dict[str, Any]) -> dict[str, Any]:
        pid, hid, ckey = item["patient_id"], item["hospital_id"], item["case_key"]
        if not item["candidate_medications"] and not args.allow_llm_free_recommendation:
            parsed = {"patient_id": pid, "hospital_id": hid, "recommended_medications": []}
            raw = {"skipped_api": True, "reason": "no_l1_l2_hits"}
        else:
            raw, parsed = call_deepseek(item["prompt"], args.timeout, args.max_retries)
        raw_pred, raw_scores = parse_recommended(
            parsed, args.max_recommendations
        )
        pred, predicted_scores = enforce_rerank_only(
            raw_pred, raw_scores, item["candidate_medications"], args.default_top_k, args.rerank_top_n, args.max_replacements, args.max_recommendations
        )
        row = {
            "patient_id": pid,
            "hospital_id": hid,
            "true_medications": item["true_medications"],
            "predicted_medications": pred,
            "predicted_scores": predicted_scores,
            "retrieved_anchor_keys": item["retrieved_anchor_keys"],
            "num_l1_hits": len(item["l1_hits"]),
            "num_l2_hits": len(item["l2_hits"]),
            "candidate_medications": item.get("candidate_medications", []),
            "used_api": not raw.get("skipped_api", False),
        }
        write_jsonl_line(output_paths["raw"], {"patient_id": pid, "hospital_id": hid, "raw_response": raw}, lock)
        write_jsonl_line(output_paths["predictions"], row, lock)
        update_checkpoint(checkpoint, ckey, pid, hid, "success", 1)
        return row

    if used_api:
        skip_success = checkpoint_successes(checkpoint) if args.resume else set()
        todo = [item for item in selected_items if item["case_key"] not in skip_success]
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            futures = {executor.submit(run_one, item): item for item in todo}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    prediction_rows.append(future.result())
                except Exception as exc:
                    if isinstance(exc, ApiFailure):
                        failure_type = "api_failure"
                        failure_counts["api_failure"] += 1
                    elif isinstance(exc, ParseFailure):
                        failure_type = "parse_failure"
                        failure_counts["parse_failure"] += 1
                    elif isinstance(exc, CheckpointFailure):
                        failure_type = "checkpoint_failure"
                        failure_counts["checkpoint_failure"] += 1
                        failure_counts["infrastructure_failure"] += 1
                    else:
                        failure_type = "infrastructure_failure"
                        failure_counts["infrastructure_failure"] += 1

                    failed_case_keys.add(item["case_key"])
                    failure = {
                        "patient_id": item["patient_id"],
                        "hospital_id": item["hospital_id"],
                        "failure_type": failure_type,
                        "error_message": str(exc),
                    }
                    failures.append(failure)
                    write_jsonl_line(output_paths["failures"], failure, lock)

                    if not isinstance(exc, CheckpointFailure):
                        try:
                            update_checkpoint(
                                checkpoint,
                                item["case_key"],
                                item["patient_id"],
                                item["hospital_id"],
                                "failed",
                                args.max_retries,
                                str(exc),
                            )
                        except CheckpointFailure as checkpoint_exc:
                            failure_counts["checkpoint_failure"] += 1
                            failure_counts["infrastructure_failure"] += 1
                            checkpoint_failure = {
                                "patient_id": item["patient_id"],
                                "hospital_id": item["hospital_id"],
                                "failure_type": "checkpoint_failure",
                                "error_message": str(checkpoint_exc),
                            }
                            failures.append(checkpoint_failure)
                            write_jsonl_line(output_paths["failures"], checkpoint_failure, lock)

    # If API was skipped, keep prediction/failure files empty and metrics unset.
    metrics_by_case = []
    if prediction_rows:
        for row in prediction_rows:
            m = metric_for_case(
                row["true_medications"],
                row["predicted_medications"],
                row.get("predicted_scores") or {},
                ddi_metric_pairs,
                vocabulary,
            )
            metrics_by_case.append({**row, **m})

    metrics_path = args.output_dir / "recommendation_metrics_by_case.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "patient_id", "hospital_id", "precision", "recall", "f1",
            "jaccard", "prauc", "safety_adjusted_jaccard", "exact_match",
            "ddi_pair_count", "total_pred_pairs", "case_ddi_rate"
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in metrics_by_case:
            writer.writerow({field: row.get(field) for field in fields})

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 6) if values else None


    candidate_recall_values = []
    candidate_true_hits = 0
    candidate_true_total = 0
    cases_all_true_in_candidates = 0
    cases_no_true_in_candidates = 0
    for item in selected_items:
        true_set = {normalize_drug_name(x) for x in item["true_medications"] if normalize_drug_name(x)}
        cand_set = {normalize_drug_name(x) for x in item["candidate_medications"] if normalize_drug_name(x)}
        hits = len(true_set & cand_set)
        candidate_true_hits += hits
        candidate_true_total += len(true_set)
        candidate_recall_values.append(hits / len(true_set) if true_set else 0.0)
        if true_set and true_set <= cand_set:
            cases_all_true_in_candidates += 1
        if true_set and not (true_set & cand_set):
            cases_no_true_in_candidates += 1
    total_ddi = sum(row["ddi_pair_count"] for row in metrics_by_case)
    total_pairs = sum(row["total_pred_pairs"] for row in metrics_by_case)
    metrics = {
        "step": "l1_l2_deepseek_medication_recommendation",
        "test_file": str(args.test_file),
        "l1_file": str(args.l1_file),
        "l2_file": str(args.l2_file),
        "ddi_rag_file": ddi_rag_file_used or (str(ddi_rag_path) if ddi_rag_path else None),
        "ddi_metric_file": ddi_metric_file_used or (str(ddi_metric_path) if ddi_metric_path else None),
        "ddi_rag_pair_count_loaded": len(ddi_rag_pairs) if ddi_rag_pairs is not None else None,
        "ddi_metric_pair_count_loaded": len(ddi_metric_pairs) if ddi_metric_pairs is not None else None,
        "ddi_description_count_loaded": len(ddi_descriptions),
        "vocabulary_file": str(args.vocabulary_file),
        "vocabulary_size": len(vocabulary),
        "num_test_cases_selected": len(selected_items),
        "num_cases_attempted": len(prediction_rows) + len(failed_case_keys),
        "num_cases_succeeded": len(prediction_rows),
        "num_cases_failed": len(failed_case_keys),
        "num_infrastructure_failures": failure_counts["infrastructure_failure"],
        "num_api_failures": failure_counts["api_failure"],
        "num_parse_failures": failure_counts["parse_failure"],
        "num_checkpoint_failures": failure_counts["checkpoint_failure"],
        "accuracy": mean([row["precision"] for row in metrics_by_case]),
        "exact_match_accuracy": mean([1.0 if row["exact_match"] else 0.0 for row in metrics_by_case]),
        "recall": mean([row["recall"] for row in metrics_by_case]),
        "f1": mean([row["f1"] for row in metrics_by_case]),
        "jaccard": mean([row["jaccard"] for row in metrics_by_case]),
        "prauc_micro": (
            round(micro_prauc(prediction_rows, vocabulary), 6)
            if prediction_rows
            else None
        ),
        "prauc_mean_case": mean([
            row["prauc"] for row in metrics_by_case if row["prauc"] is not None
        ]),
        "ddi_rate": (
            round(total_ddi / total_pairs, 6)
            if metrics_by_case and ddi_metric_pairs is not None and total_pairs
            else None
            if not metrics_by_case or ddi_metric_pairs is None
            else 0.0
        ),
        "mean_case_ddi_rate": mean([row["case_ddi_rate"] for row in metrics_by_case]) if ddi_metric_pairs is not None else None,
        "avg_true_med_count": mean([len(item["true_medications"]) for item in selected_items]),
        "avg_pred_med_count": mean([len(row["predicted_medications"]) for row in prediction_rows]),
        "avg_l1_hits": mean([len(item["l1_hits"]) for item in selected_items]),
        "avg_l2_hits": mean([len(item["l2_hits"]) for item in selected_items]),
        "num_cases_with_no_l1_l2_hits": sum(1 for item in selected_items if not item["l1_hits"] and not item["l2_hits"]),
        "num_empty_predictions": sum(1 for row in prediction_rows if not row["predicted_medications"]),
        "candidate_recall_micro": round(candidate_true_hits / candidate_true_total, 6) if candidate_true_total else None,
        "candidate_recall_mean_case": mean(candidate_recall_values),
        "cases_all_true_in_candidates": cases_all_true_in_candidates,
        "cases_no_true_in_candidates": cases_no_true_in_candidates,
        "avg_candidate_count": mean([len(item["candidate_medications"]) for item in selected_items]),
        "rerank_top_n": args.rerank_top_n,
        "default_top_k": args.default_top_k,
        "max_replacements": args.max_replacements,
        "deterministic_baselines": baseline_metrics,
        "used_api": used_api,
        "used_llm": used_api,
        "used_l1": True,
        "used_l2": True,
        "used_condition_group": False,
        "modified_original_files": False,
        "warnings": sorted(set(warnings)),
    }
    metrics["prauc"] = metrics["prauc_micro"]
    metrics["safety_adjusted_jaccard"] = (
        round(metrics["jaccard"] * (1.0 - metrics["ddi_rate"]), 6)
        if metrics["jaccard"] is not None and metrics["ddi_rate"] is not None
        else None
    )
    metrics["mean_case_safety_adjusted_jaccard"] = mean([
        row["safety_adjusted_jaccard"] for row in metrics_by_case
    ])
    (args.output_dir / "recommendation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    run_report = {
        "step": "run_l1_l2_deepseek_recommendation",
        "output_dir": str(args.output_dir),
        "max_cases": args.max_cases,
        "offset": args.offset,
        "concurrency": args.concurrency,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "max_l1_entries": args.max_l1_entries,
        "max_l2_entries": args.max_l2_entries,
        "max_candidate_meds": args.max_candidate_meds,
        "max_prior_meds": args.max_prior_meds,
        "max_fuzzy_anchors": args.max_fuzzy_anchors,
        "rerank_top_n": args.rerank_top_n,
        "default_top_k": args.default_top_k,
        "max_replacements": args.max_replacements,
        "ddi_candidate_penalty": args.ddi_candidate_penalty,
        "max_recommendations": args.max_recommendations,
        "used_api": used_api,
        "used_llm": used_api,
        "used_condition_group": False,
        "modified_original_files": False,
        "num_infrastructure_failures": failure_counts["infrastructure_failure"],
        "num_api_failures": failure_counts["api_failure"],
        "num_parse_failures": failure_counts["parse_failure"],
        "num_checkpoint_failures": failure_counts["checkpoint_failure"],
    }
    (args.output_dir / "recommendation_run_report.json").write_text(json.dumps(run_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
