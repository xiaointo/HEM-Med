#!/usr/bin/env python3
"""Shared utilities for the anonymous HEM-Med release."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_drug_name(value: Any) -> str:
    return normalize_text(value)


def extract_item_text(item: Any) -> str:
    if isinstance(item, dict):
        for key in ("text", "name", "label", "code", "value"):
            if key in item and item[key]:
                return str(item[key])
        return " ".join(str(v) for v in item.values() if v is not None)
    if isinstance(item, (list, tuple)) and item:
        return str(item[0])
    return str(item or "")


def case_anchor_keys(case: dict[str, Any]) -> list[str]:
    anchors: list[str] = []
    for field, prefix in (("dx", "diag"), ("px", "proc"), ("lab", "lab")):
        for item in case.get(field) or []:
            text = normalize_text(extract_item_text(item))
            if text:
                anchors.append(f"{prefix}:{text}")
    return anchors


def true_meds(case: dict[str, Any]) -> list[str]:
    meds: list[str] = []
    for item in case.get("med") or []:
        if isinstance(item, (list, tuple)) and item:
            meds.append(str(item[0]))
        elif isinstance(item, dict):
            meds.append(str(item.get("name") or item.get("medication") or item.get("drug") or ""))
        else:
            meds.append(str(item))
    return [m for m in meds if m]


def true_meds_norm(case: dict[str, Any]) -> set[str]:
    return {normalize_drug_name(m) for m in true_meds(case) if normalize_drug_name(m)}


def hospital_id(case: dict[str, Any]) -> str:
    return str(case.get("hid") or case.get("hospital_id") or "unknown")


def average_precision(labels: list[int], scores: list[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    positives = sum(labels)
    if positives == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, (_score, label) in enumerate(pairs, 1):
        if label:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positives


def set_metrics(truth: set[str], pred: set[str]) -> dict[str, float]:
    tp = len(truth & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    jaccard = tp / len(truth | pred) if truth or pred else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "jaccard": jaccard}


def load_ddi_pairs(path: str | Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        a_col = "drug_name_1" if "drug_name_1" in columns else columns[0]
        b_col = "drug_name_2" if "drug_name_2" in columns else columns[1]
        for row in reader:
            a = normalize_drug_name(row.get(a_col))
            b = normalize_drug_name(row.get(b_col))
            if a and b and a != b:
                pairs.add(tuple(sorted((a, b))))
    return pairs


def ddi_rate(meds: Iterable[str], ddi_pairs: set[tuple[str, str]]) -> float:
    norm = sorted({normalize_drug_name(m) for m in meds if normalize_drug_name(m)})
    total = 0
    hits = 0
    for i, a in enumerate(norm):
        for b in norm[i + 1:]:
            total += 1
            if tuple(sorted((a, b))) in ddi_pairs:
                hits += 1
    return hits / total if total else 0.0


def top_counts(counter: Counter[str], n: int) -> list[dict[str, Any]]:
    total = sum(counter.values()) or 1
    return [
        {"medication": med, "support": count, "frequency": count / total}
        for med, count in counter.most_common(n)
    ]
