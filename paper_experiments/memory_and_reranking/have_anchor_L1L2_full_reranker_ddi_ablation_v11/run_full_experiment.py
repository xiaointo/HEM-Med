#!/usr/bin/env python3
"""Train a pure-Python lightweight reranker for L1/L2 RAG medication candidates."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
import os
from typing import Any

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ranker", BASE / "run_experiment.py")
ranker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ranker)  # type: ignore[union-attr]

GENERIC_CLASSES = {
    "anti_infective", "anticoagulant", "antiplatelet", "sedative_analgesic",
    "bronchodilator", "glucose_control", "fluid_electrolyte", "cardiac_medication",
    "antihypertensive", "steroid", "diuretic", "antiarrhythmic", "acid_suppression", "unknown"
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    data_root = Path(os.environ.get("HEM_MED_DATA_DIR", BASE.parents[2] / "data"))
    split = data_root / "splits/split_8_1_1_nonempty_med_seed1203"
    p.add_argument("--train-file", type=Path, default=split / "train_80pct.jsonl")
    p.add_argument("--validation-file", type=Path, default=split / "validation_10pct.jsonl")
    p.add_argument("--test-file", type=Path, default=BASE / "inputs/test_100_have_anchor.jsonl")
    p.add_argument("--full-test-file", type=Path, default=split / "test_10pct.jsonl")
    p.add_argument("--output-dir", type=Path, default=BASE / "outputs_full_reranker_ddi_ablation")
    p.add_argument("--train-cases", type=int, default=72436)
    p.add_argument("--validation-cases", type=int, default=9054)
    p.add_argument("--test-cases", type=int, default=100)
    p.add_argument("--full-test-cases", type=int, default=9055, help="0 skips full test; set e.g. 1000/9055 to evaluate more.")
    p.add_argument("--candidate-depth-train", type=int, default=30)
    p.add_argument("--candidate-depth-predict", type=int, default=30)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=0.035)
    p.add_argument("--l2", type=float, default=0.00015)
    p.add_argument("--seed", type=int, default=1203)
    p.add_argument("--max-l1-entries", type=int, default=100)
    p.add_argument("--max-l2-entries", type=int, default=100)
    p.add_argument("--max-candidate-meds", type=int, default=60)
    p.add_argument("--max-prior-meds", type=int, default=25)
    p.add_argument("--max-fuzzy-anchors", type=int, default=80)
    p.add_argument("--ddi-candidate-penalty", type=float, default=0.04)
    p.add_argument("--generic-icu-penalty", type=float, default=0.12)
    p.add_argument("--generic-icu-prior-penalty", type=float, default=0.35)
    p.add_argument("--missed-common-boost", type=float, default=0.16)
    return p.parse_args()


def read_cases(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def true_meds_norm(case: dict[str, Any]) -> set[str]:
    return {ranker.normalize_drug_name(x) for x in ranker.true_meds(case) if ranker.normalize_drug_name(x)}


def true_classes(case: dict[str, Any], med_to_class: dict[str, str]) -> set[str]:
    out = set()
    for item in case.get("med") or []:
        if isinstance(item, (list, tuple)) and item:
            med = ranker.normalize_drug_name(item[0])
            cls = str(item[1]) if len(item) > 1 else med_to_class.get(med, "unknown")
        else:
            med = ranker.normalize_drug_name(item)
            cls = med_to_class.get(med, "unknown")
        if cls:
            out.add(cls)
    return out


def load_context(args: argparse.Namespace) -> dict[str, Any]:
    def j(name: str) -> Any:
        return json.loads((BASE / "inputs" / name).read_text(encoding="utf-8"))
    l1 = j("L1_statistical_memory_from_have_anchor.json")
    l2 = j("L2_hospital_residual_memory_from_have_anchor.json")
    vocab_payload = j("medication_vocabulary_canonical.json")
    vocab = vocab_payload.get("medications", []) if isinstance(vocab_payload, dict) else vocab_payload
    warnings = []
    ddi_pairs, _desc, _path = ranker.load_ddi_knowledge(BASE / "inputs/eicu_space_ddi_pair_set.csv", warnings)
    return {
        "l1": l1, "l2": l2, "memory_anchor_index": ranker.build_memory_anchor_index(l1, l2),
        "hospital_prior": ranker.load_prior_rows(BASE / "inputs/hospital_medication_prior.json"),
        "global_prior": ranker.load_prior_rows(BASE / "inputs/global_medication_prior.json"),
        "anchor_prior": ranker.load_prior_rows(BASE / "inputs/anchor_medication_prior.json"),
        "hospital_anchor_prior": ranker.load_prior_rows(BASE / "inputs/hospital_anchor_medication_prior.json"),
        "vocabulary": {ranker.normalize_drug_name(x) for x in vocab if ranker.normalize_drug_name(x)},
        "ddi_pairs": ddi_pairs,
    }


def collect_med_classes(cases: list[dict[str, Any]], ctx: dict[str, Any]) -> dict[str, str]:
    counts = defaultdict(Counter)
    for case in cases:
        for item in case.get("med") or []:
            if isinstance(item, (list, tuple)) and item:
                med = ranker.normalize_drug_name(item[0])
                cls = str(item[1]) if len(item) > 1 else "unknown"
                counts[med][cls] += 1
    for mem in (ctx["l1"],):
        for anchor_map in mem.values():
            for med, payload in anchor_map.items():
                cls = payload.get("medication_function") if isinstance(payload, dict) else None
                if cls:
                    counts[ranker.normalize_drug_name(med)][cls] += 1
    for hosp in ctx["l2"].values():
        for anchor_map in hosp.values():
            for med, payload in anchor_map.items():
                cls = payload.get("medication_function") if isinstance(payload, dict) else None
                if cls:
                    counts[ranker.normalize_drug_name(med)][cls] += 1
    return {med: ctr.most_common(1)[0][0] for med, ctr in counts.items() if med and ctr}


def candidate_details(case: dict[str, Any], ctx: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    hid = ranker.hospital_id(case)
    _anchors, l1_hits, l2_hits, anchor_matches = ranker.retrieve_memory(
        case, ctx["l1"], ctx["l2"], args.max_l1_entries, args.max_l2_entries, ctx["memory_anchor_index"], args.max_fuzzy_anchors
    )
    hospital_rows = ranker.prior_med_rows(ctx["hospital_prior"], hid, args.max_prior_meds)
    global_rows = ranker.prior_med_rows(ctx["global_prior"], None, args.max_prior_meds)
    anchor_rows = ranker.anchor_prior_rows(ctx["anchor_prior"], anchor_matches, args.max_prior_meds)
    hospital_anchor_rows = ranker.hospital_anchor_prior_rows(ctx["hospital_anchor_prior"], hid, anchor_matches, args.max_prior_meds)
    _meds, details = ranker.candidate_medications(
        l1_hits, l2_hits, hospital_rows, global_rows, anchor_rows, hospital_anchor_rows,
        ctx["ddi_pairs"], args.max_candidate_meds, args.ddi_candidate_penalty,
        args.generic_icu_penalty, args.generic_icu_prior_penalty, args.missed_common_boost,
    )
    return details


def text_for_case(case: dict[str, Any]) -> str:
    parts = []
    for field in ("dx", "px", "lab"):
        for x in case.get(field) or []:
            parts.append(ranker.normalize_text(ranker.extract_item_text(x)))
    return " | ".join(parts)


def med_mentioned_in_context(med_norm: str, case_text: str) -> bool:
    if not med_norm or len(med_norm) < 4:
        return False
    if med_norm in case_text:
        return True
    # Use distinctive tokens for medication names embedded in procedure strings.
    toks = [t for t in med_norm.replace(",", " ").split() if len(t) >= 5 and t not in {"unspecified", "chloride"}]
    return any(tok in case_text for tok in toks)


def features_for_candidate(case: dict[str, Any], item: dict[str, Any], rank_idx: int, med_to_class: dict[str, str]) -> dict[str, float]:
    med_norm = ranker.normalize_drug_name(item.get("medication"))
    cls = med_to_class.get(med_norm, "unknown")
    sources = item.get("sources") or []
    score = float(item.get("score", 0.0))
    raw = float(item.get("raw_score", score) or 0.0)
    feats = {
        "bias": 1.0,
        "score_log": math.log1p(max(score, 0.0)) / 4.0,
        "raw_score_log": math.log1p(max(raw, 0.0)) / 4.0,
        "rank_inv": 1.0 / (rank_idx + 1),
        "rank_top3": 1.0 if rank_idx < 3 else 0.0,
        "rank_top5": 1.0 if rank_idx < 5 else 0.0,
        "rank_top8": 1.0 if rank_idx < 8 else 0.0,
        "ddi_hits": min(float(item.get("ddi_candidate_penalty_hits", 0)), 5.0) / 5.0,
        "mentioned_current_context": 1.0 if med_mentioned_in_context(med_norm, text_for_case(case)) else 0.0,
    }
    source_counts = Counter(src.get("source") for src in sources)
    for src in ["L1", "L2", "hospital_anchor_prior", "anchor_prior", "hospital_prior", "global_prior"]:
        feats[f"src_{src}"] = min(source_counts.get(src, 0), 5) / 5.0
    prefixes = Counter()
    residuals = Counter()
    support_total = 0.0
    freq_max = 0.0
    for src in sources:
        anchor = str(src.get("anchor_key") or "")
        if ":" in anchor:
            prefixes[anchor.split(":", 1)[0]] += 1
        residual = src.get("residual_type")
        if residual:
            residuals[residual] += 1
        support_total += float(src.get("support") or 0.0)
        freq_max = max(freq_max, float(src.get("frequency") or 0.0))
    for prefix in ["diag", "proc", "lab"]:
        feats[f"anchor_{prefix}"] = min(prefixes[prefix], 5) / 5.0
    for residual in ["hospital_amplified", "hospital_specific", "l2_residual_not_in_l1", "hospital_suppressed"]:
        feats[f"residual_{residual}"] = min(residuals[residual], 5) / 5.0
    feats["support_log"] = math.log1p(support_total) / 10.0
    feats["freq_max"] = freq_max
    if cls in GENERIC_CLASSES:
        feats[f"class_{cls}"] = 1.0
    else:
        feats["class_other"] = 1.0
    return feats


def sigmoid(z: float) -> float:
    if z >= 35:
        return 1.0
    if z <= -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def dot(weights: dict[str, float], feats: dict[str, float]) -> float:
    return sum(weights.get(k, 0.0) * v for k, v in feats.items())


def train_logistic(examples: list[tuple[dict[str, float], int]], args: argparse.Namespace) -> dict[str, float]:
    rng = random.Random(args.seed)
    weights = defaultdict(float)
    for epoch in range(args.epochs):
        rng.shuffle(examples)
        lr = args.learning_rate / (1.0 + 0.2 * epoch)
        for feats, label in examples:
            pred = sigmoid(dot(weights, feats))
            err = float(label) - pred
            pos_weight = 2.0 if label else 1.0
            for k, v in feats.items():
                weights[k] += lr * (pos_weight * err * v - args.l2 * weights[k])
    return dict(weights)


def train_logistic_stream(
    cases: list[dict[str, Any]], ctx: dict[str, Any], med_to_class: dict[str, str], args: argparse.Namespace
) -> tuple[dict[str, float], int, int]:
    """Train on every case without retaining millions of feature dictionaries."""
    rng = random.Random(args.seed)
    weights = defaultdict(float)
    example_count = 0
    positive_count = 0
    indices = list(range(len(cases)))
    for epoch in range(args.epochs):
        rng.shuffle(indices)
        lr = args.learning_rate / (1.0 + 0.2 * epoch)
        for seen, case_idx in enumerate(indices, 1):
            case = cases[case_idx]
            truth = true_meds_norm(case)
            details = candidate_details(case, ctx, args)
            for rank_idx, item in enumerate(details[: args.candidate_depth_train]):
                med_norm = ranker.normalize_drug_name(item.get("medication"))
                label = 1 if med_norm in truth else 0
                feats = features_for_candidate(case, item, rank_idx, med_to_class)
                pred = sigmoid(dot(weights, feats))
                err = float(label) - pred
                pos_weight = 2.0 if label else 1.0
                for key, value in feats.items():
                    weights[key] += lr * (pos_weight * err * value - args.l2 * weights[key])
                if epoch == 0:
                    example_count += 1
                    positive_count += label
            if seen % 5000 == 0:
                print(json.dumps({"phase": "train", "epoch": epoch + 1, "cases": seen, "total": len(cases)}), flush=True)
    return dict(weights), example_count, positive_count


def build_examples(cases: list[dict[str, Any]], ctx: dict[str, Any], med_to_class: dict[str, str], args: argparse.Namespace) -> tuple[list[tuple[dict[str, float], int]], list[dict[str, Any]]]:
    examples = []
    rows = []
    for case in cases:
        truth = true_meds_norm(case)
        details = candidate_details(case, ctx, args)
        for idx, item in enumerate(details[: args.candidate_depth_train]):
            med_norm = ranker.normalize_drug_name(item.get("medication"))
            label = 1 if med_norm in truth else 0
            # Keep positives and strong/ranked negatives; top20 is small enough.
            examples.append((features_for_candidate(case, item, idx, med_to_class), label))
        rows.append({"case": case, "candidate_details": details})
    return examples, rows


def score_details(case: dict[str, Any], details: list[dict[str, Any]], weights: dict[str, float], med_to_class: dict[str, str], depth: int) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate(details[:depth]):
        copied = json.loads(json.dumps(item, ensure_ascii=False))
        feats = features_for_candidate(case, item, idx, med_to_class)
        prob = sigmoid(dot(weights, feats))
        copied["ml_probability"] = round(prob, 6)
        copied["original_rank"] = idx + 1
        copied["ml_features"] = feats
        out.append(copied)
    out.sort(key=lambda x: (-float(x.get("ml_probability", 0.0)), x.get("original_rank", 999), ranker.normalize_drug_name(x.get("medication"))))
    return out


def count_ddi_hits(med_norm: str, selected_norms: list[str], ddi_pairs: set[tuple[str, str]]) -> int:
    return sum(tuple(sorted((med_norm, prev))) in ddi_pairs for prev in selected_norms)


def predict_threshold(
    scored: list[dict[str, Any]], threshold: float, min_k: int, max_k: int,
    class_thresholds: dict[str, float], class_caps: dict[str, int], med_to_class: dict[str, str],
    ddi_pairs: set[tuple[str, str]] | None = None, ddi_penalty: float = 0.0,
    ddi_margin: float = 0.0, ddi_max_hits: int = 99,
) -> tuple[list[str], dict[str, float]]:
    selected_items: list[dict[str, Any]] = []
    selected_norms: list[str] = []
    class_counts = Counter()
    deferred: list[dict[str, Any]] = []
    for item in scored:
        med = item.get("medication")
        norm = ranker.normalize_drug_name(med)
        cls = med_to_class.get(norm, "unknown")
        cap = class_caps.get(cls, max_k)
        if class_counts[cls] >= cap:
            continue
        prob = float(item.get("ml_probability", 0.0))
        th = class_thresholds.get(cls, threshold)
        hits = count_ddi_hits(norm, selected_norms, ddi_pairs or set()) if ddi_pairs else 0
        effective = prob - ddi_penalty * min(hits, ddi_max_hits)
        ddi_blocked = hits > ddi_max_hits or (hits > 0 and prob < th + ddi_margin)
        eligible = effective >= th
        candidate = {**item, "effective_probability": effective, "selection_ddi_hits": hits}
        if ddi_blocked or not eligible:
            deferred.append(candidate)
            continue
        selected_items.append(candidate)
        selected_norms.append(norm)
        class_counts[cls] += 1
        if len(selected_items) >= max_k:
            break
    # Backfill only to the tuned minimum, preserving class caps first.
    if len(selected_items) < min_k:
        chosen = set(selected_norms)
        for item in sorted(deferred, key=lambda x: (-float(x.get("effective_probability", 0.0)), x.get("original_rank", 999))):
            norm = ranker.normalize_drug_name(item.get("medication"))
            cls = med_to_class.get(norm, "unknown")
            if norm in chosen or class_counts[cls] >= class_caps.get(cls, max_k):
                continue
            selected_items.append(item)
            selected_norms.append(norm)
            chosen.add(norm)
            class_counts[cls] += 1
            if len(selected_items) >= min_k:
                break
    selected = [item.get("medication") for item in selected_items]
    scores = {
        ranker.normalize_drug_name(item.get("medication")): float(item.get("effective_probability", item.get("ml_probability", 0.0)))
        for item in selected_items
    }
    return selected, scores


def metric_summary_from_predictions(predictions: list[dict[str, Any]], ctx: dict[str, Any], med_to_class: dict[str, str]) -> dict[str, Any]:
    exact_items = []
    class_items = []
    for row in predictions:
        case = row["case"]
        pred = row["pred"]
        scores = row["scores"]
        exact_items.append({"true_medications": ranker.true_meds(case), "pred": pred, "scores": scores})
        pred_classes = sorted({med_to_class.get(ranker.normalize_drug_name(m), "unknown") for m in pred})
        true_cls = sorted(true_classes(case, med_to_class))
        class_scores = {cls: max((scores.get(ranker.normalize_drug_name(m), 0.0) for m in pred if med_to_class.get(ranker.normalize_drug_name(m), "unknown") == cls), default=0.0) for cls in pred_classes}
        class_items.append({"true_medications": true_cls, "pred": pred_classes, "scores": class_scores})
    return {
        "exact_drug": ranker.metric_summary(exact_items, ctx["vocabulary"], ctx["ddi_pairs"]),
        "class_level": ranker.metric_summary(class_items, set(GENERIC_CLASSES), None),
    }


def prepare_scored_rows(cases: list[dict[str, Any]], weights: dict[str, float], med_to_class: dict[str, str], ctx: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for idx, case in enumerate(cases, 1):
        details = candidate_details(case, ctx, args)
        scored = score_details(case, details, weights, med_to_class, args.candidate_depth_predict)
        compact = [{"medication": x.get("medication"), "ml_probability": x.get("ml_probability"), "original_rank": x.get("original_rank")} for x in scored]
        rows.append({"case": case, "scored": compact})
        if idx % 2000 == 0:
            print(json.dumps({"phase": "score", "cases": idx, "total": len(cases)}), flush=True)
    return rows


def evaluate_scored_rows(
    rows: list[dict[str, Any]], med_to_class: dict[str, str], ctx: dict[str, Any],
    threshold: float, min_k: int, max_k: int, class_thresholds: dict[str, float],
    class_caps: dict[str, int], ddi_cfg: dict[str, Any],
) -> dict[str, Any]:
    predictions = []
    for row in rows:
        pred, scores = predict_threshold(
            row["scored"], threshold, min_k, max_k, class_thresholds, class_caps, med_to_class,
            ctx.get("ddi_pairs"), float(ddi_cfg["penalty"]), float(ddi_cfg["margin"]), int(ddi_cfg["max_hits"]),
        )
        predictions.append({"case": row["case"], "pred": pred, "scores": scores})
    return metric_summary_from_predictions(predictions, ctx, med_to_class)


def tune_global(rows: list[dict[str, Any]], med_to_class: dict[str, str], ctx: dict[str, Any]) -> dict[str, Any]:
    class_caps = {
        "anti_infective": 2, "sedative_analgesic": 2, "anticoagulant": 1,
        "antiplatelet": 1, "bronchodilator": 2, "glucose_control": 2, "fluid_electrolyte": 2,
    }
    baseline = {"penalty": 0.0, "margin": 0.0, "max_hits": 99}
    best = None
    results = []
    for threshold in [0.26, 0.34, 0.42, 0.46, 0.50, 0.55, 0.60]:
        for min_k, max_k in [(3, 6), (4, 6), (4, 7), (5, 7), (5, 8)]:
            metrics = evaluate_scored_rows(rows, med_to_class, ctx, threshold, min_k, max_k, {}, class_caps, baseline)
            exact = metrics["exact_drug"]
            score = exact["jaccard"] + exact["f1"] + exact["prauc_micro"] + exact["safety_adjusted_jaccard"]
            result = {"score": score, "config": {"threshold": threshold, "min_k": min_k, "max_k": max_k}, "metrics": metrics}
            results.append(result)
            if best is None or score > best["score"]:
                best = result
    return {"best": best, "all": results, "class_caps": class_caps}


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    ctx = load_context(args)
    train_cases = read_cases(args.train_file, args.train_cases)
    val_cases = read_cases(args.validation_file, args.validation_cases)
    test_cases = read_cases(args.test_file, args.test_cases)
    med_to_class = collect_med_classes(train_cases + val_cases, ctx)
    weights, example_count, positive_count = train_logistic_stream(train_cases, ctx, med_to_class, args)
    val_rows = prepare_scored_rows(val_cases, weights, med_to_class, ctx, args)
    tuning = tune_global(val_rows, med_to_class, ctx)
    cfg = tuning["best"]["config"]
    class_caps = tuning["class_caps"]
    test_rows = prepare_scored_rows(test_cases, weights, med_to_class, ctx, args)
    full_cases = read_cases(args.full_test_file, args.full_test_cases)
    full_rows = prepare_scored_rows(full_cases, weights, med_to_class, ctx, args)
    ddi_profiles = {
        "baseline_no_selection_ddi": {"penalty": 0.0, "margin": 0.0, "max_hits": 99},
        "mild_ddi": {"penalty": 0.03, "margin": 0.02, "max_hits": 2},
        "balanced_ddi": {"penalty": 0.08, "margin": 0.05, "max_hits": 2},
        "strong_ddi": {"penalty": 0.10, "margin": 0.06, "max_hits": 1},
    }
    test100_ablation = {}
    full_test_ablation = {}
    for name, ddi_cfg in ddi_profiles.items():
        test100_ablation[name] = evaluate_scored_rows(test_rows, med_to_class, ctx, cfg["threshold"], cfg["min_k"], cfg["max_k"], {}, class_caps, ddi_cfg)
        full_test_ablation[name] = evaluate_scored_rows(full_rows, med_to_class, ctx, cfg["threshold"], cfg["min_k"], cfg["max_k"], {}, class_caps, ddi_cfg)
    report = {
        "step": "full_lightweight_reranker_ddi_ablation_v11",
        "data": {"train_cases": len(train_cases), "validation_cases": len(val_cases), "test100_cases": len(test_cases), "full_test_cases": len(full_cases)},
        "training": {"epochs": args.epochs, "candidate_depth_train": args.candidate_depth_train, "candidate_depth_predict": args.candidate_depth_predict, "num_training_examples": example_count, "positive_training_examples": positive_count, "patient_history_available": False, "current_treatment_context_proxy": "explicit medication mentions in dx/px/lab text"},
        "fixed_issues": ["class caps apply before min_k", "scores map to actually selected medications", "train and prediction candidate depth both 30", "streaming full-training implementation"],
        "best_validation_config": cfg,
        "best_validation_metrics": tuning["best"]["metrics"],
        "class_caps": class_caps,
        "ddi_profiles": ddi_profiles,
        "test100_ablation": test100_ablation,
        "full_test_ablation": full_test_ablation,
        "top_positive_weights": sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:30],
        "top_negative_weights": sorted(weights.items(), key=lambda kv: kv[1])[:30],
    }
    (args.output_dir / "reranker_weights.json").write_text(json.dumps(weights, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "medication_class_map.json").write_text(json.dumps(med_to_class, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "validation_tuning_grid.json").write_text(json.dumps(tuning["all"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "full_reranker_ddi_ablation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
