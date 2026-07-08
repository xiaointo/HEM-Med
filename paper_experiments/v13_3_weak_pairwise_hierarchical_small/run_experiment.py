#!/usr/bin/env python3
"""Small-data v13.3 ablation with weak memory, pairwise ranking and set decoding."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("HEM_MED_DATA_DIR", ROOT.parent / "data"))
BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs_test300"
OUT.mkdir(parents=True, exist_ok=True)
V13_FULL = ROOT / "memory_and_reranking/have_anchor_L1L2_score_calibrated_v13_3_full/run_v13_3_full.py"
spec = importlib.util.spec_from_file_location("v13_full", V13_FULL)
v13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v13)
v12, v11 = v13.v12, v13.v11

TRAIN_N, VALID_N, TEST_N, SEED = 3000, 500, 300, 1203
WEAK_L1 = DATA_ROOT / "memory/L1_statistical_memory_final_cleaned_with_weak_candidates.json"
WEAK_L2 = DATA_ROOT / "memory/L2_final_merged_memory.json"
DDI_DESC = DATA_ROOT / "ddi/eicu_space_ddi_pairs_simple.csv"
HIGH_FP = {"aspirin", "heparin", "insulin", "sodium chloride", "vancomycin", "enoxaparin"}
LOW_RECALL = {"clopidogrel", "warfarin", "lorazepam", "glucagon", "hydralazine"}
CANDIDATE_CACHE = {}


def sigmoid(x):
    if x > 35: return 1.0
    if x < -35: return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def dot(w, x):
    return sum(w.get(k, 0.0) * value for k, value in x.items())


def sample_cases(path, n, seed):
    rows = v11.read_cases(path, None)
    return random.Random(seed).sample(rows, n)


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_context(args):
    ctx = v11.load_context(args)
    weak_l1 = json.loads(WEAK_L1.read_text(encoding="utf-8"))
    weak_l2 = json.loads(WEAK_L2.read_text(encoding="utf-8"))
    ctx["weak_l1"] = weak_l1
    ctx["weak_l2"] = weak_l2
    ctx["weak_memory_anchor_index"] = v11.ranker.build_memory_anchor_index(weak_l1, weak_l2)
    return ctx


def auxiliary_context(ctx):
    aux = dict(ctx)
    aux["l1"], aux["l2"] = ctx["weak_l1"], ctx["weak_l2"]
    aux["memory_anchor_index"] = ctx["weak_memory_anchor_index"]
    return aux


def relabel_weak_sources(item):
    copied = deepcopy(item)
    copied["score"] = float(copied.get("score", 0.0)) * 0.42
    copied["raw_score"] = float(copied.get("raw_score", copied["score"])) * 0.42
    for src in copied.get("sources") or []:
        src["source"] = "weak_" + str(src.get("source") or "memory")
    return copied


def candidate_details(case, ctx, args, include_weak):
    cache_key = (id(case), include_weak)
    if cache_key in CANDIDATE_CACHE:
        return CANDIDATE_CACHE[cache_key]
    primary = v11.candidate_details(case, ctx, args)
    if not include_weak:
        CANDIDATE_CACHE[cache_key] = primary[:60]
        return CANDIDATE_CACHE[cache_key]
    weak = v11.candidate_details(case, auxiliary_context(ctx), args)
    merged = {v11.ranker.normalize_drug_name(x.get("medication")): deepcopy(x) for x in primary}
    for raw in weak:
        item = relabel_weak_sources(raw)
        med = v11.ranker.normalize_drug_name(item.get("medication"))
        if not med: continue
        if med in merged:
            merged[med].setdefault("sources", []).extend(item.get("sources") or [])
            merged[med]["weak_score"] = max(float(merged[med].get("weak_score", 0.0)), float(item.get("score", 0.0)))
        else:
            item["weak_only"] = True
            merged[med] = item
    CANDIDATE_CACHE[cache_key] = sorted(merged.values(), key=lambda x: (-float(x.get("score", 0.0)) - float(x.get("weak_score", 0.0)), v11.ranker.normalize_drug_name(x.get("medication"))))[:60]
    return CANDIDATE_CACHE[cache_key]


def features(case, item, rank, med_class):
    x = v11.features_for_candidate(case, item, rank, med_class)
    med = v11.ranker.normalize_drug_name(item.get("medication"))
    sources = item.get("sources") or []
    weak = [s for s in sources if str(s.get("source", "")).startswith("weak_")]
    x["weak_evidence_count"] = min(len(weak), 6) / 6.0
    x["weak_support_log"] = math.log1p(sum(float(s.get("support") or 0) for s in weak)) / 10.0
    x["weak_frequency_max"] = max([float(s.get("frequency") or 0) for s in weak] or [0.0])
    x["weak_only"] = float(bool(item.get("weak_only")))
    x["med::" + med] = 1.0
    return x


def subtract(a, b):
    keys = set(a) | set(b)
    return {k: a.get(k, 0.0) - b.get(k, 0.0) for k in keys}


def train_pointwise(cases, ctx, med_class, args, include_weak):
    w = defaultdict(float); support = Counter()
    for case in cases: support.update(v11.true_meds_norm(case))
    rng = random.Random(SEED)
    order = list(range(len(cases)))
    for epoch in range(3):
        rng.shuffle(order); lr = 0.035 / (1 + 0.25 * epoch)
        for index in order:
            case = cases[index]; truth = v11.true_meds_norm(case)
            for rank, item in enumerate(candidate_details(case, ctx, args, include_weak)):
                med = v11.ranker.normalize_drug_name(item.get("medication")); y = float(med in truth)
                x = features(case, item, rank, med_class); p = sigmoid(dot(w, x))
                pos_weight = min(5.0, 1.5 + 12.0 / math.sqrt(max(1, support[med]))) if y else 1.0
                fp_cost = 1.65 if med in HIGH_FP and not y else 1.0
                scale = pos_weight if y else fp_cost
                for k, value in x.items(): w[k] += lr * (scale * (y - p) * value - 0.0002 * w[k])
    return dict(w), support


def train_pairwise(cases, ctx, med_class, args, include_weak, support):
    w = defaultdict(float); rng = random.Random(SEED + 1); pairs = Counter()
    order = list(range(len(cases)))
    for epoch in range(4):
        rng.shuffle(order); lr = 0.028 / (1 + 0.25 * epoch)
        for index in order:
            case = cases[index]; truth = v11.true_meds_norm(case)
            details = candidate_details(case, ctx, args, include_weak)
            rows = [(v11.ranker.normalize_drug_name(item.get("medication")), features(case, item, rank, med_class), rank) for rank, item in enumerate(details)]
            positives = [r for r in rows if r[0] in truth]
            negatives = [r for r in rows if r[0] not in truth]
            for med_p, xp, _ in positives:
                cls = med_class.get(med_p, "unknown")
                same = [r for r in negatives if med_class.get(r[0], "unknown") == cls][:3]
                hard = negatives[:4] + [r for r in negatives if r[0] in HIGH_FP][:2] + same
                seen = set()
                for med_n, xn, rank_n in hard:
                    if med_n in seen: continue
                    seen.add(med_n); diff = subtract(xp, xn); margin = dot(w, diff); grad = 1.0 - sigmoid(margin)
                    rare_boost = min(3.0, 1.0 + 8.0 / math.sqrt(max(1, support[med_p])))
                    for k, value in diff.items(): w[k] += lr * (rare_boost * grad * value - 0.00025 * w[k])
                    pairs["total"] += 1
                    if med_class.get(med_n, "unknown") == cls: pairs["same_class"] += 1
                    if rank_n < 10: pairs["top_ranked"] += 1
                    if med_n in HIGH_FP: pairs["high_fp"] += 1
    return dict(w), dict(pairs)


def train_class_model(cases, ctx, med_class, args, include_weak):
    w = defaultdict(float); rng = random.Random(SEED + 2); order = list(range(len(cases)))
    for epoch in range(3):
        rng.shuffle(order); lr = 0.025 / (1 + .25 * epoch)
        for index in order:
            case = cases[index]; true_cls = v11.true_classes(case, med_class); best = {}
            for rank, item in enumerate(candidate_details(case, ctx, args, include_weak)):
                med = v11.ranker.normalize_drug_name(item.get("medication")); cls = med_class.get(med, "unknown")
                if cls not in best: best[cls] = features(case, item, rank, med_class)
            for cls, base_x in best.items():
                x = dict(base_x); x["target_class::" + cls] = 1.0; y = float(cls in true_cls); p = sigmoid(dot(w, x))
                for k, value in x.items(): w[k] += lr * ((2.0 if y else 1.0) * (y - p) * value - .0002 * w[k])
    return dict(w)


def count_x(case):
    return v12.count_features(case)


def train_count_classifier(cases):
    weights = {k: defaultdict(float) for k in range(1, 9)}; rng = random.Random(SEED + 3); order = list(range(len(cases)))
    for epoch in range(8):
        rng.shuffle(order); lr = .035 / (1 + .2 * epoch)
        for index in order:
            case = cases[index]; x = count_x(case); target = min(8, max(1, len(v11.true_meds_norm(case))))
            logits = {k: dot(weights[k], x) for k in weights}; peak = max(logits.values()); z = sum(math.exp(v - peak) for v in logits.values())
            probs = {k: math.exp(v - peak) / z for k, v in logits.items()}
            for k in weights:
                err = float(k == target) - probs[k]
                for name, value in x.items(): weights[k][name] += lr * (err * value - .0002 * weights[k][name])
    return {str(k): dict(v) for k, v in weights.items()}


def count_probs(case, weights):
    x = count_x(case); logits = {int(k): dot(w, x) for k, w in weights.items()}; peak = max(logits.values()); z = sum(math.exp(v - peak) for v in logits.values())
    return {k: math.exp(v - peak) / z for k, v in logits.items()}


def score_cases(cases, ctx, med_class, args, point_w, pair_w, class_w, include_weak):
    rows = []
    for case in cases:
        scored = []; class_cache = {}
        details = candidate_details(case, ctx, args, include_weak)
        for rank, item in enumerate(details):
            med = v11.ranker.normalize_drug_name(item.get("medication")); cls = med_class.get(med, "unknown"); x = features(case, item, rank, med_class)
            if cls not in class_cache:
                cx = dict(x); cx["target_class::" + cls] = 1.0; class_cache[cls] = sigmoid(dot(class_w, cx))
            point = sigmoid(dot(point_w, x)); pair = sigmoid(dot(pair_w, x))
            probability = (point ** .48) * (pair ** .27) * (class_cache[cls] ** .25)
            scored.append({"medication": item.get("medication"), "probability": probability, "raw_probability": probability, "class_probability": class_cache[cls], "point_probability": point, "pair_probability": pair, "rank": rank + 1, "weak_evidence_count": x["weak_evidence_count"]})
        scored.sort(key=lambda x: (-x["probability"], x["rank"]))
        rows.append({"case": case, "scored": scored})
    return rows


def fit_platt(rows):
    by_med = defaultdict(list)
    for row in rows:
        truth = v11.true_meds_norm(row["case"])
        for item in row["scored"]: by_med[v11.ranker.normalize_drug_name(item["medication"])].append((item["probability"], int(v11.ranker.normalize_drug_name(item["medication"]) in truth)))
    params = {}
    for med, vals in by_med.items():
        pos = sum(y for _, y in vals); neg = len(vals) - pos
        if pos < 3 or neg < 8: params[med] = {"a": 1.0, "b": 0.0, "mode": "identity", "pos": pos}; continue
        a, b = 1.0, 0.0
        for _ in range(160):
            ga = gb = 0.0
            for p, y in vals:
                logit = math.log(max(1e-5, p) / max(1e-5, 1 - p)); pred = sigmoid(a * logit + b); ga += (pred - y) * logit; gb += pred - y
            a -= .04 * (ga / len(vals) + .02 * (a - 1)); b -= .04 * (gb / len(vals) + .02 * b)
            a = max(.25, min(3.0, a)); b = max(-4, min(4, b))
        params[med] = {"a": a, "b": b, "mode": "platt", "pos": pos}
    return params


def apply_platt(rows, params):
    for row in rows:
        for item in row["scored"]:
            med = v11.ranker.normalize_drug_name(item["medication"]); par = params.get(med, {"a": 1, "b": 0}); p = item["probability"]
            logit = math.log(max(1e-5, p) / max(1e-5, 1 - p)); item["probability"] = .55 * p + .45 * sigmoid(par["a"] * logit + par["b"])
        row["scored"].sort(key=lambda x: (-x["probability"], x["rank"]))


def ddi_severity():
    severity = {}
    with DDI_DESC.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            a = v11.ranker.normalize_drug_name(row["drug_name_1"]); b = v11.ranker.normalize_drug_name(row["drug_name_2"]); text = row["interaction_description"].lower()
            value = 1.0 if any(x in text for x in ("death", "fatal", "contraind", "hemorrhage", "serotonin syndrome")) else .65 if any(x in text for x in ("risk or severity", "toxicity", "qt", "bleeding")) else .35
            severity[tuple(sorted((a, b)))] = value
    return severity


def coadmin_confidence(cases, ddi_pairs):
    med_n = Counter(); pair_n = Counter()
    for case in cases:
        meds = sorted(v11.true_meds_norm(case)); med_n.update(meds)
        for i, a in enumerate(meds):
            for b in meds[i + 1:]:
                pair = tuple(sorted((a, b)))
                if pair in ddi_pairs: pair_n[pair] += 1
    return {pair: n / max(1, min(med_n[pair[0]], med_n[pair[1]])) for pair, n in pair_n.items()}


def beam_decode(row, med_class, count_w, ddi_pairs, severity, coadmin, ddi_scale):
    candidates = row["scored"][:18]
    count_p = count_probs(row["case"], count_w)
    target_sizes = {k for k, _ in sorted(count_p.items(), key=lambda x: x[1], reverse=True)[:2]}
    max_target = max(target_sizes)
    beams = {0: [(0.0, tuple(), Counter())]}
    for item in candidates:
        med = v11.ranker.normalize_drug_name(item["medication"]); cls = med_class.get(med, "unknown")
        expanded = {size: list(states) for size, states in beams.items()}
        for size, states in beams.items():
            if size >= max_target: continue
            for score, chosen, cls_n in states:
                if med in chosen: continue
                pair_penalty = 0.0
                for prev in chosen:
                    pair = tuple(sorted((med, prev)))
                    if pair in ddi_pairs: pair_penalty += ddi_scale * severity.get(pair, .5) * (1.0 - min(.85, coadmin.get(pair, 0.0)))
                duplicate = .055 * max(0, cls_n[cls] - 1)
                new_counts = cls_n.copy(); new_counts[cls] += 1
                expanded.setdefault(size + 1, []).append((score + math.log(max(item["probability"], 1e-6)) - pair_penalty - duplicate, chosen + (med,), new_counts))
        beams = {size: sorted(states, key=lambda x: x[0], reverse=True)[:30] for size, states in expanded.items()}
    finals = []
    for size in target_sizes:
        for score, chosen, cls_n in beams.get(size, []):
            finals.append((score + .65 * math.log(max(count_p.get(size, 1e-6), 1e-6)), chosen))
    chosen = max(finals, key=lambda x: x[0])[1]
    display = {v11.ranker.normalize_drug_name(x["medication"]): x for x in candidates}
    return [display[m]["medication"] for m in chosen], {m: display[m]["probability"] for m in chosen}


def topk_decode(row, count_w):
    k = max(count_probs(row["case"], count_w), key=count_probs(row["case"], count_w).get)
    chosen = row["scored"][:k]
    return [x["medication"] for x in chosen], {v11.ranker.normalize_drug_name(x["medication"]): x["probability"] for x in chosen}


def evaluate(rows, ctx, med_class, decoder):
    predictions = []; labels = []; scores = []; case_ap = []
    vocab = sorted(ctx["vocabulary"])
    for row in rows:
        pred, selected_scores = decoder(row); predictions.append({"case": row["case"], "pred": pred, "scores": selected_scores})
        truth = v11.true_meds_norm(row["case"]); all_scores = {v11.ranker.normalize_drug_name(x["medication"]): x["probability"] for x in row["scored"]}
        y = [int(m in truth) for m in vocab]; p = [all_scores.get(m, 0.0) for m in vocab]; labels.extend(y); scores.extend(p); case_ap.append(v11.ranker.average_precision(y, p))
    metrics = v11.metric_summary_from_predictions(predictions, ctx, med_class)["exact_drug"]
    metrics["drug_level_micro_prauc"] = v11.ranker.average_precision(labels, scores)
    metrics["case_level_mean_prauc"] = sum(case_ap) / len(case_ap)
    return metrics, predictions, case_ap


def main():
    args = v11.parse_args(); args.candidate_depth_train = 60; args.candidate_depth_predict = 60; args.max_candidate_meds = 60; args.epochs = 3
    ctx = load_context(args)
    train = sample_cases(args.train_file, TRAIN_N, SEED)
    valid = sample_cases(args.validation_file, VALID_N, SEED + 1)
    test = sample_cases(args.full_test_file, TEST_N, SEED + 2)
    med_class = v11.collect_med_classes(train + valid, ctx)
    # Matched small-data v13.3-style pointwise baseline.
    base_w, _ = train_pointwise(train, ctx, med_class, args, False)
    base_class = train_class_model(train, ctx, med_class, args, False)
    count_w = train_count_classifier(train)
    base_val = score_cases(valid, ctx, med_class, args, base_w, {}, base_class, False)
    base_test = score_cases(test, ctx, med_class, args, base_w, {}, base_class, False)
    base_platt = fit_platt(base_val); apply_platt(base_test, base_platt)
    baseline_metrics, _, _ = evaluate(base_test, ctx, med_class, lambda row: topk_decode(row, count_w))
    # Enhanced weak-evidence + pointwise + pairwise + two-stage model.
    point_w, support = train_pointwise(train, ctx, med_class, args, True)
    pair_w, pair_audit = train_pairwise(train, ctx, med_class, args, True, support)
    class_w = train_class_model(train, ctx, med_class, args, True)
    val_rows = score_cases(valid, ctx, med_class, args, point_w, pair_w, class_w, True)
    test_rows = score_cases(test, ctx, med_class, args, point_w, pair_w, class_w, True)
    platt = fit_platt(val_rows); apply_platt(test_rows, platt)
    severity = ddi_severity(); coadmin = coadmin_confidence(train, ctx["ddi_pairs"])
    profiles = {"no_ddi": 0.0, "mild": .12, "balanced": .22}
    results = {}
    audits = {}
    for name, scale in profiles.items():
        decoder = lambda row, s=scale: beam_decode(row, med_class, count_w, ctx["ddi_pairs"], severity, coadmin, s)
        metrics, predictions, case_ap = evaluate(test_rows, ctx, med_class, decoder)
        results[name] = metrics
        audits[name] = predictions
    report = {
        "experiment": "v13.3 small-data weak-evidence pairwise hierarchical ablation",
        "data": {"train": TRAIN_N, "validation": VALID_N, "test": TEST_N, "seed": SEED, "sampling": "seeded random without replacement from official splits"},
        "baseline": {"description": "matched 3000-case pointwise small-data baseline with count classifier and per-drug Platt", "metrics": baseline_metrics},
        "enhancements": ["decoupled primary L1/L2 plus attenuated weak-memory auxiliary channel", "pairwise logistic ranking with same-class/top-ranked/high-FP hard negatives", "P(class|patient) x P(drug|class,patient) geometric score", "per-drug Platt calibration on 500 validation cases", "multiclass P(K=1..8) medication-count model", "DDI severity and empirical coadministration calibrated beam-search set decoder", "high-FP drug negative cost and low-frequency positive weighting"],
        "pairwise_training": pair_audit,
        "profiles": profiles,
        "results": results,
        "note": "PR-AUC uses full canonical vocabulary; non-candidate drugs receive score 0. Accuracy is intentionally not reported."
    }
    write_json(OUT / "experiment_report.json", report)
    write_json(OUT / "pointwise_weights.json", point_w); write_json(OUT / "pairwise_weights.json", pair_w); write_json(OUT / "class_model_weights.json", class_w)
    write_json(OUT / "count_classifier_weights.json", count_w); write_json(OUT / "per_drug_platt_parameters.json", platt); write_json(OUT / "medication_class_map.json", med_class)
    with (OUT / "test_300_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in test: handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    with (OUT / "per_case_predictions_balanced.jsonl").open("w", encoding="utf-8") as handle:
        for row in audits["balanced"]: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
