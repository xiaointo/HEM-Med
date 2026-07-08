#!/usr/bin/env python3
"""Run Improved Balanced on the official 72,436/9,054/9,055 split."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import os


BASE = Path(__file__).resolve().parent
SOURCE = BASE.parent / "v13_3_weak_pairwise_hierarchical_small" / "run_experiment.py"
OUT = BASE / "outputs_full_split"
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("balanced_small", SOURCE)
balanced = importlib.util.module_from_spec(spec)
spec.loader.exec_module(balanced)
v11 = balanced.v11


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def score_without_retaining_candidates(cases, *args):
    rows = balanced.score_cases(cases, *args)
    balanced.CANDIDATE_CACHE.clear()
    return rows


def main() -> int:
    args = v11.parse_args()
    args.candidate_depth_train = 60
    args.candidate_depth_predict = 60
    args.max_candidate_meds = 60
    args.epochs = 3

    ctx = balanced.load_context(args)
    train = v11.read_cases(args.train_file, None)
    valid = v11.read_cases(args.validation_file, None)
    test = v11.read_cases(args.full_test_file, None)
    print(json.dumps({"phase": "load", "train": len(train), "validation": len(valid), "test": len(test)}), flush=True)

    med_class = v11.collect_med_classes(train, ctx)
    count_w = balanced.train_count_classifier(train)
    print(json.dumps({"phase": "count_model_complete"}), flush=True)

    point_w, support = balanced.train_pointwise(train, ctx, med_class, args, True)
    print(json.dumps({"phase": "pointwise_complete", "candidate_cache_cases": len(balanced.CANDIDATE_CACHE)}), flush=True)
    pair_w, pair_audit = balanced.train_pairwise(train, ctx, med_class, args, True, support)
    print(json.dumps({"phase": "pairwise_complete", "pair_audit": pair_audit}), flush=True)
    class_w = balanced.train_class_model(train, ctx, med_class, args, True)
    print(json.dumps({"phase": "class_model_complete"}), flush=True)

    # Training feature cache is no longer needed.
    balanced.CANDIDATE_CACHE.clear()

    val_rows = score_without_retaining_candidates(
        valid, ctx, med_class, args, point_w, pair_w, class_w, True
    )
    platt = balanced.fit_platt(val_rows)
    balanced.apply_platt(val_rows, platt)
    print(json.dumps({"phase": "validation_calibration_complete", "cases": len(val_rows), "calibrators": len(platt)}), flush=True)

    test_rows = score_without_retaining_candidates(
        test, ctx, med_class, args, point_w, pair_w, class_w, True
    )
    balanced.apply_platt(test_rows, platt)
    print(json.dumps({"phase": "test_scoring_complete", "cases": len(test_rows)}), flush=True)

    severity = balanced.ddi_severity()
    coadmin = balanced.coadmin_confidence(train, ctx["ddi_pairs"])
    decoder = lambda row: balanced.beam_decode(
        row, med_class, count_w, ctx["ddi_pairs"], severity, coadmin, 0.22
    )
    metrics, predictions, case_ap = balanced.evaluate(test_rows, ctx, med_class, decoder)

    report = {
        "experiment": "HEM-Med Improved Balanced full official split",
        "status": "complete",
        "data": {
            "train": len(train),
            "validation": len(valid),
            "test": len(test),
            "seed": balanced.SEED,
            "train_file": str(args.train_file),
            "validation_file": str(args.validation_file),
            "test_file": str(args.full_test_file),
        },
        "configuration": {
            "candidate_depth": 60,
            "weak_weight": 0.42,
            "pointwise_epochs": 3,
            "pairwise_epochs": 4,
            "class_model_epochs": 3,
            "count_model_epochs": 8,
            "platt_validation_cases": len(valid),
            "beam_candidate_depth": 18,
            "beam_width_per_size": 30,
            "count_target_sizes": 2,
            "count_log_probability_weight": 0.65,
            "class_redundancy_penalty": 0.055,
            "ddi_profile": "balanced",
            "ddi_scale": 0.22,
            "ddi_coadministration_cap": 0.85,
        },
        "pairwise_training": pair_audit,
        "metrics": metrics,
        "case_level_ap": {
            "count": len(case_ap),
            "mean": sum(case_ap) / len(case_ap),
        },
        "memory_provenance_warning": (
            "Current L1/L2 and weak-memory files were built from the pre-existing train-only-memory reconstruction."
        ),
        "accuracy_reported": False,
    }

    write_json(OUT / "full_split_report.json", report)
    write_json(OUT / "pointwise_weights.json", point_w)
    write_json(OUT / "pairwise_weights.json", pair_w)
    write_json(OUT / "class_model_weights.json", class_w)
    write_json(OUT / "count_classifier_weights.json", count_w)
    write_json(OUT / "per_drug_platt_parameters.json", platt)
    write_json(OUT / "medication_class_map.json", med_class)

    with (OUT / "case_level_average_precision.jsonl").open("w", encoding="utf-8") as handle:
        for case, ap in zip(test, case_ap):
            handle.write(json.dumps({"stay_id": case.get("id"), "hospital_id": case.get("hid"), "average_precision": ap}, ensure_ascii=False) + "\n")
    with (OUT / "predictions_balanced.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"phase": "complete", "report": str(OUT / "full_split_report.json"), "metrics": metrics}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
