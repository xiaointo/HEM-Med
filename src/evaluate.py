#!/usr/bin/env python3
"""Evaluate predicted medication sets against JSONL ground truth."""
from __future__ import annotations

import argparse
from pathlib import Path

from utils import ddi_rate, load_ddi_pairs, read_jsonl, set_metrics, true_meds_norm, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ddi", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    ddi_pairs = load_ddi_pairs(args.ddi) if args.ddi else set()
    totals = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0, "ddi_rate": 0.0}
    for row in rows:
        truth = true_meds_norm(row["case"])
        pred = set(row.get("pred") or [])
        metrics = set_metrics(truth, pred)
        for key, value in metrics.items():
            totals[key] += value
        totals["ddi_rate"] += ddi_rate(pred, ddi_pairs) if ddi_pairs else 0.0
    n = len(rows) or 1
    report = {key: value / n for key, value in totals.items()}
    report["cases"] = len(rows)
    report["safety_adjusted_jaccard"] = report["jaccard"] * (1.0 - report["ddi_rate"])
    write_json(args.output, report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
