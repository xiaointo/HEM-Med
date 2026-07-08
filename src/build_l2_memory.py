#!/usr/bin/env python3
"""Build a compact hospital-specific L2 residual memory from train JSONL records."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from utils import case_anchor_keys, hospital_id, normalize_drug_name, read_jsonl, top_counts, true_meds, write_json


def build_l2(cases: list[dict[str, Any]], min_support: int, top_k: int) -> dict[str, dict[str, dict[str, Any]]]:
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for case in cases:
        hid = hospital_id(case)
        anchors = case_anchor_keys(case)
        meds = {normalize_drug_name(m) for m in true_meds(case) if normalize_drug_name(m)}
        for anchor in anchors:
            counts[hid][anchor].update(meds)
    memory: dict[str, dict[str, dict[str, Any]]] = {}
    for hid, anchor_map in counts.items():
        hospital_payload: dict[str, dict[str, Any]] = {}
        for anchor, counter in anchor_map.items():
            rows = [row for row in top_counts(counter, top_k) if row["support"] >= min_support]
            if rows:
                hospital_payload[anchor] = {
                    row["medication"]: {**row, "residual_type": "hospital_specific"}
                    for row in rows
                }
        if hospital_payload:
            memory[hid] = hospital_payload
    return memory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    cases = read_jsonl(args.train_file)
    memory = build_l2(cases, args.min_support, args.top_k)
    write_json(args.output, memory)
    print(f"wrote L2 memory: {args.output} hospitals={len(memory)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
