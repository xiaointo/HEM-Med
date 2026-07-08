#!/usr/bin/env python3
"""Generate medication candidates from L1/L2 memories and optional priors."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from utils import case_anchor_keys, hospital_id, normalize_drug_name, read_json, read_jsonl, write_jsonl


def add_score(scores: Counter[str], med: str, value: float) -> None:
    med = normalize_drug_name(med)
    if med:
        scores[med] += value


def candidates_for_case(case: dict[str, Any], l1: dict[str, Any], l2: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    scores: Counter[str] = Counter()
    sources: dict[str, list[dict[str, Any]]] = {}
    anchors = case_anchor_keys(case)
    hid = hospital_id(case)

    for anchor in anchors:
        for med, payload in (l1.get(anchor) or {}).items():
            support = float(payload.get("support", 1) if isinstance(payload, dict) else 1)
            add_score(scores, med, support)
            sources.setdefault(normalize_drug_name(med), []).append({"source": "L1", "anchor_key": anchor, "support": support})
        for med, payload in ((l2.get(hid) or {}).get(anchor) or {}).items():
            support = float(payload.get("support", 1) if isinstance(payload, dict) else 1)
            add_score(scores, med, 1.25 * support)
            sources.setdefault(normalize_drug_name(med), []).append({"source": "L2", "anchor_key": anchor, "support": support})

    return [
        {"medication": med, "score": score, "sources": sources.get(med, [])}
        for med, score in scores.most_common(top_k)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSONL cases")
    parser.add_argument("--l1", required=True, type=Path)
    parser.add_argument("--l2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=60)
    args = parser.parse_args()

    l1 = read_json(args.l1)
    l2 = read_json(args.l2)
    rows = []
    for case in read_jsonl(args.input):
        rows.append({"case": case, "candidates": candidates_for_case(case, l1, l2, args.top_k)})
    write_jsonl(args.output, rows)
    print(f"wrote candidates: {args.output} cases={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
