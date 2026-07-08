#!/usr/bin/env python3
"""Decode final medication sets with a simple DDI-aware beam search."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from utils import load_ddi_pairs, normalize_drug_name, read_jsonl


def decode_one(scored: list[dict[str, Any]], ddi_pairs: set[tuple[str, str]], k: int, ddi_scale: float) -> list[str]:
    beams: list[tuple[float, tuple[str, ...]]] = [(0.0, tuple())]
    for item in scored[: max(20, k * 4)]:
        med = normalize_drug_name(item.get("medication"))
        if not med:
            continue
        expanded = list(beams)
        for score, chosen in beams:
            if len(chosen) >= k or med in chosen:
                continue
            penalty = sum(ddi_scale for prev in chosen if tuple(sorted((med, prev))) in ddi_pairs)
            expanded.append((score + math.log(max(float(item.get("probability", 1e-6)), 1e-6)) - penalty, chosen + (med,)))
        beams = sorted(expanded, key=lambda x: x[0], reverse=True)[:40]
    return list(max((b for b in beams if len(b[1]) <= k), key=lambda x: (len(x[1]), x[0]))[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", required=True, type=Path)
    parser.add_argument("--ddi", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ddi-scale", type=float, default=0.22)
    args = parser.parse_args()

    ddi_pairs = load_ddi_pairs(args.ddi)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in read_jsonl(args.scored):
            pred = decode_one(row.get("scored") or [], ddi_pairs, args.k, args.ddi_scale)
            handle.write(json.dumps({"case": row["case"], "pred": pred}, ensure_ascii=False) + "\n")
    print(f"wrote decoded predictions: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
