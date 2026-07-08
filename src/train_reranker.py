#!/usr/bin/env python3
"""Train a lightweight pointwise reranker over generated medication candidates."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from utils import normalize_drug_name, read_jsonl, true_meds_norm, write_json


def sigmoid(x: float) -> float:
    if x > 35:
        return 1.0
    if x < -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def features(item: dict[str, Any], rank: int) -> dict[str, float]:
    sources = item.get("sources") or []
    feats = {
        "bias": 1.0,
        "score_log": math.log1p(max(float(item.get("score", 0.0)), 0.0)),
        "rank_inv": 1.0 / (rank + 1),
        "rank_top5": 1.0 if rank < 5 else 0.0,
        "src_l1": sum(1 for s in sources if s.get("source") == "L1") / 5.0,
        "src_l2": sum(1 for s in sources if s.get("source") == "L2") / 5.0,
    }
    return feats


def dot(weights: dict[str, float], feats: dict[str, float]) -> float:
    return sum(weights.get(k, 0.0) * v for k, v in feats.items())


def train(rows: list[dict[str, Any]], epochs: int, lr: float, seed: int) -> dict[str, float]:
    weights: dict[str, float] = defaultdict(float)
    rng = random.Random(seed)
    order = list(range(len(rows)))
    for epoch in range(epochs):
        rng.shuffle(order)
        step = lr / (1.0 + 0.2 * epoch)
        for idx in order:
            row = rows[idx]
            truth = true_meds_norm(row["case"])
            for rank, item in enumerate(row.get("candidates") or []):
                med = normalize_drug_name(item.get("medication"))
                y = 1.0 if med in truth else 0.0
                x = features(item, rank)
                err = y - sigmoid(dot(weights, x))
                scale = 2.0 if y else 1.0
                for name, value in x.items():
                    weights[name] += step * (scale * err * value - 0.0002 * weights[name])
    return dict(weights)


def score_rows(rows: list[dict[str, Any]], weights: dict[str, float]) -> list[dict[str, Any]]:
    scored_rows = []
    for row in rows:
        scored = []
        for rank, item in enumerate(row.get("candidates") or []):
            p = sigmoid(dot(weights, features(item, rank)))
            scored.append({**item, "probability": p, "rank": rank + 1})
        scored.sort(key=lambda x: (-x["probability"], x["rank"]))
        scored_rows.append({"case": row["case"], "scored": scored})
    return scored_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-candidates", required=True, type=Path)
    parser.add_argument("--validation-candidates", type=Path)
    parser.add_argument("--weights-out", required=True, type=Path)
    parser.add_argument("--scored-out", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=1203)
    args = parser.parse_args()

    train_rows = read_jsonl(args.train_candidates)
    weights = train(train_rows, args.epochs, args.learning_rate, args.seed)
    write_json(args.weights_out, weights)
    if args.scored_out:
        rows = read_jsonl(args.validation_candidates or args.train_candidates)
        args.scored_out.parent.mkdir(parents=True, exist_ok=True)
        with args.scored_out.open("w", encoding="utf-8") as handle:
            for row in score_rows(rows, weights):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote reranker weights: {args.weights_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
