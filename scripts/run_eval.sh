#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${HEM_MED_DATA_DIR:-$ROOT/data}"
OUT="$ROOT/outputs"
SPLIT="$DATA_ROOT/splits/split_8_1_1_nonempty_med_seed1203"
MEM="$DATA_ROOT/memory"
DDI="$DATA_ROOT/ddi/eicu_space_ddi_pairs_simple.csv"
mkdir -p "$OUT"
python3 "$ROOT/src/candidate_generation.py" \
  --input "$SPLIT/test_10pct.jsonl" \
  --l1 "$MEM/L1_statistical_memory_final_cleaned_with_weak_candidates.json" \
  --l2 "$MEM/L2_final_merged_memory.json" \
  --output "$OUT/test_candidates.jsonl"
python3 "$ROOT/src/train_reranker.py" \
  --train-candidates "$OUT/test_candidates.jsonl" \
  --weights-out "$OUT/tmp_eval_weights.json" \
  --scored-out "$OUT/test_scored.jsonl"
python3 "$ROOT/src/ddi_decoding.py" \
  --scored "$OUT/test_scored.jsonl" \
  --ddi "$DDI" \
  --output "$OUT/predictions.jsonl"
python3 "$ROOT/src/evaluate.py" \
  --predictions "$OUT/predictions.jsonl" \
  --ddi "$DDI" \
  --output "$OUT/metrics.json"
