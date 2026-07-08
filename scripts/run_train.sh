#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${HEM_MED_DATA_DIR:-$ROOT/data}"
OUT="$ROOT/outputs"
SPLIT="$DATA_ROOT/splits/split_8_1_1_nonempty_med_seed1203"
MEM="$DATA_ROOT/memory"
mkdir -p "$OUT"
python3 "$ROOT/src/candidate_generation.py" \
  --input "$SPLIT/train_80pct.jsonl" \
  --l1 "$MEM/L1_statistical_memory_final_cleaned_with_weak_candidates.json" \
  --l2 "$MEM/L2_final_merged_memory.json" \
  --output "$OUT/train_candidates.jsonl"
python3 "$ROOT/src/candidate_generation.py" \
  --input "$SPLIT/validation_10pct.jsonl" \
  --l1 "$MEM/L1_statistical_memory_final_cleaned_with_weak_candidates.json" \
  --l2 "$MEM/L2_final_merged_memory.json" \
  --output "$OUT/validation_candidates.jsonl"
python3 "$ROOT/src/train_reranker.py" \
  --validation-candidates "$OUT/validation_candidates.jsonl" \
  --weights-out "$OUT/reranker_weights.json" \
  --scored-out "$OUT/validation_scored.jsonl"
