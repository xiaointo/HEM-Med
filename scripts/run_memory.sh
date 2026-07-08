#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${HEM_MED_DATA_DIR:-$ROOT/data}"
SPLIT="$DATA_ROOT/splits/split_8_1_1_nonempty_med_seed1203"
MEM="$DATA_ROOT/memory"
mkdir -p "$MEM"
python3 "$ROOT/src/build_l1_memory.py" \
  --train-file "$SPLIT/train_80pct.jsonl" \
  --output "$MEM/L1_statistical_memory_final_cleaned_with_weak_candidates.json"
python3 "$ROOT/src/build_l2_memory.py" \
  --train-file "$SPLIT/train_80pct.jsonl" \
  --output "$MEM/L2_final_merged_memory.json"
