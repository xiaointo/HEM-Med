#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../../.." && pwd)"
python3 scripts/deepseek_l3_case_level_generation/14_build_minimal_case_input_for_llm.py \
  --input outputs/stage1_preprocess_final/stay_level_all_detailed.jsonl \
  --output-dir outputs/deepseek_l3_minimal_case_inputs \
  --max-dx 12 \
  --max-px 8 \
  --max-lab 8 \
  --max-med 5
