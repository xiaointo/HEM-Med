#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../../.." && pwd)"
python3 scripts/deepseek_l3_min_token_single_anchor_run/03_recover_l3_from_raw_responses.py \
  --raw-responses outputs/deepseek_l3_min_token_single_anchor_run/raw_responses_limit100_promptv2_cleaninput.jsonl \
  --input-file outputs/deepseek_l3_minimal_case_inputs/stay_level_all_minimal_for_llm_dx_px_nonempty.jsonl \
  --output-file outputs/deepseek_l3_min_token_single_anchor_run/L3_deepseek_generated_limit100_promptv2_cleaninput_recovered.jsonl \
  --schema-error-file outputs/deepseek_l3_min_token_single_anchor_run/schema_error_records_limit100_promptv2_cleaninput_recovered.jsonl \
  --illegal-anchor-file outputs/deepseek_l3_min_token_single_anchor_run/illegal_anchor_records_limit100_promptv2_cleaninput_recovered.jsonl \
  --null-anchor-file outputs/deepseek_l3_min_token_single_anchor_run/null_anchor_records_limit100_promptv2_cleaninput_recovered.jsonl \
  --report-file outputs/deepseek_l3_min_token_single_anchor_run/run_report_limit100_promptv2_cleaninput_recovered.json
