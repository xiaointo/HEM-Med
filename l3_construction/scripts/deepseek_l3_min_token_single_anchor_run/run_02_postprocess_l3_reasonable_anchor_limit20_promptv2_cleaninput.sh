#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../../.." && pwd)"
python3 scripts/deepseek_l3_min_token_single_anchor_run/02_postprocess_l3_reasonable_anchor.py \
  --input-file outputs/deepseek_l3_min_token_single_anchor_run/L3_deepseek_generated_limit20_promptv2_cleaninput.jsonl \
  --output-file outputs/deepseek_l3_min_token_single_anchor_run/L3_deepseek_generated_limit20_promptv2_cleaninput_cleaned.jsonl \
  --audit-file outputs/deepseek_l3_min_token_single_anchor_run/anchor_postprocess_audit_limit20_promptv2_cleaninput.csv \
  --report-file outputs/deepseek_l3_min_token_single_anchor_run/run_report_limit20_promptv2_cleaninput_cleaned.json \
  --high-risk-file outputs/deepseek_l3_min_token_single_anchor_run/high_risk_anchor_records_limit20_promptv2_cleaninput_cleaned.jsonl \
  --null-anchor-file outputs/deepseek_l3_min_token_single_anchor_run/null_anchor_records_limit20_promptv2_cleaninput_cleaned.jsonl \
  --unsafe-hypothesis-file outputs/deepseek_l3_min_token_single_anchor_run/unsafe_hypothesis_records_limit20_promptv2_cleaninput_cleaned.jsonl
