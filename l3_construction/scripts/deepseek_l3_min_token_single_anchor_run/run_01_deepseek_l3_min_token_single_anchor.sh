#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../../.." && pwd)"
if [[ ! -f configs/private/deepseek_key.py ]]; then
  mkdir -p configs/private
  cp configs/private/deepseek_key_template.py configs/private/deepseek_key.py
  echo "Created configs/private/deepseek_key.py from template."
  echo "Please edit configs/private/deepseek_key.py and fill your DeepSeek API key."
fi

LIMIT="${1:-500}"
PROMPT_VERSION_ARG="${2:-promptv2}"
CONCURRENCY="${3:-5}"
OFFSET="${4:-0}"

if [[ "$PROMPT_VERSION_ARG" != "promptv2" ]]; then
  echo "Only promptv2 is supported by this runner; got: ${PROMPT_VERSION_ARG}" >&2
  exit 2
fi

OUTPUT_DIR="outputs/deepseek_l3_min_token_single_anchor_run"
MODEL="${MODEL:-deepseek-chat}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_TARGET_MEDS_PER_CASE="${MAX_TARGET_MEDS_PER_CASE:-10}"
MAX_LAB_ITEMS="${MAX_LAB_ITEMS:-20}"
RUN_TAG="limit${LIMIT}_offset${OFFSET}_${PROMPT_VERSION_ARG}_conc${CONCURRENCY}"
mkdir -p "$OUTPUT_DIR"

echo "==== Running DeepSeek L3 min-token single-anchor batch: ${RUN_TAG} ===="
python3 scripts/deepseek_l3_min_token_single_anchor_run/01_run_deepseek_l3_min_token_single_anchor.py \
  --input-file outputs/deepseek_l3_minimal_case_inputs/stay_level_all_minimal_for_llm_dx_px_nonempty.jsonl \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --temperature "$TEMPERATURE" \
  --limit "$LIMIT" \
  --offset "$OFFSET" \
  --concurrency "$CONCURRENCY" \
  --run-tag "$RUN_TAG" \
  --resume \
  --max-target-meds-per-case "$MAX_TARGET_MEDS_PER_CASE" \
  --max-lab-items "$MAX_LAB_ITEMS"

echo "==== DeepSeek L3 min-token single-anchor run report: ${RUN_TAG} ===="
python3 - "$RUN_TAG" <<'PY'
import json, sys
from pathlib import Path
run_tag = sys.argv[1]
r = json.loads(Path(f'outputs/deepseek_l3_min_token_single_anchor_run/run_report_{run_tag}.json').read_text(encoding='utf-8'))
keys = [
    'run_tag','prompt_version','api_called','model','offset','limit','concurrency',
    'parallel_api_call_enabled','async_client','run_mode',
    'num_cases_attempted','num_cases_success','num_cases_failed','stopped_early_due_to_consecutive_failures',
    'num_l3_records_generated','valid_l3_records','schema_error_records','illegal_anchor_records',
    'unsafe_causal_language_records','null_anchor_records','medication_procedure_anchor_check',
    'single_anchor_check','token_usage_summary','cost_estimate','warnings','reason'
]
print(json.dumps({k:r.get(k) for k in keys}, ensure_ascii=False, indent=2))
PY

python3 - "$LIMIT" "$PROMPT_VERSION_ARG" "$CONCURRENCY" <<'PY'
import csv
import json
import shutil
import sys
from pathlib import Path

limit = str(sys.argv[1])
prompt_version = str(sys.argv[2])
concurrency = str(sys.argv[3])
base = Path('outputs/deepseek_l3_min_token_single_anchor_run')
if limit != '500' or prompt_version != 'promptv2':
    raise SystemExit(0)
source_tags = [
    f'limit500_offset0_{prompt_version}_conc{concurrency}',
    f'limit500_offset500_{prompt_version}_conc{concurrency}',
]
reports = [base / f'run_report_{tag}.json' for tag in source_tags]
if not all(p.exists() for p in reports):
    missing = [str(p) for p in reports if not p.exists()]
    print('Merged report not generated yet; missing source report(s):', missing)
    raise SystemExit(0)

merged_tag = f'limit1000_{prompt_version}_conc{concurrency}_merged'
concat_specs = [
    ('L3_deepseek_generated', 'jsonl'),
    ('raw_responses', 'jsonl'),
]
for prefix, ext in concat_specs:
    out = base / f'{prefix}_{merged_tag}.{ext}'
    with out.open('w', encoding='utf-8') as wf:
        for tag in source_tags:
            src = base / f'{prefix}_{tag}.{ext}'
            if src.exists():
                with src.open('r', encoding='utf-8') as rf:
                    shutil.copyfileobj(rf, wf)

out_csv = base / f'token_usage_{merged_tag}.csv'
with out_csv.open('w', encoding='utf-8', newline='') as wf:
    writer = None
    for idx, tag in enumerate(source_tags):
        src = base / f'token_usage_{tag}.csv'
        if not src.exists():
            continue
        with src.open('r', encoding='utf-8', newline='') as rf:
            reader = csv.reader(rf)
            header = next(reader, None)
            if header and writer is None:
                writer = csv.writer(wf)
                writer.writerow(header)
            for row in reader:
                if writer is not None:
                    writer.writerow(row)

rs = [json.loads(p.read_text(encoding='utf-8')) for p in reports]
usage = {
    'total_prompt_tokens': sum((r.get('token_usage_summary') or {}).get('total_prompt_tokens', 0) for r in rs),
    'total_completion_tokens': sum((r.get('token_usage_summary') or {}).get('total_completion_tokens', 0) for r in rs),
    'total_tokens': sum((r.get('token_usage_summary') or {}).get('total_tokens', 0) for r in rs),
}
success = sum(r.get('num_cases_success', 0) for r in rs)
usage['mean_prompt_tokens'] = usage['total_prompt_tokens'] / success if success else 0
usage['mean_completion_tokens'] = usage['total_completion_tokens'] / success if success else 0
usage['mean_total_tokens'] = usage['total_tokens'] / success if success else 0
input_price = 0.14
output_price = 0.28
usd_rmb = 7.2
usd = usage['total_prompt_tokens'] / 1e6 * input_price + usage['total_completion_tokens'] / 1e6 * output_price
report = {
    'run_tag': merged_tag,
    'source_run_tags': source_tags,
    'num_batches': 2,
    'target_total_cases': 1000,
    'total_cases_attempted': sum(r.get('num_cases_attempted', 0) for r in rs),
    'total_cases_success': success,
    'total_cases_failed': sum(r.get('num_cases_failed', 0) for r in rs),
    'total_l3_records_generated': sum(r.get('num_l3_records_generated', 0) for r in rs),
    'total_valid_l3_records': sum(r.get('valid_l3_records', 0) for r in rs),
    'total_schema_error_records': sum(r.get('schema_error_records', 0) for r in rs),
    'total_illegal_anchor_records': sum(r.get('illegal_anchor_records', 0) for r in rs),
    'total_unsafe_causal_language_records': sum(r.get('unsafe_causal_language_records', 0) for r in rs),
    'total_null_anchor_records': sum(r.get('null_anchor_records', 0) for r in rs),
    'token_usage_summary': usage,
    'cost_estimate': {
        'input_price_per_million': input_price,
        'output_price_per_million': output_price,
        'estimated_cost_usd': usd,
        'estimated_cost_rmb_rough': usd * usd_rmb,
    },
    'prompt_version': 'v2_no_medication_procedure_anchor',
    'parallel_api_call_enabled': True,
    'async_client': 'AsyncOpenAI',
    'concurrency': int(concurrency),
    'warnings': [w for r in rs for w in r.get('warnings', [])],
}
(base / f'run_report_{merged_tag}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Merged outputs generated for {merged_tag}')
PY
