#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../../.." && pwd)"
TAG="limit200000_offset20000_promptv2_cleaninput"
INPUT="outputs/deepseek_l3_minimal_case_inputs/stay_level_all_minimal_for_llm_dx_px_nonempty.jsonl"
OUTDIR="outputs/deepseek_l3_min_token_single_anchor_run"

LIMIT=200000
OFFSET=20000
CONCURRENCY=500
MAX_ROUNDS=50
SLEEP_SECONDS=60

echo "Start full remaining L3 generation from offset=${OFFSET}"
echo "TAG=${TAG}"
echo "CONCURRENCY=${CONCURRENCY}"

for ROUND in $(seq 1 ${MAX_ROUNDS}); do
  echo "============================================================"
  echo "Round ${ROUND}/${MAX_ROUNDS}: running API generation with resume"
  echo "============================================================"

  python3 scripts/deepseek_l3_min_token_single_anchor_run/01_run_deepseek_l3_min_token_single_anchor.py \
    --input-file "${INPUT}" \
    --output-dir "${OUTDIR}" \
    --limit "${LIMIT}" \
    --offset "${OFFSET}" \
    --concurrency "${CONCURRENCY}" \
    --run-tag "${TAG}" \
    --resume || true

  CHECKPOINT="${OUTDIR}/checkpoint_${TAG}.sqlite"
  REPORT="${OUTDIR}/run_report_${TAG}.json"

  echo "Checking checkpoint/report..."

  python3 - <<PY
import json, sqlite3, sys
from pathlib import Path

checkpoint = Path("${CHECKPOINT}")
report = Path("${REPORT}")

if not checkpoint.exists():
    print("checkpoint_missing")
    sys.exit(2)

conn = sqlite3.connect(str(checkpoint))
cur = conn.cursor()
rows = cur.execute("select status, count(*) from cases group by status").fetchall()
conn.close()

status = {k: v for k, v in rows}
print("checkpoint_status:", status)

failed = status.get("failed", 0)
running = status.get("running", 0)
pending = status.get("pending", 0)
success = status.get("success", 0)

if report.exists():
    obj = json.loads(report.read_text(encoding="utf-8"))
    print("report_num_cases_attempted:", obj.get("num_cases_attempted"))
    print("report_num_cases_success:", obj.get("num_cases_success"))
    print("report_num_cases_failed:", obj.get("num_cases_failed"))
    print("stopped_early_due_to_consecutive_failures:", obj.get("stopped_early_due_to_consecutive_failures"))

if failed == 0 and running == 0 and pending == 0 and success > 0:
    print("DONE")
    sys.exit(0)

print("NOT_DONE")
sys.exit(1)
PY

  STATUS=$?

  if [ "$STATUS" -eq 0 ]; then
    echo "All checkpoint cases are successful. API generation completed."
    exit 0
  fi

  echo "Not finished yet. Sleep ${SLEEP_SECONDS}s, then resume..."
  sleep "${SLEEP_SECONDS}"
done

echo "Reached MAX_ROUNDS=${MAX_ROUNDS}. Please inspect checkpoint and failed records."
exit 1
