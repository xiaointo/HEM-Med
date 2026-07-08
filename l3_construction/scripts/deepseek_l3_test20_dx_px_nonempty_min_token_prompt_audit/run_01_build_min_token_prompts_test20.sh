#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../../.." && pwd)"
mkdir -p outputs/deepseek_l3_test20_dx_px_nonempty_min_token_prompt_audit
python3 scripts/deepseek_l3_test20_dx_px_nonempty_min_token_prompt_audit/01_build_min_token_prompts_test20.py
echo "==== DeepSeek L3 min-token prompt dry-run audit report ===="
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('outputs/deepseek_l3_test20_dx_px_nonempty_min_token_prompt_audit/min_token_prompt_audit_report.json').read_text(encoding='utf-8'))
print(json.dumps({k:r.get(k) for k in ['dry_run_only','api_called','num_cases','num_prompts','prompt_compaction','case_stats','problem_counts','quality_check','warnings']},ensure_ascii=False,indent=2))
PY
