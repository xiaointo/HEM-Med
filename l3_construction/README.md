# L3 Construction

This directory contains the code used to construct the patient-level L3 attribution records described in the paper. It is included for transparency and reproducibility of the L3 construction process.

Generated L3 JSONL files, raw API responses, token usage logs, checkpoints, and reports are not included in the anonymous release.

## Pipeline

1. Build compact stay-level inputs for the LLM:

```bash
cd l3_construction
bash scripts/deepseek_l3_case_level_generation/run_14_build_minimal_case_input_for_llm.sh
```

This expects the private preprocessed stay file at:

```text
outputs/stage1_preprocess_final/stay_level_all_detailed.jsonl
```

and writes compact L3 inputs under:

```text
outputs/deepseek_l3_minimal_case_inputs/
```

2. Run DeepSeek/OpenAI-compatible single-anchor L3 generation:

```bash
cd l3_construction
cp configs/private/deepseek_key_template.py configs/private/deepseek_key.py
# Fill configs/private/deepseek_key.py locally, or configure your environment.
bash scripts/deepseek_l3_min_token_single_anchor_run/run_01_deepseek_l3_min_token_single_anchor.sh
```

The main generated L3 file has the form:

```text
outputs/deepseek_l3_min_token_single_anchor_run/L3_deepseek_generated_<tag>.jsonl
```

3. Postprocess generated L3 records:

```bash
cd l3_construction
bash scripts/deepseek_l3_min_token_single_anchor_run/run_02_postprocess_l3_reasonable_anchor_limit20_promptv2_cleaninput.sh
```

4. Recover L3 records from saved raw responses when needed:

```bash
cd l3_construction
bash scripts/deepseek_l3_min_token_single_anchor_run/run_03_recover_l3_from_raw_responses_limit100_promptv2_cleaninput.sh
```

5. Optional prompt audit without API calls:

```bash
cd l3_construction
bash scripts/deepseek_l3_test20_dx_px_nonempty_min_token_prompt_audit/run_01_build_min_token_prompts_test20.sh
```

## Included Code

```text
scripts/deepseek_l3_case_level_generation/
  14_build_minimal_case_input_for_llm.py
  run_14_build_minimal_case_input_for_llm.sh

scripts/deepseek_l3_min_token_single_anchor_run/
  01_run_deepseek_l3_min_token_single_anchor.py
  02_postprocess_l3_reasonable_anchor.py
  03_recover_l3_from_raw_responses.py
  check_deepseek_key.py
  run_01_deepseek_l3_min_token_single_anchor.sh
  run_02_postprocess_l3_reasonable_anchor_limit20_promptv2_cleaninput.sh
  run_03_recover_l3_from_raw_responses_limit100_promptv2_cleaninput.sh
  run_full_offset20000_auto_resume.sh

scripts/deepseek_l3_test20_dx_px_nonempty_min_token_prompt_audit/
  01_build_min_token_prompts_test20.py
  run_01_build_min_token_prompts_test20.sh

configs/private/deepseek_key_template.py
```

## Excluded Generated Files

Do not commit files under `l3_construction/outputs/`, including:

```text
L3_deepseek_generated_*.jsonl
raw_responses_*.jsonl
token_usage_*.csv
checkpoint_*.sqlite
run_report_*.json
failed_records_*.jsonl
schema_error_records_*.jsonl
```

These files may contain private EHR-derived intermediate information, API responses, or run metadata.
