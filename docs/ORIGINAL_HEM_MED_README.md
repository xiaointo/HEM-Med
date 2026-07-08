# HEM-Med

HEM-Med is a hierarchical agentic memory framework for safe multi-center medication recommendation. This repository contains the core experimental code used in the paper, including L3 construction, L1/L2 memory retrieval and calibration, and the final medication recommendation pipeline.

This public release intentionally excludes raw EHR data, generated intermediate files, trained weights, full memory JSON files, predictions, API responses, and private API keys.

## Repository Layout

```text
HEM-Med/
  l3_construction/                  # L3 patient-context attribution construction
    scripts/deepseek_l3_case_level_generation/
    scripts/deepseek_l3_min_token_single_anchor_run/
    scripts/deepseek_l3_test20_dx_px_nonempty_min_token_prompt_audit/
  memory_and_reranking/             # L1/L2 retrieval, candidate generation, calibration, DDI-aware reranking
    have_anchor_L1L2_full_reranker_ddi_ablation_v11/
    have_anchor_L1L2_hierarchical_calibrated_v12/
    have_anchor_L1L2_asymmetric_calibrated_v13/
    have_anchor_L1L2_refined_v13_1/
    have_anchor_L1L2_score_calibrated_v13_3_full/
  experiments/
    v13_3_weak_pairwise_hierarchical_small/   # development ablation
    final_balanced_main/                      # final full official split experiment
  configs/private/deepseek_key_template.py
  docs/
```

## Core Experimental Pipeline

1. Build compact L3 inputs from stay-level records:

```bash
cd l3_construction
bash scripts/deepseek_l3_case_level_generation/run_14_build_minimal_case_input_for_llm.sh
```

2. Generate L3 medication-to-anchor attribution memory with DeepSeek/OpenAI-compatible API:

```bash
cd l3_construction
cp ../configs/private/deepseek_key_template.py configs/private/deepseek_key.py
# Fill in configs/private/deepseek_key.py locally.
bash scripts/deepseek_l3_min_token_single_anchor_run/run_01_deepseek_l3_min_token_single_anchor.sh
```

3. Postprocess and audit L3 records:

```bash
cd l3_construction
bash scripts/deepseek_l3_min_token_single_anchor_run/run_02_postprocess_l3_reasonable_anchor_limit20_promptv2_cleaninput.sh
bash scripts/deepseek_l3_min_token_single_anchor_run/run_03_recover_l3_from_raw_responses_limit100_promptv2_cleaninput.sh
```

4. Build or provide L1/L2 memory artifacts from the L3 corpus. The public code expects data artifacts under `data/` by default, or under `$HEM_MED_DATA_DIR` if set.

Expected data layout for reranking experiments:

```text
data/
  splits/split_8_1_1_nonempty_med_seed1203/
    train_80pct.jsonl
    validation_10pct.jsonl
    test_10pct.jsonl
  memory/
    L1_statistical_memory_final_cleaned_with_weak_candidates.json
    L2_final_merged_memory.json
  ddi/
    eicu_space_ddi_pairs_simple.csv
```

The v11 candidate/reranker code also expects its private experiment inputs, such as L1/L2 memory snapshots and priors, in `memory_and_reranking/have_anchor_L1L2_full_reranker_ddi_ablation_v11/inputs/` when running the historical v11-v13 chain.

5. Run the final Improved Balanced recommendation experiment:

```bash
export HEM_MED_DATA_DIR=/path/to/hem_med_data
bash experiments/final_balanced_main/run_full.sh
```

## Final Method

The final paper method is `Improved Balanced`, implemented by:

- `experiments/v13_3_weak_pairwise_hierarchical_small/run_experiment.py`
- `experiments/final_balanced_main/run_full_experiment.py`

It combines decoupled L1/L2 memory retrieval, attenuated weak-memory features, pointwise and pairwise logistic reranking, two-stage class-drug scoring, per-drug Platt calibration, medication-count prediction, and DDI-aware beam search.

## Notes for Anonymous Release

- No patient data are included.
- No generated L3/L1/L2 memory result files are included.
- No API keys are included.
- See `docs/EXPERIMENT_SUMMARY.md` for the paper experiment summary and metric table.
