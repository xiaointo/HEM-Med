<img src="Pictures/hem_med.svg" alt="HEM-Med" border="0">

<p float="left">
  <img src="https://img.shields.io/badge/python-v3.9+-red">
  <img src="https://img.shields.io/badge/reranker-standard_library-blue">
  <img src="https://img.shields.io/badge/API-OpenAI%20compatible-green">
  <img src="https://img.shields.io/badge/data-private_EHR-lightgrey">
</p>

# HEM-Med

This repository provides the anonymous implementation of **HEM-Med**, a hierarchical experience memory framework for safe multi-center medication recommendation. HEM-Med explicitly disentangles cross-hospital consensus experience from hospital-specific residual prescribing patterns, and generates medication sets through evidence retrieval, calibrated reranking, and DDI-aware decoding.

This public release contains the core experimental code used in the paper, including L3 attribution construction, L1/L2 memory retrieval, weak-memory augmentation, candidate reranking, medication-count prediction, calibration, and the final DDI-aware medication recommendation pipeline.

This repository **does not include** raw EHR data, generated L3/L1/L2 memory files, model outputs, predictions, API responses, private keys, or other non-redistributable artifacts derived from private clinical data.

<div align="center">
   <img src="Pictures/fig1.png" alt="Motivating Example" width="70%" border="0">
</div>

**Figure 1**: Motivating example of multi-center medication recommendation. Hospitals may show distinct medication distributions for the same disease condition. Direct cross-hospital transfer may suffer from negative transfer, while ignoring local feasibility and DDI risks may lead to unsafe medication combinations.

## ✨ Overview

<div align="center">
   <img src="Pictures/fig2.png" alt="Framework" width="80%" border="0">
</div>

**Figure 2**: Overall framework of HEM-Med. HEM-Med disentangles patient-level EHR experience into cross-hospital consensus memory and hospital-specific residual memory, retrieves multi-source evidence for candidate medication scoring, and generates medication sets through calibrated prediction and DDI-aware decoding.

HEM-Med is designed for safe medication recommendation in multi-center ICU scenarios. Existing medication recommendation methods often mix together generalizable medical knowledge and hospital-specific prescribing preferences, which may cause negative transfer when knowledge learned from one hospital is directly applied to another. HEM-Med addresses this problem through a hierarchical experience memory design:

* **L3 patient-level attribution memory** records medication-context associations from historical ICU stays.
* **L1 cross-hospital consensus memory** captures stable medication-context associations shared across hospitals.
* **L2 hospital-specific residual memory** models local prescribing deviations and hospital-specific medication preferences.
* **Weak auxiliary experience** preserves low-support associations as down-weighted evidence for long-tail medications.
* **DDI-aware decoding** generates safer medication sets by penalizing potentially risky drug-drug interactions.

## 🧠 Method Components

HEM-Med contains the following major components:

1. **L3 Attribution Construction**

   Patient-level EHR records are converted into compact stay-level inputs. DeepSeek/OpenAI-compatible API scripts are then used to construct medication-to-anchor attribution records. These records are used only as offline intermediate evidence and are not retrieved during inference.

2. **L1 Cross-Hospital Memory**

   L1 memory aggregates reliable medication-context associations that appear consistently across multiple hospitals. It is used to capture transferable clinical experience.

3. **L2 Hospital-Specific Residual Memory**

   L2 memory captures hospital-specific amplified, suppressed, or localized medication patterns. It helps the model adapt to local prescribing feasibility and hospital-specific treatment pathways.

4. **Weak Memory Augmentation**

   Low-support but potentially useful medication-context associations are retained as weak auxiliary evidence. This is especially useful for improving candidate recall for long-tail medications and rare clinical contexts.

5. **Calibrated Reranking**

   The final reranker combines retrieval features, evidence features, medication features, safety features, priors, and class-level information. It uses pointwise and pairwise ranking signals, per-drug calibration, and medication-count prediction.

6. **DDI-Aware Set Decoding**

   Instead of selecting medications independently, HEM-Med performs set-level decoding with DDI penalties and redundancy control to produce safer medication combinations.

## 📊 Experiments

<div align="center">
   <img src="Pictures/fig3.png" alt="Main Results" width="70%" border="0">
</div>

**Figure 3**: Main experimental results on the multi-center eICU medication recommendation task.

<div align="center">
   <img src="Pictures/fig4.png" alt="Ablation Study" width="70%" border="0">
</div>

**Figure 4**: Ablation study of key HEM-Med components, including L1/L2 memory, weak experience, reranking, calibration, and DDI-aware decoding.

<div align="center">
   <img src="Pictures/fig5.png" alt="Parameter Analysis" width="70%" border="0">
</div>

**Figure 5**: Parameter analysis of weak auxiliary experience weight and candidate depth.

For more detailed experimental results, please refer to:

```text
docs/EXPERIMENT_SUMMARY.md
```

## 📁 Repository Layout

```text
HEM-Med/
  l3_construction/                  # L3 patient-context attribution construction
    scripts/deepseek_l3_case_level_generation/
    scripts/deepseek_l3_min_token_single_anchor_run/
    scripts/deepseek_l3_test20_dx_px_nonempty_min_token_prompt_audit/

  memory_and_reranking/             # L1/L2 retrieval, candidate generation, calibration, and reranking
    have_anchor_L1L2_full_reranker_ddi_ablation_v11/
    have_anchor_L1L2_hierarchical_calibrated_v12/
    have_anchor_L1L2_asymmetric_calibrated_v13/
    have_anchor_L1L2_refined_v13_1/
    have_anchor_L1L2_score_calibrated_v13_3_full/

  experiments/
    v13_3_weak_pairwise_hierarchical_small/   # Development ablation experiment
    final_balanced_main/                      # Final full official split experiment

  configs/
    private/
      deepseek_key_template.py                # Template for local API key configuration

  docs/
    EXPERIMENT_SUMMARY.md
    DATA_AND_ARTIFACT_GUIDE.md
```

## 📦 Data and Artifact Guide

This release does not include raw EHR data, generated L3/L1/L2 memory files, model outputs, predictions, API responses, or private keys. The final reranker code is designed to run from repository-local artifacts under `data/`, or from another location specified by `HEM_MED_DATA_DIR`.

Set `HEM_MED_DATA_DIR` to a directory with the layout below, or place the same layout under `HEM-Med/data/`.

```text
data/
  splits/split_8_1_1_nonempty_med_seed1203/
    train_80pct.jsonl
    validation_10pct.jsonl
    test_10pct.jsonl

  memory/
    L1_statistical_memory_final_cleaned.json
    L1_statistical_memory_final_cleaned_with_weak_candidates.json
    L2_final_merged_memory.json

  ddi/
    eicu_space_ddi_pairs_simple.csv
```

The historical v11-v13 reranking chain may additionally expect private experiment inputs under:

```text
memory_and_reranking/have_anchor_L1L2_full_reranker_ddi_ablation_v11/inputs/
  L1_statistical_memory_from_have_anchor.json
  L2_hospital_residual_memory_from_have_anchor.json
  eicu_space_ddi_pairs_simple.csv
  eicu_space_ddi_pair_set.csv
  medication_vocabulary.json
  medication_vocabulary_canonical.json
  global_medication_prior.json
  hospital_medication_prior.json
  anchor_medication_prior.json
  hospital_anchor_medication_prior.json
  test_100_have_anchor.jsonl
```

### Required Artifacts

| Artifact                   | Expected release path                                                       | Format / required fields                                                                | Used by                                                                                |
| -------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Training split             | `data/splits/split_8_1_1_nonempty_med_seed1203/train_80pct.jsonl`           | JSONL. One stay per line with keys `id`, `hid`, `age`, `sex`, `dx`, `px`, `lab`, `med`. | Final reranker training, medication-count model, priors, co-administration confidence. |
| Validation split           | `data/splits/split_8_1_1_nonempty_med_seed1203/validation_10pct.jsonl`      | Same JSONL schema as training.                                                          | Platt calibration and validation-time tuning.                                          |
| Test split                 | `data/splits/split_8_1_1_nonempty_med_seed1203/test_10pct.jsonl`            | Same JSONL schema as training.                                                          | Independent final evaluation.                                                          |
| Primary L1 memory          | `data/memory/L1_statistical_memory_final_cleaned.json`                      | JSON dict keyed by normalized anchors such as `diag:*`, `proc:*`, or `lab:*`.           | Main L1 retrieval channel.                                                             |
| Primary L2 memory          | `data/memory/L2_final_merged_memory.json`                                   | JSON dict keyed by hospital id, anchor key, and medication.                             | Main hospital-specific L2 retrieval channel.                                           |
| Weak L1 memory             | `data/memory/L1_statistical_memory_final_cleaned_with_weak_candidates.json` | L1 schema plus weak candidate associations.                                             | Auxiliary weak-memory channel.                                                         |
| DDI descriptions           | `data/ddi/eicu_space_ddi_pairs_simple.csv`                                  | CSV with columns `drug_name_1`, `drug_name_2`, `interaction_description`.               | DDI severity and text-level interaction descriptions.                                  |
| Medication vocabulary      | `memory_and_reranking/.../inputs/medication_vocabulary_canonical.json`      | JSON object with at least `medications` and `canonicalized`.                            | Candidate scoring and full-vocabulary PR-AUC.                                          |
| Global prior JSON          | `memory_and_reranking/.../inputs/global_medication_prior.json`              | JSON object with medication frequency statistics.                                       | Candidate prior features.                                                              |
| Hospital prior JSON        | `memory_and_reranking/.../inputs/hospital_medication_prior.json`            | JSON object with hospital-level medication frequency statistics.                        | Hospital-specific prior features.                                                      |
| Anchor prior JSON          | `memory_and_reranking/.../inputs/anchor_medication_prior.json`              | JSON object with anchor-medication statistics.                                          | Anchor-level prior features.                                                           |
| Hospital-anchor prior JSON | `memory_and_reranking/.../inputs/hospital_anchor_medication_prior.json`     | JSON object with hospital-anchor-medication statistics.                                 | Hospital-anchor prior features.                                                        |

The official split contains 72,436 training records, 9,054 validation records, and 9,055 test records. These files are derived from the de-identified stay-level EHR table after filtering to non-empty medication cases with the fixed seed 1203 split. Raw EHR data cannot be redistributed.

## 🚀 Usage

You can reproduce the main pipeline according to the following steps.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

The final Improved Balanced reranker is implemented with the Python standard library. The L3 DeepSeek/OpenAI-compatible API construction scripts require:

```text
openai>=1.0.0
httpx>=0.24.0
```

### 2. Build compact L3 inputs

```bash
cd l3_construction
bash scripts/deepseek_l3_case_level_generation/run_14_build_minimal_case_input_for_llm.sh
```

### 3. Generate L3 medication-to-anchor attribution memory

```bash
cd l3_construction
cp ../configs/private/deepseek_key_template.py configs/private/deepseek_key.py
# Fill in configs/private/deepseek_key.py locally.
bash scripts/deepseek_l3_min_token_single_anchor_run/run_01_deepseek_l3_min_token_single_anchor.sh
```

The API key file is intentionally excluded from the anonymous release.

### 4. Postprocess and audit L3 records

```bash
cd l3_construction
bash scripts/deepseek_l3_min_token_single_anchor_run/run_02_postprocess_l3_reasonable_anchor_limit20_promptv2_cleaninput.sh
bash scripts/deepseek_l3_min_token_single_anchor_run/run_03_recover_l3_from_raw_responses_limit100_promptv2_cleaninput.sh
```

### 5. Provide L1/L2 memory artifacts

The public code expects data artifacts under `data/` by default, or under `$HEM_MED_DATA_DIR` if set.

```bash
export HEM_MED_DATA_DIR=/path/to/hem_med_data
```

Expected layout:

```text
$HEM_MED_DATA_DIR/
  splits/split_8_1_1_nonempty_med_seed1203/
    train_80pct.jsonl
    validation_10pct.jsonl
    test_10pct.jsonl
  memory/
    L1_statistical_memory_final_cleaned.json
    L1_statistical_memory_final_cleaned_with_weak_candidates.json
    L2_final_merged_memory.json
  ddi/
    eicu_space_ddi_pairs_simple.csv
```

### 6. Run the final Improved Balanced experiment

```bash
bash experiments/final_balanced_main/run_full.sh
```

## 🧪 Minimal `sample_data/` Smoke Test

A tiny `sample_data/` directory may be added for reviewers who only need to check that the pipeline imports and executes. The values can be synthetic or heavily downsampled. Metrics from this directory are meaningless and must not be reported as paper results.

Recommended layout:

```text
sample_data/
  splits/split_8_1_1_nonempty_med_seed1203/
    train_80pct.jsonl
    validation_10pct.jsonl
    test_10pct.jsonl
  memory/
    L1_statistical_memory_final_cleaned.json
    L1_statistical_memory_final_cleaned_with_weak_candidates.json
    L2_final_merged_memory.json
  ddi/
    eicu_space_ddi_pairs_simple.csv
```

Smoke-test command:

```bash
HEM_MED_DATA_DIR=sample_data bash experiments/final_balanced_main/run_full.sh
```

For this command to run end-to-end, the historical v11 input directory must also contain matching tiny versions of the vocabulary and prior JSON files listed above. If the anonymous release omits `sample_data/`, this command should be documented as a schema check target rather than a reproduced result.

## 🔬 Final Method

The final paper method is implemented by:

```text
experiments/v13_3_weak_pairwise_hierarchical_small/run_experiment.py
experiments/final_balanced_main/run_full_experiment.py
```

The final pipeline combines:

* Decoupled L1/L2 experience memory retrieval.
* Down-weighted weak-memory evidence.
* Candidate generation with global, hospital, anchor, and hospital-anchor priors.
* Pointwise and pairwise logistic reranking.
* Two-stage class-drug scoring.
* Per-drug Platt calibration.
* Medication-count prediction.
* DDI-aware beam search.
* Redundancy-aware medication set decoding.

## 📤 Generated Outputs

The following files and directories are generated outputs and are intentionally not included in the anonymous release:

```text
experiments/final_balanced_main/outputs_full_split/
experiments/v13_3_weak_pairwise_hierarchical_small/outputs_test300/
memory_and_reranking/**/outputs*/
```

The release also excludes:

* L3 API raw responses.
* Checkpoints.
* Token-usage CSV files.
* Prediction files.
* Full L1/L2 memory snapshots when they contain non-redistributable information derived from the private EHR corpus.

Paths mentioned in experiment README files that point to `outputs*/` directories identify the provenance of completed internal runs. They are not required files in the anonymous code release unless explicitly regenerated by the reviewer.

## 🔐 Notes for Anonymous Release

* No patient-level raw EHR data are included.
* No generated L3/L1/L2 memory result files are included.
* No API keys are included.
* No model predictions or internal evaluation outputs are included.
* All private artifacts should be regenerated locally or provided through `HEM_MED_DATA_DIR`.
* The released code is intended for reproducibility, schema checking, and reviewer inspection under the constraints of private clinical data redistribution.

## 🌟 Contributions and Suggestions

Contributions, suggestions, and issue reports are welcome. Please make sure that any shared files do not contain raw patient data, private API keys, generated private EHR-derived artifacts, or non-redistributable clinical information.
