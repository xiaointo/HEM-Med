# Data and Artifact Guide

This repository release does not include raw EHR data, generated L3/L1/L2 memory files, model outputs, predictions, API responses, or private keys. The final reranker code is designed to run from repository-local artifacts under `data/`, or from another location provided through `HEM_MED_DATA_DIR`.

## Required Artifacts

Set `HEM_MED_DATA_DIR` to a directory with the layout below, or place the same layout under `HEM-Med/data/`.

| Artifact | Expected release path | Format / required fields | Used by | How to obtain or generate |
|---|---|---|---|---|
| Training split | `data/splits/split_8_1_1_nonempty_med_seed1203/train_80pct.jsonl` | JSONL. One stay per line with keys `id`, `hid`, `age`, `sex`, `dx`, `px`, `lab`, `med`. The official split contains 72,436 records. | Final reranker training, medication-count model, priors, co-administration confidence. | Derived from the de-identified stay-level EHR table after filtering to non-empty medication cases with the fixed seed 1203 split. Raw EHR data cannot be redistributed. |
| Validation split | `data/splits/split_8_1_1_nonempty_med_seed1203/validation_10pct.jsonl` | Same JSONL schema as training. The official split contains 9,054 records. | Platt calibration and validation-time tuning. | Same split procedure as above. |
| Test split | `data/splits/split_8_1_1_nonempty_med_seed1203/test_10pct.jsonl` | Same JSONL schema as training. The official split contains 9,055 records. | Independent final evaluation. | Same split procedure as above. |
| Primary L1 memory | `data/memory/L1_statistical_memory_final_cleaned.json` or `paper_experiments/memory_and_reranking/have_anchor_L1L2_full_reranker_ddi_ablation_v11/inputs/L1_statistical_memory_from_have_anchor.json` | JSON dict keyed by normalized anchor keys such as `diag:*`, `proc:*`, or `lab:*`. Each anchor maps to medication entries with support/frequency/function metadata. | Main L1 retrieval channel. | Generated from train-only L3 attribution records by the L1/L2 memory construction scripts. |
| Primary L2 memory | `data/memory/L2_final_merged_memory.json` or `paper_experiments/memory_and_reranking/have_anchor_L1L2_full_reranker_ddi_ablation_v11/inputs/L2_hospital_residual_memory_from_have_anchor.json` | JSON dict keyed by hospital id, then anchor key, then medication. Entries include residual type, support, frequency, and medication-function metadata. | Main hospital-specific L2 retrieval channel. | Generated from train-only L3 attribution records and merged hospital residual memory. |
| Weak L1 memory | `data/memory/L1_statistical_memory_final_cleaned_with_weak_candidates.json` | JSON dict with the L1 schema plus weak candidate associations. | Auxiliary weak-memory channel in `paper_experiments/v13_3_weak_pairwise_hierarchical_small/run_experiment.py`. | Built by merging weak candidate anchors into the cleaned L1 memory. |
| Weak L2 memory | `data/memory/L2_final_merged_memory.json` | JSON dict with the L2 schema after merging cleaned/canonical L2 and weak candidate memories. | Auxiliary weak-memory channel and primary L2 lookup. | Built by the final L2 merge step from train-only memory reconstruction artifacts. |
| DDI descriptions | `data/ddi/eicu_space_ddi_pairs_simple.csv` | CSV with columns `drug_name_1`, `drug_name_2`, `interaction_description`. | DDI severity and text-level interaction descriptions. | Built by mapping the medication vocabulary to DrugBank/DDInter-style interaction pairs, then filtering to the eICU medication space. |
| DDI pair set | `paper_experiments/memory_and_reranking/have_anchor_L1L2_full_reranker_ddi_ablation_v11/inputs/eicu_space_ddi_pair_set.csv` | CSV pair set. Two medication columns are sufficient for metric lookup. | Historical v11-v13 chain and DDI-rate metric. | Generated from the same cleaned DDI mapping as the description CSV. |
| Medication vocabulary | `paper_experiments/memory_and_reranking/have_anchor_L1L2_full_reranker_ddi_ablation_v11/inputs/medication_vocabulary_canonical.json` | JSON object with at least `medications` and `canonicalized`. | Candidate scoring and full-vocabulary PR-AUC. | Built from the train split medication field after normalization/canonicalization. |
| Global prior JSON | `paper_experiments/memory_and_reranking/have_anchor_L1L2_full_reranker_ddi_ablation_v11/inputs/global_medication_prior.json` | JSON object with `source_train_file`, `train_case_count`, `canonicalized`, and `medications`. | Candidate prior features. | Computed from medication frequencies in the train split. |
| Hospital prior JSON | `paper_experiments/memory_and_reranking/have_anchor_L1L2_full_reranker_ddi_ablation_v11/inputs/hospital_medication_prior.json` | JSON object with `source_train_file`, `canonicalized`, `hospital_case_counts`, and `hospitals`. | Hospital-specific prior features. | Computed from medication frequencies by hospital in the train split. |
| Anchor prior JSON | `paper_experiments/memory_and_reranking/have_anchor_L1L2_full_reranker_ddi_ablation_v11/inputs/anchor_medication_prior.json` | JSON object with `source_train_file`, `canonicalized`, and `anchors`. | Anchor-level prior features. | Computed from train split context anchors and medications. |
| Hospital-anchor prior JSON | `paper_experiments/memory_and_reranking/have_anchor_L1L2_full_reranker_ddi_ablation_v11/inputs/hospital_anchor_medication_prior.json` | JSON object with `source_train_file`, `canonicalized`, and `hospital_anchors`. Keys combine hospital and anchor, for example `180::diag:*`. | Hospital-anchor prior features. | Computed from train split hospital, anchor, and medication co-occurrence. |

## Expected Directory Layout

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

paper_experiments/memory_and_reranking/have_anchor_L1L2_full_reranker_ddi_ablation_v11/inputs/
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

The `data/` directory is the preferred public-release location for split, weak memory, and DDI artifacts. The `memory_and_reranking/.../inputs/` directory is retained because the historical v11-v13 import chain expects these private experiment inputs.

## Minimal `sample_data/` Smoke Test

A tiny `sample_data/` directory may be added for reviewers who only need to check that the pipeline imports and executes. The values can be synthetic or heavily downsampled; metrics from this directory are meaningless and must not be reported as paper results.

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

The smoke-test command is:

```bash
HEM_MED_DATA_DIR=sample_data bash paper_experiments/final_balanced_main/run_full.sh
```

For the command above to run end-to-end, the historical v11 input directory must also contain matching tiny versions of the vocabulary and prior JSON files listed in the table. If the anonymous release omits `sample_data/`, this command should be documented as a schema check target rather than a reproduced result.

## Dependency Note

The final Improved Balanced reranker is implemented with the Python standard library. The L3 DeepSeek/OpenAI-compatible API construction scripts require:

```text
openai>=1.0.0
httpx>=0.24.0
```

No additional package is required for the pure-Python reranking path unless a downstream user adds optional plotting or external baseline scripts.

## Generated Outputs

The following files and directories are generated outputs and are intentionally not included in the anonymous release:

- `paper_experiments/final_balanced_main/outputs_full_split/`
- `paper_experiments/v13_3_weak_pairwise_hierarchical_small/outputs_test300/`
- `paper_experiments/memory_and_reranking/**/outputs*/`
- L3 API raw responses, checkpoints, token-usage CSV files, and predictions
- Full L1/L2 memory snapshots when they contain non-redistributable information derived from the private EHR corpus

Paths mentioned in experiment READMEs that point to `outputs*/` directories identify provenance of completed internal runs. They are not required files in the anonymous code release unless explicitly regenerated by the reviewer.
