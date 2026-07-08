# HEM-Med Final Main Experiment: Improved Balanced

## Status

- Final proposed method: **Improved Balanced**.
- Development evaluation completed on 3,000 train / 500 validation / 300 test cases.
- Full 72,436 / 9,054 / 9,055 official split confirmation is complete.
- Existing v13.3 files remain unchanged.

## Frozen Method

1. Decoupled L1/L2 is the primary retrieval channel.
2. Weak and non-decoupled associations form an attenuated auxiliary channel (`weak_weight=0.42`).
3. Top-60 candidates are scored by pointwise and pairwise logistic models.
4. Hard negatives include same-class, top-ranked false-positive, and frequent false-positive drugs.
5. The score combines `P(class|patient)` and `P(drug|class,patient)`.
6. Per-drug Platt calibration is fitted on validation data.
7. Medication count uses multiclass `P(K=1..8)`.
8. Final sets use DDI-aware beam search with Balanced `ddi_scale=0.22`.
9. DDI penalties incorporate interaction severity and empirical co-administration confidence.

## Development Metrics

| Metric | Value |
|---|---:|
| Precision | 0.441389 |
| Recall | 0.442556 |
| F1 | 0.440616 |
| Jaccard | 0.316791 |
| Drug-level micro PR-AUC | 0.410467 |
| Case-level mean PR-AUC | 0.516856 |
| DDI Rate | 0.077457 |
| SAJ | 0.292253 |

These are development results on the fixed 300-case subset and must be labeled accordingly.

## Full Official Split Metrics

Full split rerun uses 72,436 training cases, 9,054 validation cases for calibration, and 9,055 independent test cases.

| Metric | Value |
|---|---:|
| Precision | 0.451114 |
| Recall | 0.461351 |
| F1 | 0.453299 |
| Jaccard | 0.324601 |
| Drug-level micro PR-AUC | 0.416100 |
| Case-level mean PR-AUC | 0.530116 |
| DDI Rate | 0.083097 |
| SAJ | 0.297628 |


Important provenance note: current L1/L2 and weak-memory files were built from the pre-existing train-only-memory reconstruction. 

## Source Artifacts

- Implementation: `paper_experiments/v13_3_weak_pairwise_hierarchical_small/run_experiment.py`
- Report: `paper_experiments/v13_3_weak_pairwise_hierarchical_small/outputs_test300/experiment_report.json`
- Test cases: `paper_experiments/v13_3_weak_pairwise_hierarchical_small/outputs_test300/test_300_cases.jsonl`
- Predictions: `paper_experiments/v13_3_weak_pairwise_hierarchical_small/outputs_test300/per_case_predictions_balanced.jsonl`
- Full split implementation: `paper_experiments/final_balanced_main/run_full_experiment.py`
- Full split report: `paper_experiments/final_balanced_main/outputs_full_split/full_split_report.json`
- Full split predictions: `paper_experiments/final_balanced_main/outputs_full_split/predictions_balanced.jsonl`

Use Improved Balanced as the proposed method. For publication-level exact-drug results, prefer the full official split metrics above over the 300-case development metrics.

