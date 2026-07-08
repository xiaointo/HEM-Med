# HEM-Med L1/L2 Medication Recommendation Experiment Summary

## 1. Document Purpose

This document summarizes the current HEM-Med L1/L2 memory-enhanced medication recommendation experiments. It records experimental paths, data scale, methodological changes, main metrics, experiment types, and conclusions, and can serve as foundational material for the "Experimental Settings, Parameter Analysis, Ablation Study, and Result Analysis" sections of the paper.

## 2. Metric Definitions

- **Jaccard**: The average Jaccard coefficient between the predicted medication set and the ground-truth medication set for each case.
- **F1 / Precision / Recall**: The average case-level set metrics.
- **DDI Rate**: The proportion of drug pairs in the final predicted medication combinations that hit the DDI knowledge base.
- **SAJ**: `Jaccard × (1 - DDI Rate)`, used to jointly measure therapeutic matching and safety.
- **Drug-level micro PR-AUC**: Calculated by flattening the `[number of cases, medication vocabulary size]` labels and continuous probabilities and then computing `average_precision_score(y_true.ravel(), y_score.ravel())`. This is the PR-AUC adopted in the later main reports.
- **Case-level mean PR-AUC**: AP is calculated within the full medication vocabulary for each case and then averaged across cases.
- **Selection PR-AUC**: Calculated in early experiments based on the final recommendation set or sparse prediction scores, and should not be directly compared horizontally with the later full-vocabulary micro PR-AUC.
- **Class-level metrics**: Calculated after mapping medications to therapeutic categories. These metrics reflect the ability to identify treatment directions, but cannot replace exact-drug metrics.

## 3. Data and Knowledge Sources

- Data split: `have_anchor/split_8_1_1_nonempty_med_seed1203`
- Full training set: 72,436 records; validation set: 9,054 records; test set: 9,055 records.
- Early API / ranking experiments: fixed 100 test cases.
- v13.3 full and Improved Balanced full: full 9,055 test cases.
- L1: Global diagnosis/procedure-medication statistical memory.
- L2: Hospital-level residual, amplified, or specific medication memory.
- DDI: `eicu_space_ddi_pair_set.csv` and `eicu_space_ddi_pairs_simple.csv`.
- Medication vocabulary: approximately 161 medications after normalization.

## 4. Overview of the Experimental Lineage

### 4.1 Lightweight Statistical Reranker Main Line

| Experiment | Test Scale | Type | Core Processing | Jaccard | F1 | Recall | PR-AUC | DDI | SAJ |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| v8 lightweight reranker | 9,055 | Main model | 3,000 training cases; pointwise logistic; L1/L2/hospital/anchor prior features | 0.2684 | 0.4002 | 0.4896 | 0.2622† | 13.63% | 0.2318 |
| v9 DDI-aware | 9,055 | Parameter/safety | DDI penalty, margin, and maximum number of conflicts | 0.2664 | 0.3955 | 0.4572 | 0.2516† | 9.23% | 0.2418 |
| v11 basic | 9,055 | Main model | Full 72,436 training cases, Top30, fixed class cap and score mapping | 0.2651 | 0.3983 | 0.5066 | 0.2684† | 13.62% | 0.2290 |
| v11 mild DDI | 9,055 | DDI parameter | penalty=0.03, margin=0.02, max hits=2 | 0.2645 | 0.3972 | 0.4971 | 0.2655† | 11.03% | 0.2353 |
| v11 balanced DDI | 9,055 | DDI parameter | penalty=0.08, margin=0.05, max hits=2 | 0.2637 | 0.3958 | 0.4862 | 0.2613† | 8.96% | 0.2401 |
| v11 strong DDI | 9,055 | DDI parameter | penalty=0.10, margin=0.06, max hits=1 | 0.2616 | 0.3928 | 0.4763 | 0.2577† | 6.60% | 0.2444 |
| v12 hierarchical balanced | 1,000 | Main model | Top60, dynamic medication count, per-drug/class threshold, experience-based co-use DDI | 0.2811 | 0.4106 | 0.4454 | 0.3631‡ | 10.47% | 0.2517 |
| v13 asymmetric mild | 1,000 | Main model | Asymmetric thresholds for high-frequency-drug precision and low-recall-drug recall | 0.2849 | 0.4177 | 0.4875 | 0.3631‡ | 11.58% | 0.2519 |
| v13 asymmetric balanced | 1,000 | DDI parameter | v13 + balanced DDI | 0.2849 | 0.4175 | 0.4860 | 0.3631‡ | 11.09% | 0.2533 |
| v13.1 offset1-extra0 | 1,000 | Parameter/refinement | Reduced fixed output of 7 medications, within-class reranking, balanced DDI | 0.2840 | 0.4121 | 0.4479 | 0.3631‡ | 9.58% | 0.2568 |
| v13.2 offset1-extra0 | 1,000 | Parameter/rule | Dynamic Top5/6 truncation and conditional release of high-frequency drugs | 0.2835 | 0.4097 | 0.4275 | 0.3631‡ | 9.05% | 0.2578 |
| v13.3 offset1-extra0 | 1,000 | Main model | Per-drug Platt calibration and threshold retuning | 0.2838 | 0.4108 | 0.4415 | **0.3642**‡ | 8.74% | **0.2590** |
| v14 safety profile | 1,000 | Exploration | Full 165-drug scoring, pairwise training, class decomposition, set-level safety selection | 0.2738 | 0.4013 | 0.4516 | **0.3753**‡ | 9.28% | 0.2484 |
| Improved Balanced full | 9,055 | Final main experiment | Weak-evidence assistance, pairwise hard negatives, two-stage class-drug probability, Platt, count classifier, balanced DDI beam search | **0.3246** | **0.4533** | 0.4614 | **0.4161**‡ | 8.31% | **0.2976** |

‡ The underlying probability ranking is the same for v12-v13.2, so the raw drug-level micro PR-AUC remains 0.3631; post-processing thresholds only change Jaccard/F1/DDI. After probability recalibration, v13.3 increases to 0.3642. v14 full-vocabulary scoring increases to 0.3753, but the final set metrics decrease. The main PR-AUC of Improved Balanced full uses the full-vocabulary drug-level micro definition and is 0.4161; the sparse candidate PR-AUC retained in the same report is 0.2895 and is not used as the main reporting definition.

### 4.2 Main Results on the Full Test Set

Full testing is evaluated on 72,436 training cases, 9,054 validation cases, and 9,055 independent test cases. Improved Balanced is the final main experiment; v13.3 full is retained as the previous full-test reference.

| Method/Strategy | Precision | Recall | F1 | Jaccard | Drug micro PR-AUC | Case mean PR-AUC | DDI | SAJ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Improved Balanced (final main experiment) | **0.4511** | **0.4614** | **0.4533** | **0.3246** | **0.4161** | **0.5301** | **8.31%** | **0.2976** |
| v13.3 offset1-extra0 (previous safety strategy) | 0.3904 | 0.4438 | 0.4117 | 0.2824 | 0.3677 | 0.4796 | 8.86% | 0.2574 |
| v13.3 offset1-extra1 (previous high-recall strategy) | 0.3824 | 0.4572 | 0.4112 | 0.2811 | 0.3677 | 0.4796 | 8.90% | 0.2561 |

Improved Balanced path: `paper_experiments/final_balanced_main/outputs_full_split/full_split_report.json`

v13.3 full path: `paper_experiments/memory_and_reranking/have_anchor_L1L2_score_calibrated_v13_3_full/outputs_v13_3_full_test/v13_3_full_case_level_prauc_report.json`


### 4.3 RES-MR External Baselines

| Model | train/val/test | Jaccard | F1 | PR-AUC | DDI | SAJ |
|---|---|---:|---:|---:|---:|---:|
| RES-MR all-minimal | 99,282/12,410/12,411 | 0.2866 | 0.3991 | 0.3803 | 7.48% | 0.2647 |
| RES-MR + L1/L2 | 99,282/12,410/12,411 | 0.2926 | 0.4069 | 0.3908 | 7.25% | 0.2713 |


## 7. Main Result Organization

1. **Final proposed method**: Use Improved Balanced (weak-evidence assistance, pairwise, two-stage ranking, Platt, count classifier, and balanced DDI beam search).
2. **Full-test result of the main method**: On the official 72,436/9,054/9,055 split, Improved Balanced achieves Jaccard 0.3246, F1 0.4533, drug-level micro PR-AUC 0.4161, DDI 8.31%, and SAJ 0.2976.

## 8. Key Report Paths

- Improved Balanced full: `paper_experiments/final_balanced_main/outputs_full_split/full_split_report.json`
