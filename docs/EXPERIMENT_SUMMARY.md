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
- v12-v14 main development experiments: 1,000 test cases.
- v13.3 full and Improved Balanced full: full 9,055 test cases.
- L1: Global diagnosis/procedure-medication statistical memory.
- L2: Hospital-level residual, amplified, or specific medication memory.
- DDI: `eicu_space_ddi_pair_set.csv` and `eicu_space_ddi_pairs_simple.csv`.
- Medication vocabulary: approximately 161 medications after normalization.

## 4. Overview of the Experimental Lineage

### 4.1 DeepSeek RAG and Deterministic Ranking Exploration (100 Test Cases)

| Experiment | Type | Core Processing | Jaccard | F1 | Recall | PR-AUC* | DDI | SAJ | Conclusion |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Initial L1+L2+STI DeepSeek | Exploration | Free recommendation by the LLM after L1/L2 RAG | 0.1621 | 0.2590 | 0.4650 | 0.1840 | 14.78% | 0.1381 | Recall is relatively high, but false positives and DDI are frequent |
| Strong L2 + DDI RAG v2 | Exploration/safety | Compressed L2, non-empty medication cases, DDI RAG | 0.1686 | 0.2671 | 0.2813 | 0.1074 | 10.72% | 0.1505 | Safety improves, but DDI filtering harms recall |
| Have-anchor DeepSeek v1 | Main-line starting point | Anchored L1/L2 + DDI RAG, free generation | 0.1640 | 0.2578 | 0.2595 | 0.0993 | 5.88% | 0.1543 | Free-generation performance is low, and the model falls back to high-frequency ICU medications |
| Candidate boost v2 | Parameter/candidate | Added Top-60, hospital prior, global prior, and fuzzy anchors | 0.1931 | 0.3013 | 0.3233 | 0.1344 | 5.30% | 0.1829 | Candidate recall increases to 95.17%, indicating that candidate enhancement is effective |
| DeepSeek rerank Top12 v3 | LLM rerank | Only Top12 are provided, default Top5, with at most 2 replacements | 0.2353 | 0.3561 | 0.3803 | 0.1650 | 16.40% | 0.1967 | Clearly better than free generation, but lower than deterministic Top-k |
| Deterministic Top5 v4 | Main baseline | Non-specificity penalty, canonicalization, fixed Top5 | 0.2648 | 0.3879 | 0.4118 | 0.1885 | 15.80% | 0.2230 | Jaccard/SAJ outperform LLM rerank |
| Adaptive Top-k v4 | Parameter experiment | Dynamic truncation based on the score breakpoints at ranks 5-7 | 0.2614 | 0.3907 | 0.4658 | 0.2074 | 15.92% | 0.2198 | Recall/PR-AUC are higher, while Precision slightly decreases |
| Pairwise LLM judge v5 | LLM ablation | The LLM only compares boundary candidates and does not change the Top3 | 0.2634 | 0.3859 | 0.4098 | 0.1876 | 16.00% | 0.2213 | Close to but does not exceed deterministic Top5 |
| Pairwise + evidence gate v6 | LLM ablation | Accepts LLM decisions only when the statistical score gap is small and the evidence type permits it | 0.2665 | 0.3899 | 0.4138 | 0.1894 | 15.90% | 0.2241 | Slightly improves over v5, but the LLM gain is limited |
| Per-drug calibration v7 Top6 | Parameter experiment | Per-drug multiplier/threshold calibration | 0.2526 | 0.3776 | 0.4408 | 0.1938 | 12.20% | 0.2218 | Reduces DDI, but is overall inferior to v4 Top5 |

\* The PR-AUC in this table follows the early selection/sparse-score definition.

### 4.2 Lightweight Statistical Reranker Main Line

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

### 4.3 Main Results on the Full Test Set

Full testing is evaluated on 72,436 training cases, 9,054 validation cases, and 9,055 independent test cases. Improved Balanced is the final main experiment; v13.3 full is retained as the previous full-test reference.

| Method/Strategy | Precision | Recall | F1 | Jaccard | Drug micro PR-AUC | Case mean PR-AUC | DDI | SAJ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Improved Balanced (final main experiment) | **0.4511** | **0.4614** | **0.4533** | **0.3246** | **0.4161** | **0.5301** | **8.31%** | **0.2976** |
| v13.3 offset1-extra0 (previous safety strategy) | 0.3904 | 0.4438 | 0.4117 | 0.2824 | 0.3677 | 0.4796 | 8.86% | 0.2574 |
| v13.3 offset1-extra1 (previous high-recall strategy) | 0.3824 | 0.4572 | 0.4112 | 0.2811 | 0.3677 | 0.4796 | 8.90% | 0.2561 |

Improved Balanced path: `paper_experiments/final_balanced_main/outputs_full_split/full_split_report.json`

v13.3 full path: `paper_experiments/memory_and_reranking/have_anchor_L1L2_score_calibrated_v13_3_full/outputs_v13_3_full_test/v13_3_full_case_level_prauc_report.json`


### 4.4 RES-MR External Baselines

| Model | train/val/test | Jaccard | F1 | PR-AUC | DDI | SAJ |
|---|---|---:|---:|---:|---:|---:|
| RES-MR all-minimal | 99,282/12,410/12,411 | 0.2866 | 0.3991 | 0.3803 | 7.48% | 0.2647 |
| RES-MR + L1/L2 | 99,282/12,410/12,411 | 0.2926 | 0.4069 | 0.3908 | 7.25% | 0.2713 |

## 5. Parameter Experiments

### 5.1 Candidate Ranking Parameter Grid

- Directory: `have_anchor_L1L2_score_calibrated_topk_v4/grid_outputs/`
- Search parameters: generic ICU penalty, commonly missed medication boost, and adaptive gap threshold.
- Role: Confirms that candidate recall has reached approximately 97.7%, and that the bottleneck has shifted from candidate recall to within-candidate ranking.
- Best tendency: Top5 optimizes Jaccard/SAJ; adaptive Top5-7 optimizes Recall/PR-AUC.

### 5.2 LLM Pairwise Parameter Grid

- Directory: `have_anchor_L1L2_pairwise_borderline_rerank_v5/grid_outputs/`
- Parameters: boundary score gap, non-specificity penalty, candidate boost, LLM confidence, and minimum number of wins.
- Conclusion: The LLM only provides small value in the uncertain zone, and overall it does not outperform the strong deterministic ranker.

### 5.3 Per-drug Threshold Calibration

- Directory: `have_anchor_L1L2_statistical_per_drug_calibrated_v7/outputs_per_drug_calibrated/`
- Processing: Raise thresholds for high-frequency false-positive medications and lower thresholds for low-recall medications.
- Conclusion: It can change the precision-recall balance, but is stable only after the underlying scores have been well calibrated.

### 5.4 DDI Strength Parameters

- v9: DDI penalty/margin/max hits grid.
- v11: Four settings: no constraint, mild, balanced, and strong DDI.
- Conclusion: When DDI decreases from 13.62% to 6.60%, Jaccard drops by only about 0.0035. If the main metric emphasizes overall safety, strong DDI has the highest SAJ, but balanced DDI is more suitable as the default clinical trade-off.

### 5.5 Medication Count Parameters

- Directory: `have_anchor_L1L2_v13_count_policy_v13_1/`
- Comparison: `count_offset` and `max_extra`.
- Conclusion: A fixed tendency toward 7 medications improves Recall but introduces FPs; `offset1-extra0` is more balanced in Jaccard, DDI, and SAJ.

### 5.6 v13.1/v13.2 Rule Parameters

- v13.1: Within-class reranking, precision target for high-frequency false-positive medications, and balanced DDI.
- v13.2: Dynamic Top5/6 truncation and conditional release of high-frequency medications.
- Conclusion: Post-processing can improve Jaccard/DDI, but it does not change the underlying raw PR-AUC; improving PR-AUC requires retraining or calibrating the probability ranking model.

## 6. Ablation Experiments

### 6.1 LLM Free Generation, Constrained Rerank, and Deterministic Ranking

- Free generation v1: Jaccard 0.1640.
- Constrained Top12 rerank v3: Jaccard 0.2353.
- Deterministic Top5 v4: Jaccard 0.2648.
- Pairwise judge v5/v6: Jaccard 0.2634/0.2665.
- Conclusion: LLM free generation falls back to generic high-frequency ICU medications; constraining the LLM to boundary judgment is clearly more stable, but the core decision should still be handled by the statistical ranker.

### 6.2 DDI Constraint Ablation (v11 full test)

- No DDI: Jaccard 0.2651, DDI 13.62%.
- Mild: Jaccard 0.2645, DDI 11.03%.
- Balanced: Jaccard 0.2637, DDI 8.96%.
- Strong constraint: Jaccard 0.2616, DDI 6.60%.
- Conclusion: Set-level DDI constraints can significantly improve safety with limited performance loss; frequently co-used drugs in training are assigned lower penalties.

### 6.3 Weak Evidence + Pairwise + Two-stage Model Ablation (3,000/500/300)

| Setting | Jaccard | F1 | Recall | Drug micro PR-AUC | Case mean PR-AUC | DDI | SAJ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Matched small-sample pointwise baseline | 0.2717 | 0.3938 | 0.4133 | 0.3495 | 0.4733 | 10.57% | 0.2430 |
| Improved version, no DDI | **0.3187** | **0.4420** | **0.4443** | **0.4105** | **0.5169** | 11.97% | 0.2805 |
| Improved version, mild DDI | 0.3153 | 0.4385 | 0.4406 | 0.4105 | 0.5169 | 9.57% | 0.2851 |
| Improved version, balanced DDI | 0.3168 | 0.4406 | 0.4426 | 0.4105 | 0.5169 | **7.75%** | **0.2923** |

The processing includes a decoupled memory main channel, an attenuated weak-evidence auxiliary channel, 357,200 pairwise hard-negative training pairs, two-stage class-drug probabilities, per-drug Platt calibration, a medication count classifier, and DDI-aware beam search.

Conclusion: Weakly related evidence should not be directly merged into the main memory; instead, it should be used as an auxiliary feature with attenuated weights. With pairwise ranking, PR-AUC, Jaccard, F1, and DDI can improve simultaneously. This 300-case experiment is for method development and ablation; the full 72,436/9,054/9,055 replication result is reported in `final_balanced_main`.

Path: `paper_experiments/v13_3_weak_pairwise_hierarchical_small/outputs_test300/experiment_report.json`


## 7. Main Result Organization

1. **Final proposed method**: Use Improved Balanced (weak-evidence assistance, pairwise, two-stage ranking, Platt, count classifier, and balanced DDI beam search).
2. **Full-test result of the main method**: On the official 72,436/9,054/9,055 split, Improved Balanced achieves Jaccard 0.3246, F1 0.4533, drug-level micro PR-AUC 0.4161, DDI 8.31%, and SAJ 0.2976.
3. **Development-set result**: On the fixed 300 test cases, Jaccard 0.3168, F1 0.4406, drug-level micro PR-AUC 0.4105, DDI 7.75%, and SAJ 0.2923 are used only as method development records.
4. **Safety parameter table**: Use the four v11 DDI-constraint settings to show the performance-safety trade-off.
5. **Model component ablation table**: Deterministic Top5, LLM rerank, pairwise evidence gate, v12 hierarchy, v13 calibration, and the weak-evidence pairwise small-sample experiment.
6. **L1/L2 decoupling table**: Report decoupled and non-decoupled settings separately, emphasizing predictive gains and safety costs.
7. **Class-level results**: Use them as auxiliary analysis to show that the model more easily predicts treatment directions, while exact-drug discrimination remains the main bottleneck.

## 8. Current Overall Conclusions


- DeepSeek is more suitable as a boundary evidence judge and is not suitable for freely generating the final prescription set.
- L1/L2 statistical memory, hospital prior, and anchor prior are effective for candidate recall; weakly related information should be used through an independent attenuation channel.
- Probability calibration directly affects PR-AUC; thresholding and Top-k post-processing mainly affect Jaccard/F1/DDI and do not change raw PR-AUC.
- DDI should be handled at the combination level, incorporating severity and training co-use frequency to avoid incorrectly removing clinically common co-medications.
- The final proposed method is determined to be Improved Balanced; both the 3,000/500/300 development experiment and the full 72,436/9,054/9,055 split replication have been completed.

## 9. Key Report Paths

- v8: `.../have_anchor_L1L2_lightweight_reranker_v8/outputs_lightweight_reranker/lightweight_reranker_report.json`
- v9: `.../have_anchor_L1L2_lightweight_reranker_ddi_aware_v9/outputs_lightweight_reranker/ddi_aware_grid_summary.json`
- v11: `.../have_anchor_L1L2_full_reranker_ddi_ablation_v11/outputs_full_reranker_ddi_ablation/full_reranker_ddi_ablation_report.json`
- v12: `.../have_anchor_L1L2_hierarchical_calibrated_v12/outputs_v12_test1000/v12_report.json`
- v13: `.../have_anchor_L1L2_asymmetric_calibrated_v13/outputs_v13_test1000/v13_report.json`
- v13.1: `.../have_anchor_L1L2_refined_v13_1/outputs_v13_1_test1000/v13_1_report.json`
- v13.2: `.../have_anchor_L1L2_refined_v13_2/outputs_v13_2_test1000/v13_2_report.json`
- Improved Balanced full: `paper_experiments/final_balanced_main/outputs_full_split/full_split_report.json`
- v14: `.../have_anchor_L1L2_full_vocabulary_pairwise_v14/outputs_v14_test1000/v14_report.json`
- Decoupling ablation: `experiments/l1_l2_undecoupled_ablation_v13_3/outputs_undecoupled_full_test/decoupled_vs_undecoupled_v13_3_comparison.json`
- Weak-evidence ablation: `paper_experiments/v13_3_weak_pairwise_hierarchical_small/outputs_test300/experiment_report.json`
