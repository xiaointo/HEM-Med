<img src="Pictures/hem-med.png" alt="title" border="0">

<p float="left"> <img src="https://img.shields.io/badge/python-v3.10+-red"> <img src="https://img.shields.io/badge/reranker-pure%20python-blue"> <img src="https://img.shields.io/badge/API-DeepSeek%2FOpenAI--compatible-green"> </p>

HEM-Med

## ✨ Overview

<div align="center">
   <img src="Pictures/Gap.png" alt="motivation" width="70%" border="0">
</div>

**Figure 1**: Motivation of HEM-Med. Multi-center medication recommendation faces both cross-hospital knowledge transfer challenges and medication safety constraints.

HEM-Med is motivated by two key gaps in safe multi-center medication recommendation.

### Gap 1: Entangled Cross-Hospital Knowledge

Existing methods for multi-center medication recommendation often overlook a subtle but critical clinical reality: prescribing decisions are influenced not only by generalizable medical knowledge, but also by hospital resources, institution-specific treatment pathways, local medication protocols, departmental expertise, and physician prescribing preferences.

As illustrated in the motivation figure, different hospitals may exhibit distinct medication distributions even for the same disease condition. In complex ICU scenarios with multiple coexisting conditions, directly transferring such entangled knowledge across hospitals may lead to negative transfer. In other words, knowledge acquired from one hospital may not always be clinically effective or locally feasible when directly applied to another hospital.

### Gap 2: Insufficient Safety Awareness

Existing transfer-based recommendation methods mainly focus on representation alignment, while paying insufficient attention to the safety constraints inherent in clinical medication recommendation. Unlike general cross-domain recommendation, medication recommendation is a clinically constrained multi-label prediction problem that must jointly consider treatment requirements, medication safety, potential drug-drug interactions, and local formulary feasibility.

If multi-hospital recommendation only accounts for distributional discrepancies without explicitly modeling DDI risks and local feasibility, it may generate clinically inappropriate or unsafe medication combinations, posing significant risks to real-world clinical deployment.

## 🧠 Method Framework

To address these challenges, HEM-Med proposes a **hierarchical agentic memory framework** for safe multi-center medication recommendation.

<div align="center">
   <img src="Pictures/framework.png" alt="framework" width="70%" border="0">
</div>

**Figure 2**: Overview of HEM-Med. The framework constructs patient-level attribution memory, aggregates it into cross-hospital consensus memory and hospital-specific residual memory, and generates the final medication set through calibrated prediction and DDI-aware decoding.

HEM-Med contains three memory levels:

| Memory Level                            | Role                                                                                 | Purpose                                                                                        |
| --------------------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| **L3 Patient-level Attribution Memory** | Builds patient-context attribution records from stay-medication observations.        | Captures fine-grained medication-context associations from historical ICU stays.               |
| **L1 Consensus Memory**                 | Aggregates cross-hospital statistical experience shared across centers.              | Models transferable medical experience and reduces noisy direct hospital-to-hospital transfer. |
| **L2 Residual Memory**                  | Captures hospital-specific amplified, suppressed, or localized prescribing patterns. | Preserves local feasibility and hospital-specific medication preferences.                      |

During online recommendation, the original L3 records are **not directly retrieved**. Instead, L3 is used offline to construct L1 consensus memory, L2 hospital residual memory, and attenuated weak auxiliary evidence.

## ✅ How HEM-Med Addresses the Gaps

| Gap                                           | HEM-Med Solution                                                                                                                                                                                                                               |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Gap 1: Entangled cross-hospital knowledge** | HEM-Med disentangles transferable cross-hospital consensus experience from hospital-specific residual prescribing patterns. L1 memory captures generalizable medication-context associations, while L2 memory models local hospital residuals. |
| **Gap 2: Insufficient safety awareness**      | HEM-Med integrates DDI-aware candidate scoring and DDI-aware beam-search decoding. The final medication set is selected by considering both recommendation accuracy and potential drug-drug interaction risks.                                 |

In this way, HEM-Med does not simply align representations across hospitals. Instead, it explicitly separates what should be transferred across hospitals from what should remain hospital-specific, and then performs safety-aware medication set generation.

## 🔬 Experimental Workflow

The paper experiments follow the pipeline below:

1. **Construct L3 patient-level attribution records**
   DeepSeek/OpenAI-compatible API calls are used to select constrained patient-context anchors for stay-medication observations.

2. **Build hierarchical memory**
   L3 records are aggregated into L1 cross-hospital consensus memory and L2 hospital-specific residual memory.

3. **Generate candidate medications**
   Candidate medications are retrieved from L1/L2 memory, attenuated weak evidence, global priors, hospital priors, anchor priors, and DDI-aware features.

4. **Train calibrated reranking models**
   HEM-Med trains pointwise and pairwise logistic rerankers, followed by two-stage class-drug scoring and per-drug Platt calibration.

5. **Predict medication count**
   A medication-count classifier estimates the expected size of the final medication set.

6. **Decode the final medication set**
   DDI-aware beam search generates the final recommendation set by balancing medication relevance, predicted set size, redundancy, and DDI risk.

## 📊 Main Reported Result

The final paper method is **Improved Balanced**, which combines:

* decoupled L1/L2 memory retrieval;
* attenuated weak-memory auxiliary evidence;
* pointwise and pairwise logistic reranking;
* two-stage class-drug scoring;
* per-drug Platt calibration;
* medication-count prediction;
* DDI-aware beam-search decoding.

The full official split result is evaluated on the official **72,436 / 9,054 / 9,055** train / validation / test split.

| Metric                  |  Value |
| ----------------------- | -----: |
| Precision               | 0.4511 |
| Recall                  | 0.4614 |
| F1                      | 0.4533 |
| Jaccard                 | 0.3246 |
| Drug-level micro PR-AUC | 0.4161 |
| Case-level mean PR-AUC  | 0.5301 |
| DDI Rate                |  8.31% |
| SAJ                     | 0.2976 |
| SafeScore               | 0.3649 |

These results show that HEM-Med improves medication recommendation accuracy while maintaining a favorable accuracy-safety trade-off. More detailed experimental records are provided in [`docs/EXPERIMENT_SUMMARY.md`](docs/EXPERIMENT_SUMMARY.md).

## 🚀 Reproducing the Paper Experiments

This section describes how to reproduce the main paper experiment and how to run lightweight checks in the anonymous release.

### 1. Prepare the Environment

Python 3.10+ is recommended. Create a virtual environment and install the optional API dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The final reranking pipeline uses only the Python standard library. The packages in `requirements.txt` are mainly required for DeepSeek/OpenAI-compatible L3 construction.

### 2. Prepare Data and Artifacts

Raw EHR data and generated memory artifacts are not redistributed in this anonymous release. To run the paper code, prepare the required artifacts locally and place them under `data/`, or set `HEM_MED_DATA_DIR` to an external artifact directory.

Expected local layout:

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

The historical v11-v13 dependency chain also expects private experiment inputs under:

```text
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

For detailed artifact descriptions, schemas, expected paths, and generation notes, see [`docs/DATA_AND_ARTIFACTS.md`](docs/DATA_AND_ARTIFACTS.md).

### 3. Reproduce the Final Paper Result

After preparing the required artifacts, run the final **Improved Balanced** experiment:

```bash
export HEM_MED_DATA_DIR=/path/to/hem_med_data
bash paper_experiments/final_balanced_main/run_full.sh
```

This run corresponds to the full official **72,436 / 9,054 / 9,055** train / validation / test split reported in the paper.

Generated outputs are written to:

```text
paper_experiments/final_balanced_main/outputs_full_split/
```

Important generated files include:

```text
full_split_report.json
predictions_balanced.jsonl
case_level_average_precision.jsonl
pointwise_weights.json
pairwise_weights.json
class_model_weights.json
count_classifier_weights.json
per_drug_platt_parameters.json
medication_class_map.json
```

These files are generated artifacts and are therefore not included in the anonymous release.

### 4. Run the Development Ablation

The weak-evidence pairwise development ablation can be run with:

```bash
export HEM_MED_DATA_DIR=/path/to/hem_med_data
python3 paper_experiments/v13_3_weak_pairwise_hierarchical_small/run_experiment.py
```

It writes outputs under:

```text
paper_experiments/v13_3_weak_pairwise_hierarchical_small/outputs_test300/
```

This experiment uses a smaller **3,000 / 500 / 300** development setting and is mainly intended for method development and ablation analysis. Publication-level exact-drug results should use the full official split run above.

### 5. Optional: Run L3 Construction

The L3 construction code is provided for transparency. It uses DeepSeek/OpenAI-compatible API calls to construct patient-level attribution records.

Build compact stay-level L3 inputs:

```bash
cd l3_construction
bash scripts/deepseek_l3_case_level_generation/run_14_build_minimal_case_input_for_llm.sh
```

Run single-anchor L3 generation:

```bash
cd l3_construction
cp configs/private/deepseek_key_template.py configs/private/deepseek_key.py
# Fill configs/private/deepseek_key.py locally, or configure your environment.
bash scripts/deepseek_l3_min_token_single_anchor_run/run_01_deepseek_l3_min_token_single_anchor.sh
```

Postprocess generated L3 records:

```bash
cd l3_construction
bash scripts/deepseek_l3_min_token_single_anchor_run/run_02_postprocess_l3_reasonable_anchor_limit20_promptv2_cleaninput.sh
```

Files under `l3_construction/outputs/` are not included in the anonymous release because they may contain private EHR-derived intermediate information, API responses, or run metadata.

### 6. Optional: Run Lightweight Smoke Tests

The `src/` and `scripts/` directories provide a simplified reference pipeline for schema checking and small local smoke tests. These scripts are not used to reproduce the final paper metrics.

Build compact L1/L2 memories:

```bash
bash scripts/run_memory.sh
```

Generate candidates and train the lightweight reranker:

```bash
bash scripts/run_train.sh
```

Run reference candidate generation, DDI-aware decoding, and evaluation:

```bash
bash scripts/run_eval.sh
```

If artifacts are outside the repository, set:

```bash
export HEM_MED_DATA_DIR=/path/to/hem_med_data
```

### 7. Check the Output Report

After running the final experiment, check:

```text
paper_experiments/final_balanced_main/outputs_full_split/full_split_report.json
```

The expected main reported result is:

| Metric                  |  Value |
| ----------------------- | -----: |
| Precision               | 0.4511 |
| Recall                  | 0.4614 |
| F1                      | 0.4533 |
| Jaccard                 | 0.3246 |
| Drug-level micro PR-AUC | 0.4161 |
| Case-level mean PR-AUC  | 0.5301 |
| DDI Rate                |  8.31% |
| SAJ                     | 0.2976 |
| SafeScore               | 0.3649 |
