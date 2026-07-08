<img src="Pictures/hem-med.svg" alt="title" border="0">

<p float="left">
  <img src="https://img.shields.io/badge/python-v3.10+-red">
  <img src="https://img.shields.io/badge/reranker-pure%20python-blue">
  <img src="https://img.shields.io/badge/API-DeepSeek%2FOpenAI--compatible-green">
</p>

# HEM-Med

This repository provides the anonymous implementation of **“HEM-Med: A Hierarchical Agentic Memory Framework for Safe Multi-Center Medication Recommendation”**. HEM-Med is designed for safe multi-center ICU medication recommendation by disentangling cross-hospital consensus experience from hospital-specific residual prescribing patterns and generating medication sets through memory retrieval, calibrated reranking, and DDI-aware decoding.

This anonymous release contains two complementary code views:

* `paper_experiments/`: the original core experiment scripts that support the paper results, including the final **Improved Balanced** full-split run and its v11-v13.3 dependency chain.
* `src/`: a cleaner, modular reference implementation for reading the method, checking schemas, and running lightweight smoke tests.

Raw EHR data, generated L3/L1/L2 memories, trained outputs, predictions, API responses, and private API keys are intentionally excluded from this anonymous release.

<div align="center">
   <img src="Pictures/framework.png" alt="framework" width="70%" border="0">
</div>

**Figure 1**: Overview of HEM-Med. The framework constructs patient-level attribution memory, aggregates it into cross-hospital consensus memory and hospital-specific residual memory, and generates the final medication set through calibrated prediction and DDI-aware decoding.

## ✨ Overview

HEM-Med addresses two key challenges in multi-center medication recommendation:

1. **Cross-hospital transfer risk**: Directly transferring prescribing knowledge across hospitals may introduce negative transfer because hospitals differ in patient populations, formularies, treatment routines, and local prescribing preferences.
2. **Medication safety**: ICU medication recommendation is a constrained multi-label prediction problem. The final medication set should consider both therapeutic matching and drug-drug interaction risk.

HEM-Med introduces a hierarchical agentic memory framework with three levels:

* **L3 patient-level attribution memory**: patient-context attribution records constructed from stay-medication observations.
* **L1 consensus memory**: cross-hospital statistical experience shared across centers.
* **L2 residual memory**: hospital-specific amplified, suppressed, or localized prescribing patterns.

During online recommendation, the original L3 records are not directly retrieved. Instead, L3 is used offline to construct L1 consensus memory, L2 hospital residual memory, and attenuated weak auxiliary evidence.

## 📁 Repository Layout

```text
anonymous-HEM-Med/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml
├── paper_experiments/                       # exact paper experiment lineage
│   ├── README.md
│   ├── final_balanced_main/                 # final Improved Balanced full split run
│   ├── v13_3_weak_pairwise_hierarchical_small/
│   └── memory_and_reranking/                # v11-v13.3 dependency chain
├── src/                                     # clean reference implementation
│   ├── build_l1_memory.py
│   ├── build_l2_memory.py
│   ├── candidate_generation.py
│   ├── train_reranker.py
│   ├── ddi_decoding.py
│   ├── evaluate.py
│   └── utils.py
├── scripts/                                 # lightweight reference scripts
│   ├── run_memory.sh
│   ├── run_train.sh
│   └── run_eval.sh
├── data/
│   └── README.md
└── docs/
    ├── DATA_AND_ARTIFACTS.md
    ├── EXPERIMENT_SUMMARY.md
    ├── L3_DEEPSEEK_CONSTRUCTION_METHOD_SECTION.md
    └── ORIGINAL_HEM_MED_README.md
```

## 🧪 Which Code Supports the Paper Results?

Use `paper_experiments/` for paper-result reproduction.

The final paper method is implemented by:

```text
paper_experiments/final_balanced_main/run_full_experiment.py
paper_experiments/v13_3_weak_pairwise_hierarchical_small/run_experiment.py
paper_experiments/memory_and_reranking/**
```

The full paper result reported in the experiment summary is the **Improved Balanced** run on the official 72,436 / 9,054 / 9,055 train / validation / test split.

The `src/` directory is intentionally simpler. It mirrors the main ideas in readable modules but is **not** the exact code used to produce the final paper metrics.

## 📊 Main Reported Result

The final paper method is **Improved Balanced**, which combines:

* decoupled L1/L2 memory retrieval;
* attenuated weak-memory auxiliary evidence;
* pointwise and pairwise logistic reranking;
* two-stage class-drug scoring;
* per-drug Platt calibration;
* medication-count prediction;
* DDI-aware beam-search decoding.

The full official split result is summarized in `docs/EXPERIMENT_SUMMARY.md`:

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

**For more detailed experimental results, please see [`docs/EXPERIMENT_SUMMARY.md`](docs/EXPERIMENT_SUMMARY.md).**

## 🧠 Method Pipeline

<div align="center">
   <img src="Pictures/pipeline.png" alt="pipeline" width="70%" border="0">
</div>

**Figure 2**: Core experimental pipeline of HEM-Med, including L3 construction, L1/L2 memory aggregation, candidate generation, calibrated reranking, medication-count prediction, and DDI-aware decoding.

The main pipeline consists of the following stages:

1. Construct patient-level L3 attribution records from stay-medication observations.
2. Aggregate L3 records into L1 cross-hospital consensus memory and L2 hospital-specific residual memory.
3. Retrieve candidate medications from L1/L2 memory, weak auxiliary evidence, priors, and DDI-aware features.
4. Train pointwise and pairwise reranking models.
5. Apply two-stage class-drug scoring and per-drug Platt calibration.
6. Predict medication count and decode the final medication set with DDI-aware beam search.

## 🛠️ Environment

Python 3.10+ is recommended.

Create an environment and install optional API dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` intentionally contains only:

```text
openai>=1.0.0
httpx>=0.24.0
```

These packages are needed only for L3 DeepSeek/OpenAI-compatible API construction. The pure-Python reranking path does not require PyTorch, scikit-learn, NumPy, or pandas.

## 📦 Data and Artifacts

This repository does not redistribute raw EHR data or private generated memory artifacts. To run the paper code, prepare artifacts locally and either place them under `data/` or point `HEM_MED_DATA_DIR` to an external artifact directory.

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

**For the full artifact table, schemas, expected paths, and generation notes, please see [`docs/DATA_AND_ARTIFACTS.md`](docs/DATA_AND_ARTIFACTS.md).**

## 🚀 Reproduce the Final Paper Experiment

After preparing the required artifacts, run:

```bash
export HEM_MED_DATA_DIR=/path/to/hem_med_data
bash paper_experiments/final_balanced_main/run_full.sh
```

The run writes generated outputs under:

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

These files are not included in the anonymous release because they are generated artifacts.

## 🔬 Development Ablation

The development ablation for the weak-evidence pairwise method is available at:

```bash
export HEM_MED_DATA_DIR=/path/to/hem_med_data
python3 paper_experiments/v13_3_weak_pairwise_hierarchical_small/run_experiment.py
```

It writes outputs under:

```text
paper_experiments/v13_3_weak_pairwise_hierarchical_small/outputs_test300/
```

This 3,000 / 500 / 300 experiment is for method development and ablation. Publication-level exact-drug results should use the full official split run above.

## 🧩 Lightweight Reference Pipeline

The scripts under `scripts/` run the simplified modular implementation in `src/`. They are useful for schema checks and small local smoke tests, not for reproducing the final paper numbers.

Build compact L1/L2 memories from a local train split:

```bash
bash scripts/run_memory.sh
```

Generate candidates and train the lightweight reference reranker:

```bash
bash scripts/run_train.sh
```

Run reference candidate generation, DDI-aware decoding, and evaluation:

```bash
bash scripts/run_eval.sh
```

If artifacts are outside the repository:

```bash
export HEM_MED_DATA_DIR=/path/to/hem_med_data
bash scripts/run_train.sh
```

## 📝 L3 Construction Notes

The L3 construction method uses DeepSeek/OpenAI-compatible API calls to select constrained patient-context anchors. The method text and prompt/cleaning logic are documented in:

```text
docs/L3_DEEPSEEK_CONSTRUCTION_METHOD_SECTION.md
```

Private API keys are not included. If running L3 construction locally, set `DEEPSEEK_API_KEY` in the environment or use a private local key file that is not committed.

## 🔒 Anonymous Release Policy

The following are intentionally not included:

* raw EHR data;
* generated L3/L1/L2 memory files;
* trained weights and generated reports;
* prediction files;
* API raw responses and token-usage logs;
* private API keys;
* PDF manuscripts or identifying metadata;
* `outputs/`, `__pycache__/`, `*.pyc`, `*.sqlite`, generated `*.jsonl`, and generated `*.csv` files.

The `.gitignore` is configured to exclude these categories.

## 🧭 Troubleshooting

* If `run_full.sh` fails with a missing split file, check `HEM_MED_DATA_DIR` and the `data/splits/split_8_1_1_nonempty_med_seed1203/` layout.
* If it fails with a missing memory or prior JSON, check `docs/DATA_AND_ARTIFACTS.md` and populate the historical v11 input directory.
* If only a code smoke test is needed, use the lightweight `scripts/` pipeline with a tiny synthetic `sample_data/` directory.
* If using git, confirm `git status --short` before upload and make sure no generated `outputs/`, data, cache, or key files are staged.

## 📖 Citation

This is an anonymous review release. Citation metadata can be added after de-anonymization.

## 🌟 Contributions and Suggestions

Contributions and suggestions are welcome after the anonymous review period.
