# HEM-Med

HEM-Med is a hierarchical agentic memory framework for safe multi-center medication recommendation. This anonymous release contains the paper experiment code for the final Improved Balanced method.

This public release excludes raw EHR data, generated intermediate files, trained weights, full memory JSON files, predictions, API responses, and private API keys.

## Repository Layout

```text
anonymous-HEM-Med/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml
├── paper_experiments/                       # original paper experiment lineage
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

## Which Code Supports the Paper?

Use `paper_experiments/` for the paper results. It preserves the original final method implementation and the v11-v13.3 dependency chain used by the full official split experiment.

The final paper method is implemented by:

- `paper_experiments/final_balanced_main/run_full_experiment.py`
- `paper_experiments/v13_3_weak_pairwise_hierarchical_small/run_experiment.py`
- `paper_experiments/memory_and_reranking/**`

The `src/` directory is a clean reference implementation that mirrors the main concepts in smaller modules. It is useful for reading, testing schemas, and smoke tests, but it is not the exact code used to produce the final paper metrics.

## Data

Raw EHR data and generated memory artifacts are not redistributed. Place local artifacts under `data/`, or set `HEM_MED_DATA_DIR` to an external artifact directory.

Expected local layout:

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

See `docs/DATA_AND_ARTIFACTS.md` for the full artifact table, fields, and generation guidance.

## Reproducing the Final Paper Run

After obtaining or regenerating the required artifacts:

```bash
export HEM_MED_DATA_DIR=/path/to/hem_med_data
bash paper_experiments/final_balanced_main/run_full.sh
```

This run corresponds to the Improved Balanced full official split result described in `docs/EXPERIMENT_SUMMARY.md`.

## Lightweight Reference Scripts

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

## Dependencies

The final reranking path uses only the Python standard library. The L3 DeepSeek/OpenAI-compatible construction path requires the optional packages listed in `requirements.txt`.

## Anonymous Release Notes

- No patient data are included.
- No generated L3/L1/L2 result files are included.
- No API keys are included.
