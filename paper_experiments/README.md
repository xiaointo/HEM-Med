# Paper Experiment Code

This directory contains the original core scripts used for the paper experiment lineage and the final Improved Balanced run. It is kept separate from `src/`, which is a cleaner reference implementation for reading and lightweight smoke tests.

## Layout

```text
paper_experiments/
  final_balanced_main/
    run_full_experiment.py
    run_full.sh
    README.md
  v13_3_weak_pairwise_hierarchical_small/
    run_experiment.py
  memory_and_reranking/
    have_anchor_L1L2_full_reranker_ddi_ablation_v11/
    have_anchor_L1L2_hierarchical_calibrated_v12/
    have_anchor_L1L2_asymmetric_calibrated_v13/
    have_anchor_L1L2_refined_v13_1/
    have_anchor_L1L2_score_calibrated_v13_3_full/
```

## Final Paper Method

The final paper method is implemented by:

```bash
HEM_MED_DATA_DIR=/path/to/hem_med_data \
  bash paper_experiments/final_balanced_main/run_full.sh
```

The expected artifact layout is documented in `../docs/DATA_AND_ARTIFACTS.md`.

## Notes

- No raw EHR data, generated outputs, predictions, checkpoints, or API keys are included.
- The historical v11-v13 import chain is preserved because the final run imports those scripts.
