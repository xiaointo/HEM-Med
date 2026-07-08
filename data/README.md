# Data Directory

This directory is intentionally empty in the anonymous GitHub release.

Place locally obtained or regenerated artifacts here when running the code:

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

The raw EHR data and full generated memory artifacts are not redistributed. See `docs/DATA_AND_ARTIFACTS.md` for schemas, expected paths, and generation notes.
