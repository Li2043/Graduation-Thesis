# Merged Original-observation evaluation data

These three CSVs are the merged (all-shard) held-out evaluation outputs for
the Original (18D) observation condition, sourced from `F:\正式训练\outputs\welfare_analysis\`:

- `taskonly_evaluation_merged.csv` — task-only Baseline.
- `formal_welfare_evaluation_merged.csv` — Mean / GGI / Maximin welfare outcomes.
- `formal_behavioral_evaluation_merged.csv` — behavioural/mechanism measures.

They are the direct inputs consumed by the `ch5_baseline/` and `pooled12/`
analysis scripts. Per-shard files (`*_shard{N}.csv`) were intentionally not
copied — they are redundant with these merged files.
