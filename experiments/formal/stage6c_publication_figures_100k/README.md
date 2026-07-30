# Stage 6C — Publication-quality formal result figures (100k)

Read-only consumer of Stage 6B analysis artefacts tagged `formal-analysis-100k-complete`.

## Run

```text
python experiments/formal/stage6c_publication_figures_100k/scripts/run_stage6c_figures.py \
  --analysis-tag formal-analysis-100k-complete \
  --analysis-worktree <verified-analysis-worktree> \
  --output-root experiments/formal/stage6c_publication_figures_100k \
  --formats pdf svg png \
  --png-dpi 600
```

## Constraints

- Statistical unit = formal training seed
- No policy retraining
- No evaluation environment re-execution
- No convention zero-fill
- No checkpoint interpolation
