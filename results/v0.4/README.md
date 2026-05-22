# v0.4 Results (Ego-neighbourhood Rawlsian)

**Current active results.** Generated with:

```python
RAWLSIAN_SCOPE = "ego_neighbourhood"
EGO_NEIGHBOURHOOD_RADIUS = 50.0
```

## Subfolders

| Folder | Contents |
|--------|----------|
| `random/` | 100-episode random baseline vs Rawlsian |
| `trained/` | 50-episode trained DQN evaluation (**primary**) |
| `diagnostic/` | Scope & least-advantaged diagnostics |

## Key file for thesis

`trained/trained_summary.csv`

## Re-generate

```bash
python train_dqn_baseline.py
python train_dqn_rawlsian.py
python evaluate_trained.py
python run_random_baseline.py
python run_random_rawlsian.py
python evaluate_random.py
python diagnose_neighbourhood_scope.py
```
