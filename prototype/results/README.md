# Results Index

Experiment outputs are grouped by prototype version.

```text
results/
├── README.md                 ← this file
├── v0.2/                     ← historical random-policy (see GitHub tag v0.2)
├── v0.3/                     ← historical global Rawlsian DQN (see GitHub tag v0.3)
└── v0.4/                     ← current: ego-neighbourhood scope
    ├── random/               ← random baseline vs Rawlsian CSV + plots
    ├── trained/              ← trained DQN evaluation CSV + plots
    └── diagnostic/           ← least-advantaged & scope diagnostics
```

## Current active outputs (v0.4)

| Path | Description |
|------|-------------|
| `v0.4/random/random_summary.csv` | Random policy comparison |
| `v0.4/trained/trained_summary.csv` | **Main trained DQN comparison** |
| `v0.4/diagnostic/neighbourhood_scope_diagnostics.csv` | Global vs neighbourhood scope |
| `v0.4/diagnostic/least_advantaged_diagnostics.csv` | Global least-advantaged identity |

Scripts write to these paths via `config.py`.

## Historical versions (restored locally)

| Version | Path | Source |
|---------|------|--------|
| v0.2 | `v0.2/` | Git tag `v0.2` — random policy only |
| v0.3 | `v0.3/random/`, `v0.3/trained/` | Git tag `refs/tags/v0.3` — global Rawlsian DQN |
| v0.4 | `v0.4/` | Latest local runs — ego_neighbourhood scope |

To re-restore from Git:

```powershell
git show refs/tags/v0.2:results/random_summary.csv
git show refs/tags/v0.3:results/trained_summary.csv
```
