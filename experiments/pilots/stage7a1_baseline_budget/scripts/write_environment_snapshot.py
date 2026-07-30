#!/usr/bin/env python3
"""Write environment / requirements snapshot for Stage 7A-1."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
PILOT_ROOT = SCRIPT.parents[1]
REPO_ROOT = SCRIPT.parents[4]


def main() -> int:
    import numpy
    import pandas
    import torch

    try:
        import gymnasium
        gym_v = gymnasium.__version__
    except Exception:
        gym_v = "unknown"
    try:
        import scipy
        scipy_v = scipy.__version__
    except Exception:
        scipy_v = "unknown"
    try:
        import matplotlib
        mpl_v = matplotlib.__version__
    except Exception:
        mpl_v = "unknown"

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=str(REPO_ROOT), text=True
    ).strip()
    protocol = PILOT_ROOT / "configs" / "stage7a1_baseline_budget_protocol.yaml"
    import hashlib

    h = hashlib.sha256(protocol.read_bytes()).hexdigest()
    (PILOT_ROOT / "source_commit.txt").write_text(head + "\n", encoding="utf-8")
    (PILOT_ROOT / "configs" / "protocol_hash.txt").write_text(h + "\n", encoding="utf-8")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (PILOT_ROOT / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    req = "\n".join(
        [
            f"torch=={torch.__version__}",
            f"numpy=={numpy.__version__}",
            f"pandas=={pandas.__version__}",
            f"scipy=={scipy_v}",
            f"matplotlib=={mpl_v}",
            f"gymnasium=={gym_v}",
        ]
    )
    (PILOT_ROOT / "pilot_requirements.txt").write_text(req + "\n", encoding="utf-8")
    snap = {
        "python": sys.version,
        "os": platform.platform(),
        "cpu": platform.processor(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy_v,
        "matplotlib": mpl_v,
        "gymnasium": gym_v,
        "git_commit": head,
        "branch": branch,
        "protocol_sha256": h,
    }
    (PILOT_ROOT / "environment_snapshot.json").write_text(
        json.dumps(snap, indent=2), encoding="utf-8"
    )
    print(json.dumps(snap, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
