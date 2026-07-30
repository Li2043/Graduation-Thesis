"""Policy-transition braking acceleration from physics substeps (Stage 3B-R1)."""

from __future__ import annotations

import math
from typing import Sequence


def policy_braking_acceleration(substep_realised: Sequence[float | None]) -> float:
    """Most negative realised acceleration across active physics substeps.

    ``None`` entries (inactive / exited placeholders) are ignored.
    If no active substeps remain, returns 0.0.
    """
    active = [float(a) for a in substep_realised if a is not None and math.isfinite(float(a))]
    if not active:
        return 0.0
    return float(min(active))


def braking_magnitude(a_policy: float) -> float:
    return float(max(0.0, -float(a_policy)))


__all__ = ["braking_magnitude", "policy_braking_acceleration"]
