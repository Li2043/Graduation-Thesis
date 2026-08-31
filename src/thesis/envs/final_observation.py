"""Markovian learner observations for MergeEnvCandidateV3 (Stage 4A-0R).

Fixed dimension and documented ordering (index → feature):

 0  own normalised route progress ρ ∈ [0, 1] (clipped)
 1  own speed / target_speed (clipped to [0, 2])
 2  own distance_to_merge / L_ref (clipped to [-1, 2])
 3  own distance_to_exit / L_ref (clipped to [0, 2])
 4  own traffic-role indicator (+1 mainline, -1 ramp)
 5  own active_on_road (1) / completed (0)  — active flag
 6  peer normalised route progress
 7  peer speed / target_speed
 8  peer distance_to_merge / L_ref
 9  peer distance_to_exit / L_ref
10  peer traffic-role indicator
11  peer active flag
12  B_front normalised progress
13  B_front speed / target_speed
14  B_front active flag
15  B_rear normalised progress
16  B_rear speed / target_speed
17  B_rear active flag
18  nearest-front bumper gap / L_ref (0 if invalid)
19  nearest-front relative speed / v_ref (0 if invalid)
20  nearest-rear bumper gap / L_ref (0 if invalid)
21  nearest-rear relative speed / v_ref (0 if invalid)
22  front-gap validity mask ∈ {0, 1}
23  rear-gap validity mask ∈ {0, 1}
24  relative world-x to peer / L_ref (clipped [-2, 2])
25  relative world-y to peer / L_ref (clipped [-2, 2])
26  relative speed to peer / v_ref (clipped [-2, 2])

Clip policy: only the features listed above are clipped; masks and role
indicators are exact. Speed/completion inputs for Mean/Min PBRS are recoverable
from the environment Markov state (route progress, speed, completed flags).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

OBSERVATION_DIM = 27
OBSERVATION_L_REF = 100.0
OBSERVATION_V_REF = 20.0


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _finite(x: float, default: float = 0.0) -> float:
    return float(x) if math.isfinite(x) else default


def build_learner_observation(
    *,
    own: Mapping[str, Any],
    peer: Mapping[str, Any],
    b_front: Mapping[str, Any],
    b_rear: Mapping[str, Any],
    front_gap: float | None,
    front_rel_speed: float | None,
    rear_gap: float | None,
    rear_rel_speed: float | None,
    L_ref: float = OBSERVATION_L_REF,
    v_ref: float = OBSERVATION_V_REF,
) -> np.ndarray:
    """Assemble the fixed-order observation vector for one learner."""

    def pack_agent(a: Mapping[str, Any]) -> list[float]:
        role = 1.0 if a["role"] == "mainline" else -1.0
        active = 1.0 if a.get("active_on_road", not a.get("completed", False)) else 0.0
        vt = max(float(a.get("target_speed", v_ref)), 1e-6)
        return [
            _clip(_finite(float(a["rho"])), 0.0, 1.0),
            _clip(_finite(float(a["speed"]) / vt), 0.0, 2.0),
            _clip(_finite(float(a["dist_to_merge"]) / L_ref), -1.0, 2.0),
            _clip(_finite(float(a["dist_to_exit"]) / L_ref), 0.0, 2.0),
            role,
            active,
        ]

    def pack_bg(a: Mapping[str, Any]) -> list[float]:
        vt = max(float(a.get("target_speed", v_ref)), 1e-6)
        active = 1.0 if a.get("active_on_road", not a.get("completed", False)) else 0.0
        return [
            _clip(_finite(float(a["rho"])), 0.0, 1.0),
            _clip(_finite(float(a["speed"]) / vt), 0.0, 2.0),
            active,
        ]

    front_valid = 1.0 if front_gap is not None else 0.0
    rear_valid = 1.0 if rear_gap is not None else 0.0
    vec = (
        pack_agent(own)
        + pack_agent(peer)
        + pack_bg(b_front)
        + pack_bg(b_rear)
        + [
            _clip(_finite((front_gap or 0.0) / L_ref), -1.0, 2.0) * front_valid,
            _clip(_finite((front_rel_speed or 0.0) / v_ref), -2.0, 2.0) * front_valid,
            _clip(_finite((rear_gap or 0.0) / L_ref), -1.0, 2.0) * rear_valid,
            _clip(_finite((rear_rel_speed or 0.0) / v_ref), -2.0, 2.0) * rear_valid,
            front_valid,
            rear_valid,
            _clip(_finite((float(peer["world_x"]) - float(own["world_x"])) / L_ref), -2.0, 2.0),
            _clip(_finite((float(peer["world_y"]) - float(own["world_y"])) / L_ref), -2.0, 2.0),
            _clip(_finite((float(peer["speed"]) - float(own["speed"])) / v_ref), -2.0, 2.0),
        ]
    )
    arr = np.asarray(vec, dtype=np.float32)
    assert arr.shape == (OBSERVATION_DIM,)
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite observation")
    return arr
