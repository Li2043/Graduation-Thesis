"""Permanent regression test for the WSC observation-to-network semantic
column mapping bug (see wsc_formal_campaign_incident_diagnosis.md).

ROOT CAUSE THIS TEST GUARDS AGAINST: local_observation.build_local_observation
INTERLEAVES the new WSC features into the 22D vector (M_i right after the
self block, each M_j at the end of its own neighbour block) -- it does NOT
append them as a contiguous suffix. An earlier version of
wsc_checkpoint_expansion.py assumed a simple prefix/suffix split
(old -> new[:18], zeros -> new[18:22]), which silently fed every original
weight column the WRONG semantic feature from index 6 onward. This was not
caught by the original validation suite because that suite's Q-equivalence
test used np.concatenate([o, m]) -- a SYNTHETIC vector built with the SAME
wrong prefix/suffix assumption as the buggy expansion code, so the test
"passed" by making the identical mistake twice.

This test suite closes that blind spot with three independent layers:
  1. Sentinel-value checks on the OBSERVATION layout itself (no network
     involved) -- confirms build_local_observation's real index layout.
  2. Sentinel-value checks on the NETWORK WEIGHT expansion (no environment
     involved) -- confirms expand_state_dict copies each old column to the
     semantically-correct new column, not a naive prefix.
  3. An end-to-end check using REAL observations from
     build_local_observation on BOTH sides (Original and WSC) for the
     SAME underlying vehicle state, fed through an expanded network --
     this is the exact check that would have caught the original bug,
     since it independently exercises the observation-construction code
     and the network-expansion code and requires them to agree.

Any of the following must make this test suite FAIL:
  - an old feature shifted to the wrong input column;
  - a neighbour-slot feature mapped into the wrong slot;
  - a WSC feature (M_i/M_j) overwriting an old feature's column;
  - weights copied by raw index instead of semantic correspondence.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from thesis.agents.independent_dqn_v2 import QNetwork
from thesis.study_b.local_observation import (
    LOCAL_OBS_DIM,
    LOCAL_OBS_DIM_WSC,
    NEIGHBOUR_OBS_DIM,
    NEIGHBOUR_OBS_DIM_WSC,
    NEIGHBOUR_SLOTS,
    SELF_OBS_DIM,
    SELF_OBS_DIM_WSC,
    VehicleSnapshot,
    build_local_observation,
)
from thesis.study_b.wsc_checkpoint_expansion import (
    NEW_ONLY_COLUMNS,
    OLD_TO_NEW_COLUMN_MAP,
    ORIGINAL_OBS_DIM,
    WSC_OBS_DIM,
    expand_state_dict,
)

MERGE_START = 200.0

# ======================================================================
# Layer 1: sentinel checks on the OBSERVATION layout (no network)
# ======================================================================

FEATURE_NAMES_OLD = (
    ["role", "speed", "target_speed", "acceleration", "dist_to_merge", "prev_action"]
    + [f"n{k}.{f}" for k in range(NEIGHBOUR_SLOTS) for f in ("presence", "delta_d", "delta_v", "lane_relation")]
)
FEATURE_NAMES_WSC = (
    ["role", "speed", "target_speed", "acceleration", "dist_to_merge", "prev_action", "M_i"]
    + [f"n{k}.{f}" for k in range(NEIGHBOUR_SLOTS) for f in ("presence", "delta_d", "delta_v", "lane_relation", "M_j")]
)


def _ego(welfare_state: float = 1.0) -> VehicleSnapshot:
    return VehicleSnapshot(
        vehicle_id="V0", role="ramp", speed=18.0, route_position=150.0,
        acceleration=0.0, target_speed=18.0, active=True, welfare_state=welfare_state,
    )


def _others(welfare_states: tuple[float, float, float] = (0.7, 0.8, 0.9)) -> list[VehicleSnapshot]:
    # route_position chosen so |delta_d| order is V1 < V3 < V2 (matches
    # test_local_observation_leakage.py's own established ordering fixture).
    return [
        VehicleSnapshot(vehicle_id="V1", role="mainline", speed=20.0, route_position=160.0,
                         acceleration=0.5, target_speed=20.0, active=True, welfare_state=welfare_states[0]),
        VehicleSnapshot(vehicle_id="V2", role="ramp", speed=18.0, route_position=90.0,
                         acceleration=0.0, target_speed=18.0, active=True, welfare_state=welfare_states[1]),
        VehicleSnapshot(vehicle_id="V3", role="mainline", speed=22.0, route_position=110.0,
                         acceleration=0.0, target_speed=22.0, active=True, welfare_state=welfare_states[2]),
    ]


def test_layout_dims_match_module_constants():
    assert LOCAL_OBS_DIM == ORIGINAL_OBS_DIM == 18
    assert LOCAL_OBS_DIM_WSC == WSC_OBS_DIM == 22
    assert len(FEATURE_NAMES_OLD) == 18
    assert len(FEATURE_NAMES_WSC) == 22


def test_M_i_lands_exactly_at_index_6_not_appended_at_end():
    """The specific bug this whole incident hinged on: M_i is INSERTED
    after the self block (index 6), not appended as a suffix (index 18)."""
    ego = _ego(welfare_state=0.4242)
    obs_wsc = build_local_observation(ego, _others(), merge_start=MERGE_START, include_welfare_state=True)
    assert obs_wsc[SELF_OBS_DIM] == pytest.approx(0.4242)  # index 6
    assert obs_wsc[SELF_OBS_DIM] != pytest.approx(obs_wsc[-1])  # sanity: not accidentally at the end too


def test_each_M_j_lands_at_end_of_its_own_neighbour_block_not_grouped_at_end():
    ego = _ego()
    others = _others(welfare_states=(0.11, 0.22, 0.33))
    obs_wsc = build_local_observation(ego, others, merge_start=MERGE_START, include_welfare_state=True)
    # Nearest-first order established by route positions: V1 (0.11), V3 (0.33), V2 (0.22).
    expected_mj_by_slot = [0.11, 0.33, 0.22]
    for slot in range(NEIGHBOUR_SLOTS):
        base = SELF_OBS_DIM_WSC + slot * NEIGHBOUR_OBS_DIM_WSC
        mj_index = base + NEIGHBOUR_OBS_DIM  # last field in this slot's 5-tuple
        assert obs_wsc[mj_index] == pytest.approx(expected_mj_by_slot[slot]), (
            f"slot {slot}: M_j not at the expected interleaved index {mj_index}"
        )


def test_old_semantic_features_appear_at_the_shifted_indices_not_index_i():
    """Direct sentinel check: build an Original 18D observation and a WSC
    22D observation for the IDENTICAL underlying state, and confirm every
    old feature appears at OLD_TO_NEW_COLUMN_MAP[i], NOT at index i."""
    ego = _ego()
    others = _others()
    obs_old = build_local_observation(ego, others, merge_start=MERGE_START, include_welfare_state=False)
    obs_wsc = build_local_observation(ego, others, merge_start=MERGE_START, include_welfare_state=True)
    assert obs_old.shape == (18,)
    assert obs_wsc.shape == (22,)
    for i in range(18):
        target = OLD_TO_NEW_COLUMN_MAP[i]
        assert obs_wsc[target] == pytest.approx(obs_old[i]), (
            f"old feature '{FEATURE_NAMES_OLD[i]}' (old index {i}) not found at its correct "
            f"semantic WSC index {target} (found value {obs_wsc[target]}, expected {obs_old[i]})"
        )
    # And confirm the 4 NEW-only columns are exactly the welfare-state
    # features (M_i / M_j), not some old feature landing at the wrong index.
    for new_idx in NEW_ONLY_COLUMNS:
        name = FEATURE_NAMES_WSC[new_idx]
        assert name == "M_i" or name.endswith(".M_j"), (
            f"new-only column {new_idx} expected to be a welfare-state feature, found '{name}'"
        )


def test_absent_neighbour_slot_M_j_is_zero_not_arbitrary():
    ego = _ego()
    others = _others()
    others[0] = VehicleSnapshot(
        vehicle_id="V1", role="mainline", speed=20.0, route_position=160.0,
        acceleration=0.5, target_speed=20.0, active=False, welfare_state=0.99,
    )
    obs_wsc = build_local_observation(ego, others, merge_start=MERGE_START, include_welfare_state=True)
    # Only 2 active others -> slot 2 (farthest) must be fully masked, including M_j.
    base = SELF_OBS_DIM_WSC + 2 * NEIGHBOUR_OBS_DIM_WSC
    assert obs_wsc[base] == pytest.approx(0.0)  # presence
    assert obs_wsc[base + NEIGHBOUR_OBS_DIM] == pytest.approx(0.0)  # M_j padding, not 0.99 or any leaked value


# ======================================================================
# Layer 2: sentinel checks on the NETWORK WEIGHT expansion (no environment)
# ======================================================================

def test_expand_state_dict_copies_each_old_column_to_its_semantic_target():
    """Build a first-layer weight with a UNIQUE, unmistakable sentinel
    value in each of the 18 old columns (column i = value 1000+i for
    every row), expand it, and verify column OLD_TO_NEW_COLUMN_MAP[i]
    of the expanded weight equals that exact sentinel -- and that the 4
    NEW_ONLY_COLUMNS are exactly zero. This fails immediately under a
    naive prefix/suffix (or any other wrong) mapping, since a wrong
    target column would either contain the wrong sentinel or a stray
    nonzero value where zero is required."""
    hidden = 8
    sentinel_weight = torch.zeros((hidden, ORIGINAL_OBS_DIM))
    for i in range(ORIGINAL_OBS_DIM):
        sentinel_weight[:, i] = 1000.0 + i
    state_dict = {
        "net.0.weight": sentinel_weight,
        "net.0.bias": torch.zeros(hidden),
        "net.2.weight": torch.zeros(hidden, hidden),
        "net.2.bias": torch.zeros(hidden),
        "net.4.weight": torch.zeros(3, hidden),
        "net.4.bias": torch.zeros(3),
    }
    expanded = expand_state_dict(state_dict)
    new_w = expanded["net.0.weight"]
    assert new_w.shape == (hidden, WSC_OBS_DIM)

    for i in range(ORIGINAL_OBS_DIM):
        target = OLD_TO_NEW_COLUMN_MAP[i]
        expected = 1000.0 + i
        actual = new_w[0, target].item()
        assert actual == pytest.approx(expected), (
            f"old column {i} ('{FEATURE_NAMES_OLD[i]}', sentinel {expected}) expected at new column "
            f"{target} ('{FEATURE_NAMES_WSC[target]}'), found {actual} instead"
        )

    for new_idx in NEW_ONLY_COLUMNS:
        assert torch.all(new_w[:, new_idx] == 0.0), (
            f"new-only column {new_idx} ('{FEATURE_NAMES_WSC[new_idx]}') must be exactly zero, "
            f"found {new_w[:, new_idx].tolist()}"
        )

    # Every column of the expanded weight must be accounted for by EITHER
    # an old-column sentinel or a new-only zero -- no column silently
    # dropped or duplicated.
    covered = set(OLD_TO_NEW_COLUMN_MAP) | set(NEW_ONLY_COLUMNS)
    assert covered == set(range(WSC_OBS_DIM))


def test_expand_state_dict_rejects_wrong_input_dim():
    bad = {"net.0.weight": torch.zeros((8, 17))}  # not 18
    with pytest.raises(ValueError):
        expand_state_dict(bad)


# ======================================================================
# Layer 3: end-to-end -- real observations through an expanded network
# ======================================================================

def test_end_to_end_zero_init_q_equivalence_on_real_observations():
    """The check that would have caught the original bug directly: a
    QNetwork with RANDOM (not sentinel) weights, expanded via
    expand_state_dict, must produce IDENTICAL Q-values on a REAL WSC
    observation (from build_local_observation(..., True)) as the
    original network produces on the REAL Original observation
    (build_local_observation(..., False)) for the SAME underlying
    state -- not a synthetic np.concatenate, which is what hid the bug
    originally."""
    torch.manual_seed(0)
    old_net = QNetwork(ORIGINAL_OBS_DIM, 3, (16, 16))
    old_sd = old_net.state_dict()

    new_sd = expand_state_dict(old_sd)
    new_net = QNetwork(WSC_OBS_DIM, 3, (16, 16))
    new_net.load_state_dict(new_sd)

    ego = _ego(welfare_state=0.55)
    others = _others(welfare_states=(0.1, 0.6, 0.95))
    obs_old = build_local_observation(ego, others, merge_start=MERGE_START, include_welfare_state=False)
    obs_wsc = build_local_observation(ego, others, merge_start=MERGE_START, include_welfare_state=True)

    with torch.no_grad():
        q_old = old_net(torch.as_tensor(obs_old, dtype=torch.float32).unsqueeze(0)).squeeze(0)
        q_new = new_net(torch.as_tensor(obs_wsc, dtype=torch.float32).unsqueeze(0)).squeeze(0)

    torch.testing.assert_close(q_old, q_new, atol=1e-5, rtol=1e-5)
    assert torch.argmax(q_old).item() == torch.argmax(q_new).item()


def test_end_to_end_detects_a_reintroduced_prefix_suffix_bug():
    """Negative control: deliberately rebuild the OLD, WRONG prefix/suffix
    expansion inline and confirm THIS test suite's methodology actually
    detects it (i.e. this file's own checks are not vacuously passing).
    If this test ever fails, the other tests in this file are not
    trustworthy guards against regression."""
    torch.manual_seed(1)
    old_net = QNetwork(ORIGINAL_OBS_DIM, 3, (16, 16))
    old_sd = old_net.state_dict()

    # Reproduce the ORIGINAL BUG on purpose: naive prefix/suffix split.
    buggy_w = torch.zeros((old_sd["net.0.weight"].shape[0], WSC_OBS_DIM))
    buggy_w[:, :ORIGINAL_OBS_DIM] = old_sd["net.0.weight"]
    buggy_sd = dict(old_sd)
    buggy_sd["net.0.weight"] = buggy_w
    buggy_net = QNetwork(WSC_OBS_DIM, 3, (16, 16))
    buggy_net.load_state_dict(buggy_sd)

    ego = _ego(welfare_state=0.55)
    others = _others(welfare_states=(0.1, 0.6, 0.95))
    obs_old = build_local_observation(ego, others, merge_start=MERGE_START, include_welfare_state=False)
    obs_wsc = build_local_observation(ego, others, merge_start=MERGE_START, include_welfare_state=True)

    with torch.no_grad():
        q_old = old_net(torch.as_tensor(obs_old, dtype=torch.float32).unsqueeze(0)).squeeze(0)
        q_buggy = buggy_net(torch.as_tensor(obs_wsc, dtype=torch.float32).unsqueeze(0)).squeeze(0)

    # The buggy (prefix/suffix) expansion must NOT reproduce Q_old on a
    # real, semantically-interleaved WSC observation -- if it did, this
    # test file's methodology would be too weak to have caught the
    # original incident.
    assert not torch.allclose(q_old, q_buggy, atol=1e-5, rtol=1e-5), (
        "the deliberately-reintroduced prefix/suffix bug was NOT detected -- "
        "this test file's methodology is not a valid regression guard"
    )
