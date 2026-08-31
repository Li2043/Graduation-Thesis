"""Unit + smoke tests for train_joint_dqn_diagnostic.py --
VDN_Conditional_Amendment_Protocol.md sec 11 (Diagnostic 5). Focuses on
the slot-ordering correctness the module's own docstring flags as the
main implementation risk."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


diagnostic = _load_script("train_joint_dqn_diagnostic")


def test_role_major_slot_order_puts_ramp_before_mainline():
    roles = {"ramp": ["V2", "V0"], "mainline": ["V3", "V1"]}
    order = diagnostic.role_major_slot_order(roles)
    assert order == ("V0", "V2", "V1", "V3")  # ramp members sorted, then mainline members sorted


def test_role_major_slot_order_is_deterministic_regardless_of_input_list_order():
    roles_a = {"ramp": ["V0", "V1"], "mainline": ["V2", "V3"]}
    roles_b = {"ramp": ["V1", "V0"], "mainline": ["V3", "V2"]}
    assert diagnostic.role_major_slot_order(roles_a) == diagnostic.role_major_slot_order(roles_b)


def test_role_major_slot_order_rejects_wrong_role_sizes():
    with pytest.raises(ValueError):
        diagnostic.role_major_slot_order({"ramp": ["V0"], "mainline": ["V1", "V2", "V3"]})


def test_reorder_joint_observation_identity_when_slot_order_matches_vehicle_id_order():
    vehicle_id_order = ("V0", "V1", "V2", "V3")
    global_state = np.arange(24, dtype=np.float64)  # 4 chunks of 6: [0-5]=V0, [6-11]=V1, ...
    reordered = diagnostic.reorder_joint_observation(
        global_state, vehicle_id_order=vehicle_id_order, slot_order=vehicle_id_order,
    )
    np.testing.assert_array_equal(reordered, global_state)


def test_reorder_joint_observation_moves_chunks_to_role_major_slots():
    vehicle_id_order = ("V0", "V1", "V2", "V3")
    global_state = np.arange(24, dtype=np.float64)
    # Suppose V1, V3 are ramp and V0, V2 are mainline this episode.
    slot_order = ("V1", "V3", "V0", "V2")
    reordered = diagnostic.reorder_joint_observation(
        global_state, vehicle_id_order=vehicle_id_order, slot_order=slot_order,
    )
    expected = np.concatenate([global_state[6:12], global_state[18:24], global_state[0:6], global_state[12:18]])
    np.testing.assert_array_equal(reordered, expected)


def test_reorder_joint_observation_rejects_non_permutation_slot_order():
    vehicle_id_order = ("V0", "V1", "V2", "V3")
    global_state = np.arange(24, dtype=np.float64)
    with pytest.raises(ValueError):
        diagnostic.reorder_joint_observation(
            global_state, vehicle_id_order=vehicle_id_order, slot_order=("V0", "V1", "V2", "V9"),
        )


def test_slot_mapping_self_consistent_across_many_real_resets():
    """The main risk this module's docstring flags: role assignment is
    re-randomised every reset, so slot_order must be recomputed fresh each
    episode, not assumed fixed. This exercises many real resets and checks
    the round-trip (reorder into slots, then reorder back by vehicle_id
    order) always recovers the original global_state exactly."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from thesis.study_b.heterogeneous_env import StudyBEnvConfig, StudyBHeterogeneousEnv

    env = StudyBHeterogeneousEnv(StudyBEnvConfig(episode_max_steps=50))
    vid_order = ("V0", "V1", "V2", "V3")
    for seed in range(20):
        _obs, info = env.reset(seed=seed)
        slot_order = diagnostic.role_major_slot_order(info["roles"])
        assert set(slot_order) == set(vid_order)
        global_state = env.global_state()
        reordered = diagnostic.reorder_joint_observation(global_state, vehicle_id_order=vid_order, slot_order=slot_order)
        round_trip = diagnostic.reorder_joint_observation(reordered, vehicle_id_order=slot_order, slot_order=vid_order)
        np.testing.assert_array_equal(round_trip, global_state)


def test_smoke_short_training_run_end_to_end(tmp_path):
    argv = [
        "--condition", "mean", "--master-seed", "3",
        "--output-root", str(tmp_path / "out"), "--checkpoint-root", str(tmp_path / "ckpt"),
        "--max-steps", "300", "--episode-max-steps", "40", "--checkpoint-every", "150",
        "--replay-warmup", "64", "--device", "cpu",
    ]
    rc = diagnostic.main(argv)
    assert rc == 0
    manifest_path = tmp_path / "out" / "seed_3_mean_manifest.json"
    assert manifest_path.exists()
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["final_step"] == 300
    assert (tmp_path / "ckpt" / "seed_3" / "ckpt_step_300.pt").exists()
