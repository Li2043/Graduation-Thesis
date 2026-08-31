"""Stage 11B -- N-vehicle (N>2) generalisation of the v12 joint-centralised
training loop (see
``experiments/pilots/stage11_dyad_merge_pilot_v12_joint_network/STAGE11B_MULTI_VEHICLE_PILOT_DESIGN.md``
for the full architecture design this implements).

These are pilot-scoped tests, not part of Study A's frozen protocol suite
(``test_stage11_v12_joint_network.py``) -- kept in a separate file per the
design doc's testing strategy, given the volume of new N=4/6-specific
cases. Seeds used here are arbitrary (90000+), well outside any real
pilot/formal seed block, and every run uses ``strict=False`` since these
are short smoke runs, not protocol-governed formal training.
"""

from __future__ import annotations

import json

import pytest

from thesis.pilots.stage11_dyad_merge_runner import EpisodeWindowStats, run_stage11_pilot_training_job


# --------------------------------------------------------------- U_by_vid regression test


def test_episode_window_stats_does_not_collide_same_role_vehicles():
    """Direct regression test for the bug identified in
    STAGE11B_MULTI_VEHICLE_PILOT_DESIGN.md Sec 3.2: a role-keyed welfare
    dict would silently overwrite same-role peers at n_vehicles>2 (only the
    last-iterated same-role vehicle's outcome survives), corrupting
    mean_welfare/min_welfare without raising any error. This test
    constructs a 4-vehicle (2 ramp + 2 mainline) welfare dict directly,
    with each of the 4 vehicles given a distinct, deliberately different
    value, and asserts the recorded aggregate reflects all 4 -- not 2."""
    window = EpisodeWindowStats()
    # Two ramp vehicles with very different outcomes, two mainline vehicles
    # likewise -- if any two same-role vehicles silently collided under a
    # role-keyed dict, the mean/min below would be computed from only 2
    # values (one per role), not 4.
    welfare_by_vid = {"V0": 0.9, "V1": 0.1, "V2": 0.8, "V3": 0.2}
    window.episodes += 1
    window.completions += 1
    window.record_episode(welfare_by_stakeholder=welfare_by_vid, first_crosser=("ramp", "V0"))

    d = window.as_dict()
    expected_mean = sum(welfare_by_vid.values()) / 4
    expected_min = min(welfare_by_vid.values())
    assert d["mean_U_mean"] == pytest.approx(expected_mean)
    assert d["min_U_mean"] == pytest.approx(expected_min)
    # A role-keyed collision would have produced mean=(0.1+0.2)/2=0.15 or
    # similar 2-value aggregate -- well outside a reasonable tolerance of
    # the true 4-value mean (0.5) computed above from all 4 vehicles.
    assert d["mean_U_mean"] != pytest.approx(0.15, abs=0.05)


def test_episode_window_stats_min_reflects_the_true_worst_of_four_not_two():
    window = EpisodeWindowStats()
    # The true minimum (0.05) belongs to a SECOND ramp vehicle -- if the
    # first ramp vehicle's entry silently overwrote it under a role-keyed
    # dict, this minimum would never be seen.
    welfare_by_vid = {"V0": 0.9, "V1": 0.05, "V2": 0.7, "V3": 0.6}
    window.episodes += 1
    window.completions += 1
    window.record_episode(welfare_by_stakeholder=welfare_by_vid, first_crosser=None)
    d = window.as_dict()
    assert d["min_U_mean"] == pytest.approx(0.05)


# --------------------------------------------------------------- slot ordering


def test_n_vehicles_validation_rejects_non_2_4_6():
    with pytest.raises(ValueError, match="n_vehicles"):
        run_stage11_pilot_training_job(
            master_seed=90000,
            output_root="unused",
            checkpoint_root="unused",
            max_steps=10,
            strict=False,
            checkpoint_steps=(0, 10),
            episode_max_steps=10,
            enable_stage9_based_reward_v5=True,
            enable_joint_network_v12=True,
            n_vehicles=3,
        )


def test_n_vehicles_without_joint_network_raises():
    with pytest.raises(ValueError, match="n_vehicles"):
        run_stage11_pilot_training_job(
            master_seed=90000,
            output_root="unused",
            checkpoint_root="unused",
            max_steps=10,
            strict=False,
            checkpoint_steps=(0, 10),
            episode_max_steps=10,
            n_vehicles=4,
        )


# --------------------------------------------------------------- end-to-end smoke runs


@pytest.mark.parametrize("n_vehicles", [2, 4, 6])
def test_v12_short_run_completes_at_each_n_vehicles(tmp_path, n_vehicles):
    """Runs the full step/replay/update loop together at each supported N
    -- catches shape mismatches that only manifest once training actually
    runs, not from unit-testing individual functions alone. n_vehicles=2
    included to confirm this test file's own scaffolding reproduces the
    existing N=2 behaviour (a second, independent check alongside
    test_stage11_v12_joint_network.py's own N=2 coverage)."""
    output_root = tmp_path / f"output_{n_vehicles}"
    checkpoint_root = tmp_path / f"checkpoints_{n_vehicles}"
    seed = 90001 + n_vehicles
    manifest = run_stage11_pilot_training_job(
        master_seed=seed,
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 300, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_joint_network_v12=True,
        n_vehicles=n_vehicles,
    )
    assert manifest["final_step"] == 600
    assert manifest["n_vehicles"] == n_vehicles
    assert len(manifest["checkpoints"]) == 3

    traj_path = output_root / "trajectories" / f"seed_{seed}.jsonl"
    assert traj_path.exists()
    lines = traj_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 600
    first = json.loads(lines[0])
    assert len(first["vehicles"]) == n_vehicles
    roles_seen = sorted(v["role"] for v in first["vehicles"])
    assert roles_seen == sorted(["ramp"] * (n_vehicles // 2) + ["mainline"] * (n_vehicles // 2))

    last_ckpt = manifest["checkpoints"][-1]
    window = last_ckpt["window"]
    assert "completion_rate" in window
    assert last_ckpt["learner"]["update_count"] >= 0


@pytest.mark.parametrize("pbrs_condition", ["mean", "min"])
def test_v12_n4_pbrs_conditions_run_without_error(tmp_path, pbrs_condition):
    """The mean_pbrs/min_pbrs shaping paths (welfare aggregated over all 4
    vehicles, not just 2) run end-to-end at n_vehicles=4 without error."""
    output_root = tmp_path / "output"
    checkpoint_root = tmp_path / "checkpoints"
    manifest = run_stage11_pilot_training_job(
        master_seed=90050,
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        max_steps=400,
        strict=False,
        checkpoint_steps=(0, 400),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_joint_network_v12=True,
        pbrs_condition_v12=pbrs_condition,
        n_vehicles=4,
    )
    assert manifest["final_step"] == 400


def test_v12_n4_prioritized_replay_and_n_step_run_without_error(tmp_path):
    """PER's n-way max combination and n-step windowing both exercised at
    n_vehicles=4 together, matching the actual launch configuration this
    pilot is expected to use (mirroring Study A's own
    --enable-prioritized-replay-v12 --n-step-v12 3 launch flags)."""
    output_root = tmp_path / "output"
    checkpoint_root = tmp_path / "checkpoints"
    manifest = run_stage11_pilot_training_job(
        master_seed=90060,
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        max_steps=400,
        strict=False,
        checkpoint_steps=(0, 400),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_joint_network_v12=True,
        enable_prioritized_replay_v12=True,
        n_step_v12=3,
        n_vehicles=4,
    )
    assert manifest["final_step"] == 400


def test_v12_checkpoint_payload_n4_has_four_heads(tmp_path):
    import torch

    output_root = tmp_path / "output"
    checkpoint_root = tmp_path / "checkpoints"
    seed = 90070
    run_stage11_pilot_training_job(
        master_seed=seed,
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        max_steps=300,
        strict=False,
        checkpoint_steps=(0, 300),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_joint_network_v12=True,
        n_vehicles=4,
    )
    ckpt_path = checkpoint_root / f"seed_{seed}" / "ckpt_step_300.pt"
    payload = torch.load(ckpt_path, weights_only=False)
    online_keys = payload["learner"]["online"].keys()
    head_indices = {int(k.split(".")[1]) for k in online_keys if k.startswith("heads.")}
    assert head_indices == {0, 1, 2, 3}
