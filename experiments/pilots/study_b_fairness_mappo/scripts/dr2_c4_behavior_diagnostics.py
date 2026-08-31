#!/usr/bin/env python3
"""DR2 -- C4 four-seed behavior/observation/control diagnostics
(RUNBOOK C4_STATUS=NOT_YET_QUALIFIED diagnostic freeze, 2026-08-16).

Reads DR1's per-episode CSV for each seed, picks one representative
SUCCESS episode and one representative COLLISION episode per seed
(TIMEOUT too, if any occurred), and REPLAYS that exact episode
deterministically (same salted RNG streams, same episode ordering as
DR1 -- episode N in DR1 is bit-identical to episode N here as long as
the checkpoint and code are unchanged) with full per-step tracing:
local observation vectors, action taken vs. the greedy action at the
same state (to flag exploration-induced deviations), desired/target
speed (static scenario value and the vehicle's own dynamic
target_speed attribute), requested vs. realized (post-clip) physical
acceleration, nearest same-lane gap/TTC, active/completion state, and
merge timestamps (derived post-hoc from the x-position trace).

Live regression checks performed directly against the traced data
(invariants 1/2/3/4/7/8/9 from the user's list -- the ones observable
from a step trace):
  1/2 completed-vehicle inactivity: once a vehicle's exit_event fires,
      its target_speed attribute must never change again this episode.
  3   realized acceleration must lie in [-3.0, +2.0] on every step.
  4   realized == clip(requested, -3.0, +2.0) on every step.
  7   collision geometry: at the reported failure timestep, the
      colliding pair must satisfy the exact collision_pairs rule
      (|dx|<=collision_distance_longitudinal_m and
      |dy|<collision_lateral_threshold_m).
  8   timeout episodes must have episode_length==episode_max_steps and
      term_reason=="truncation" with no collision_event.
  9   active-mask consistency: once inactive, a vehicle stays inactive
      for the remainder of the episode.

Invariants 5/6/10/11/12/13 (local-information restriction, hidden
other-agent target speed, reward decomposition, replay-terminal
handling, target-network sync, absolute LR/epsilon schedules) are
structural/architectural properties already covered by the existing
`tests/study_b/` regression suite and by this eval harness's own
guarantees (no store_transition/maybe_update call anywhere in DR1/DR2,
so replay/target-sync are structurally inapplicable to a frozen-
checkpoint eval) -- this script re-runs that suite once and records the
result rather than re-deriving those checks from raw trace data.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(REPO_SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from thesis.study_b.envs.highwayenv_action import ACCELERATE, BRAKE, HOLD  # noqa: E402
from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig  # noqa: E402
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig  # noqa: E402
from thesis.study_b.oracle_controller import DEFAULT_LATERAL_SAME_LANE_THRESHOLD_M  # noqa: E402
from thesis.study_b.shared_local_dqn import SharedLocalDQNAgent, build_study_b_dqn_config, epsilon_at_step_v12  # noqa: E402
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402

_ENV_CONFIG = ThesisHighwayMergeEnvConfig()
_MERGE_START_X = _ENV_CONFIG.before_merge_length
_MERGE_END_X = _ENV_CONFIG.before_merge_length + _ENV_CONFIG.converge_merge_length
_ACTION_NAME = {HOLD: "HOLD", ACCELERATE: "ACCELERATE", BRAKE: "BRAKE"}


def nearest_same_lane_gap_ttc(vid: str, x: dict, y: dict, speed: dict, vids: tuple[str, ...]) -> tuple[float, float]:
    best_gap, best_ttc = float("inf"), float("inf")
    for other in vids:
        if other == vid:
            continue
        if abs(y[vid] - y[other]) >= DEFAULT_LATERAL_SAME_LANE_THRESHOLD_M:
            continue
        gap = abs(x[vid] - x[other])
        if gap < best_gap:
            best_gap = gap
        if x[vid] <= x[other]:
            rear, front = vid, other
        else:
            rear, front = other, vid
        closing = speed[rear] - speed[front]
        if closing > 1e-6:
            ttc = gap / closing
            if ttc < best_ttc:
                best_ttc = ttc
    return best_gap, best_ttc


def run_regression_suite() -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/study_b/", "-q"],
        cwd=str(Path(__file__).resolve().parents[4]),
        capture_output=True, text=True, timeout=1800,
    )
    tail = "\n".join(result.stdout.strip().splitlines()[-15:])
    return {"returncode": result.returncode, "summary_tail": tail}


def pick_representative_episodes(dr1_csv: Path) -> dict[str, dict[str, int | None]]:
    rows_by_seed: dict[int, list[dict]] = defaultdict(list)
    with open(dr1_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_by_seed[int(row["seed"])].append(row)

    picks: dict[str, dict[str, int | None]] = {}
    for seed, rows in rows_by_seed.items():
        successes = [r for r in rows if r["term_reason"] == "success"]
        collisions = [r for r in rows if r["term_reason"] == "collision"]
        timeouts = [r for r in rows if r["term_reason"] == "truncation"]
        picks[str(seed)] = {
            "success": int(successes[len(successes) // 2]["episode_index"]) if successes else None,
            "collision": int(collisions[len(collisions) // 2]["episode_index"]) if collisions else None,
            "timeout": int(timeouts[0]["episode_index"]) if timeouts else None,
        }
    return picks


def trace_episode(
    *, seed: int, checkpoint: Path, scenario_bank_path: Path, scenario_ids: list[str],
    absolute_step: int, eps_decay_steps: int, target_episode_index: int, episode_max_steps: int, device: str,
) -> dict:
    """Deterministically replays episodes 0..target_episode_index in the
    SAME order/RNG streams DR1 used, capturing full step detail only for
    target_episode_index."""
    scenarios = load_scenario_bank(scenario_bank_path)
    by_id = {s.scenario_id: s for s in scenarios}
    stage_scenarios = [by_id[sid] for sid in scenario_ids]

    env_config = ThesisHighwayMergeEnvConfig(episode_max_steps=episode_max_steps, action_representation="meta_speed")
    env = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=env_config))

    dqn_config = build_study_b_dqn_config(device=device)
    agent = SharedLocalDQNAgent(dqn_config, seed=seed * 1_000_003 + 555_001)  # same salt as DR1
    ckpt = torch.load(checkpoint, map_location=device)
    agent.learner.online.load_state_dict(ckpt["online"])

    eps = epsilon_at_step_v12(absolute_step, decay_steps=eps_decay_steps)
    scenario_rng = np.random.default_rng(seed * 1_000_003 + 555_002)  # same salt as DR1

    trace_steps: list[dict] = []
    exit_step: dict[str, int | None] = {}
    merge_zone_entry_step: dict[str, int | None] = {}
    merge_zone_exit_step: dict[str, int | None] = {}
    scenario = None
    term_reason = "ongoing"
    failure_timestep = None
    collision_pairs: list[tuple[str, str]] = []
    episode_length = 0

    for episode_index in range(target_episode_index + 1):
        scenario = stage_scenarios[int(scenario_rng.integers(0, len(stage_scenarios)))]
        obs, _info = env.reset(seed=0, scenario=scenario)
        is_target = episode_index == target_episode_index
        exit_step = dict.fromkeys(env.active_vehicle_ids)
        merge_zone_entry_step = dict.fromkeys(env.active_vehicle_ids)
        merge_zone_exit_step = dict.fromkeys(env.active_vehicle_ids)
        term_reason = "ongoing"
        failure_timestep = None
        collision_pairs = []
        episode_length = 0

        for t in range(episode_max_steps):
            vids = env.active_vehicle_ids
            x = {vid: env._env.world_xy(env._env._vehicle_by_id[vid])[0] for vid in vids}  # noqa: SLF001
            y = {vid: env._env.world_xy(env._env._vehicle_by_id[vid])[1] for vid in vids}  # noqa: SLF001
            speed = {vid: float(env._env._vehicle_by_id[vid].speed) for vid in vids}  # noqa: SLF001

            actions = agent.select_actions(obs, epsilon=eps)
            greedy_actions = agent.select_actions(obs, epsilon=0.0, greedy=True) if is_target else {}

            if is_target:
                step_record: dict = {"t": t, "vehicles": {}}
                for vid in vids:
                    vehicle = env._env._vehicle_by_id[vid]  # noqa: SLF001
                    gap, ttc = nearest_same_lane_gap_ttc(vid, x, y, speed, vids)
                    requested = getattr(vehicle, "last_requested_acceleration", None)
                    realized = getattr(vehicle, "last_realized_acceleration", vehicle.action.get("acceleration"))
                    step_record["vehicles"][vid] = {
                        "obs": obs[vid].tolist(),
                        "action": int(actions[vid]), "action_name": _ACTION_NAME[int(actions[vid])],
                        "greedy_action": int(greedy_actions[vid]), "greedy_action_name": _ACTION_NAME[int(greedy_actions[vid])],
                        "exploration_deviation": bool(int(actions[vid]) != int(greedy_actions[vid])),
                        "target_speed_static": scenario.vehicles[vid].target_speed,
                        "target_speed_dynamic": float(getattr(vehicle, "target_speed", float("nan"))),
                        "requested_acceleration": (None if requested is None else float(requested)),
                        "realized_acceleration": float(realized) if realized is not None else None,
                        "speed": speed[vid], "x": x[vid], "y": y[vid],
                        "same_lane_gap": (None if gap == float("inf") else round(gap, 4)),
                        "same_lane_ttc": (None if ttc == float("inf") else round(ttc, 4)),
                        "active": not env._env._completed[vid],  # noqa: SLF001
                    }
                trace_steps.append(step_record)

            for vid in vids:
                if merge_zone_entry_step[vid] is None and x[vid] >= _MERGE_START_X:
                    merge_zone_entry_step[vid] = t
                if merge_zone_exit_step[vid] is None and x[vid] >= _MERGE_END_X:
                    merge_zone_exit_step[vid] = t

            obs, _reward, terminated, truncated, step_info = env.step(actions)
            episode_length = t + 1

            if is_target:
                for vid in vids:
                    x_after, y_after = env._env.world_xy(env._env._vehicle_by_id[vid])  # noqa: SLF001
                    step_record["vehicles"][vid]["x_after"] = x_after
                    step_record["vehicles"][vid]["y_after"] = y_after

            for vid, done in step_info["exit_event"].items():
                if done and exit_step[vid] is None:
                    exit_step[vid] = t + 1

            if step_info["collision_event"]:
                term_reason = "collision"
                failure_timestep = t + 1
                collision_pairs = step_info["collision_pairs"]
                break
            if terminated:
                term_reason = "success"
                break
            if truncated:
                term_reason = "truncation"
                break

    return {
        "seed": seed, "episode_index": target_episode_index, "scenario_id": scenario.scenario_id if scenario else None,
        "term_reason": term_reason, "episode_length": episode_length, "failure_timestep": failure_timestep,
        "collision_pairs": collision_pairs, "exit_step": exit_step,
        "merge_zone_entry_step": merge_zone_entry_step, "merge_zone_exit_step": merge_zone_exit_step,
        "trace_steps": trace_steps,
    }


def run_live_checks(episode: dict) -> list[dict]:
    findings = []
    seed, ep_idx = episode["seed"], episode["episode_index"]

    for step in episode["trace_steps"]:
        for vid, v in step["vehicles"].items():
            ra = v["realized_acceleration"]
            if ra is not None and not (-3.0 - 1e-6 <= ra <= 2.0 + 1e-6):
                findings.append({"seed": seed, "episode_index": ep_idx, "invariant": "3_accel_clip",
                                  "detail": f"t={step['t']} vid={vid} realized_acceleration={ra} outside [-3,2]"})
            if ra is not None and v["requested_acceleration"] is not None:
                expected = float(np.clip(v["requested_acceleration"], -3.0, 2.0))
                if abs(expected - ra) > 1e-6:
                    findings.append({"seed": seed, "episode_index": ep_idx, "invariant": "4_clip_consistency",
                                      "detail": f"t={step['t']} vid={vid} realized={ra} expected_clip={expected}"})

    for vid, exit_t in episode["exit_step"].items():
        if exit_t is None:
            continue
        post_completion_ts = [
            step["vehicles"][vid]["target_speed_dynamic"]
            for step in episode["trace_steps"]
            if step["t"] + 1 > exit_t and vid in step["vehicles"]
        ]
        if len(set(round(v, 6) for v in post_completion_ts)) > 1:
            findings.append({"seed": seed, "episode_index": ep_idx, "invariant": "1_2_completed_inactivity",
                              "detail": f"vid={vid} target_speed_dynamic changed after exit_step={exit_t}: {sorted(set(post_completion_ts))}"})

    if episode["term_reason"] == "collision" and episode["collision_pairs"]:
        last_step = episode["trace_steps"][-1] if episode["trace_steps"] else None
        if last_step is not None:
            for pair in episode["collision_pairs"]:
                a, b = pair
                if a in last_step["vehicles"] and b in last_step["vehicles"]:
                    # POST-step ("_after") positions: collision_event is
                    # determined by the env from the physics state AFTER
                    # this step's action was applied, not the pre-step
                    # snapshot the rest of this step's fields were logged
                    # from -- confirmed to matter after an initial dr2
                    # smoketest false-positive traced to this exact
                    # pre/post timing offset.
                    dx = abs(last_step["vehicles"][a]["x_after"] - last_step["vehicles"][b]["x_after"])
                    dy = abs(last_step["vehicles"][a]["y_after"] - last_step["vehicles"][b]["y_after"])
                    if not (dx <= _ENV_CONFIG.collision_distance_longitudinal_m and dy < _ENV_CONFIG.collision_lateral_threshold_m):
                        findings.append({"seed": seed, "episode_index": ep_idx, "invariant": "7_collision_geometry",
                                          "detail": f"pair={pair} dx={dx} dy={dy} does not satisfy collision rule at final traced step"})

    if episode["term_reason"] == "truncation":
        if episode["episode_length"] != 200 or episode["failure_timestep"] is not None:
            findings.append({"seed": seed, "episode_index": ep_idx, "invariant": "8_timeout_semantics",
                              "detail": f"episode_length={episode['episode_length']} failure_timestep={episode['failure_timestep']}"})

    for vid in episode["exit_step"]:
        exit_t = episode["exit_step"][vid]
        if exit_t is None:
            continue
        for step in episode["trace_steps"]:
            if step["t"] + 1 > exit_t and step["vehicles"].get(vid, {}).get("active", False):
                findings.append({"seed": seed, "episode_index": ep_idx, "invariant": "9_active_mask",
                                  "detail": f"vid={vid} still marked active at t={step['t']} after exit_step={exit_t}"})
                break

    return findings


def classify_collision_causes(episode: dict) -> dict:
    if episode["term_reason"] != "collision" or not episode["trace_steps"]:
        return {}
    window = episode["trace_steps"][-10:]  # last up to 10 policy steps (2s at 5Hz -- widened below if shorter)
    exploration_deviation_in_window = any(
        v["exploration_deviation"] for step in window for v in step["vehicles"].values()
    )
    ttc_series = [
        min((v["same_lane_ttc"] for v in step["vehicles"].values() if v["same_lane_ttc"] is not None), default=None)
        for step in window
    ]
    ttc_series = [t for t in ttc_series if t is not None]
    gradual_approach = len(ttc_series) >= 3 and ttc_series[0] > ttc_series[-1]
    return {
        "exploration_deviation_in_final_window": exploration_deviation_in_window,
        "ttc_series_final_window": ttc_series,
        "gradual_ttc_decrease": gradual_approach,
        "note": (
            "exploration_deviation_in_final_window=True -> category E (exploration-induced) is plausible for this episode; "
            "gradual_ttc_decrease=True with no exploration deviation -> consistent with D/F (undertrained/coordination) rather than a sudden anomaly; "
            "this is a heuristic flag, not a definitive causal classification -- see live_check findings for the authoritative defect/no-defect determination"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-bank", type=Path, required=True)
    p.add_argument("--scenario-ids", type=str, nargs="+", required=True)
    p.add_argument("--checkpoints", type=str, nargs="+", required=True, help="seed:path pairs")
    p.add_argument("--dr1-csv", type=Path, required=True)
    p.add_argument("--absolute-step", type=int, default=600_000)
    p.add_argument("--eps-decay-steps", type=int, default=640_000)
    p.add_argument("--episode-max-steps", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--skip-regression-suite", action="store_true")
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    regression_result = {"skipped": True}
    if not args.skip_regression_suite:
        print("[DR2] running tests/study_b/ regression suite (covers invariants 5/6/10/11/12/13)...")
        regression_result = run_regression_suite()
        print(f"[DR2] regression suite returncode={regression_result['returncode']}")

    picks = pick_representative_episodes(args.dr1_csv)

    all_findings: list[dict] = []
    all_classifications: dict = {}
    checkpoint_by_seed = {}
    for pair in args.checkpoints:
        seed_str, ckpt_str = pair.split(":", 1)
        checkpoint_by_seed[int(seed_str)] = Path(ckpt_str)

    for seed_str, target in picks.items():
        seed = int(seed_str)
        ckpt = checkpoint_by_seed[seed]
        for kind, ep_idx in target.items():
            if ep_idx is None:
                continue
            print(f"[DR2] tracing seed={seed} kind={kind} episode_index={ep_idx} (replaying 0..{ep_idx})")
            episode = trace_episode(
                seed=seed, checkpoint=ckpt, scenario_bank_path=args.scenario_bank, scenario_ids=args.scenario_ids,
                absolute_step=args.absolute_step, eps_decay_steps=args.eps_decay_steps,
                target_episode_index=ep_idx, episode_max_steps=args.episode_max_steps, device=args.device,
            )
            assert episode["term_reason"] == kind or (kind == "timeout" and episode["term_reason"] == "truncation"), \
                f"replay mismatch: expected {kind}, got {episode['term_reason']} (seed={seed}, ep={ep_idx})"

            findings = run_live_checks(episode)
            all_findings.extend(findings)

            classification = classify_collision_causes(episode) if kind == "collision" else {}
            if classification:
                all_classifications[f"seed{seed}_{kind}_ep{ep_idx}"] = classification

            ep_out = args.output_dir / f"C4_DR2_TRACE_seed{seed}_{kind}_ep{ep_idx}.json"
            ep_out.write_text(json.dumps(episode, indent=2, default=str), encoding="utf-8")

    report = {
        "regression_suite": regression_result,
        "representative_episodes_picked": picks,
        "live_check_findings": all_findings,
        "n_live_check_findings": len(all_findings),
        "collision_cause_classification_heuristics": all_classifications,
    }
    (args.output_dir / "C4_DR2_SUMMARY.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# DR2 -- C4 four-seed behavior/observation/control diagnostics\n"]
    lines.append(f"Regression suite: returncode={regression_result.get('returncode')}\n```\n{regression_result.get('summary_tail','')}\n```\n")
    lines.append(f"Live-check findings (invariants 1/2/3/4/7/8/9 checked directly against traced episodes): "
                 f"**{len(all_findings)}**\n")
    if all_findings:
        for f in all_findings:
            lines.append(f"- seed={f['seed']} ep={f['episode_index']} invariant={f['invariant']}: {f['detail']}")
    else:
        lines.append("(none -- all traced episodes satisfied invariants 1/2/3/4/7/8/9)\n")
    lines.append("\n## Representative episodes picked\n")
    lines.append(json.dumps(picks, indent=2))
    lines.append("\n## Collision-cause heuristic flags (last-window TTC/exploration analysis)\n")
    lines.append(json.dumps(all_classifications, indent=2))
    (args.output_dir / "C4_DR2_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote DR2 outputs to {args.output_dir}")
    print(f"n_live_check_findings={len(all_findings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
