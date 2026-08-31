#!/usr/bin/env python3
"""AUTHORITATIVE_C4_GATE_EVALUATION (RUNBOOK Amendment 9, 2026-08-17).

Runs the frozen greedy policy (epsilon=0, argmax, no exploration) for
each seed's frozen 600K checkpoint against the frozen C4 scenario bank,
and applies RUNBOOK sec 40's ORIGINAL numeric gate to the result.

Sample-size note (verified empirically, see tests/study_b/
test_c4_greedy_gate_eval.py::test_greedy_evaluation_is_deterministic_
across_repeated_runs): epsilon=0 execution against a FIXED scenario is
100% deterministic (same checkpoint + same scenario_id always produces
the exact same trajectory -- confirmed by running the same pair twice
and diffing). C4's scenario bank has exactly 4 fixed scenarios, so the
greedy gate is an EXACT, complete characterization of the frozen
policy's behavior on the frozen evaluation bank -- N=4 per seed, not a
statistical sample, and there is no way to manufacture additional
independent greedy replicates of the same 4 scenarios (they would just
be exact duplicates). This is the "existing frozen evaluation-bank
protocol specifies a different deterministic replication structure"
case the user's own item 4 anticipated.

Reuses dr1_c4_failure_map.py's collision-type classification and
min-TTC helpers so the per-episode diagnostic columns (collision type,
role/speed_class/ttc_slot, merge order, min TTC) are directly
comparable to DR1's exploration-on diagnostic output.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (REPO_SRC, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402

from dr1_c4_failure_map import classify_collision_type, min_pairwise_ttc  # noqa: E402
from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig  # noqa: E402
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig  # noqa: E402
from thesis.study_b.shared_local_dqn import SharedLocalDQNAgent, build_study_b_dqn_config  # noqa: E402
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402

EPISODE_CSV_FIELDS = [
    "seed", "scenario_id", "term_reason", "episode_length",
    "min_ttc", "failure_timestep", "collision_type",
    "collision_vehicle_ids", "collision_vehicle_roles", "collision_vehicle_speed_classes",
    "collision_vehicle_ttc_slots", "merge_order",
]


def run_greedy_gate_for_seed(
    *, seed: int, checkpoint: Path, scenario_bank_path: Path, scenario_ids: list[str],
    episode_max_steps: int, device: str, expected_step: int,
) -> list[dict]:
    """epsilon=0, greedy=True, no training. Loads ONLY the online
    network (via a fresh agent instance -- no optimiser/target/replay
    state exists to touch), never calls store_transition/maybe_update,
    never calls hard_sync_target -- structurally incapable of mutating
    any training state, matching evaluate_policy_highwayenv.py's own
    guarantee."""
    scenarios = load_scenario_bank(scenario_bank_path)
    by_id = {s.scenario_id: s for s in scenarios}
    stage_scenarios = [by_id[sid] for sid in scenario_ids]

    env_config = ThesisHighwayMergeEnvConfig(episode_max_steps=episode_max_steps, action_representation="meta_speed")
    env = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=env_config))

    dqn_config = build_study_b_dqn_config(device=device)
    agent = SharedLocalDQNAgent(dqn_config, seed=0)  # seed is irrelevant: greedy=True never touches the RNG
    ckpt = torch.load(checkpoint, map_location=device)
    agent.learner.online.load_state_dict(ckpt["online"])
    assert int(ckpt["step"]) == expected_step, f"expected checkpoint step={expected_step}, got step={ckpt['step']} from {checkpoint}"

    rows: list[dict] = []
    for scenario in stage_scenarios:
        obs, _info = env.reset(seed=0, scenario=scenario)
        term_reason = "ongoing"
        failure_timestep = None
        collision_pairs: list[tuple[str, str]] = []
        completion_step: dict[str, int | None] = dict.fromkeys(env.active_vehicle_ids)
        episode_min_ttc = float("inf")
        steps_taken = 0

        for t in range(episode_max_steps):
            vids = env.active_vehicle_ids
            x_positions = {vid: env._env.world_xy(env._env._vehicle_by_id[vid])[0] for vid in vids}  # noqa: SLF001
            y_positions = {vid: env._env.world_xy(env._env._vehicle_by_id[vid])[1] for vid in vids}  # noqa: SLF001
            speeds = {vid: float(env._env._vehicle_by_id[vid].speed) for vid in vids}  # noqa: SLF001
            step_ttc = min_pairwise_ttc(x_positions, y_positions, speeds, vids)
            if step_ttc < episode_min_ttc:
                episode_min_ttc = step_ttc

            actions = agent.select_actions(obs, epsilon=0.0, greedy=True)
            obs, _reward, terminated, truncated, step_info = env.step(actions)
            steps_taken += 1

            for vid, done in step_info["exit_event"].items():
                if done and completion_step[vid] is None:
                    completion_step[vid] = t + 1

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

        collision_vehicle_ids: list[str] = []
        collision_vehicle_roles: list[str] = []
        collision_vehicle_speed_classes: list[str] = []
        collision_vehicle_ttc_slots: list[str] = []
        collision_type = ""
        if term_reason == "collision" and collision_pairs:
            pair = collision_pairs[0]
            x_final = {vid: env._env.world_xy(env._env._vehicle_by_id[vid])[0] for vid in env.active_vehicle_ids}  # noqa: SLF001
            collision_type = classify_collision_type(x_final, pair)
            colliding_ids = sorted({vid for p in collision_pairs for vid in p})
            collision_vehicle_ids = colliding_ids
            collision_vehicle_roles = [scenario.vehicles[vid].role for vid in colliding_ids]
            collision_vehicle_speed_classes = [scenario.vehicles[vid].speed_class for vid in colliding_ids]
            collision_vehicle_ttc_slots = [scenario.vehicles[vid].ttc_slot for vid in colliding_ids]

        merge_order = sorted(
            (vid for vid in completion_step if completion_step[vid] is not None),
            key=lambda vid: completion_step[vid],
        )
        merge_order_str = ">".join(merge_order) if merge_order else "DNF"

        rows.append({
            "seed": seed, "scenario_id": scenario.scenario_id, "term_reason": term_reason,
            "episode_length": steps_taken,
            "min_ttc": (None if episode_min_ttc == float("inf") else round(episode_min_ttc, 4)),
            "failure_timestep": failure_timestep, "collision_type": collision_type,
            "collision_vehicle_ids": ";".join(collision_vehicle_ids),
            "collision_vehicle_roles": ";".join(collision_vehicle_roles),
            "collision_vehicle_speed_classes": ";".join(collision_vehicle_speed_classes),
            "collision_vehicle_ttc_slots": ";".join(collision_vehicle_ttc_slots),
            "merge_order": merge_order_str,
        })

    return rows


def apply_original_c4_gate(completion: float) -> str:
    if completion >= 0.90:
        return "PASS"
    if completion >= 0.75:
        return "SOFT_PASS"
    return "FAIL"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-bank", type=Path, required=True)
    p.add_argument("--scenario-ids", type=str, nargs="+", required=True)
    p.add_argument("--checkpoints", type=str, nargs="+", required=True, help="seed:path pairs")
    p.add_argument("--episode-max-steps", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-step", type=int, default=600_000)
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    per_seed_summary: dict[str, dict] = {}

    for pair in args.checkpoints:
        seed_str, ckpt_str = pair.split(":", 1)
        seed = int(seed_str)
        rows = run_greedy_gate_for_seed(
            seed=seed, checkpoint=Path(ckpt_str), scenario_bank_path=args.scenario_bank,
            scenario_ids=args.scenario_ids, episode_max_steps=args.episode_max_steps, device=args.device,
            expected_step=args.expected_step,
        )
        all_rows.extend(rows)
        n = len(rows)
        completion = sum(r["term_reason"] == "success" for r in rows) / n
        collision = sum(r["term_reason"] == "collision" for r in rows) / n
        timeout = sum(r["term_reason"] == "truncation" for r in rows) / n
        per_scenario = {
            r["scenario_id"]: {"term_reason": r["term_reason"], "episode_length": r["episode_length"]}
            for r in rows
        }
        gate = apply_original_c4_gate(completion)
        per_seed_summary[str(seed)] = {
            "n_episodes": n, "completion": completion, "collision": collision, "timeout": timeout,
            "gate": gate, "per_scenario": per_scenario,
        }
        print(f"[C4_GREEDY_GATE] seed={seed} n={n} completion={completion:.3f} collision={collision:.3f} "
              f"timeout={timeout:.3f} gate={gate}")

    step_tag = f"{args.expected_step // 1000}K"
    csv_path = args.output_dir / f"C4_GREEDY_GATE_{step_tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    n_total = len(all_rows)
    pooled = {
        "n_episodes": n_total,
        "completion": sum(r["term_reason"] == "success" for r in all_rows) / n_total,
        "collision": sum(r["term_reason"] == "collision" for r in all_rows) / n_total,
        "timeout": sum(r["term_reason"] == "truncation" for r in all_rows) / n_total,
    }
    gate_counts = {"PASS": 0, "SOFT_PASS": 0, "FAIL": 0}
    for s in per_seed_summary.values():
        gate_counts[s["gate"]] += 1

    report = {
        "label": "AUTHORITATIVE_C4_GATE_EVALUATION",
        "epsilon": 0.0, "greedy": True, "checkpoint_step": args.expected_step,
        "sample_size_note": (
            "N=4 per seed (one deterministic trajectory per C4 scenario) -- epsilon=0 against a "
            "fixed scenario is 100% deterministic (verified: tests/study_b/test_c4_greedy_gate_eval.py), "
            "so this is an EXACT characterization of the frozen policy on the frozen evaluation bank, "
            "not a statistical sample; additional replicates of the same scenario would be exact duplicates."
        ),
        "pooled": pooled,
        "per_seed": per_seed_summary,
        "gate_counts": gate_counts,
        "original_gate_criteria": {"PASS": "completion>=0.90", "SOFT_PASS": "0.75<=completion<0.90", "FAIL": "completion<0.75"},
    }
    (args.output_dir / f"C4_GREEDY_GATE_{step_tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [f"# AUTHORITATIVE_C4_GATE_EVALUATION -- C4 {step_tag} greedy (epsilon=0) gate\n"]
    lines.append(f"N={n_total} (N=4/seed, exact deterministic characterization, not a sample -- see sample_size_note in the JSON)\n")
    lines.append(f"Pooled: completion={pooled['completion']:.3f} collision={pooled['collision']:.3f} timeout={pooled['timeout']:.3f}\n")
    lines.append(f"Gate counts: {gate_counts}\n")
    lines.append("\n| seed | n | completion | collision | timeout | gate |")
    lines.append("|---|---|---|---|---|---|")
    for seed, s in per_seed_summary.items():
        lines.append(f"| {seed} | {s['n_episodes']} | {s['completion']:.3f} | {s['collision']:.3f} | {s['timeout']:.3f} | {s['gate']} |")
    lines.append("\n## Per-seed per-scenario outcome\n")
    lines.append("| seed | scenario_id | term_reason | episode_length |")
    lines.append("|---|---|---|---|")
    for seed, s in per_seed_summary.items():
        for sid, v in sorted(s["per_scenario"].items()):
            lines.append(f"| {seed} | {sid} | {v['term_reason']} | {v['episode_length']} |")
    (args.output_dir / f"C4_GREEDY_GATE_{step_tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {csv_path} and companion JSON/MD to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
