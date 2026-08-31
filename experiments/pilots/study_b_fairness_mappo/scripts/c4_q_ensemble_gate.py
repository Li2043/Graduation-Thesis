#!/usr/bin/env python3
"""C4_Q_ENSEMBLE_GATE (RUNBOOK Amendment 12, 2026-08-17):
PRE_FORMAL_STABILIZATION_GATE.

Evaluates the fixed equal-weight Q-ensemble (550K/600K/clean-650K/
700K, same seed lineage) against the frozen C4 scenario bank.
epsilon=0 by construction (the ensemble API has no epsilon parameter).
Deterministic -> N=4/seed (one trajectory per scenario), same
methodology/rationale as c4_greedy_gate_eval.py.

Also produces the required diagnostic-only comparison against each
component checkpoint's own action/margin at every step (section 14 --
does not affect the decision rule)."""

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

from dr1_c4_failure_map import classify_collision_type, min_pairwise_ttc  # noqa: E402
from thesis.study_b.envs.highwayenv_action import ACCELERATE, BRAKE, HOLD  # noqa: E402
from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig  # noqa: E402
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig  # noqa: E402
from thesis.study_b.q_ensemble import (  # noqa: E402
    EXPECTED_ENSEMBLE_STEPS,
    load_ensemble_agents,
    q_component_values,
    q_ensemble_values,
    select_ensemble_actions,
)
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402

_ACTION_NAME = {HOLD: "HOLD", ACCELERATE: "ACCELERATE", BRAKE: "BRAKE"}
_ACTION_ORDER = [HOLD, ACCELERATE, BRAKE]

EPISODE_CSV_FIELDS = [
    "seed", "scenario_id", "term_reason", "episode_length",
    "min_ttc", "failure_timestep", "collision_type",
    "collision_vehicle_ids", "collision_vehicle_roles", "collision_vehicle_speed_classes",
    "collision_vehicle_ttc_slots", "merge_order",
]


def _make_env(episode_max_steps: int) -> StudyBHeterogeneousHighwayEnv:
    cfg = ThesisHighwayMergeEnvConfig(episode_max_steps=episode_max_steps, action_representation="meta_speed")
    return StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=cfg))


def run_ensemble_gate_for_seed(
    *, seed: int, checkpoint_paths: dict[int, Path], scenario_bank_path: Path, scenario_ids: list[str],
    episode_max_steps: int, device: str,
) -> tuple[list[dict], list[dict]]:
    agents = load_ensemble_agents(seed=seed, checkpoint_paths=checkpoint_paths, device=device)

    scenarios = load_scenario_bank(scenario_bank_path)
    by_id = {s.scenario_id: s for s in scenarios}
    stage_scenarios = [by_id[sid] for sid in scenario_ids]

    env = _make_env(episode_max_steps)
    episode_rows: list[dict] = []
    diagnostic_rows: list[dict] = []

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

            ensemble_actions = select_ensemble_actions(agents, obs)

            for vid in vids:
                q_ens = q_ensemble_values(agents, obs[vid])
                q_comp = q_component_values(agents, obs[vid])
                q_ens_sorted = sorted(q_ens, reverse=True)
                component_actions = {step: int(sorted(range(3), key=lambda i: q_comp[step][i])[-1]) for step in EXPECTED_ENSEMBLE_STEPS}
                diagnostic_rows.append({
                    "seed": seed, "scenario_id": scenario.scenario_id, "t": t, "vehicle_id": vid,
                    "action_550K": _ACTION_NAME[component_actions[550_000]],
                    "action_600K": _ACTION_NAME[component_actions[600_000]],
                    "action_650K": _ACTION_NAME[component_actions[650_000]],
                    "action_700K": _ACTION_NAME[component_actions[700_000]],
                    "ensemble_action": _ACTION_NAME[int(ensemble_actions[vid])],
                    "q_ensemble_HOLD": float(q_ens[0]), "q_ensemble_ACCELERATE": float(q_ens[1]), "q_ensemble_BRAKE": float(q_ens[2]),
                    "ensemble_margin": float(q_ens_sorted[0] - q_ens_sorted[1]),
                    "component_margin_550K": float(sorted(q_comp[550_000], reverse=True)[0] - sorted(q_comp[550_000], reverse=True)[1]),
                    "component_margin_600K": float(sorted(q_comp[600_000], reverse=True)[0] - sorted(q_comp[600_000], reverse=True)[1]),
                    "component_margin_650K": float(sorted(q_comp[650_000], reverse=True)[0] - sorted(q_comp[650_000], reverse=True)[1]),
                    "component_margin_700K": float(sorted(q_comp[700_000], reverse=True)[0] - sorted(q_comp[700_000], reverse=True)[1]),
                })

            obs, _reward, terminated, truncated, step_info = env.step(ensemble_actions)
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

        episode_rows.append({
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

    return episode_rows, diagnostic_rows


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
    p.add_argument("--seed-checkpoint-roots", type=str, nargs="+", required=True,
                    help="seed:C4_550K_path:C4ext2_root_dir (C4ext2_root_dir contains ckpt_step_{600000,650000,700000}.pt)")
    p.add_argument("--episode-max-steps", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_episode_rows: list[dict] = []
    all_diagnostic_rows: list[dict] = []
    per_seed_summary: dict[str, dict] = {}

    for entry in args.seed_checkpoint_roots:
        seed_str, ckpt_550k, c4ext2_dir = entry.split(":", 2)
        seed = int(seed_str)
        c4ext2_dir_path = Path(c4ext2_dir)
        checkpoint_paths = {
            550_000: Path(ckpt_550k),
            600_000: c4ext2_dir_path / "ckpt_step_600000.pt",
            650_000: c4ext2_dir_path / "ckpt_step_650000.pt",
            700_000: c4ext2_dir_path / "ckpt_step_700000.pt",
        }
        episode_rows, diagnostic_rows = run_ensemble_gate_for_seed(
            seed=seed, checkpoint_paths=checkpoint_paths, scenario_bank_path=args.scenario_bank,
            scenario_ids=args.scenario_ids, episode_max_steps=args.episode_max_steps, device=args.device,
        )
        all_episode_rows.extend(episode_rows)
        all_diagnostic_rows.extend(diagnostic_rows)

        n = len(episode_rows)
        completion = sum(r["term_reason"] == "success" for r in episode_rows) / n
        collision = sum(r["term_reason"] == "collision" for r in episode_rows) / n
        timeout = sum(r["term_reason"] == "truncation" for r in episode_rows) / n
        gate = apply_original_c4_gate(completion)
        per_seed_summary[str(seed)] = {
            "n_episodes": n, "completion": completion, "collision": collision, "timeout": timeout, "gate": gate,
            "per_scenario": {r["scenario_id"]: {"term_reason": r["term_reason"], "episode_length": r["episode_length"]} for r in episode_rows},
        }
        print(f"[C4_Q_ENSEMBLE_GATE] seed={seed} n={n} completion={completion:.3f} collision={collision:.3f} "
              f"timeout={timeout:.3f} gate={gate}")

    ep_csv = args.output_dir / "C4_Q_ENSEMBLE_GATE.csv"
    with open(ep_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_episode_rows)

    diag_csv = args.output_dir / "C4_Q_ENSEMBLE_VS_COMPONENT_DIAGNOSTIC.csv"
    with open(diag_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_diagnostic_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_diagnostic_rows)

    gate_counts = {"PASS": 0, "SOFT_PASS": 0, "FAIL": 0}
    for s in per_seed_summary.values():
        gate_counts[s["gate"]] += 1
    n_strict_pass = gate_counts["PASS"]

    if n_strict_pass >= 3:
        case = "CASE_A"; classification = "SUCCESS"; c4_status = "QUALIFIED_FOR_NEXT_CURRICULUM_STAGE"
    elif n_strict_pass == 2:
        case = "CASE_B"; classification = "INSUFFICIENT"; c4_status = "NOT_YET_QUALIFIED"
    else:
        case = "CASE_C"; classification = "FAILED"; c4_status = "LOCAL_DQN_DID_NOT_MEET_STABLE_C4_COMPETENCE_REQUIREMENT"

    component_margins = [all_diagnostic_rows[i][f"component_margin_{s}K"] for i in range(len(all_diagnostic_rows)) for s in ("550", "600", "650", "700")]
    ensemble_margins = [r["ensemble_margin"] for r in all_diagnostic_rows]

    report = {
        "label": "PRE_FORMAL_STABILIZATION_GATE",
        "ensemble_definition": "Q_ensemble(o,a) = (1/4) * sum over {550K,600K,clean-650K,700K} of Q_theta_k(o,a); a = argmax_a Q_ensemble(o,a); epsilon=0 (no exploration knob exists)",
        "n_strict_pass": n_strict_pass, "gate_counts": gate_counts,
        "case": case, "C4_Q_ENSEMBLE_STABILIZATION": classification, "C4_status": c4_status,
        "per_seed": per_seed_summary,
        "original_gate_criteria": {"PASS": "completion>=0.90 and collision<=0.05 and timeout<=0.05", "SOFT_PASS": "0.75<=completion<0.90", "FAIL": "completion<0.75"},
        "diagnostic_margin_comparison": {
            "note": "diagnostic only, does not affect the decision rule",
            "component_margin_min": min(component_margins), "component_margin_median": sorted(component_margins)[len(component_margins) // 2],
            "ensemble_margin_min": min(ensemble_margins), "ensemble_margin_median": sorted(ensemble_margins)[len(ensemble_margins) // 2],
            "ensemble_margin_larger_than_component_median": sorted(ensemble_margins)[len(ensemble_margins) // 2] > sorted(component_margins)[len(component_margins) // 2],
        },
    }
    (args.output_dir / "C4_Q_ENSEMBLE_GATE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# C4_Q_ENSEMBLE_GATE -- PRE_FORMAL_STABILIZATION_GATE\n"]
    lines.append(f"**Case: {case} -- C4_Q_ENSEMBLE_STABILIZATION = {classification}, C4 = {c4_status}**\n")
    lines.append(f"Strict PASS count: {n_strict_pass}/4. Gate counts: {gate_counts}\n")
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
    lines.append(f"\n## Diagnostic margin comparison (informational only, does not affect the decision)\n")
    lines.append(json.dumps(report["diagnostic_margin_comparison"], indent=2))
    (args.output_dir / "C4_Q_ENSEMBLE_GATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {ep_csv}, {diag_csv}, and companion JSON/MD to {args.output_dir}")
    print(f"CASE={case} C4_Q_ENSEMBLE_STABILIZATION={classification} C4={c4_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
