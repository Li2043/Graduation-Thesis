#!/usr/bin/env python3
"""new_protocol.md Sec 29 behavioural diagnostic, run on the ORIGINAL six
formal seeds (920xxx seeds don't exist yet). Read-only against existing
frozen checkpoints -- no training.

Reference event (Sec 29.1, decided BEFORE looking at any result, applied
identically to whichever seeds it's run against): the timestep at which
the FIRST of the four vehicles exits the merge region (min completion_step
over the 4 vehicles). This is an "already available merge-event variable"
(step_info["exit_event"], the same field stage_q_ensemble_gate.py and
evaluate_formal_behavioral.py already use for merge_order) -- nothing new
invented for the definition itself.

"Principal conflict vehicle" (Sec 29.2): the other vehicle sharing the
same ttc_slot ("front"/"rear") -- this is the scenario generator's own
matched-TTC conflict-pair structure (scenario_generator.py's
matched_ttc_deltas groups vehicles by ttc_slot into exactly these pairs),
not an invented pairing.

At the reference event, among the three vehicles still active, the one
with the largest ACCUMULATED burden so far (dt * sum(1-attainment) over
active steps 0..ref_step) is the "high-burden vehicle" for that episode.
Episodes with no reference event (no vehicle ever exits, e.g. an
early collision before anyone crosses) are excluded and counted.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# DENSE-REWARD COPY (2026-08-26): see evaluate_formal_welfare.py's identical
# comment -- was a stale hardcoded Path(r"F:\正式训练"), replaced with
# portable parents[]-based resolution. No scientific logic touched.
BUNDLE_ROOT = Path(__file__).resolve().parents[5]
PROJECT_ROOT = BUNDLE_ROOT / "project"
SB_SCRIPTS = PROJECT_ROOT / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SB_SCRIPTS))

from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig  # noqa: E402
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig  # noqa: E402
from thesis.study_b.q_ensemble import ensemble_window_for_stage_end, load_ensemble_agents, select_ensemble_actions  # noqa: E402
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402

SEEDS = [900101, 900102, 900103, 900104, 910101, 910102]
CONDITIONS = ["mean", "ggi", "maximin"]
STAGE_END = 2_000_000
WINDOW = ensemble_window_for_stage_end(STAGE_END)
CKPT_ROOT = BUNDLE_ROOT / "checkpoints" / "formal_runs"
BANK_ROOT = BUNDLE_ROOT / "scenario_banks"
OUT_ROOT = BUNDLE_ROOT / "outputs" / "welfare_analysis"
HARD_BRAKE_THRESHOLD = -3.0  # frozen physical floor, consistent with Sec 5.6.5's corrected threshold
VIDS = ("V0", "V1", "V2", "V3")

CSV_FIELDS = [
    "run_id", "condition", "seed", "scenario_id", "has_reference_event",
    "reference_step", "first_exit_vehicle", "high_burden_vehicle",
    "high_burden_role", "high_burden_speed_class", "burden_at_reference",
    "principal_conflict_vehicle", "conflict_never_exits",
    "high_burden_completion_step", "conflict_completion_step",
    "high_burden_goes_before_conflict", "subsequent_burden_increment",
    "bystander_hard_brake_before_high_burden_exit",
]


def checkpoint_paths_for(run_id: str, seed: int) -> dict[int, Path]:
    d = CKPT_ROOT / run_id / f"seed_{seed}_Formal_{run_id.split('_', 1)[0]}"
    return {s: d / f"ckpt_step_{s}.pt" for s in WINDOW}


def run_one(run_id: str, condition: str, seed: int, bank_name: str) -> list[dict]:
    stage_name = f"Formal_{condition}"
    agents = load_ensemble_agents(
        seed=seed, checkpoint_paths=checkpoint_paths_for(run_id, seed), expected_steps=WINDOW,
        expected_stage_by_step=dict.fromkeys(WINDOW, stage_name),
    )
    scenarios = load_scenario_bank(BANK_ROOT / f"{bank_name}.json")
    env_config = ThesisHighwayMergeEnvConfig(episode_max_steps=200, action_representation="meta_speed")
    env = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=env_config, local_sensing_range_m=50.0))
    dt = env.dt()

    rows = []
    for scenario in scenarios:
        obs, _info = env.reset(seed=0, scenario=scenario)
        completion_step: dict[str, int | None] = dict.fromkeys(env.active_vehicle_ids)
        for t in range(200):
            actions = select_ensemble_actions(agents, obs)
            obs, _reward, terminated, truncated, step_info = env.step(actions)
            for vid, done in step_info["exit_event"].items():
                if done and completion_step[vid] is None:
                    completion_step[vid] = t + 1
            if terminated or truncated:
                break

        traces = env.episode_traces()
        exited = {vid: s for vid, s in completion_step.items() if s is not None}
        row = {"run_id": run_id, "condition": condition, "seed": seed, "scenario_id": scenario.scenario_id}

        if not exited:
            row.update({"has_reference_event": 0})
            for f in CSV_FIELDS:
                row.setdefault(f, "")
            rows.append(row)
            continue

        ref_step = min(exited.values())
        first_exit_vid = min(exited, key=lambda v: exited[v])
        active_at_ref = [v for v in VIDS if v != first_exit_vid]

        def accumulated_burden(vid: str, upto_step: int) -> float:
            tr = traces[vid]
            attain = tr.attainments()
            n = min(upto_step, len(attain))
            return dt * sum((1.0 - attain[t]) for t in range(n) if tr.active_flags[t])

        burdens_at_ref = {v: accumulated_burden(v, ref_step) for v in active_at_ref}
        high_burden_vid = max(burdens_at_ref, key=lambda v: burdens_at_ref[v])
        conflict_vid = scenario.vehicles[
            next(v for v in VIDS if v != high_burden_vid and scenario.vehicles[v].ttc_slot == scenario.vehicles[high_burden_vid].ttc_slot)
        ].vehicle_id if any(
            v != high_burden_vid and scenario.vehicles[v].ttc_slot == scenario.vehicles[high_burden_vid].ttc_slot for v in VIDS
        ) else None

        hb_step = completion_step[high_burden_vid]
        conflict_step = completion_step.get(conflict_vid) if conflict_vid else None
        conflict_never_exits = int(conflict_vid is not None and conflict_step is None)
        goes_before = None
        if hb_step is not None and conflict_step is not None:
            goes_before = int(hb_step < conflict_step)

        total_burden_hb = accumulated_burden(high_burden_vid, len(traces[high_burden_vid].speeds))
        subsequent_increment = total_burden_hb - burdens_at_ref[high_burden_vid]

        bystanders = [v for v in active_at_ref if v != high_burden_vid]
        end_window = hb_step if hb_step is not None else len(traces[high_burden_vid].speeds)
        bystander_brake = 0
        for v in bystanders:
            accel = traces[v].accelerations
            window = accel[ref_step:end_window] if end_window > ref_step else []
            if any(a <= HARD_BRAKE_THRESHOLD for a in window):
                bystander_brake = 1
                break

        row.update({
            "has_reference_event": 1, "reference_step": ref_step, "first_exit_vehicle": first_exit_vid,
            "high_burden_vehicle": high_burden_vid,
            "high_burden_role": scenario.vehicles[high_burden_vid].role,
            "high_burden_speed_class": scenario.vehicles[high_burden_vid].speed_class,
            "burden_at_reference": round(burdens_at_ref[high_burden_vid], 6),
            "principal_conflict_vehicle": conflict_vid or "",
            "conflict_never_exits": conflict_never_exits,
            "high_burden_completion_step": hb_step if hb_step is not None else "",
            "conflict_completion_step": conflict_step if conflict_step is not None else "",
            "high_burden_goes_before_conflict": "" if goes_before is None else goes_before,
            "subsequent_burden_increment": round(subsequent_increment, 6),
            "bystander_hard_brake_before_high_burden_exit": bystander_brake,
        })
        rows.append(row)
    return rows


def all_jobs() -> list[tuple[str, str, int, str]]:
    jobs = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            jobs.append((f"{condition}_{seed}", condition, seed, "H1"))
    return jobs


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = all_jobs()
    my_jobs = [j for i, j in enumerate(jobs) if i % args.num_shards == args.shard_index]

    all_rows = []
    for run_id, condition, seed, bank in my_jobs:
        print(f"[high_burden_diagnostic shard {args.shard_index}] {run_id} ...", flush=True)
        rows = run_one(run_id, condition, seed, bank)
        all_rows.extend(rows)
        n_ref = sum(r["has_reference_event"] for r in rows)
        n_before = sum(1 for r in rows if r.get("high_burden_goes_before_conflict") == 1)
        n_after = sum(1 for r in rows if r.get("high_burden_goes_before_conflict") == 0)
        print(f"  n={len(rows)} with_reference_event={n_ref} goes_before_conflict={n_before} goes_after={n_after}", flush=True)

    out_csv = OUT_ROOT / f"high_burden_diagnostic_shard{args.shard_index}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} rows -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
