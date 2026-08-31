"""Task A2 (followup prompt): evaluate all twelve C64 (curriculum, pre-branch)
checkpoint-Q ensembles on the frozen H1 held-out bank (256 scenarios),
epsilon_eval=0. This is the pre-treatment measurement for the Section 5.7.4
moderation diagnostic -- it must NOT reuse the Mean/GGI/Maximin/Baseline
2.0M endpoints, and must NOT reuse the C64-on-Q(64) curriculum-gate numbers
(those used the 64-scenario curriculum gate bank, not H1).

Reuses evaluate_formal_welfare.run_one()/CSV_FIELDS unchanged; only the
checkpoint-locating and window logic are monkeypatched, following the same
pattern already established by evaluate_replication.py's _patch_module().
Read-only against frozen checkpoints; writes new CSVs only.
"""
from __future__ import annotations
import os

import csv
import sys
from pathlib import Path

FINAL_NEW = Path(os.environ.get("FINAL_NEW_BUNDLE", ""))  # raw checkpoints not distributed with this repo; set env var
SEED_REPL = Path(os.environ.get("SEED_REPL_BUNDLE", ""))  # raw checkpoints not distributed with this repo; set env var
PROJECT_ROOT = SEED_REPL / "project"
SB_SCRIPTS = PROJECT_ROOT / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SB_SCRIPTS))

from thesis.study_b.q_ensemble import ensemble_window_for_stage_end  # noqa: E402

import evaluate_formal_welfare as efw  # noqa: E402

STAGE_END = 1_200_000
WINDOW = ensemble_window_for_stage_end(STAGE_END)  # (1050000, 1100000, 1150000, 1200000)
assert WINDOW == (1_050_000, 1_100_000, 1_150_000, 1_200_000)

SEEDS12 = [900101, 900102, 900103, 900104, 910101, 910102,
           920101, 920102, 920103, 920104, 920105, 920106]

CKPT_DIR_FOR_SEED = {
    900101: FINAL_NEW / "checkpoints" / "formal_init" / "900101" / "C64_R50" / "seed_900101_C64_R50",
    900102: FINAL_NEW / "checkpoints" / "formal_init" / "900102" / "C64_R50" / "seed_900102_C64_R50",
    900103: FINAL_NEW / "checkpoints" / "formal_init" / "900103" / "C64_R50" / "seed_900103_C64_R50",
    900104: FINAL_NEW / "checkpoints" / "formal_init" / "900104" / "C64_R50" / "seed_900104_C64_R50",
    910101: FINAL_NEW / "checkpoints" / "curriculum_910101_910102" / "910101" / "C64_R50" / "seed_910101_C64_R50",
    910102: FINAL_NEW / "checkpoints" / "curriculum_910101_910102" / "910102" / "C64_R50" / "seed_910102_C64_R50",
    920101: SEED_REPL / "checkpoints" / "seed_replication_v1" / "curriculum" / "920101" / "C64_R50" / "seed_920101_C64_R50",
    920102: SEED_REPL / "checkpoints" / "seed_replication_v1" / "curriculum" / "920102" / "C64_R50" / "seed_920102_C64_R50",
    920103: SEED_REPL / "checkpoints" / "seed_replication_v1" / "curriculum" / "920103" / "C64_R50" / "seed_920103_C64_R50",
    920104: SEED_REPL / "checkpoints" / "seed_replication_v1" / "curriculum" / "920104" / "C64_R50" / "seed_920104_C64_R50",
    920105: SEED_REPL / "checkpoints" / "seed_replication_v1" / "curriculum" / "920105" / "C64_R50" / "seed_920105_C64_R50",
    920106: SEED_REPL / "checkpoints" / "seed_replication_v1" / "curriculum" / "920106" / "C64_R50" / "seed_920106_C64_R50",
}

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def checkpoint_paths_for(run_id: str, seed: int) -> dict[int, Path]:
    d = CKPT_DIR_FOR_SEED[seed]
    return {s: d / f"ckpt_step_{s}.pt" for s in WINDOW}


def all_jobs() -> list[tuple[str, str, int, str]]:
    return [(f"c64_{seed}", "c64", seed, "H1") for seed in SEEDS12]


def verify_checkpoints() -> list[str]:
    missing = []
    for seed in SEEDS12:
        d = CKPT_DIR_FOR_SEED[seed]
        for s in WINDOW:
            p = d / f"ckpt_step_{s}.pt"
            if not p.exists():
                missing.append(str(p))
    return missing


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    missing = verify_checkpoints()
    if missing:
        print(f"[evaluate_c64_prebranch_h1] MISSING {len(missing)} checkpoints:")
        for p in missing:
            print(" ", p)
        return 2

    # Monkeypatch evaluate_formal_welfare's module-level checkpoint resolution
    # and window; run_one()/CSV_FIELDS/episode-eval logic are reused unchanged.
    efw.WINDOW = WINDOW
    efw.checkpoint_paths_for = checkpoint_paths_for
    efw.BANK_ROOT = FINAL_NEW / "scenario_banks"

    # Patch stage_name resolution inside run_one: run_one() builds
    # stage_name = f"Formal_{condition}" and passes expected_stage_by_step
    # keyed on that. C64 checkpoints carry stage="C64_R50", not "Formal_c64",
    # so we cannot call run_one() unmodified -- reimplement the same body
    # with the correct stage_name, reusing every other piece of run_one.
    from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig
    from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig
    from thesis.study_b.q_ensemble import load_ensemble_agents, select_ensemble_actions
    from thesis.study_b.training_common import load_scenario_bank
    from thesis.study_b.utility import episode_burdens, episode_utilities, generalized_gini_welfare, gini_coefficient

    def run_one_c64(seed: int, bank_name: str) -> list[dict]:
        stage_name = "C64_R50"
        ckpt_paths = checkpoint_paths_for(f"c64_{seed}", seed)
        agents = load_ensemble_agents(
            seed=seed, checkpoint_paths=ckpt_paths, expected_steps=WINDOW,
            expected_stage_by_step=dict.fromkeys(WINDOW, stage_name),
        )
        scenarios = load_scenario_bank(efw.BANK_ROOT / f"{bank_name}.json")
        env_config = ThesisHighwayMergeEnvConfig(episode_max_steps=200, action_representation="meta_speed")
        env = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=env_config, local_sensing_range_m=50.0))

        rows = []
        for scenario in scenarios:
            obs, _info = env.reset(seed=0, scenario=scenario)
            term_reason = "truncation"
            steps_taken = 0
            for _t in range(200):
                actions = select_ensemble_actions(agents, obs)
                obs, _reward, terminated, truncated, step_info = env.step(actions)
                steps_taken += 1
                if terminated:
                    term_reason = "collision" if step_info["collision_event"] else "success"
                    break
                if truncated:
                    term_reason = "truncation"
                    break

            traces = env.episode_traces()
            utilities = episode_utilities(traces)
            burdens = episode_burdens(traces, dt=env.dt())
            u_values = list(utilities.values())
            min_vid = min(utilities, key=lambda v: utilities[v])

            row = {
                "run_id": f"c64_{seed}", "condition": "c64", "seed": seed, "bank": bank_name,
                "scenario_id": scenario.scenario_id, "traffic_type": scenario.traffic_type,
                "term_reason": term_reason,
                "completion": int(term_reason == "success"), "collision": int(term_reason == "collision"),
                "timeout": int(term_reason == "truncation"),
                "mean_U": sum(u_values) / len(u_values), "min_U": utilities[min_vid],
                "min_U_vehicle": min_vid, "min_U_role": scenario.vehicles[min_vid].role,
                "min_U_speed_class": scenario.vehicles[min_vid].speed_class,
                "ggi": generalized_gini_welfare(u_values), "gini": gini_coefficient(u_values),
                "C_max": max(burdens.values()), "C_mean": sum(burdens.values()) / len(burdens),
                "episode_length": steps_taken,
            }
            for vid in ("V0", "V1", "V2", "V3"):
                row[f"role_{vid}"] = scenario.vehicles[vid].role
                row[f"speed_class_{vid}"] = scenario.vehicles[vid].speed_class
                row[f"U_{vid}"] = utilities[vid]
                row[f"C_{vid}"] = burdens[vid]
            rows.append(row)
        return rows

    jobs = all_jobs()
    my_jobs = [j for i, j in enumerate(jobs) if i % args.num_shards == args.shard_index]

    all_rows = []
    for run_id, condition, seed, bank in my_jobs:
        print(f"[evaluate_c64_prebranch_h1 shard {args.shard_index}] {run_id} on {bank} ...", flush=True)
        rows = run_one_c64(seed, bank)
        all_rows.extend(rows)
        n = len(rows)
        comp = sum(r["completion"] for r in rows) / n
        coll = sum(r["collision"] for r in rows) / n
        to = sum(r["timeout"] for r in rows) / n
        print(f"  n={n} completion={comp:.4f} collision={coll:.4f} timeout={to:.4f}", flush=True)

    out_csv = OUT_DIR / f"c64_prebranch_h1_shard{args.shard_index}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=efw.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} rows -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
