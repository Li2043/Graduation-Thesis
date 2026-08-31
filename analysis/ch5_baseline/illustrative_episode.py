"""5.6.7: render the illustrative matched episode (seed 920103, scenario
H1_00004, selected per the documented rule) under all four conditions."""
from __future__ import annotations
import os
import sys, json
from pathlib import Path

BUNDLE_ROOT = Path(os.environ.get("FINAL_NEW_BUNDLE", ""))  # raw checkpoints not distributed with this repo; set env var
sys.path.insert(0, str(BUNDLE_ROOT / "project" / "src"))
sys.path.insert(0, str(BUNDLE_ROOT / "project" / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"))

from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig
from thesis.study_b.q_ensemble import ensemble_window_for_stage_end, load_ensemble_agents, select_ensemble_actions
from thesis.study_b.training_common import load_scenario_bank
from thesis.study_b.utility import episode_utilities, episode_burdens
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = Path(os.environ.get("THESIS_FIGURES_DIR", str(Path(__file__).resolve().parent / "outputs" / "figures")))
OUT = Path(__file__).resolve().parent / "outputs"
SEED = 920103
SCENARIO_ID = "H1_00004"
CONDS4 = ["baseline", "mean", "ggi", "maximin"]
COND_LABELS = {"baseline": "Baseline", "mean": "Mean", "ggi": "GGI", "maximin": "Maximin"}
VIDS = ["V0", "V1", "V2", "V3"]
VCOLORS = {"V0": "#1f77b4", "V1": "#ff7f0e", "V2": "#2ca02c", "V3": "#d62728"}
WINDOW = ensemble_window_for_stage_end(2_000_000)


def checkpoint_paths_for(condition, seed):
    if condition == "baseline":
        d = BUNDLE_ROOT / "checkpoints" / "taskonly_arm" / str(seed) / f"seed_{seed}_Formal_taskonly"
        return {s: d / f"ckpt_step_{s}.pt" for s in WINDOW}
    # checkpoints not distributed with this repo; set SEED_REPL_WELFARE_CKPT to your local bundle's welfare checkpoint dir
    d = Path(os.environ.get("SEED_REPL_WELFARE_CKPT", "")) / str(seed) / \
        {"mean": "Mean", "ggi": "GGI", "maximin": "Maximin"}[condition] / f"seed_{seed}_Formal_{condition}"
    return {s: d / f"ckpt_step_{s}.pt" for s in WINDOW}


scenarios = load_scenario_bank(BUNDLE_ROOT / "scenario_banks" / "H1.json")
scenario = next(s for s in scenarios if s.scenario_id == SCENARIO_ID)

results = {}
for cond in CONDS4:
    stage_name = "Formal_taskonly" if cond == "baseline" else f"Formal_{cond}"
    agents = load_ensemble_agents(seed=SEED, checkpoint_paths=checkpoint_paths_for(cond, SEED),
                                   expected_steps=WINDOW, expected_stage_by_step=dict.fromkeys(WINDOW, stage_name))
    env_config = ThesisHighwayMergeEnvConfig(episode_max_steps=200, action_representation="meta_speed")
    env = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=env_config, local_sensing_range_m=50.0))
    obs, _ = env.reset(seed=0, scenario=scenario)

    traj = {v: {"u": [], "x": []} for v in VIDS}
    hard_brake_steps = {v: [] for v in VIDS}
    entry_step = {v: None for v in VIDS}
    exit_step = {v: None for v in VIDS}
    pre_active = {v: True for v in VIDS}
    term_reason = "truncation"
    for t in range(200):
        actions = select_ensemble_actions(agents, obs)
        obs, _r, terminated, truncated, step_info = env.step(actions)
        for v in VIDS:
            if not pre_active[v]:
                continue
            vehicle = env._env._vehicle_by_id[v]  # noqa: SLF001
            x, _y = env._env.world_xy(vehicle)  # noqa: SLF001
            target = scenario.vehicles[v].target_speed
            u = float(np.clip(vehicle.speed / target, 0.0, 1.0))
            traj[v]["u"].append(u); traj[v]["x"].append(x)
            if entry_step[v] is None and x >= 220: entry_step[v] = t
            if exit_step[v] is None and x >= 380: exit_step[v] = t
            accel = float(vehicle.action["acceleration"])
            if accel <= -3.0:
                hard_brake_steps[v].append(t)
        pre_active = dict(step_info["active"])
        if terminated:
            term_reason = "collision" if step_info["collision_event"] else "success"
            break
        if truncated:
            term_reason = "truncation"; break

    traces = env.episode_traces()
    utilities = episode_utilities(traces)
    burdens = episode_burdens(traces, dt=env.dt())
    finished = sorted([(v, exit_step[v]) for v in VIDS if exit_step[v] is not None], key=lambda p: p[1])
    merge_order = ">".join(v for v, _ in finished) if finished else "DNF"

    results[cond] = {"traj": traj, "hard_brake_steps": hard_brake_steps, "entry_step": entry_step,
                      "exit_step": exit_step, "term_reason": term_reason, "merge_order": merge_order,
                      "U": {v: round(utilities[v], 4) for v in VIDS}, "C": {v: round(burdens[v], 4) for v in VIDS}}
    print(f"{cond}: term={term_reason} merge_order={merge_order} U={results[cond]['U']} C={results[cond]['C']}")

with open(OUT / "illustrative_episode_stats.json", "w", encoding="utf-8") as f:
    json.dump({"seed": SEED, "scenario_id": SCENARIO_ID,
               "results": {c: {k: v for k, v in r.items() if k != "traj"} for c, r in results.items()}},
              f, indent=2, default=str)

fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
for ax, cond in zip(axes, CONDS4):
    r = results[cond]
    for v in VIDS:
        u = r["traj"][v]["u"]
        role = scenario.vehicles[v].role; sc = scenario.vehicles[v].speed_class
        ax.plot(np.arange(len(u)) * 0.2, u, color=VCOLORS[v], linewidth=1.8, label=f"{v} ({role[:1].upper()}-{sc[:1].upper()})")
        for hb_t in r["hard_brake_steps"][v]:
            ax.axvline(hb_t * 0.2, color=VCOLORS[v], alpha=0.15, linewidth=4, ymin=0, ymax=0.08)
    ax.set_title(f"{COND_LABELS[cond]}\n({r['term_reason']}, order: {r['merge_order']})", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.25, linewidth=0.6)
axes[0].set_ylabel("Normalized instantaneous mobility $u_{i,t}$")
axes[0].set_ylim(-0.05, 1.15)
axes[-1].legend(fontsize=8, loc="lower left", bbox_to_anchor=(1.01, 0.0))
fig.suptitle(f"Illustrative matched episode -- seed {SEED}, scenario {SCENARIO_ID}\nnot an inferential sample; shaded ticks near x-axis mark hard-brake steps (accel <= -3.0 m/s²)", fontsize=10.5)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_13_illustrative_episode.png", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig5_13_illustrative_episode.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote fig5_13_illustrative_episode")
