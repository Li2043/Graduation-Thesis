"""WSC v2 behavioural mechanism trajectory run. Evaluation-only: loads frozen
Original (18D) and WSC v2 (22D, corrected mapping) checkpoints read-only,
replays them on the frozen H1 bank, and extracts pre-specified behavioural
events (see outputs/wsc_behavioural_metric_definitions.json, frozen before
any result was inspected).

Does NOT launch, resume, or modify training. Does NOT modify any frozen
source file. Checkpoint-path resolution for Original conditions is copied
verbatim from F:\\正式训练\\scripts\\evaluate_behavioral_window.py
(checkpoint_paths_for) -- not re-derived.

ENGINEERING NOTE on event granularity (documented, not a silent substitution):
a naive per-timestep-per-pair event log would produce on the order of 10-15
million rows across all 96 (seed x condition x regime) combinations, which
is neither necessary nor appropriate given section 4's own instruction that
"you may aggregate many behavioural events within each seed, but formal
comparisons must first produce one seed-level statistic" and its warning
against pseudo-replicating timestep-level events. This script therefore
aggregates the primitive opportunity/yield/priority/burden/recovery counters
ONLINE, per (seed, condition, regime, group), during simulation, and writes
only the resulting aggregated counts to wsc_behavioural_events.csv (one row
per seed x condition x regime x group x metric-family) -- the seed-level
probabilities/ratios used for inference (wsc_behavioural_seed_summary.csv)
are a deterministic function of these counts, computed in a separate
aggregation script (wsc_v2_behavioural_aggregate.py) for a clean separation
of concerns. No raw per-timestep table is persisted; it is exactly
regenerable from the frozen checkpoints + H1 bank + this script.

M_i / M_j for BOTH regimes are read from the wrapper's own internal
`_traces` dict via `running_active_attainment()` (utility.py, unchanged) --
this dict is populated unconditionally by `_append_pre_step_trace_sample()`
regardless of `include_welfare_state`, so the SAME function computes the
SAME welfare-state value whether or not the policy itself observes it.
Original policies never receive M_i/M_j as input (include_welfare_state is
never set True for an Original agent below).

Usage:
    python wsc_v2_behavioural_run.py --seeds 900101 900102 ... --out-suffix shard0
"""
from __future__ import annotations
import os

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

FINAL_NEW = Path(os.environ.get("FINAL_NEW_BUNDLE", ""))  # raw checkpoints not distributed with this repo; set env var
REPO_ROOT = Path(os.environ.get("SEED_REPL_BUNDLE", ""))  # raw checkpoints not distributed with this repo; set env var
PROJECT_ROOT = REPO_ROOT / "project"
SB_SCRIPTS = PROJECT_ROOT / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SB_SCRIPTS))

from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig  # noqa: E402
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig  # noqa: E402
from thesis.study_b.local_observation import LOCAL_OBS_DIM, LOCAL_OBS_DIM_WSC  # noqa: E402
from thesis.study_b.q_ensemble import ensemble_window_for_stage_end, load_ensemble_agents, select_ensemble_actions  # noqa: E402
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402
from thesis.study_b.utility import running_active_attainment  # noqa: E402
from thesis.pilots.stage11_welfare import target_speed_attainment  # noqa: E402

# ---------------------------------------------------------------- frozen constants (see metric definitions JSON)
SEEDS_ORIG = [900101, 900102, 900103, 900104, 910101, 910102]
SEEDS_NEW = [920101, 920102, 920103, 920104, 920105, 920106]
SEEDS_12 = SEEDS_ORIG + SEEDS_NEW
CONDITIONS = ["baseline", "mean", "ggi", "maximin"]
REGIMES = ["original", "wsc"]
STAGE_END = 2_000_000
WINDOW = ensemble_window_for_stage_end(STAGE_END)
BANK_PATH = FINAL_NEW / "scenario_banks" / "H1.json"

X_CONVERGE_START = 220.0
X_MERGE_END = 380.0
R_OBS = 50.0
HARD_BRAKE_THRESH = -3.0
DECELERATE = 2
DT = 0.2
VIDS = ["V0", "V1", "V2", "V3"]
RECOVERY_HORIZONS = [10, 25, 50]
TIE_TOL = 1e-9

WSC_ROOT = REPO_ROOT / "checkpoints" / "wsc_formal_runs_v2"

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Checkpoint path resolution
# ============================================================================
def original_checkpoint_paths(condition: str, seed: int) -> dict[int, Path]:
    """Copied verbatim from F:\\正式训练\\scripts\\evaluate_behavioral_window.py."""
    if condition == "baseline":
        d = FINAL_NEW / "checkpoints" / "taskonly_arm" / str(seed) / f"seed_{seed}_Formal_taskonly"
        return {s: d / f"ckpt_step_{s}.pt" for s in WINDOW}
    if seed in SEEDS_ORIG:
        run_id = f"{condition}_{seed}"
        d = FINAL_NEW / "checkpoints" / "formal_runs" / run_id / f"seed_{seed}_Formal_{condition}"
        return {s: d / f"ckpt_step_{s}.pt" for s in WINDOW}
    cond_dir = {"mean": "Mean", "ggi": "GGI", "maximin": "Maximin"}[condition]
    d = (REPO_ROOT / "checkpoints" / "seed_replication_v1" / "welfare"
         / str(seed) / cond_dir / f"seed_{seed}_Formal_{condition}")
    return {s: d / f"ckpt_step_{s}.pt" for s in WINDOW}


def wsc_checkpoint_paths(condition: str, seed: int) -> dict[int, Path]:
    stage = f"Formal_{condition}_WSC_v2"
    d = WSC_ROOT / f"{condition}_wsc_{seed}" / f"seed_{seed}_{stage}"
    return {s: d / f"ckpt_step_{s}.pt" for s in WINDOW}


def stage_name_for(condition: str, regime: str) -> str:
    if regime == "original":
        return "Formal_taskonly" if condition == "baseline" else f"Formal_{condition}"
    return f"Formal_{condition}_WSC_v2"


# ============================================================================
# Counters -- nested defaultdicts keyed by (seed, condition, regime, group)
# group == "ALL" for the primary (non-stratified) metric, else a
# role-speed_class label for the group-analysis breakdown.
# ============================================================================
def new_counter_tree():
    return defaultdict(lambda: defaultdict(float))


def group_label(role: str, speed_class: str) -> str:
    return f"{role}-{speed_class}"


class Counters:
    def __init__(self):
        # yielding (keyed by group of vehicle i = the potential yielder)
        self.opp_worse = new_counter_tree()
        self.opp_better = new_counter_tree()
        self.yield_worse = new_counter_tree()
        self.yield_better = new_counter_tree()
        # yielding RECEIVED (keyed by group of vehicle j = the recipient), only when i yielded
        self.recv_opp_worse = new_counter_tree()   # j was worse-off and had an opportunity (i.e. i could have yielded to it)
        self.recv_yield_worse = new_counter_tree()  # j was worse-off and received a yield
        # merge priority (keyed by group of the currently-worse-off vehicle in the pair)
        self.priority_pairs = new_counter_tree()
        self.priority_to_worse_off = new_counter_tree()
        # burden transfer / costly cooperative action (keyed by group of vehicle i)
        self.burden_opp_worse = new_counter_tree()
        self.burden_opp_better = new_counter_tree()
        self.burden_event_worse = new_counter_tree()
        self.burden_event_better = new_counter_tree()
        # worst-off recovery (keyed by group of the worst-off vehicle w)
        self.recovery_n = {k: new_counter_tree() for k in RECOVERY_HORIZONS}
        self.recovery_sum = {k: new_counter_tree() for k in RECOVERY_HORIZONS}
        self.gapclosure_sum = {k: new_counter_tree() for k in RECOVERY_HORIZONS}
        # worst-off persistence (how often each group is the worst-off vehicle, sampled)
        self.worst_off_samples = new_counter_tree()
        # episode-level bookkeeping
        self.n_episodes = 0
        self.n_completion = 0
        self.n_collision = 0


_VALIDATED_INCREMENTAL_MI = [False]  # module-level one-shot correctness check flag


def run_combo(condition: str, seed: int, regime: str, counters: Counters, scenarios) -> None:
    include_wsc = regime == "wsc"
    obs_dim = LOCAL_OBS_DIM_WSC if include_wsc else LOCAL_OBS_DIM
    ckpt_paths = wsc_checkpoint_paths(condition, seed) if include_wsc else original_checkpoint_paths(condition, seed)
    for s, p in ckpt_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"missing checkpoint: regime={regime} cond={condition} seed={seed} step={s}: {p}")
    agents = load_ensemble_agents(
        seed=seed, checkpoint_paths=ckpt_paths, expected_steps=WINDOW,
        expected_stage_by_step=dict.fromkeys(WINDOW, stage_name_for(condition, regime)),
        obs_dim=obs_dim,
    )
    env_config = ThesisHighwayMergeEnvConfig(episode_max_steps=200, action_representation="meta_speed")
    env = StudyBHeterogeneousHighwayEnv(
        StudyBHighwayWrapperConfig(env_config=env_config, local_sensing_range_m=R_OBS, include_welfare_state=include_wsc)
    )

    key = (seed, condition, regime)

    for ep_idx, scenario in enumerate(scenarios):
        obs, _info = env.reset(seed=0, scenario=scenario)
        pre_active = {v: True for v in VIDS}
        groups = {v: group_label(scenario.vehicles[v].role, scenario.vehicles[v].speed_class) for v in VIDS}
        target_speeds = {v: scenario.vehicles[v].target_speed for v in VIDS}
        # incremental M_i(t) accumulator -- O(1)/step instead of O(T) via
        # running_active_attainment(trace) recomputing the whole history each
        # call. Mathematically identical (mean of active target_speed_attainment
        # samples so far, neutral=1.0 if none yet) -- see one-shot validation below.
        running_sum = {v: 0.0 for v in VIDS}
        running_n = {v: 0 for v in VIDS}
        do_validate = not _VALIDATED_INCREMENTAL_MI[0]

        exit_step: dict[str, int | None] = {v: None for v in VIDS}
        hb_in_run: dict[str, bool] = {v: False for v in VIDS}
        # pair -> welfare state of each member the FIRST time the pair became an opportunity
        pair_first_state: dict[tuple[str, str], tuple[float, float, int]] = {}
        # buffered worst-off samples: list of (t, w, Mw, active_others_M) to later compute recovery at t+k
        worst_off_log: list[tuple[int, str, float, dict[str, float]]] = []
        # per-step active-flag history for recovery validity checks
        active_history: list[dict[str, bool]] = []
        # per-step M history for recovery computation
        m_history: list[dict[str, float]] = []
        x_history: list[dict[str, float]] = []

        term_reason = "truncation"
        for t in range(200):
            actions = select_ensemble_actions(agents, obs)

            # --- pre-step introspection (decision-time state) ---
            xs, Ms, active_now = {}, {}, {}
            for v in VIDS:
                if not pre_active[v]:
                    continue
                vehicle = env._env._vehicle_by_id[v]  # noqa: SLF001
                x, _y = env._env.world_xy(vehicle)  # noqa: SLF001
                xs[v] = x
                Ms[v] = (running_sum[v] / running_n[v]) if running_n[v] > 0 else 1.0
                if do_validate:
                    ref = float(running_active_attainment(env._traces[v]))  # noqa: SLF001
                    if abs(ref - Ms[v]) > 1e-9:
                        raise AssertionError(
                            f"incremental M_i mismatch vs running_active_attainment: "
                            f"incremental={Ms[v]!r} reference={ref!r} vehicle={v} step={t}"
                        )
                active_now[v] = True
                # append THIS step's sample, matching _append_pre_step_trace_sample's
                # placement (before physics) so the NEXT step's Ms reflects it, exactly
                # as running_active_attainment(trace) would after that same append.
                running_sum[v] += target_speed_attainment(float(vehicle.speed), target_speeds[v])
                running_n[v] += 1

            active_history.append(dict(active_now))
            m_history.append(dict(Ms))
            x_history.append(dict(xs))

            # yielding opportunities / actions (directed pairs i -> j)
            for i in xs:
                if xs[i] >= X_MERGE_END:
                    continue
                gi = groups[i]
                for j in xs:
                    if j == i or j not in xs:
                        continue
                    if abs(xs[i] - xs[j]) > R_OBS:
                        continue
                    worse_off_j = Ms[j] < Ms[i]
                    yielded = int(actions[i]) == DECELERATE
                    if worse_off_j:
                        counters.opp_worse[key][gi] += 1
                        counters.recv_opp_worse[key][groups[j]] += 1
                        if yielded:
                            counters.yield_worse[key][gi] += 1
                            counters.recv_yield_worse[key][groups[j]] += 1
                        # burden/costly-action opportunity ledger shares the same opportunity definition
                        counters.burden_opp_worse[key][gi] += 1
                    else:
                        counters.opp_better[key][gi] += 1
                        if yielded:
                            counters.yield_better[key][gi] += 1
                        counters.burden_opp_better[key][gi] += 1
                    # merge-priority pair bookkeeping: record first co-occurrence state only
                    pair = (i, j) if i < j else (j, i)
                    if pair not in pair_first_state:
                        pair_first_state[pair] = (Ms.get(pair[0], float("nan")), Ms.get(pair[1], float("nan")), t)

            # hard-brake (costly action) event starts, classified by opportunity state at this step
            for v in xs:
                vehicle = env._env._vehicle_by_id[v]  # noqa: SLF001
                accel = float(vehicle.action["acceleration"])
                is_hb = accel <= HARD_BRAKE_THRESH
                has_worse_opp = any(
                    (w != v and w in xs and abs(xs[v] - xs[w]) <= R_OBS and Ms[w] < Ms[v]) for w in xs
                )
                has_better_only_opp = (not has_worse_opp) and any(
                    (w != v and w in xs and abs(xs[v] - xs[w]) <= R_OBS) for w in xs
                )
                if is_hb and not hb_in_run[v]:
                    if has_worse_opp:
                        counters.burden_event_worse[key][groups[v]] += 1
                    elif has_better_only_opp:
                        counters.burden_event_better[key][groups[v]] += 1
                hb_in_run[v] = is_hb

            # worst-off identity (sampled at every step where >=2 active vehicles and
            # at least one is inside the merge-relevant window, per definitions JSON)
            if len(xs) >= 2 and any(X_CONVERGE_START <= x < X_MERGE_END for x in xs.values()):
                m_min = min(Ms.values())
                tied = [v for v in Ms if abs(Ms[v] - m_min) < TIE_TOL]
                if len(tied) < len(Ms):  # exclude degenerate all-tied
                    wgt = 1.0 / len(tied)
                    for w in tied:
                        counters.worst_off_samples[key][groups[w]] += wgt
                    # log first (lowest-index) tied vehicle as the representative w for recovery tracking
                    w_rep = sorted(tied)[0]
                    worst_off_log.append((t, w_rep, Ms[w_rep], {v: m for v, m in Ms.items() if v != w_rep}))

            obs, _reward, terminated, truncated, step_info = env.step(actions)

            for v in VIDS:
                if not pre_active[v]:
                    continue
                vehicle = env._env._vehicle_by_id[v]  # noqa: SLF001
                x, _y = env._env.world_xy(vehicle)  # noqa: SLF001
                if exit_step[v] is None and x >= X_MERGE_END:
                    exit_step[v] = t

            if terminated:
                term_reason = "collision" if step_info["collision_event"] else "success"
            elif truncated:
                term_reason = "truncation"
            pre_active = dict(step_info["active"])
            if terminated or truncated:
                break

        if do_validate:
            _VALIDATED_INCREMENTAL_MI[0] = True
            print("  [validation] incremental M_i matched running_active_attainment() exactly for episode 0", flush=True)

        counters.n_episodes += 1
        counters.n_completion += int(term_reason == "success")
        counters.n_collision += int(term_reason == "collision")

        # merge-priority resolution using logged first-co-occurrence state + exit steps
        for (a, b), (Ma0, Mb0, _t0) in pair_first_state.items():
            if exit_step[a] is None or exit_step[b] is None:
                continue  # need both to resolve who passed first
            if Ma0 != Ma0 or Mb0 != Mb0:  # NaN guard
                continue
            worse = a if Ma0 < Mb0 else (b if Mb0 < Ma0 else None)
            if worse is None:
                continue  # exact tie at first encounter -- not informative for this metric
            g_worse = groups[worse]
            counters.priority_pairs[key][g_worse] += 1
            worse_exits_first = exit_step[worse] < exit_step[a if worse == b else b]
            if worse_exits_first:
                counters.priority_to_worse_off[key][g_worse] += 1

        # worst-off recovery: for each sampled (t, w, Mw, others) look ahead k steps
        for (t0, w, Mw0, others0) in worst_off_log:
            for k in RECOVERY_HORIZONS:
                tk = t0 + k
                if tk >= len(m_history):
                    continue
                if not active_history[tk].get(w, False):
                    continue
                Mw_k = m_history[tk].get(w)
                if Mw_k is None:
                    continue
                others_k = {v: m for v, m in m_history[tk].items() if v != w}
                if not others0 or not others_k:
                    continue
                mean_other_0 = sum(others0.values()) / len(others0)
                mean_other_k = sum(others_k.values()) / len(others_k)
                gapclosure = (mean_other_0 - Mw0) - (mean_other_k - Mw_k)
                g_w = groups[w]
                counters.recovery_n[k][key][g_w] += 1
                counters.recovery_sum[k][key][g_w] += (Mw_k - Mw0)
                counters.gapclosure_sum[k][key][g_w] += gapclosure


# ============================================================================
# Serialization: flatten Counters -> wsc_behavioural_events.csv rows
# ============================================================================
def counters_to_rows(counters: Counters, seeds_done: list[tuple[int, str, str]]) -> list[dict]:
    rows = []
    for (seed, condition, regime) in seeds_done:
        key = (seed, condition, regime)
        groups_seen = set()
        for d in (counters.opp_worse, counters.opp_better, counters.recv_opp_worse,
                  counters.priority_pairs, counters.burden_opp_worse, counters.worst_off_samples):
            groups_seen |= set(d[key].keys())
        for k in RECOVERY_HORIZONS:
            groups_seen |= set(counters.recovery_n[k][key].keys())
        groups_seen.add("ALL")

        def total(d):
            return sum(d[key].values())

        for g in sorted(groups_seen):
            def gv(d):
                return d[key].get(g, 0.0) if g != "ALL" else total(d)

            row = {
                "seed": seed, "condition": condition, "regime": regime, "group": g,
                "opp_worse": gv(counters.opp_worse), "opp_better": gv(counters.opp_better),
                "yield_worse": gv(counters.yield_worse), "yield_better": gv(counters.yield_better),
                "recv_opp_worse": gv(counters.recv_opp_worse), "recv_yield_worse": gv(counters.recv_yield_worse),
                "priority_pairs": gv(counters.priority_pairs), "priority_to_worse_off": gv(counters.priority_to_worse_off),
                "burden_opp_worse": gv(counters.burden_opp_worse), "burden_opp_better": gv(counters.burden_opp_better),
                "burden_event_worse": gv(counters.burden_event_worse), "burden_event_better": gv(counters.burden_event_better),
                "worst_off_samples": gv(counters.worst_off_samples),
            }
            for k in RECOVERY_HORIZONS:
                row[f"recovery_n_k{k}"] = gv(counters.recovery_n[k])
                row[f"recovery_sum_k{k}"] = gv(counters.recovery_sum[k])
                row[f"gapclosure_sum_k{k}"] = gv(counters.gapclosure_sum[k])
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out-suffix", type=str, required=True)
    args = ap.parse_args()

    for s in args.seeds:
        if s not in SEEDS_12:
            raise SystemExit(f"seed {s} is not in the frozen 12-seed set")

    scenarios = load_scenario_bank(BANK_PATH)
    assert len(scenarios) == 256, f"expected 256 H1 scenarios, got {len(scenarios)}"

    counters = Counters()
    combos_done: list[tuple[int, str, str]] = []
    episode_summary_rows = []

    for seed in args.seeds:
        for condition in CONDITIONS:
            for regime in REGIMES:
                n_before = counters.n_episodes
                comp_before, coll_before = counters.n_completion, counters.n_collision
                print(f"[behavioural_run] seed={seed} condition={condition} regime={regime} ...", flush=True)
                run_combo(condition, seed, regime, counters, scenarios)
                combos_done.append((seed, condition, regime))
                n_ep = counters.n_episodes - n_before
                comp = counters.n_completion - comp_before
                coll = counters.n_collision - coll_before
                print(f"  n_episodes={n_ep} completion_rate={comp/n_ep:.3f} collision_rate={coll/n_ep:.3f}", flush=True)
                episode_summary_rows.append({
                    "seed": seed, "condition": condition, "regime": regime,
                    "n_episodes": n_ep, "n_completion": comp, "n_collision": coll,
                })

    rows = counters_to_rows(counters, combos_done)
    out_csv = OUT_DIR / f"wsc_behavioural_events_{args.out_suffix}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv} ({len(rows)} rows)")

    ep_csv = OUT_DIR / f"wsc_behavioural_episode_counts_{args.out_suffix}.csv"
    with open(ep_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(episode_summary_rows[0].keys()))
        w.writeheader()
        w.writerows(episode_summary_rows)
    print(f"wrote {ep_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
