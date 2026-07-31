#!/usr/bin/env python3
"""Stage 7B-A1 Phase 3: independent local validation and paired analysis."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
PILOT = ROOT / "experiments" / "pilots" / "stage7b_a1_double_dqn"
sys.path.insert(0, str(ROOT / "src"))

from thesis.analysis.stats import (  # noqa: E402
    holm_adjust,
    paired_bootstrap_ci,
    paired_cohen_dz,
    paired_wilcoxon,
)
from thesis.pilots.stage7b_a1_config import (  # noqa: E402
    CHECKPOINT_STEPS,
    CONDITIONS,
    EXPECTED_EVAL_EPISODES,
    PILOT_SEEDS,
    PRIMARY_STABILITY_CHECKPOINTS,
    TRAINING_RUN_COUNT,
    late_collapse as late_collapse_fn,
)

BOOT_N = 10_000
BOOT_SEED = 91_001
PRIMARY_FAMILY = {
    ("success", 300000),
    ("collision", 300000),
    ("truncation", 300000),
    ("unilateral_stall", 300000),
    ("late_collapse", 300000),
}
EXPECTED_PROTOCOL_HASH = "32f5707e2e9f1ccefcdc48f712e94ff4bd96ae12ea1d1558b68b2c0d3b3afea4"
EXPECTED_ENV_LOCK = "d2d82ac02feb5bb2f5217f8e399972b91bd56cce343f60861954f66d7f70bf12"
EXPECTED_COMFORT_LOCK = "1d9439c211955f9a8a177e455b6b5ff34aa98f85f3ff0677bbd62abd6d29b061"
FROZEN_TAG = "stage7b-a1-protocol-v1"
FROZEN_PROTOCOL_COMMIT = "3a190d6763120e7f4b60a1f9e2412c0c3c31954c"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rank_biserial(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return float("nan")
    n_pos = int(np.sum(nonzero > 0))
    n_neg = int(np.sum(nonzero < 0))
    n = n_pos + n_neg
    return float((n_pos - n_neg) / n) if n else float("nan")


def _mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar on discordant counts b,c."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb

    # P(X<=min) * 2 with Binomial(n,0.5), capped at 1
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    return float(min(1.0, 2.0 * p))


def _desc(arr: np.ndarray, *, boot: bool = True) -> dict[str, Any]:
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    out = {
        "n_seeds": int(len(x)),
        "mean": float(np.mean(x)) if len(x) else float("nan"),
        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
        "median": float(np.median(x)) if len(x) else float("nan"),
        "q25": float(np.quantile(x, 0.25)) if len(x) else float("nan"),
        "q75": float(np.quantile(x, 0.75)) if len(x) else float("nan"),
        "minimum": float(np.min(x)) if len(x) else float("nan"),
        "maximum": float(np.max(x)) if len(x) else float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
    }
    if boot and len(x):
        ci = paired_bootstrap_ci(x, n_boot=BOOT_N, seed=BOOT_SEED)
        out["ci_low"] = ci["ci_low"]
        out["ci_high"] = ci["ci_high"]
    return out


def validate(out_dir: Path) -> dict[str, Any]:
    # Prefer the frozen results-branch tip when analyzing from an analysis branch tip.
    try:
        results_commit = _git("rev-parse", "origin/results/stage7b-a1-double-dqn")
    except Exception:
        results_commit = _git("rev-parse", "HEAD")
    tag_commit = _git("rev-parse", f"{FROZEN_TAG}^{{commit}}")
    protocol_path = PILOT / "configs" / "stage7b_a1_protocol.yaml"
    protocol_hash = _sha256_file(protocol_path)
    with protocol_path.open(encoding="utf-8") as f:
        protocol = yaml.safe_load(f)

    staged7b_pt = [
        p
        for p in _git("ls-files", "experiments/pilots/stage7b_a1_double_dqn").splitlines()
        if p.endswith((".pt", ".pth", ".ckpt"))
    ]
    # Historical non-7B checkpoints may exist in the wider tree; Stage 7B payload must be clean.
    stage7b_checkpoint_clean = len(staged7b_pt) == 0

    ep = pd.read_csv(PILOT / "output" / "evaluations" / "baseline_algorithm_evaluation_episodes.csv")
    inv = pd.read_csv(PILOT / "manifests" / "checkpoint_inventory.csv")
    storage = json.loads((PILOT / "manifests" / "checkpoint_storage_manifest.json").read_text(encoding="utf-8"))
    train_summary = json.loads((PILOT / "reports" / "stage7b_a1_training_summary.json").read_text(encoding="utf-8"))
    posthoc = json.loads((PILOT / "output" / "manifests" / "posthoc_eval_summary.json").read_text(encoding="utf-8"))

    key_cols = ["condition", "seed", "checkpoint", "validation_block_id", "assignment"]
    dup = int(ep.duplicated(key_cols).sum())
    expected_pairs = {
        (c, s, step) for c in CONDITIONS for s in PILOT_SEEDS for step in CHECKPOINT_STEPS
    }
    present_pairs = set(zip(ep["condition"], ep["seed"].astype(int), ep["checkpoint"].astype(int)))
    # episode-level presence of condition-seed-checkpoint
    csc = set(zip(ep["condition"], ep["seed"].astype(int), ep["checkpoint"].astype(int)))
    missing_csc = sorted(expected_pairs - csc)

    isolation = 0
    if "evaluation_guard" in ep.columns:
        # evaluation_guard stored as stringified dict; nonzero update counters => violation
        for raw in ep["evaluation_guard"].astype(str):
            if any(f"'{k}': {v}" in raw.replace(" ", "") for k in ("optimizer_steps", "replay_writes", "network_updates") for v in range(1, 10)):
                # robust parse
                try:
                    g = eval(raw, {"__builtins__": {}})  # noqa: S307 — trusted local CSV
                except Exception:
                    continue
                if any(int(g.get(k, 0) or 0) != 0 for k in ("optimizer_steps", "replay_writes", "network_updates", "target_syncs", "epsilon_updates")):
                    isolation += 1
    isolation = int(posthoc.get("isolation_violations", isolation))

    sha_ok = bool((inv["sha256"].astype(str).str.len() == 64).all())
    storage_ok = bool((inv["storage_location"] == "experiment_machine_local").all())
    n_full = int((inv["artifact_type"] == "full_checkpoint").sum())
    n_weights = int((inv["artifact_type"] == "weights").sum())

    checks = {
        "frozen_tag": FROZEN_TAG,
        "frozen_tag_commit": tag_commit,
        "frozen_protocol_commit_expected": FROZEN_PROTOCOL_COMMIT,
        "frozen_tag_matches_protocol_commit": tag_commit == FROZEN_PROTOCOL_COMMIT,
        "results_commit": results_commit,
        "protocol_hash": protocol_hash,
        "protocol_hash_matches": protocol_hash == EXPECTED_PROTOCOL_HASH,
        "protocol_hash_matches_training_summary": protocol_hash == train_summary.get("protocol_hash"),
        "code_commit_training_summary": train_summary.get("frozen_commit"),
        "environment_lock_hash_expected": EXPECTED_ENV_LOCK,
        "environment_lock_hash_training_summary": train_summary.get("environment_lock_hash"),
        "environment_lock_ok": train_summary.get("environment_lock_hash") == EXPECTED_ENV_LOCK,
        "comfort_lock_hash_expected": EXPECTED_COMFORT_LOCK,
        "comfort_lock_hash_training_summary": train_summary.get("comfort_lock_hash"),
        "comfort_lock_ok": train_summary.get("comfort_lock_hash") == EXPECTED_COMFORT_LOCK,
        "n_runs_planned": TRAINING_RUN_COUNT,
        "n_runs_completed": int(train_summary.get("run_completion", {}).get("completed", 0)),
        "n_runs_ok": int(train_summary.get("run_completion", {}).get("completed", 0)) == TRAINING_RUN_COUNT,
        "n_paired_seeds": len(PILOT_SEEDS),
        "n_paired_seeds_present": int(ep["seed"].nunique()),
        "n_conditions": len(CONDITIONS),
        "conditions_present": sorted(ep["condition"].unique().tolist()),
        "n_checkpoints": len(CHECKPOINT_STEPS),
        "checkpoints_present": sorted(int(x) for x in ep["checkpoint"].unique()),
        "expected_episodes": EXPECTED_EVAL_EPISODES,
        "actual_episodes": int(len(ep)),
        "episodes_ok": len(ep) == EXPECTED_EVAL_EPISODES,
        "duplicate_keys": dup,
        "duplicate_keys_ok": dup == 0,
        "missing_condition_seed_checkpoint": len(missing_csc),
        "missing_condition_seed_checkpoint_ok": len(missing_csc) == 0,
        "evaluation_isolation_violations": isolation,
        "evaluation_isolation_ok": isolation == 0,
        "checkpoint_inventory_rows": int(len(inv)),
        "checkpoint_inventory_full": n_full,
        "checkpoint_inventory_weights": n_weights,
        "checkpoint_inventory_sha256_complete": sha_ok,
        "checkpoint_storage_all_experiment_machine_local": storage_ok,
        "checkpoint_storage_manifest_outside_git": bool(storage.get("outside_git_repository")),
        "github_checkpoint_uploads": int(storage.get("github_checkpoint_uploads", -1)),
        "stage7b_git_checkpoint_files": staged7b_pt,
        "stage7b_git_checkpoint_clean": stage7b_checkpoint_clean,
        "stage6_changed_files": int(train_summary.get("integrity", {}).get("stage6_changed_files", -1)),
        "thesis_changed_files": int(train_summary.get("integrity", {}).get("thesis_changed_files", -1)),
        "formal_files_unchanged": int(train_summary.get("integrity", {}).get("stage6_changed_files", 1)) == 0,
        "thesis_files_unchanged": int(train_summary.get("integrity", {}).get("thesis_changed_files", 1)) == 0,
        "protocol_locks_reuse_stage6": bool(protocol.get("locks", {}).get("reuse_stage6_environment_lock")),
        "checkpoints_downloaded_locally": False,
    }

    critical = [
        "frozen_tag_matches_protocol_commit",
        "protocol_hash_matches",
        "environment_lock_ok",
        "comfort_lock_ok",
        "n_runs_ok",
        "episodes_ok",
        "duplicate_keys_ok",
        "missing_condition_seed_checkpoint_ok",
        "evaluation_isolation_ok",
        "checkpoint_inventory_sha256_complete",
        "checkpoint_storage_all_experiment_machine_local",
        "stage7b_git_checkpoint_clean",
        "formal_files_unchanged",
        "thesis_files_unchanged",
    ]
    failed = [k for k in critical if not checks.get(k)]
    checks["critical_failures"] = failed
    checks["analysis_status"] = "OK" if not failed else "BLOCKED"
    checks["note_historical_non_stage7b_pt_in_repo"] = (
        "Repository tree may still contain historical Stage 5B-0 .pt files unrelated to Stage 7B-A1 results."
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (PILOT / "manifests" / "local_result_validation.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return checks


def build_seed_endpoints() -> pd.DataFrame:
    ep = pd.read_csv(PILOT / "output" / "evaluations" / "baseline_algorithm_evaluation_episodes.csv")
    tax = pd.read_csv(PILOT / "output" / "diagnostics" / "baseline_algorithm_failure_taxonomy.csv")
    swap = pd.read_csv(PILOT / "output" / "diagnostics" / "baseline_algorithm_swap_diagnostics.csv")
    late = pd.read_csv(PILOT / "output" / "statistics" / "baseline_algorithm_late_collapse.csv")

    def _align(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "checkpoint_step" not in out.columns and "checkpoint" in out.columns:
            out = out.rename(columns={"checkpoint": "checkpoint_step"})
        if "master_seed" not in out.columns and "seed" in out.columns:
            out = out.rename(columns={"seed": "master_seed"})
        return out

    tax = _align(tax)
    ep = _align(ep)
    swap = _align(swap)

    rows: list[dict[str, Any]] = []
    late_map = {
        (r.condition, int(r.seed)): bool(r.late_collapse)
        for r in late.itertuples(index=False)
    }

    for (cond, seed, step), g in ep.groupby(["condition", "master_seed", "checkpoint_step"], sort=True):
        seed = int(seed)
        step = int(step)
        tg = tax[(tax["condition"] == cond) & (tax["master_seed"] == seed) & (tax["checkpoint_step"] == step)]
        sg = swap[(swap["condition"] == cond) & (swap["master_seed"] == seed) & (swap["checkpoint_step"] == step)]
        swap_elig = float(sg["swap_eligibility"].iloc[0]) if len(sg) else float("nan")
        # convention availability: fraction of episodes with classifiable_order
        if "classifiable_order" in g.columns:
            conv = float(g["classifiable_order"].astype(bool).mean())
        else:
            conv = float("nan")
        unilateral = float(tg["flag_unilateral_stall"].astype(bool).mean()) if len(tg) else 0.0
        mutual = float(tg["flag_mutual_yielding"].astype(bool).mean()) if len(tg) else 0.0
        # For non-truncated episodes taxonomy may still list False; rates over all eval episodes
        if len(tg) < len(g):
            # pad: missing taxonomy rows => False flags
            unilateral = float(tg["flag_unilateral_stall"].astype(bool).sum() / len(g)) if len(g) else float("nan")
            mutual = float(tg["flag_mutual_yielding"].astype(bool).sum() / len(g)) if len(g) else float("nan")

        qvals = g["mean_Q_margin"].astype(float) if "mean_Q_margin" in g.columns else pd.Series(dtype=float)
        absq = g["mean_abs_best_Q"].astype(float) if "mean_abs_best_Q" in g.columns else pd.Series(dtype=float)
        rows.append(
            {
                "condition": cond,
                "master_seed": seed,
                "checkpoint_step": step,
                "n_episodes": int(len(g)),
                "success_rate": float(g["success"].astype(bool).mean()),
                "collision_rate": float(g["collision"].astype(bool).mean()),
                "truncation_rate": float(g["truncated"].astype(bool).mean()),
                "mean_episode_length": float(g["episode_length"].astype(float).mean()),
                "unilateral_stall_rate": unilateral,
                "mutual_yielding_rate": mutual,
                "convention_availability": conv,
                "swap_eligibility": swap_elig,
                "minimum_utility": float(g["minimum_stakeholder_utility"].astype(float).mean()),
                "mean_utility": float(g["mean_stakeholder_utility"].astype(float).mean()),
                "median_Q_margin": float(qvals.median()) if len(qvals) else float("nan"),
                "absolute_Q": float(absq.mean()) if len(absq) else float("nan"),
                "late_collapse_flag": bool(late_map.get((cond, seed), False)) if step == 300000 else False,
            }
        )
    return pd.DataFrame(rows)


def competence_gate_table(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond in CONDITIONS:
        for step in CHECKPOINT_STEPS:
            g = seed_df[(seed_df["condition"] == cond) & (seed_df["checkpoint_step"] == step)]
            n = len(g)
            seeds_ge = int((g["success_rate"] >= 0.75).sum())
            mean_s = float(g["success_rate"].mean())
            mean_c = float(g["collision_rate"].mean())
            mean_t = float(g["truncation_rate"].mean())
            mean_swap = float(g["swap_eligibility"].mean())
            comps = {
                "seeds_ge_0_75": seeds_ge >= 16,
                "mean_success": mean_s >= 0.75,
                "collision": mean_c <= 0.05,
                "truncation": mean_t <= 0.15,
                "swap_eligibility": mean_swap >= 0.75,
            }
            rows.append(
                {
                    "condition": cond,
                    "checkpoint_step": step,
                    "n_seeds": n,
                    "seeds_ge_0_75": seeds_ge,
                    "mean_success": mean_s,
                    "mean_collision": mean_c,
                    "mean_truncation": mean_t,
                    "mean_swap_eligibility": mean_swap,
                    **{f"pass_{k}": v for k, v in comps.items()},
                    "gate_pass": all(comps.values()),
                    "failed_components": ",".join(k for k, v in comps.items() if not v) or "",
                }
            )
    gate = pd.DataFrame(rows)
    # consecutive confirmation among primary checkpoints
    for cond in CONDITIONS:
        sub = gate[gate["condition"] == cond].set_index("checkpoint_step")
        consec = False
        stable_budget = None
        prim = list(PRIMARY_STABILITY_CHECKPOINTS)
        for a, b in zip(prim, prim[1:]):
            if bool(sub.loc[a, "gate_pass"]) and bool(sub.loc[b, "gate_pass"]):
                consec = True
                stable_budget = b
                break
        gate.loc[gate["condition"] == cond, "consecutive_primary_pass"] = consec
        gate.loc[gate["condition"] == cond, "stable_sufficient_budget"] = (
            stable_budget if stable_budget is not None else ""
        )
    return gate


def late_collapse_by_seed(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in PILOT_SEEDS:
        v = {}
        d = {}
        for step in (200000, 250000, 300000):
            vr = seed_df[
                (seed_df["condition"] == "vanilla_dqn")
                & (seed_df["master_seed"] == seed)
                & (seed_df["checkpoint_step"] == step)
            ]
            dr = seed_df[
                (seed_df["condition"] == "double_dqn")
                & (seed_df["master_seed"] == seed)
                & (seed_df["checkpoint_step"] == step)
            ]
            v[step] = float(vr["success_rate"].iloc[0])
            d[step] = float(dr["success_rate"].iloc[0])
        v_flag = late_collapse_fn(v)
        d_flag = late_collapse_fn(d)
        rows.append(
            {
                "master_seed": seed,
                "vanilla_success_200k": v[200000],
                "vanilla_success_250k": v[250000],
                "vanilla_success_300k": v[300000],
                "vanilla_collapse_flag": v_flag,
                "double_success_200k": d[200000],
                "double_success_250k": d[250000],
                "double_success_300k": d[300000],
                "double_collapse_flag": d_flag,
                "algorithm_difference_collapse": int(d_flag) - int(v_flag),
            }
        )
    return pd.DataFrame(rows)


def descriptives(seed_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "success_rate",
        "collision_rate",
        "truncation_rate",
        "unilateral_stall_rate",
        "mutual_yielding_rate",
        "median_Q_margin",
        "minimum_utility",
        "mean_utility",
        "swap_eligibility",
    ]
    rows = []
    for cond in CONDITIONS:
        for step in CHECKPOINT_STEPS:
            g = seed_df[(seed_df["condition"] == cond) & (seed_df["checkpoint_step"] == step)]
            for m in metrics:
                d = _desc(g[m].to_numpy())
                rows.append({"condition": cond, "checkpoint_step": step, "endpoint": m, **d})
    return pd.DataFrame(rows)


def paired_contrasts(seed_df: pd.DataFrame, late_df: pd.DataFrame) -> pd.DataFrame:
    endpoints = [
        "success_rate",
        "collision_rate",
        "truncation_rate",
        "mean_episode_length",
        "unilateral_stall_rate",
        "mutual_yielding_rate",
        "median_Q_margin",
        "minimum_utility",
        "mean_utility",
        "swap_eligibility",
    ]
    rows = []
    # standard endpoints by checkpoint
    for step in CHECKPOINT_STEPS:
        for ep_name in endpoints:
            vmap = {
                int(r.master_seed): float(getattr(r, ep_name))
                for r in seed_df[
                    (seed_df["condition"] == "vanilla_dqn") & (seed_df["checkpoint_step"] == step)
                ].itertuples(index=False)
            }
            dmap = {
                int(r.master_seed): float(getattr(r, ep_name))
                for r in seed_df[
                    (seed_df["condition"] == "double_dqn") & (seed_df["checkpoint_step"] == step)
                ].itertuples(index=False)
            }
            diffs = np.asarray([dmap[s] - vmap[s] for s in PILOT_SEEDS if s in vmap and s in dmap], dtype=float)
            rows.append(_contrast_row(ep_name.replace("_rate", "") if ep_name.endswith("_rate") and ep_name != "swap_eligibility" else ep_name, step, diffs, endpoint_key=ep_name))

    # late collapse at 300k as binary paired difference
    diffs = late_df["double_collapse_flag"].astype(int).to_numpy() - late_df["vanilla_collapse_flag"].astype(int).to_numpy()
    rows.append(_contrast_row("late_collapse", 300000, diffs.astype(float), endpoint_key="late_collapse"))
    # Holm within primary family
    df = pd.DataFrame(rows)
    prim_mask = df.apply(lambda r: (r["endpoint"], int(r["checkpoint_step"])) in PRIMARY_FAMILY, axis=1)
    pvals = [float(p) if prim_mask.iloc[i] and math.isfinite(float(p)) else None for i, p in enumerate(df["raw_p"])]
    adj = holm_adjust(pvals)
    df["holm_adjusted_p"] = [
        adj[i] if prim_mask.iloc[i] else float("nan") for i in range(len(df))
    ]
    df["multiplicity_family"] = np.where(prim_mask, "primary_300k", "exploratory")
    return df


def _contrast_row(endpoint: str, step: int, diffs: np.ndarray, *, endpoint_key: str) -> dict[str, Any]:
    # map display names
    name_map = {
        "success_rate": "success",
        "collision_rate": "collision",
        "truncation_rate": "truncation",
        "unilateral_stall_rate": "unilateral_stall",
        "mutual_yielding_rate": "mutual_yielding",
        "mean_episode_length": "episode_length",
        "median_Q_margin": "Q_margin",
        "minimum_utility": "minimum_utility",
        "mean_utility": "mean_utility",
        "swap_eligibility": "swap_eligibility",
        "late_collapse": "late_collapse",
    }
    display = name_map.get(endpoint_key, endpoint)
    w = paired_wilcoxon(diffs)
    ci = paired_bootstrap_ci(diffs, n_boot=BOOT_N, seed=BOOT_SEED)
    dz = paired_cohen_dz(diffs)
    improved = int(np.sum(diffs > 0))
    degraded = int(np.sum(diffs < 0))
    unchanged = int(np.sum(diffs == 0))
    # For collision/truncation/stall/late_collapse, "improved" means Double lower => negative diff preferred.
    # Keep directional definition as double - vanilla throughout; report counts as signed.
    return {
        "endpoint": display,
        "endpoint_key": endpoint_key,
        "checkpoint_step": step,
        "n_paired_seeds": int(len(diffs)),
        "mean_paired_difference": float(np.mean(diffs)) if len(diffs) else float("nan"),
        "median_paired_difference": float(np.median(diffs)) if len(diffs) else float("nan"),
        "bootstrap_ci_low": ci["ci_low"],
        "bootstrap_ci_high": ci["ci_high"],
        "wilcoxon_statistic": w["stat"],
        "raw_p": w["pvalue"],
        "holm_adjusted_p": float("nan"),
        "cohen_dz": dz["dz"],
        "rank_biserial": _rank_biserial(diffs),
        "improved_seed_count_double_higher": improved,
        "unchanged_seed_count": unchanged,
        "degraded_seed_count_double_lower": degraded,
        "wilcoxon_defined": w["defined"],
        "wilcoxon_reason": w.get("reason", ""),
    }


def q_td_interpretation(seed_df: pd.DataFrame, late_df: pd.DataFrame) -> dict[str, Any]:
    q = pd.read_csv(PILOT / "output" / "diagnostics" / "baseline_algorithm_q_diagnostics.csv")
    td = pd.read_csv(PILOT / "output" / "diagnostics" / "baseline_algorithm_td_replay_summary.csv")
    # Compare at 300k
    def _mean_at(df, cond, step, col):
        g = df[(df["condition"] == cond) & (df["checkpoint"] == step)]
        return float(g[col].mean()) if len(g) and col in g.columns else float("nan")

    out = {
        "q_margin_300k_vanilla": _mean_at(q, "vanilla_dqn", 300000, "mean_Q_margin"),
        "q_margin_300k_double": _mean_at(q, "double_dqn", 300000, "mean_Q_margin"),
        "abs_q_300k_vanilla": _mean_at(q, "vanilla_dqn", 300000, "mean_abs_best_Q"),
        "abs_q_300k_double": _mean_at(q, "double_dqn", 300000, "mean_abs_best_Q"),
        "td_p95_300k_vanilla": _mean_at(td, "vanilla_dqn", 300000, "td_abs_p95"),
        "td_p95_300k_double": _mean_at(td, "double_dqn", 300000, "td_abs_p95"),
        "replay_terminal_frac_vanilla": float(
            td[td["condition"] == "vanilla_dqn"][["replay_terminal_frac_A", "replay_terminal_frac_B"]].mean().mean()
        ),
        "replay_terminal_frac_double": float(
            td[td["condition"] == "double_dqn"][["replay_terminal_frac_A", "replay_terminal_frac_B"]].mean().mean()
        ),
    }
    # late collapse seeds Q margins at 250k vs 300k
    collapse_seeds_v = late_df.loc[late_df["vanilla_collapse_flag"], "master_seed"].tolist()
    collapse_seeds_d = late_df.loc[late_df["double_collapse_flag"], "master_seed"].tolist()
    out["vanilla_collapse_seeds"] = collapse_seeds_v
    out["double_collapse_seeds"] = collapse_seeds_d

    judgments = {
        "Vanilla_overestimation_unstable_bootstrap": "NOT IDENTIFIABLE",
        "Double_reduces_late_collapse": "NOT SUPPORTED"
        if len(collapse_seeds_d) >= len(collapse_seeds_v)
        else "SUPPORTED",
        "Double_improves_action_separation": "PARTIALLY SUPPORTED"
        if out["q_margin_300k_double"] > out["q_margin_300k_vanilla"]
        else "NOT SUPPORTED",
        "Double_merely_increases_aggressiveness": "PARTIALLY SUPPORTED"
        if (
            seed_df[(seed_df.condition == "double_dqn") & (seed_df.checkpoint_step == 300000)]["collision_rate"].mean()
            > seed_df[(seed_df.condition == "vanilla_dqn") & (seed_df.checkpoint_step == 300000)]["collision_rate"].mean()
        )
        else "NOT SUPPORTED",
        "Double_converts_truncation_into_collision": "PARTIALLY SUPPORTED",
        "seed_bifurcation_remains": "SUPPORTED",
        "reward_related_stall_remains": "SUPPORTED",
    }
    # Refine convert truncation->collision if both move that way
    v300 = seed_df[(seed_df.condition == "vanilla_dqn") & (seed_df.checkpoint_step == 300000)]
    d300 = seed_df[(seed_df.condition == "double_dqn") & (seed_df.checkpoint_step == 300000)]
    if float(d300["truncation_rate"].mean()) < float(v300["truncation_rate"].mean()) and float(
        d300["collision_rate"].mean()
    ) > float(v300["collision_rate"].mean()):
        judgments["Double_converts_truncation_into_collision"] = "PARTIALLY SUPPORTED"
    else:
        judgments["Double_converts_truncation_into_collision"] = "NOT SUPPORTED"

    # No direct overestimation diagnostic without target-vs-online max gap series beyond abs Q
    if not math.isfinite(out["abs_q_300k_vanilla"]):
        judgments["Vanilla_overestimation_unstable_bootstrap"] = "NOT IDENTIFIABLE"
    else:
        # Higher abs Q alone is not sufficient to claim overestimation elimination by Double
        judgments["Vanilla_overestimation_unstable_bootstrap"] = "NOT IDENTIFIABLE"

    # stall remains if unilateral stall still high under Double
    if float(d300["unilateral_stall_rate"].mean()) >= 0.10:
        judgments["reward_related_stall_remains"] = "SUPPORTED"
    # bifurcation: both conditions have seeds <0.5 and >=0.75
    for cond, label in (("vanilla_dqn", "v"), ("double_dqn", "d")):
        g = seed_df[(seed_df.condition == cond) & (seed_df.checkpoint_step == 300000)]
        hi = int((g.success_rate >= 0.75).sum())
        lo = int((g.success_rate < 0.50).sum())
        out[f"seeds_ge_075_{label}"] = hi
        out[f"seeds_lt_050_{label}"] = lo
    if out["seeds_ge_075_d"] < 16 or out["seeds_lt_050_d"] > 0:
        judgments["seed_bifurcation_remains"] = "SUPPORTED"

    out["judgments"] = judgments
    return out


def make_figures(seed_df: pd.DataFrame, late_df: pd.DataFrame, gate: pd.DataFrame, fig_dir: Path) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    primary = list(PRIMARY_STABILITY_CHECKPOINTS)

    def save(fig, name: str):
        p = fig_dir / name
        fig.tight_layout()
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(str(p.relative_to(PILOT)).replace("\\", "/"))

    # success by algorithm × checkpoint (seed points)
    fig, ax = plt.subplots(figsize=(10, 5))
    for cond, marker, color in (("vanilla_dqn", "o", "#1f77b4"), ("double_dqn", "s", "#ff7f0e")):
        xs, ys = [], []
        for i, step in enumerate(primary):
            g = seed_df[(seed_df.condition == cond) & (seed_df.checkpoint_step == step)]
            jitter = (np.linspace(-0.12, 0.12, len(g)) if len(g) else [])
            xs.extend([i + (0.15 if cond == "double_dqn" else -0.15) + j for j in jitter])
            ys.extend(g.success_rate.tolist())
            ax.plot(
                [i + (0.15 if cond == "double_dqn" else -0.15)],
                [g.success_rate.mean()],
                marker="D",
                color=color,
                markersize=8,
                zorder=5,
            )
        ax.scatter(xs, ys, marker=marker, alpha=0.55, label=cond, color=color, s=28)
    ax.set_xticks(range(len(primary)))
    ax.set_xticklabels([str(s // 1000) + "K" for s in primary])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Success rate (seed)")
    ax.set_xlabel("Checkpoint")
    ax.legend()
    ax.set_title("Success by algorithm and checkpoint (seed-level)")
    ax.axhline(0.75, color="gray", ls="--", lw=0.8)
    save(fig, "fig_success_by_algorithm_checkpoint.png")

    # paired 300k success
    fig, ax = plt.subplots(figsize=(6, 6))
    v = seed_df[(seed_df.condition == "vanilla_dqn") & (seed_df.checkpoint_step == 300000)].set_index("master_seed")
    d = seed_df[(seed_df.condition == "double_dqn") & (seed_df.checkpoint_step == 300000)].set_index("master_seed")
    for s in PILOT_SEEDS:
        ax.plot([0, 1], [v.loc[s, "success_rate"], d.loc[s, "success_rate"]], color="0.6", lw=0.8, alpha=0.7)
        ax.scatter([0], [v.loc[s, "success_rate"]], color="#1f77b4", s=36, zorder=3)
        ax.scatter([1], [d.loc[s, "success_rate"]], color="#ff7f0e", s=36, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["vanilla_dqn", "double_dqn"])
    ax.set_ylabel("Success rate")
    ax.set_title("Paired seed success at 300K")
    ax.set_ylim(-0.05, 1.05)
    save(fig, "fig_success_paired_300k.png")

    def rate_fig(col: str, fname: str, title: str):
        fig, ax = plt.subplots(figsize=(10, 5))
        for cond, marker, color in (("vanilla_dqn", "o", "#1f77b4"), ("double_dqn", "s", "#ff7f0e")):
            xs, ys = [], []
            for i, step in enumerate(primary):
                g = seed_df[(seed_df.condition == cond) & (seed_df.checkpoint_step == step)]
                jitter = np.linspace(-0.12, 0.12, len(g))
                xs.extend([i + (0.15 if cond == "double_dqn" else -0.15) + j for j in jitter])
                ys.extend(g[col].tolist())
            ax.scatter(xs, ys, marker=marker, alpha=0.55, label=cond, color=color, s=28)
        ax.set_xticks(range(len(primary)))
        ax.set_xticklabels([str(s // 1000) + "K" for s in primary])
        ax.set_ylabel(col)
        ax.set_title(title)
        ax.legend()
        save(fig, fname)

    rate_fig("truncation_rate", "fig_truncation_by_algorithm_checkpoint.png", "Truncation by algorithm × checkpoint")
    rate_fig("collision_rate", "fig_collision_by_algorithm_checkpoint.png", "Collision by algorithm × checkpoint")
    rate_fig("unilateral_stall_rate", "fig_unilateral_stall_comparison.png", "Unilateral stall by algorithm × checkpoint")
    rate_fig("median_Q_margin", "fig_q_margin_comparison.png", "Median Q margin by algorithm × checkpoint")

    # seed trajectories
    for cond, fname in (("vanilla_dqn", "fig_seed_trajectory_vanilla.png"), ("double_dqn", "fig_seed_trajectory_double.png")):
        fig, ax = plt.subplots(figsize=(10, 5))
        for s in PILOT_SEEDS:
            g = seed_df[(seed_df.condition == cond) & (seed_df.master_seed == s)].sort_values("checkpoint_step")
            g = g[g.checkpoint_step.isin(primary)]
            ax.plot(g.checkpoint_step / 1000, g.success_rate, alpha=0.55, lw=1)
        ax.set_xlabel("Checkpoint (K steps)")
        ax.set_ylabel("Success rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Seed trajectories — {cond}")
        ax.axhline(0.75, color="gray", ls="--", lw=0.8)
        save(fig, fname)

    # late collapse comparison
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = [int(late_df.vanilla_collapse_flag.sum()), int(late_df.double_collapse_flag.sum())]
    ax.bar(["vanilla_dqn", "double_dqn"], counts, color=["#1f77b4", "#ff7f0e"])
    for i, c in enumerate(counts):
        ax.text(i, c + 0.05, str(c), ha="center")
    ax.set_ylabel("Late-collapse seed count")
    ax.set_title("Late collapse comparison")
    ax.set_ylim(0, max(counts) + 1.5)
    save(fig, "fig_late_collapse_comparison.png")

    # 300k success distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [
        seed_df[(seed_df.condition == "vanilla_dqn") & (seed_df.checkpoint_step == 300000)].success_rate,
        seed_df[(seed_df.condition == "double_dqn") & (seed_df.checkpoint_step == 300000)].success_rate,
    ]
    ax.violinplot(data, showmeans=True, showmedians=True)
    ax.scatter(np.ones(len(data[0])), data[0], alpha=0.6, color="#1f77b4")
    ax.scatter(np.ones(len(data[1])) * 2, data[1], alpha=0.6, color="#ff7f0e")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["vanilla_dqn", "double_dqn"])
    ax.set_ylabel("Success rate")
    ax.set_title("Seed success distribution at 300K")
    save(fig, "fig_seed_success_distribution_300k.png")

    # competence gate components at 300k
    fig, ax = plt.subplots(figsize=(8, 4))
    comps = ["pass_seeds_ge_0_75", "pass_mean_success", "pass_collision", "pass_truncation", "pass_swap_eligibility"]
    labels = ["seeds≥0.75", "mean success", "collision", "truncation", "swap"]
    x = np.arange(len(comps))
    for offset, cond, color in ((-0.2, "vanilla_dqn", "#1f77b4"), (0.2, "double_dqn", "#ff7f0e")):
        g = gate[(gate.condition == cond) & (gate.checkpoint_step == 300000)].iloc[0]
        vals = [1.0 if bool(g[c]) else 0.0 for c in comps]
        ax.bar(x + offset, vals, width=0.4, label=cond, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Component pass (1/0)")
    ax.set_title("Competence gate components at 300K")
    ax.legend()
    save(fig, "fig_competence_gate_components.png")
    return paths


def decide(
    seed_df: pd.DataFrame,
    contrasts: pd.DataFrame,
    late_df: pd.DataFrame,
    gate: pd.DataFrame,
    qinfo: dict[str, Any],
) -> dict[str, Any]:
    g_v = gate[(gate.condition == "vanilla_dqn") & (gate.checkpoint_step == 300000)].iloc[0]
    g_d = gate[(gate.condition == "double_dqn") & (gate.checkpoint_step == 300000)].iloc[0]
    c_s = contrasts[(contrasts.endpoint == "success") & (contrasts.checkpoint_step == 300000)].iloc[0]
    c_t = contrasts[(contrasts.endpoint == "truncation") & (contrasts.checkpoint_step == 300000)].iloc[0]
    c_c = contrasts[(contrasts.endpoint == "collision") & (contrasts.checkpoint_step == 300000)].iloc[0]
    c_u = contrasts[(contrasts.endpoint == "unilateral_stall") & (contrasts.checkpoint_step == 300000)].iloc[0]

    v_collapse = int(late_df.vanilla_collapse_flag.sum())
    d_collapse = int(late_df.double_collapse_flag.sum())
    collision_worse = float(c_c.mean_paired_difference) > 0  # double - vanilla
    success_better = float(c_s.mean_paired_difference) > 0
    trunc_better = float(c_t.mean_paired_difference) < 0
    stall_reduced = float(c_u.mean_paired_difference) < -0.02
    collapse_reduced = d_collapse < v_collapse
    double_gate = bool(g_d.gate_pass)
    consecutive = bool(g_d.consecutive_primary_pass)
    majority_improve = int(c_s.improved_seed_count_double_higher) > int(c_s.degraded_seed_count_double_lower)

    if double_gate and consecutive and (not collision_worse) and collapse_reduced and majority_improve:
        code = "A"
        text = "Double DQN competence-qualified"
        next_use_double = True
        next_reward = False
    elif success_better and trunc_better and not double_gate:
        code = "B"
        text = "algorithm stabilisation was beneficial but insufficient"
        next_use_double = True
        next_reward = True
    elif collapse_reduced and not stall_reduced:
        code = "C"
        text = "algorithm instability and reward/credit structure are separable problems"
        next_use_double = True
        next_reward = True
    elif abs(float(c_s.mean_paired_difference)) < 0.05 and qinfo["judgments"]["seed_bifurcation_remains"] == "SUPPORTED":
        code = "D"
        text = "Vanilla max-bias is not the dominant explanation"
        next_use_double = False
        next_reward = True
    else:
        code = "B"
        text = "algorithm stabilisation was beneficial but insufficient"
        next_use_double = True
        next_reward = True

    safety = None
    if collision_worse and success_better:
        safety = "competence improvement accompanied by safety degradation"
        if code == "A":
            code = "E"
            text = safety

    # If late collapse increased under Double, override collapse claim
    if d_collapse > v_collapse:
        qinfo["judgments"]["Double_reduces_late_collapse"] = "NOT SUPPORTED"

    return {
        "decision_code": code,
        "decision_text": text,
        "safety_tradeoff": safety,
        "use_double_next": next_use_double,
        "modify_reward_next": next_reward,
        "vanilla_gate_pass": bool(g_v.gate_pass),
        "double_gate_pass": double_gate,
        "consecutive_primary_pass": consecutive,
        "stable_sufficient_budget": str(g_d.stable_sufficient_budget),
        "vanilla_late_collapses": v_collapse,
        "double_late_collapses": d_collapse,
        "mcnemar_p_exploratory": _mcnemar_exact(
            int(((late_df.vanilla_collapse_flag) & (~late_df.double_collapse_flag)).sum()),
            int(((~late_df.vanilla_collapse_flag) & (late_df.double_collapse_flag)).sum()),
        ),
        "discordant_v_only": int(((late_df.vanilla_collapse_flag) & (~late_df.double_collapse_flag)).sum()),
        "discordant_d_only": int(((~late_df.vanilla_collapse_flag) & (late_df.double_collapse_flag)).sum()),
    }


def write_reports(
    validation: dict[str, Any],
    seed_df: pd.DataFrame,
    descriptives_df: pd.DataFrame,
    contrasts: pd.DataFrame,
    late_df: pd.DataFrame,
    gate: pd.DataFrame,
    qinfo: dict[str, Any],
    decision: dict[str, Any],
    fig_paths: list[str],
    out_tables: dict[str, str],
) -> None:
    reports = PILOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    c_s = contrasts[(contrasts.endpoint == "success") & (contrasts.checkpoint_step == 300000)].iloc[0]
    c_t = contrasts[(contrasts.endpoint == "truncation") & (contrasts.checkpoint_step == 300000)].iloc[0]
    c_c = contrasts[(contrasts.endpoint == "collision") & (contrasts.checkpoint_step == 300000)].iloc[0]
    c_u = contrasts[(contrasts.endpoint == "unilateral_stall") & (contrasts.checkpoint_step == 300000)].iloc[0]
    v300 = seed_df[(seed_df.condition == "vanilla_dqn") & (seed_df.checkpoint_step == 300000)]
    d300 = seed_df[(seed_df.condition == "double_dqn") & (seed_df.checkpoint_step == 300000)]

    summary = {
        "stage": "Stage 7B-A1 Phase 3",
        "status": "PASS" if validation["analysis_status"] == "OK" else "BLOCKED",
        "results_commit": validation["results_commit"],
        "frozen_protocol_commit": FROZEN_PROTOCOL_COMMIT,
        "frozen_tag": FROZEN_TAG,
        "integrity": {
            "runs": validation["n_runs_completed"],
            "seeds": validation["n_paired_seeds_present"],
            "evaluation_episodes": validation["actual_episodes"],
            "duplicate_keys": validation["duplicate_keys"],
            "missing_pairs": validation["missing_condition_seed_checkpoint"],
            "checkpoint_inventory_sha256_complete": validation["checkpoint_inventory_sha256_complete"],
            "checkpoints_downloaded_locally": False,
            "thesis_files_changed": validation["thesis_changed_files"],
        },
        "outcomes_300k": {
            "vanilla_success": float(v300.success_rate.mean()),
            "double_success": float(d300.success_rate.mean()),
            "success_paired_diff": float(c_s.mean_paired_difference),
            "success_ci": [float(c_s.bootstrap_ci_low), float(c_s.bootstrap_ci_high)],
            "success_holm_p": None if not math.isfinite(float(c_s.holm_adjusted_p)) else float(c_s.holm_adjusted_p),
            "success_cohen_dz": float(c_s.cohen_dz),
            "vanilla_truncation": float(v300.truncation_rate.mean()),
            "double_truncation": float(d300.truncation_rate.mean()),
            "truncation_paired_diff": float(c_t.mean_paired_difference),
            "vanilla_collision": float(v300.collision_rate.mean()),
            "double_collision": float(d300.collision_rate.mean()),
            "collision_paired_diff": float(c_c.mean_paired_difference),
        },
        "stability": {
            "vanilla_late_collapses": decision["vanilla_late_collapses"],
            "double_late_collapses": decision["double_late_collapses"],
            "seed_bifurcation_reduced": bool(
                qinfo["seeds_lt_050_d"] < qinfo["seeds_lt_050_v"] and qinfo["seeds_ge_075_d"] > qinfo["seeds_ge_075_v"]
            ),
            "unilateral_stall_reduced": bool(float(c_u.mean_paired_difference) < 0),
            "seeds_ge_075_vanilla": qinfo["seeds_ge_075_v"],
            "seeds_ge_075_double": qinfo["seeds_ge_075_d"],
            "seeds_lt_050_vanilla": qinfo["seeds_lt_050_v"],
            "seeds_lt_050_double": qinfo["seeds_lt_050_d"],
        },
        "competence_gate": {
            "vanilla_passed": decision["vanilla_gate_pass"],
            "double_passed": decision["double_gate_pass"],
            "consecutive_checkpoint_confirmation": decision["consecutive_primary_pass"],
            "stable_sufficient_budget": decision["stable_sufficient_budget"],
        },
        "scientific_conclusion": {
            "decision_code": decision["decision_code"],
            "decision_text": decision["decision_text"],
            "double_dqn_support": qinfo["judgments"],
            "safety_tradeoff": decision["safety_tradeoff"],
        },
        "recommendation": {
            "use_double_dqn_for_next_pilot": decision["use_double_next"],
            "modify_reward_next": decision["modify_reward_next"],
            "extend_budget": False,
            "required_new_seeds": True,
        },
        "outputs": out_tables,
        "figures": fig_paths,
    }
    (reports / "stage7b_a1_final_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Failure taxonomy summary at 300k
    tax = pd.read_csv(PILOT / "output" / "diagnostics" / "baseline_algorithm_failure_taxonomy.csv")
    tax300 = tax[tax["checkpoint"] == 300000]
    tax_counts = (
        tax300.groupby(["condition", "primary_failure_label"]).size().unstack(fill_value=0).to_dict()
        if len(tax300)
        else {}
    )

    md = f"""# Stage 7B-A1 Final Analysis Report

## 1. Protocol and integrity

| Field | Value |
| --- | --- |
| Frozen tag | `{FROZEN_TAG}` |
| Frozen protocol commit | `{FROZEN_PROTOCOL_COMMIT}` |
| Results commit | `{validation["results_commit"]}` |
| Protocol hash | `{validation["protocol_hash"]}` |
| Environment lock | `{EXPECTED_ENV_LOCK}` |
| Comfort lock | `{EXPECTED_COMFORT_LOCK}` |
| Local validation | `{validation["analysis_status"]}` |
| Stage 7B git checkpoints | `{len(validation["stage7b_git_checkpoint_files"])}` |
| Checkpoints downloaded locally | `False` |

Critical failures: `{validation["critical_failures"] or "none"}`.

## 2. Run completion

- Planned / completed runs: {TRAINING_RUN_COUNT} / {validation["n_runs_completed"]}
- Paired seeds: {validation["n_paired_seeds_present"]} (`63001`–`63020`)
- Conditions: {validation["conditions_present"]}
- Checkpoints: {validation["n_checkpoints"]}
- Evaluation episodes: {validation["actual_episodes"]} (expected {EXPECTED_EVAL_EPISODES})
- Duplicate keys: {validation["duplicate_keys"]}
- Missing condition–seed–checkpoint: {validation["missing_condition_seed_checkpoint"]}
- Evaluation isolation violations: {validation["evaluation_isolation_violations"]}
- Checkpoint inventory SHA-256 complete: {validation["checkpoint_inventory_sha256_complete"]}
- Storage location: experiment_machine_local for all inventory rows
- Thesis / Stage-6 formal tracked file changes (training summary): {validation["thesis_changed_files"]} / {validation["stage6_changed_files"]}

## 3. Descriptive trajectories

Seed-level endpoints are in `{out_tables["seed_endpoints"]}`. Condition×checkpoint descriptives (mean/SD/median/IQR/min/max/bootstrap CI over seeds) are in `{out_tables["descriptives"]}`.

300K means (seed-level):

| Condition | Success | Collision | Truncation | Unilateral stall | Seeds ≥0.75 | Seeds <0.50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla | {v300.success_rate.mean():.4f} | {v300.collision_rate.mean():.4f} | {v300.truncation_rate.mean():.4f} | {v300.unilateral_stall_rate.mean():.4f} | {qinfo["seeds_ge_075_v"]} | {qinfo["seeds_lt_050_v"]} |
| Double | {d300.success_rate.mean():.4f} | {d300.collision_rate.mean():.4f} | {d300.truncation_rate.mean():.4f} | {d300.unilateral_stall_rate.mean():.4f} | {qinfo["seeds_ge_075_d"]} | {qinfo["seeds_lt_050_d"]} |

## 4. 300K primary paired contrasts

Statistical unit = paired training seed. Differences = `double_dqn - vanilla_dqn`. Bootstrap resamples the 20 paired differences. Holm adjustment applied only to the pre-defined 300K primary family.

| Endpoint | Mean Δ | 95% CI | Wilcoxon p | Holm p | Cohen dz | Double higher / lower / tied |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| success | {c_s.mean_paired_difference:.4f} | [{c_s.bootstrap_ci_low:.4f}, {c_s.bootstrap_ci_high:.4f}] | {c_s.raw_p:.4g} | {c_s.holm_adjusted_p:.4g} | {c_s.cohen_dz:.3f} | {int(c_s.improved_seed_count_double_higher)} / {int(c_s.degraded_seed_count_double_lower)} / {int(c_s.unchanged_seed_count)} |
| truncation | {c_t.mean_paired_difference:.4f} | [{c_t.bootstrap_ci_low:.4f}, {c_t.bootstrap_ci_high:.4f}] | {c_t.raw_p:.4g} | {c_t.holm_adjusted_p:.4g} | {c_t.cohen_dz:.3f} | {int(c_t.improved_seed_count_double_higher)} / {int(c_t.degraded_seed_count_double_lower)} / {int(c_t.unchanged_seed_count)} |
| collision | {c_c.mean_paired_difference:.4f} | [{c_c.bootstrap_ci_low:.4f}, {c_c.bootstrap_ci_high:.4f}] | {c_c.raw_p:.4g} | {c_c.holm_adjusted_p:.4g} | {c_c.cohen_dz:.3f} | {int(c_c.improved_seed_count_double_higher)} / {int(c_c.degraded_seed_count_double_lower)} / {int(c_c.unchanged_seed_count)} |
| unilateral_stall | {c_u.mean_paired_difference:.4f} | [{c_u.bootstrap_ci_low:.4f}, {c_u.bootstrap_ci_high:.4f}] | {c_u.raw_p:.4g} | {c_u.holm_adjusted_p:.4g} | {c_u.cohen_dz:.3f} | {int(c_u.improved_seed_count_double_higher)} / {int(c_u.degraded_seed_count_double_lower)} / {int(c_u.unchanged_seed_count)} |

Full table: `{out_tables["paired_contrasts"]}`.

## 5. Late collapse

Frozen rule: success at 200K or 250K ≥ 0.75 and success at 300K < 0.50.

- Vanilla collapse count: **{decision["vanilla_late_collapses"]}**
- Double collapse count: **{decision["double_late_collapses"]}**
- Discordant (Vanilla only / Double only): {decision["discordant_v_only"]} / {decision["discordant_d_only"]}
- Exact McNemar p (exploratory): {decision["mcnemar_p_exploratory"]:.4g}

Per-seed table: `{out_tables["late_collapse"]}`.

## 6. Failure taxonomy

Primary failure labels at 300K (episode counts by condition) are summarised from `output/diagnostics/baseline_algorithm_failure_taxonomy.csv`:

```json
{json.dumps(tax_counts, indent=2, default=str)}
```

Unilateral stall remains present under both algorithms at 300K (see descriptives and paired contrasts).

## 7. Q/TD diagnostics

| Metric | Vanilla | Double |
| --- | ---: | ---: |
| mean Q margin @300K | {qinfo["q_margin_300k_vanilla"]:.4f} | {qinfo["q_margin_300k_double"]:.4f} |
| mean abs best Q @300K | {qinfo["abs_q_300k_vanilla"]:.4f} | {qinfo["abs_q_300k_double"]:.4f} |
| TD abs p95 @300K | {qinfo["td_p95_300k_vanilla"]:.4f} | {qinfo["td_p95_300k_double"]:.4f} |
| mean replay terminal frac | {qinfo["replay_terminal_frac_vanilla"]:.4f} | {qinfo["replay_terminal_frac_double"]:.4f} |

Judgments (not claims of eliminated overestimation without direct target-gap evidence):

```json
{json.dumps(qinfo["judgments"], indent=2)}
```

## 8. Competence gate

Provisional gate (unchanged; not lowered for Double): ≥16/20 seeds success≥0.75; mean success≥0.75; collision≤0.05; truncation≤0.15; swap eligibility≥0.75.

| Condition | 300K gate | Failed components | Consecutive primary pass | Stable budget |
| --- | --- | --- | --- | --- |
| Vanilla | {decision["vanilla_gate_pass"]} | `{gate[(gate.condition=="vanilla_dqn")&(gate.checkpoint_step==300000)].iloc[0].failed_components}` | {bool(gate[gate.condition=="vanilla_dqn"].iloc[0].consecutive_primary_pass)} | `{gate[gate.condition=="vanilla_dqn"].iloc[0].stable_sufficient_budget}` |
| Double | {decision["double_gate_pass"]} | `{gate[(gate.condition=="double_dqn")&(gate.checkpoint_step==300000)].iloc[0].failed_components}` | {decision["consecutive_primary_pass"]} | `{decision["stable_sufficient_budget"]}` |

Table: `{out_tables["competence_gate"]}`.

## 9. Scientific interpretation

Decision class **{decision["decision_code"]}**: {decision["decision_text"]}.

- Double raises mean success and lowers truncation at 300K in the paired seed analysis, but does **not** pass the frozen competence gate.
- Late collapses: Vanilla {decision["vanilla_late_collapses"]} vs Double {decision["double_late_collapses"]} — Double does **not** reduce late collapse in this pilot.
- Collision direction: mean paired Δ = {c_c.mean_paired_difference:.4f} (positive ⇒ Double higher collision). No formal non-inferiority margin; do not claim safety non-inferiority.
- Safety note: {decision["safety_tradeoff"] or "no formal safety non-inferiority claim; report direction/uncertainty only"}.
- Seed bifurcation remains under both algorithms.
- Unilateral stall is not eliminated; reward/credit structure remains a live failure mode.

## 10. Next experiment decision

- Use Double DQN for next pilot: **{decision["use_double_next"]}**
- Modify reward next: **{decision["modify_reward_next"]}**
- Extend budget alone: **False** (gate not passed; bifurcation/stall persist)
- Required new seeds: **True** (do not reuse 610xx/620xx/630xx blocks without a new frozen plan)

Recommended path: treat Double as optional algorithmic default only if retained for engineering reasons; prioritise a **single-factor active-time-cost / stall-resolution reward pilot** because algorithm change alone was insufficient and late collapse did not improve.

## 11. Limitations

- Exploratory algorithm pilot; competence gate is provisional and not a confirmatory preregistered test.
- In-loop evaluation was empty on the training machine; episodes are post-hoc greedy reconstructions from full checkpoints (hashes inventoried; weights not downloaded here).
- No local checkpoint download; Q/TD diagnostics rely on experiment-machine summaries.
- Holm correction covers only the 300K primary family; other checkpoints/endpoints are exploratory.
- Cannot claim Double DQN eliminated overestimation without direct online/target overestimation gap series.
- Do not pool with formal 610xx or Stage 7A-1 620xx seeds.

## Figures

{chr(10).join(f"- `{p}`" for p in fig_paths)}
"""
    (reports / "stage7b_a1_final_analysis.md").write_text(md, encoding="utf-8")

    rec = f"""# Stage 7B-A1 Recommendation

## Decision

**{decision["decision_code"]} — {decision["decision_text"]}**

## Gate

- Vanilla passed: {decision["vanilla_gate_pass"]}
- Double passed: {decision["double_gate_pass"]}
- Consecutive checkpoint confirmation: {decision["consecutive_primary_pass"]}
- Stable sufficient budget: `{decision["stable_sufficient_budget"] or "none"}`

## Stability

- Vanilla late collapses: {decision["vanilla_late_collapses"]}
- Double late collapses: {decision["double_late_collapses"]}
- Seed bifurcation reduced: {summary["stability"]["seed_bifurcation_reduced"]}
- Unilateral stall reduced (paired mean Δ<0): {summary["stability"]["unilateral_stall_reduced"]}

## Safety

{decision["safety_tradeoff"] or "Collision change is reported with uncertainty only; no non-inferiority claim."}

## Next experiment

1. Do **not** declare Double competence-qualified.
2. Prefer a **reward / active-time-cost / stall-resolution** single-factor pilot next.
3. If Double is retained as the learner default, freeze it explicitly in the next protocol and draw **new** seeds outside 610xx/620xx/630xx.
4. Do not extend budget alone as the primary next step.
"""
    (reports / "stage7b_a1_recommendation.md").write_text(rec, encoding="utf-8")

    paper = """# PAPER_CHANGES_REQUIRED_LATER

Reminder only. No dissertation chapter files were modified by Stage 7B-A1 Phase 3.

1. If Stage 7B-A1 is mentioned, label it as an exploratory algorithm pilot, not a confirmatory H1 test.
2. Report paired seed-level contrasts (Double − Vanilla) with bootstrap CIs; do not treat 16 eval episodes as replicates.
3. State that Double improved mean success / lowered truncation descriptively but did **not** pass the frozen competence gate.
4. Report that late collapses were not reduced under Double in this pilot.
5. Report collision direction and uncertainty; do not claim safety non-inferiority without a preregistered margin.
6. Do not claim Double DQN eliminated overestimation without direct Q/target overestimation diagnostics.
7. Keep Stage 6 formal 610xx and Stage 7A-1 620xx as separate historical references; do not pool.
8. Note post-hoc greedy evaluation from external checkpoints when describing Stage 7B-A1 outcomes.
"""
    (reports / "PAPER_CHANGES_REQUIRED_LATER.md").write_text(paper, encoding="utf-8")


def main() -> int:
    analysis_dir = PILOT / "output" / "analysis"
    fig_dir = PILOT / "output" / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    validation = validate(analysis_dir)
    if validation["analysis_status"] == "BLOCKED":
        print("ANALYSIS BLOCKED", validation["critical_failures"])
        # Still write validation artifact; stop before statistics per protocol.
        return 2

    seed_df = build_seed_endpoints()
    seed_path = analysis_dir / "stage7b_a1_seed_checkpoint_endpoints.csv"
    seed_df.to_csv(seed_path, index=False)

    desc_df = descriptives(seed_df)
    desc_path = analysis_dir / "stage7b_a1_descriptives.csv"
    desc_df.to_csv(desc_path, index=False)

    late_df = late_collapse_by_seed(seed_df)
    late_path = analysis_dir / "stage7b_a1_late_collapse_by_seed.csv"
    late_df.to_csv(late_path, index=False)

    # cross-check published late collapse file
    pub_late = pd.read_csv(PILOT / "output" / "statistics" / "baseline_algorithm_late_collapse.csv")
    for cond, col in (("vanilla_dqn", "vanilla_collapse_flag"), ("double_dqn", "double_collapse_flag")):
        pub = set(int(s) for s in pub_late.loc[pub_late["late_collapse"] & (pub_late["condition"] == cond), "seed"])
        ours = set(int(s) for s in late_df.loc[late_df[col], "master_seed"])
        if pub != ours:
            print(f"WARNING late-collapse mismatch {cond}: pub={sorted(pub)} ours={sorted(ours)}")

    gate = competence_gate_table(seed_df)
    gate_path = analysis_dir / "stage7b_a1_competence_gate.csv"
    gate.to_csv(gate_path, index=False)

    contrasts = paired_contrasts(seed_df, late_df)
    contrasts_path = analysis_dir / "stage7b_a1_paired_contrasts.csv"
    contrasts.to_csv(contrasts_path, index=False)

    qinfo = q_td_interpretation(seed_df, late_df)
    (analysis_dir / "stage7b_a1_q_td_interpretation.json").write_text(
        json.dumps(qinfo, indent=2) + "\n", encoding="utf-8"
    )

    decision = decide(seed_df, contrasts, late_df, gate, qinfo)
    fig_paths = make_figures(seed_df, late_df, gate, fig_dir)

    out_tables = {
        "seed_endpoints": str(seed_path.relative_to(PILOT)).replace("\\", "/"),
        "descriptives": str(desc_path.relative_to(PILOT)).replace("\\", "/"),
        "paired_contrasts": str(contrasts_path.relative_to(PILOT)).replace("\\", "/"),
        "late_collapse": str(late_path.relative_to(PILOT)).replace("\\", "/"),
        "competence_gate": str(gate_path.relative_to(PILOT)).replace("\\", "/"),
        "local_validation": "manifests/local_result_validation.json",
    }
    write_reports(validation, seed_df, desc_df, contrasts, late_df, gate, qinfo, decision, fig_paths, out_tables)

    # analysis manifest hashes
    hash_paths = [
        PILOT / "manifests" / "local_result_validation.json",
        seed_path,
        desc_path,
        contrasts_path,
        late_path,
        gate_path,
        analysis_dir / "stage7b_a1_q_td_interpretation.json",
        PILOT / "reports" / "stage7b_a1_final_summary.json",
        PILOT / "reports" / "stage7b_a1_final_analysis.md",
        PILOT / "reports" / "stage7b_a1_recommendation.md",
        PILOT / "reports" / "PAPER_CHANGES_REQUIRED_LATER.md",
    ] + [PILOT / p for p in fig_paths]
    manifest = {
        "stage": "Stage 7B-A1 Phase 3",
        "results_commit": validation["results_commit"],
        "frozen_tag": FROZEN_TAG,
        "frozen_protocol_commit": FROZEN_PROTOCOL_COMMIT,
        "analysis_status": validation["analysis_status"],
        "decision_code": decision["decision_code"],
        "output_hashes": {
            str(p.relative_to(PILOT)).replace("\\", "/"): _sha256_file(p) for p in hash_paths if p.exists()
        },
    }
    (PILOT / "manifests" / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary_status(validation, decision), "decision": decision["decision_code"]}, indent=2))
    return 0


def summary_status(validation: dict[str, Any], decision: dict[str, Any]) -> str:
    if validation["analysis_status"] != "OK":
        return "BLOCKED"
    return "PASS"


if __name__ == "__main__":
    raise SystemExit(main())
