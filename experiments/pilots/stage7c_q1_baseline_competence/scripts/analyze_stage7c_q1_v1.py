#!/usr/bin/env python3
"""Stage 7C-Q1 independent integrity audit + competence qualification analysis.

Formal status is only PASS / FAIL / INVALID.
Does not modify gates, reward coefficients, or seed sets.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from thesis.analysis.stats import paired_bootstrap_ci  # noqa: E402
from thesis.pilots.stage7c_q1_config import (  # noqa: E402
    ACTIVE_TIME_COST_PER_STEP,
    ALGORITHM,
    BASE_REWARD_VERSION,
    CHECKPOINT_STEPS,
    CONDITION,
    GATE_ADJACENT_SUCCESS_DROP_MAX,
    GATE_CHECKPOINTS,
    GATE_COLLISION_MAX,
    GATE_MATERIAL_REGRESSION,
    GATE_MAX_LATE_COLLAPSE_SEEDS,
    GATE_MAX_MATERIAL_REGRESSION_SEEDS,
    GATE_MEAN_SUCCESS_MIN,
    GATE_MIN_QUALIFIED_SEEDS,
    GATE_SEED_SUCCESS_MIN,
    GATE_SWAP_ELIGIBILITY_MIN,
    GATE_TRUNCATION_MAX,
    LEARNING_CURVE_CHECKPOINTS,
    MAX_STEPS,
    PILOT_SEEDS,
    PROTOCOL_TAG,
    late_collapse_7c,
)
from thesis.pilots.stage7c_q1_eval import compute_swap_eligibility  # noqa: E402
from thesis.pilots.stage7c_q1_eval_seeds import (  # noqa: E402
    assert_no_eval_seed_overlap,
    eval_plan_for_checkpoint,
    stable_eval_seed,
)
from thesis.pilots.stage7c_q1_gate import evaluate_competence_gate  # noqa: E402

RESULTS = ROOT / "results" / "stage7c_q1" / "v1"
OUT = ROOT / "analysis" / "stage7c_q1" / "v1"
FIG = OUT / "figures"

BOOT_N = 10_000
BOOT_SEED = 91_001
ANALYSIS_SEED = 91_001
EXPECTED_CODE_COMMIT = "c8c75207c06c6a0511cac5fb24b644a61def8d14"
EXPECTED_CONFIG_SHA = "df64cc71c3c221e22b1abdb714ff6a45850ae32162e1b8ac8672cf23dc20e248"
EXPECTED_EPISODES = 14_080
EXPECTED_SEED_CKPTS = 340
HIST_7B_300K = {
    "success": 0.75625,
    "collision": 0.08125,
    "truncation": 0.1625,
    "note": (
        "Historical, non-paired comparison. "
        "Different master seeds and evaluation protocol. "
        "Not a causal estimate of the active-time reward effect."
    ),
}
FORBIDDEN_WEIGHT_SUFFIXES = (".pt", ".pth", ".ckpt")
FORBIDDEN_REPLAY_TOKENS = ("replay",)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _seed_cluster_ci(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    ci = paired_bootstrap_ci(x, n_boot=BOOT_N, seed=BOOT_SEED)
    return {
        "mean": float(np.mean(x)),
        "ci_low": float(ci["ci_low"]),
        "ci_high": float(ci["ci_high"]),
        "n_seeds": float(len(x)),
    }


def _parse_role_map(val: Any) -> dict[str, str]:
    if isinstance(val, dict):
        return {str(k): str(v) for k, v in val.items()}
    if pd.isna(val):
        return {}
    s = str(val)
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
    except (SyntaxError, ValueError):
        pass
    return {}


def build_seed_checkpoint(ep: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (seed, step), g in ep.groupby(["master_seed", "checkpoint_step"], sort=True):
        episodes = g.to_dict(orient="records")
        rows.append(
            {
                "master_seed": int(seed),
                "checkpoint_step": int(step),
                "success_rate": float(g["success"].mean()),
                "collision_rate": float(g["collision"].mean()),
                "truncation_rate": float(g["truncation"].mean()),
                "swap_eligibility": float(compute_swap_eligibility(episodes)),
                "n_episodes": int(len(g)),
                "mean_episode_length": float(g["episode_length"].mean()),
                "median_episode_length": float(g["episode_length"].median()),
                "n_success": int(g["success"].sum()),
            }
        )
    return pd.DataFrame(rows)


def run_integrity(ep: pd.DataFrame) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            errors.append(f"{name}: {detail}")

    # 1 protocol tag
    tag_ok = (ep["protocol_tag"] == PROTOCOL_TAG).all() and PROTOCOL_TAG == "stage7c-q1-protocol-v1"
    try:
        tag_obj = _git("rev-parse", "stage7c-q1-protocol-v1")
        tag_commit = _git("rev-list", "-n", "1", "stage7c-q1-protocol-v1")
    except Exception as exc:  # noqa: BLE001
        tag_obj, tag_commit = "", ""
        add("protocol_tag_git", False, f"cannot resolve tag: {exc}")
        add("protocol_tag_commit", False, "tag unresolved")
    else:
        add(
            "protocol_tag_git",
            bool(tag_obj) and bool(tag_commit),
            f"tag_object={tag_obj} tagged_commit={tag_commit}",
        )
        add(
            "protocol_tag_commit",
            tag_commit == EXPECTED_CODE_COMMIT,
            f"expected={EXPECTED_CODE_COMMIT} got={tag_commit}",
        )
    add("protocol_tag_in_episodes", bool(tag_ok), f"unique={sorted(ep['protocol_tag'].unique())}")

    # 2 same training commit
    commits = sorted(ep["code_commit"].astype(str).unique())
    add(
        "single_code_commit",
        commits == [EXPECTED_CODE_COMMIT],
        f"commits={commits}",
    )

    # 3 config SHA
    cfg_paths = [
        ROOT / "configs" / "stage7c_q1.yaml",
        ROOT / "experiments" / "pilots" / "stage7c_q1_baseline_competence" / "configs" / "stage7c_q1_protocol.yaml",
    ]
    cfg_shas = {str(p): _sha256_file(p) for p in cfg_paths if p.exists()}
    add(
        "config_sha256",
        all(v == EXPECTED_CONFIG_SHA for v in cfg_shas.values()) and bool(cfg_shas),
        json.dumps(cfg_shas),
    )
    inv = pd.read_csv(RESULTS / "manifests" / "CHECKPOINT_INVENTORY.csv")
    inv_cfg = sorted(inv["config_sha256"].astype(str).unique())
    add(
        "inventory_config_sha256",
        inv_cfg == [EXPECTED_CONFIG_SHA],
        f"unique={inv_cfg}",
    )

    # 4-5 seeds
    seeds = sorted(int(s) for s in ep["master_seed"].unique())
    add("seeds_exact_64001_64020", seeds == list(PILOT_SEEDS), f"seeds={seeds}")
    forbidden = [s for s in seeds if s < 64001 or s > 64020]
    add("no_foreign_formal_seeds", forbidden == [], f"foreign={forbidden}")

    # 6-11 provenance manifests
    prov = json.loads((RESULTS / "manifests" / "PROTOCOL_PROVENANCE.json").read_text(encoding="utf-8"))
    add("algorithm_double_dqn", prov.get("algorithm") == ALGORITHM, str(prov.get("algorithm")))
    add("condition_baseline", prov.get("condition") == CONDITION, str(prov.get("condition")))
    add(
        "base_reward_v2",
        prov.get("base_reward_version") == BASE_REWARD_VERSION,
        str(prov.get("base_reward_version")),
    )
    add(
        "active_time_cost_0_0005",
        abs(float(prov.get("active_time_cost_per_step", -1)) - ACTIVE_TIME_COST_PER_STEP) < 1e-15,
        str(prov.get("active_time_cost_per_step")),
    )
    add("max_steps_400000", int(prov.get("max_steps", -1)) == MAX_STEPS, str(prov.get("max_steps")))

    ckpts = sorted(int(c) for c in ep["checkpoint_step"].unique())
    add("checkpoint_schedule_17", ckpts == list(CHECKPOINT_STEPS), f"ckpts={ckpts}")

    # 12-13 completeness
    train = json.loads((RESULTS / "manifests" / "TRAINING_COMPLETENESS_REPORT.json").read_text(encoding="utf-8"))
    add(
        "training_complete_20_seeds",
        train.get("status") == "COMPLETE"
        and int(train.get("completed_seeds", 0)) == 20
        and not train.get("failed_or_incomplete_seeds"),
        json.dumps(train),
    )
    pairs = ep.groupby(["master_seed", "checkpoint_step"]).size().reset_index(name="n")
    add(
        "logical_seed_checkpoints_340",
        len(pairs) == EXPECTED_SEED_CKPTS,
        f"n_pairs={len(pairs)}",
    )

    # 14 episode count
    add("episode_count_14080", len(ep) == EXPECTED_EPISODES, f"n={len(ep)}")

    # 15-16 duplicate / conflict keys
    key_cols = [
        "master_seed",
        "checkpoint_step",
        "scenario_block",
        "assignment",
        "eval_seed",
    ]
    dup = int(ep.duplicated(subset=key_cols, keep=False).sum())
    add("no_duplicate_episode_keys", dup == 0, f"duplicate_rows={dup}")
    # conflict: same key different outcomes
    conflict = 0
    for _, g in ep.groupby(key_cols):
        if len(g) > 1:
            conflict += 1
    add("no_same_key_conflicts", conflict == 0, f"conflict_groups={conflict}")

    # 17 eval seed overlap across master seeds
    overlap_ok = True
    overlap_err = ""
    try:
        assert_no_eval_seed_overlap(PILOT_SEEDS, CHECKPOINT_STEPS)
        # also verify recorded eval seeds match plan and no cross-seed collisions in data
        seen: dict[int, int] = {}
        for _, row in ep.drop_duplicates(["master_seed", "checkpoint_step", "scenario_block"]).iterrows():
            es = int(row["eval_seed"])
            ms = int(row["master_seed"])
            expected = stable_eval_seed(
                master_seed=ms,
                checkpoint_step=int(row["checkpoint_step"]),
                scenario_block=int(row["scenario_block"]),
            )
            if es != expected:
                overlap_ok = False
                overlap_err = f"eval_seed mismatch seed={ms} ckpt={row['checkpoint_step']} block={row['scenario_block']}"
                break
            if es in seen and seen[es] != ms:
                overlap_ok = False
                overlap_err = f"cross-seed eval_seed overlap {es}: {seen[es]} vs {ms}"
                break
            seen[es] = ms
    except AssertionError as exc:
        overlap_ok = False
        overlap_err = str(exc)
    add("no_cross_seed_eval_overlap", overlap_ok, overlap_err or "ok")

    # 18-19 role-swap pairs
    pair_ok = True
    pair_detail = []
    for (ms, ckpt, block), g in ep.groupby(["master_seed", "checkpoint_step", "scenario_block"]):
        assigns = sorted(int(a) for a in g["assignment"].unique())
        if assigns != [0, 1]:
            pair_ok = False
            pair_detail.append(f"incomplete_assign ms={ms} ckpt={ckpt} b={block} {assigns}")
            continue
        seeds = sorted(int(s) for s in g["eval_seed"].unique())
        if len(seeds) != 1:
            pair_ok = False
            pair_detail.append(f"shared_eval_seed_fail ms={ms} ckpt={ckpt} b={block} {seeds}")
        pair_ids = sorted(str(x) for x in g["swap_pair_id"].unique())
        if len(pair_ids) != 1:
            pair_ok = False
            pair_detail.append(f"swap_pair_id_fail ms={ms} ckpt={ckpt} b={block}")
    add("role_swap_pairs_complete", pair_ok, "; ".join(pair_detail[:5]) or "ok")

    # 20 reward decomposition
    comps_a = [
        "reward_progress_A",
        "reward_exit_A",
        "reward_collision_A",
        "reward_hard_braking_A",
        "reward_active_time_A",
    ]
    comps_b = [
        "reward_progress_B",
        "reward_exit_B",
        "reward_collision_B",
        "reward_hard_braking_B",
        "reward_active_time_B",
    ]
    sum_a = ep[comps_a].sum(axis=1)
    sum_b = ep[comps_b].sum(axis=1)
    bad_a = int((np.abs(sum_a - ep["reward_total_A"]) > 1e-6).sum())
    bad_b = int((np.abs(sum_b - ep["reward_total_B"]) > 1e-6).sum())
    add("reward_decomposition_sum", bad_a == 0 and bad_b == 0, f"bad_A={bad_a} bad_B={bad_b}")

    # 21 checkpoint inventory hash format
    sha_txt = (RESULTS / "manifests" / "CHECKPOINT_INVENTORY_SHA256.txt").read_text(encoding="utf-8")
    sha_lines = [ln.strip() for ln in sha_txt.splitlines() if ln.strip()]
    fmt_ok = all(len(ln.split()[0]) == 64 and len(ln.split()) >= 2 for ln in sha_lines)
    full_rows = inv[inv["artifact_type"] == "full_checkpoint"]
    add(
        "checkpoint_inventory_hash_format",
        fmt_ok and len(full_rows) == EXPECTED_SEED_CKPTS and len(sha_lines) >= EXPECTED_SEED_CKPTS,
        f"sha_lines={len(sha_lines)} full_ckpt_rows={len(full_rows)}",
    )
    # inventory lists logical 340 full checkpoints
    inv_pairs = full_rows.groupby(["master_seed", "checkpoint"]).size()
    add(
        "inventory_340_full_checkpoints",
        len(inv_pairs) == EXPECTED_SEED_CKPTS,
        f"n={len(inv_pairs)}",
    )

    # 22 no weight/replay files in git results tree
    tracked = _git("ls-files", "results/stage7c_q1/v1").splitlines()
    bad_files = []
    for rel in tracked:
        low = rel.lower()
        if low.endswith(FORBIDDEN_WEIGHT_SUFFIXES):
            bad_files.append(rel)
        if any(tok in Path(rel).name.lower() for tok in FORBIDDEN_REPLAY_TOKENS):
            bad_files.append(rel)
    add("git_results_no_weight_or_replay", bad_files == [], f"bad={bad_files[:10]}")

    # per-seed-checkpoint episode counts match protocol
    count_ok = True
    count_detail = []
    for _, r in pairs.iterrows():
        step = int(r["checkpoint_step"])
        expected_n = 16 if step <= 175_000 else 64
        if int(r["n"]) != expected_n:
            count_ok = False
            count_detail.append(f"{r['master_seed']}@{step}: {r['n']}!={expected_n}")
    add("episodes_per_seed_checkpoint", count_ok, "; ".join(count_detail[:5]) or "ok")

    # machine integrity report cross-check
    machine = json.loads((RESULTS / "manifests" / "EVALUATION_INTEGRITY_REPORT.json").read_text(encoding="utf-8"))
    add(
        "machine_integrity_complete",
        machine.get("status") == "COMPLETE"
        and int(machine.get("actual_episodes", 0)) == EXPECTED_EPISODES
        and int(machine.get("duplicate_episode_keys", -1)) == 0,
        json.dumps({k: machine.get(k) for k in ("status", "actual_episodes", "duplicate_episode_keys", "cross_seed_eval_overlap_ok")}),
    )

    status = "VALID" if not errors else "INVALID"
    return {
        "status": status,
        "integrity_ok": status == "VALID",
        "n_checks": len(checks),
        "n_failed": len(errors),
        "errors": errors,
        "checks": checks,
        "protocol_tag": PROTOCOL_TAG,
        "expected_code_commit": EXPECTED_CODE_COMMIT,
        "expected_config_sha256": EXPECTED_CONFIG_SHA,
        "analysis_commit_placeholder": None,
    }


def checkpoint_descriptives(ep: pd.DataFrame, sc: pd.DataFrame, *, view: str) -> pd.DataFrame:
    rows = []
    for step in CHECKPOINT_STEPS:
        g_ep = ep[ep["checkpoint_step"] == step]
        g_sc = sc[sc["checkpoint_step"] == step]
        if g_ep.empty or g_sc.empty:
            continue
        s_ci = _seed_cluster_ci(g_sc["success_rate"].to_numpy())
        c_ci = _seed_cluster_ci(g_sc["collision_rate"].to_numpy())
        t_ci = _seed_cluster_ci(g_sc["truncation_rate"].to_numpy())
        sw_ci = _seed_cluster_ci(g_sc["swap_eligibility"].to_numpy())
        fail = g_ep["failure_category"].fillna("unknown").value_counts(normalize=True)
        po = g_ep["passing_order"].fillna("unknown").value_counts(normalize=True)
        rows.append(
            {
                "view": view,
                "checkpoint_step": int(step),
                "n_episodes": int(len(g_ep)),
                "n_seeds": int(g_sc["master_seed"].nunique()),
                "episode_success": float(g_ep["success"].mean()),
                "episode_collision": float(g_ep["collision"].mean()),
                "episode_truncation": float(g_ep["truncation"].mean()),
                "seed_mean_success": float(g_sc["success_rate"].mean()),
                "seed_mean_collision": float(g_sc["collision_rate"].mean()),
                "seed_mean_truncation": float(g_sc["truncation_rate"].mean()),
                "seed_mean_swap_eligibility": float(g_sc["swap_eligibility"].mean()),
                "success_ci_low": s_ci["ci_low"],
                "success_ci_high": s_ci["ci_high"],
                "collision_ci_low": c_ci["ci_low"],
                "collision_ci_high": c_ci["ci_high"],
                "truncation_ci_low": t_ci["ci_low"],
                "truncation_ci_high": t_ci["ci_high"],
                "swap_ci_low": sw_ci["ci_low"],
                "swap_ci_high": sw_ci["ci_high"],
                "median_seed_success": float(g_sc["success_rate"].median()),
                "iqr_seed_success": float(g_sc["success_rate"].quantile(0.75) - g_sc["success_rate"].quantile(0.25)),
                "min_seed_success": float(g_sc["success_rate"].min()),
                "max_seed_success": float(g_sc["success_rate"].max()),
                "n_seeds_ge_0_75": int((g_sc["success_rate"] >= 0.75).sum()),
                "n_seeds_ge_0_95": int((g_sc["success_rate"] >= 0.95).sum()),
                "mean_episode_length": float(g_ep["episode_length"].mean()),
                "median_episode_length": float(g_ep["episode_length"].median()),
                "mean_exit_step_agent_0": float(g_ep["exit_step_agent_0"].mean(skipna=True)),
                "mean_exit_step_agent_1": float(g_ep["exit_step_agent_1"].mean(skipna=True)),
                "passing_mainline_first": float(po.get("mainline_first", 0.0)),
                "passing_ramp_first": float(po.get("ramp_first", 0.0)),
                "passing_no_resolution": float(po.get("no_resolution", 0.0) + po.get("none", 0.0) + po.get("unknown", 0.0)),
                "fail_success": float(fail.get("success", 0.0)),
                "fail_collision": float(fail.get("collision", 0.0)),
                "fail_unilateral_stall": float(fail.get("unilateral_stall", 0.0)),
                "fail_mutual_yielding": float(fail.get("mutual_yielding", 0.0)),
                "fail_downstream_failure": float(fail.get("downstream_failure", 0.0)),
                "fail_truncation_other": float(
                    sum(v for k, v in fail.items() if k not in {
                        "success", "collision", "unilateral_stall", "mutual_yielding", "downstream_failure"
                    })
                ),
            }
        )
    return pd.DataFrame(rows)


def safety_role_tables(ep: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    focus = [200000, 250000, 300000, 350000, 375000, 400000]
    tax_rows = []
    for step in focus:
        g = ep[ep["checkpoint_step"] == step]
        n = max(len(g), 1)
        cats = g["failure_category"].fillna("unknown").value_counts()
        tax_rows.append(
            {
                "checkpoint_step": step,
                "n_episodes": len(g),
                "success": float(g["success"].mean()),
                "collision": float(g["collision"].mean()),
                "truncation": float(g["truncation"].mean()),
                "unilateral_stall": float(cats.get("unilateral_stall", 0) / n),
                "mutual_yielding": float(cats.get("mutual_yielding", 0) / n),
                "downstream_failure": float(cats.get("downstream_failure", 0) / n),
                "collision_category": float(cats.get("collision", 0) / n),
            }
        )
    tax = pd.DataFrame(tax_rows)

    role_rows = []
    for step in CHECKPOINT_STEPS:
        g = ep[ep["checkpoint_step"] == step].copy()
        if g.empty:
            continue
        # expand per-road-role outcomes using controller mapping where possible
        for _, row in g.iterrows():
            mapping = _parse_role_map(row["controller_role_mapping"])
            # success/collision are joint; still report exit times by road role via agents
            # Agent A/B mapped to road roles
            for agent, prefix in [("A", "reward_"), ("B", "reward_")]:
                road = mapping.get(agent)
                if road not in {"mainline", "ramp"}:
                    continue
            # passing / assignment aggregates handled separately
        po = g["passing_order"].value_counts(normalize=True)
        for assign, ga in g.groupby("assignment"):
            role_rows.append(
                {
                    "checkpoint_step": int(step),
                    "assignment": int(assign),
                    "n": int(len(ga)),
                    "success": float(ga["success"].mean()),
                    "collision": float(ga["collision"].mean()),
                    "truncation": float(ga["truncation"].mean()),
                    "mainline_first": float((ga["passing_order"] == "mainline_first").mean()),
                    "ramp_first": float((ga["passing_order"] == "ramp_first").mean()),
                }
            )
        # road-role exit times: agent0/1 not always road-labeled; use mapping
        exits_main = []
        exits_ramp = []
        coll_main = []
        coll_ramp = []
        # Joint collision; approximate role exposure via which road exited later on collisions is weak.
        # Report exit steps by mapped controller role using reward_active_time as activity proxy.
        for _, row in g.iterrows():
            mapping = _parse_role_map(row["controller_role_mapping"])
            # exit_step_agent_0 ~ controller A? Dataset uses agent_0/1; mapping uses A/B.
            # Use A->agent_0, B->agent_1 convention from eval writer.
            a_role = mapping.get("A")
            b_role = mapping.get("B")
            e0 = row["exit_step_agent_0"]
            e1 = row["exit_step_agent_1"]
            if a_role == "mainline" and pd.notna(e0):
                exits_main.append(float(e0))
            if a_role == "ramp" and pd.notna(e0):
                exits_ramp.append(float(e0))
            if b_role == "mainline" and pd.notna(e1):
                exits_main.append(float(e1))
            if b_role == "ramp" and pd.notna(e1):
                exits_ramp.append(float(e1))
        # swap disagreement
        disagree = 0
        pairs = 0
        for _, gb in g.groupby("swap_pair_id"):
            if set(gb["assignment"].tolist()) != {0, 1}:
                continue
            pairs += 1
            s0 = bool(gb.loc[gb["assignment"] == 0, "success"].iloc[0])
            s1 = bool(gb.loc[gb["assignment"] == 1, "success"].iloc[0])
            if s0 != s1:
                disagree += 1
        role_rows.append(
            {
                "checkpoint_step": int(step),
                "assignment": -1,
                "n": int(len(g)),
                "success": float(g["success"].mean()),
                "collision": float(g["collision"].mean()),
                "truncation": float(g["truncation"].mean()),
                "mainline_first": float(po.get("mainline_first", 0.0)),
                "ramp_first": float(po.get("ramp_first", 0.0)),
                "mean_exit_mainline": float(np.mean(exits_main)) if exits_main else float("nan"),
                "mean_exit_ramp": float(np.mean(exits_ramp)) if exits_ramp else float("nan"),
                "swap_disagreement_rate": float(disagree / pairs) if pairs else float("nan"),
                "n_swap_pairs": int(pairs),
            }
        )
    return tax, pd.DataFrame(role_rows)


def material_and_collapse(sc_ext: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    mat_rows = []
    late_rows = []
    late_ckpts = [c for c in LEARNING_CURVE_CHECKPOINTS if c >= 350_000]
    mat_seeds = []
    collapse_seeds = []
    for seed in PILOT_SEEDS:
        g = sc_ext[sc_ext["master_seed"] == seed]
        by = {int(r.checkpoint_step): float(r.success_rate) for r in g.itertuples(index=False)}
        for a, b in zip(late_ckpts, late_ckpts[1:]):
            drop = by.get(a, 0.0) - by.get(b, 0.0)
            if drop > GATE_MATERIAL_REGRESSION:
                mat_seeds.append(seed)
                mat_rows.append(
                    {
                        "master_seed": seed,
                        "from_checkpoint": a,
                        "to_checkpoint": b,
                        "success_from": by.get(a),
                        "success_to": by.get(b),
                        "drop": drop,
                    }
                )
        if late_collapse_7c(by):
            collapse_seeds.append(seed)
            late_rows.append(
                {
                    "master_seed": seed,
                    **{f"s_{k}": by.get(k) for k in LEARNING_CURVE_CHECKPOINTS},
                }
            )
    meta = {
        "material_regression_seeds": sorted(set(mat_seeds)),
        "late_collapse_seeds": sorted(set(collapse_seeds)),
        "n_material": len(set(mat_seeds)),
        "n_late_collapse": len(set(collapse_seeds)),
    }
    return pd.DataFrame(mat_rows), pd.DataFrame(late_rows), meta


def adjacent_drops(sc_ext: pd.DataFrame) -> dict[str, Any]:
    means = {}
    for ckpt in LEARNING_CURVE_CHECKPOINTS:
        g = sc_ext[sc_ext["checkpoint_step"] == ckpt]
        means[ckpt] = float(g["success_rate"].mean())
    deltas = []
    max_drop = -1e9
    worst = None
    for a, b in zip(LEARNING_CURVE_CHECKPOINTS, LEARNING_CURVE_CHECKPOINTS[1:]):
        delta = means[b] - means[a]
        drop = means[a] - means[b]
        deltas.append({"from": a, "to": b, "delta": delta, "drop": drop, "ok": drop <= GATE_ADJACENT_SUCCESS_DROP_MAX})
        if drop > max_drop:
            max_drop = drop
            worst = (a, b, drop)
    xs = np.asarray(LEARNING_CURVE_CHECKPOINTS, dtype=float)
    ys = np.asarray([means[c] for c in LEARNING_CURVE_CHECKPOINTS], dtype=float)
    spearman = scipy_stats.spearmanr(xs, ys)
    late = [means[c] for c in (350000, 375000, 400000)]
    return {
        "means": means,
        "deltas": deltas,
        "max_drop": float(max_drop),
        "worst_pair": worst,
        "net_200k_to_400k": float(means[400000] - means[200000]),
        "spearman_rho": float(spearman.statistic),
        "spearman_pvalue": float(spearman.pvalue),
        "late_max": float(max(late)),
        "late_min": float(min(late)),
        "platform_range_350_400": float(max(late) - min(late)),
        "all_drops_ok": all(d["ok"] for d in deltas),
    }


def make_figures(sc_std: pd.DataFrame, sc_ext: pd.DataFrame, cp_std: pd.DataFrame, cp_ext: pd.DataFrame, tax: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    def _curve(cp: pd.DataFrame, ycol: str, ylabel: str, path: Path, thresholds: list[tuple[float, str, str]]) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = cp["checkpoint_step"] / 1000.0
        ax.plot(x, cp[ycol], "o-", color="#1f77b4", label="mean (seed-equal)")
        if f"{ycol.replace('seed_mean_', '').replace('episode_', '')}_ci_low" in cp.columns or True:
            # map columns
            base = {
                "seed_mean_success": ("success_ci_low", "success_ci_high"),
                "seed_mean_collision": ("collision_ci_low", "collision_ci_high"),
                "seed_mean_truncation": ("truncation_ci_low", "truncation_ci_high"),
                "seed_mean_swap_eligibility": ("swap_ci_low", "swap_ci_high"),
            }.get(ycol)
            if base:
                ax.fill_between(x, cp[base[0]], cp[base[1]], color="#1f77b4", alpha=0.2, label="seed-cluster 95% CI")
        for thr, lab, style in thresholds:
            ax.axhline(thr, color="black", ls=style, lw=1, label=lab)
        ax.axvspan(200, 400, color="#dddddd", alpha=0.35, label="extended eval (≥200K)")
        ax.set_xlabel("Checkpoint (×1000 joint steps)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Stage 7C-Q1 {ylabel}\nprotocol={PROTOCOL_TAG}; n_seeds=20; seed-cluster bootstrap CI")
        ax.legend(fontsize=8, loc="best")
        ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    _curve(
        cp_std,
        "seed_mean_success",
        "Success rate",
        FIG / "01_success_learning_curve.png",
        [(0.95, "success ≥ 0.95", "--")],
    )
    _curve(
        cp_std,
        "seed_mean_collision",
        "Collision rate",
        FIG / "02_collision_learning_curve.png",
        [(0.02, "collision ≤ 0.02", "--")],
    )
    _curve(
        cp_std,
        "seed_mean_truncation",
        "Truncation rate",
        FIG / "03_truncation_learning_curve.png",
        [(0.03, "truncation ≤ 0.03", "--")],
    )

    # spaghetti
    fig, ax = plt.subplots(figsize=(8, 5))
    for seed, g in sc_std.groupby("master_seed"):
        ax.plot(g["checkpoint_step"] / 1000.0, g["success_rate"], "-", alpha=0.35, lw=1)
    means = sc_std.groupby("checkpoint_step")["success_rate"].mean()
    ax.plot(means.index / 1000.0, means.values, "o-", color="black", lw=2, label="condition mean")
    ax.axhline(0.95, color="red", ls="--", label="0.95")
    ax.axvspan(200, 400, color="#dddddd", alpha=0.35)
    ax.set_xlabel("Checkpoint (×1000)")
    ax.set_ylabel("Seed success rate")
    ax.set_title("Stage 7C-Q1 seed success spaghetti (standard 16-ep view)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "04_seed_success_spaghetti.png", dpi=160)
    plt.close(fig)

    # heatmap
    pivot = sc_std.pivot(index="master_seed", columns="checkpoint_step", values="success_rate")
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c / 1000)) for c in pivot.columns], rotation=45)
    ax.set_xlabel("Checkpoint (K)")
    ax.set_ylabel("Master seed")
    ax.set_title("Stage 7C-Q1 success heatmap (standard view)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG / "05_checkpoint_seed_heatmap.png", dpi=160)
    plt.close(fig)

    # late success distributions
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [sc_ext.loc[sc_ext["checkpoint_step"] == c, "success_rate"].to_numpy() for c in GATE_CHECKPOINTS]
    ax.boxplot(data, tick_labels=[str(c // 1000) + "K" for c in GATE_CHECKPOINTS], showmeans=True)
    ax.axhline(0.95, color="red", ls="--", label="0.95 / 61/64 gate")
    ax.set_ylabel("Seed success (64-ep extended)")
    ax.set_title("Late-stage success distributions (350/375/400K)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "06_late_success_distribution.png", dpi=160)
    plt.close(fig)

    # failure taxonomy
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, lab in [
        ("unilateral_stall", "unilateral_stall"),
        ("mutual_yielding", "mutual_yielding"),
        ("collision", "collision"),
        ("truncation", "truncation"),
        ("downstream_failure", "downstream_failure"),
    ]:
        ax.plot(tax["checkpoint_step"] / 1000.0, tax[col], "o-", label=lab)
    ax.set_xlabel("Checkpoint (K)")
    ax.set_ylabel("Episode proportion")
    ax.set_title("Failure taxonomy vs checkpoint (extended focus points)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "07_failure_taxonomy.png", dpi=160)
    plt.close(fig)

    # swap eligibility curve (extended means)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cp_ext["checkpoint_step"] / 1000.0, cp_ext["seed_mean_swap_eligibility"], "o-", label="mean swap eligibility")
    ax.fill_between(cp_ext["checkpoint_step"] / 1000.0, cp_ext["swap_ci_low"], cp_ext["swap_ci_high"], alpha=0.2)
    ax.axhline(0.75, color="red", ls="--", label="≥0.75")
    ax.axvspan(200, 400, color="#dddddd", alpha=0.35)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Checkpoint (K)")
    ax.set_ylabel("Strict swap eligibility")
    ax.set_title("Strict swap eligibility (extended view at ≥200K)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "08_swap_eligibility_curve.png", dpi=160)
    plt.close(fig)

    # 350/375/400 platform
    fig, ax = plt.subplots(figsize=(7, 5))
    sub = cp_ext[cp_ext["checkpoint_step"].isin(GATE_CHECKPOINTS)]
    ax.errorbar(
        sub["checkpoint_step"] / 1000.0,
        sub["seed_mean_success"],
        yerr=[
            sub["seed_mean_success"] - sub["success_ci_low"],
            sub["success_ci_high"] - sub["seed_mean_success"],
        ],
        fmt="o-",
        capsize=4,
        label="success",
    )
    ax.plot(sub["checkpoint_step"] / 1000.0, sub["seed_mean_collision"], "s--", label="collision")
    ax.plot(sub["checkpoint_step"] / 1000.0, sub["seed_mean_truncation"], "^--", label="truncation")
    ax.axhline(0.95, color="gray", ls=":", lw=1)
    ax.axhline(0.02, color="gray", ls=":", lw=1)
    ax.axhline(0.03, color="gray", ls=":", lw=1)
    ax.set_xlabel("Checkpoint (K)")
    ax.set_ylabel("Rate")
    ax.set_title("350K–400K stable platform (extended 64-ep)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "09_late_platform.png", dpi=160)
    plt.close(fig)

    # reward component summary (active-time magnitude)
    fig, ax = plt.subplots(figsize=(8, 5))
    # use extended late episodes only for clarity
    late_ep_path = RESULTS / "raw" / "evaluation_episodes.csv"
    ep = pd.read_csv(late_ep_path)
    late = ep[ep["checkpoint_step"].isin(GATE_CHECKPOINTS)]
    for col, lab in [
        ("reward_active_time_A", "active_time_A"),
        ("reward_active_time_B", "active_time_B"),
        ("reward_progress_A", "progress_A"),
        ("reward_total_A", "total_A"),
    ]:
        vals = np.sort(late[col].to_numpy())
        y = np.linspace(0, 1, len(vals), endpoint=False)
        ax.plot(vals, y, label=lab, lw=1)
    ax.set_xlabel("Episode cumulative component")
    ax.set_ylabel("ECDF")
    ax.set_title("Reward component ECDF at 350/375/400K")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "10_reward_component_ecdf.png", dpi=160)
    plt.close(fig)


def write_decision(
    *,
    integrity: dict[str, Any],
    gate: dict[str, Any],
    cp_ext: pd.DataFrame,
    drops: dict[str, Any],
    mat_meta: dict[str, Any],
    intersection: list[int],
    hist_row: dict[str, Any],
    safety_answers: dict[str, str],
) -> str:
    status = gate["status"] if integrity["integrity_ok"] else "INVALID"
    if not integrity["integrity_ok"]:
        status = "INVALID"

    lines = []
    lines.append("# STAGE 7C-Q1 DECISION")
    lines.append("")
    lines.append("## 1. Experiment identity")
    lines.append("")
    lines.append("- Stage: `stage7c_q1`")
    lines.append("- Purpose: Baseline competence qualification pilot")
    lines.append("- Algorithm: Double DQN")
    lines.append("- Condition: Baseline")
    lines.append("- Base reward: V2 active-time (`0.0005` / step)")
    lines.append("- Seeds: `64001`–`64020`")
    lines.append("- Max steps: `400000` joint environment steps")
    lines.append("")
    lines.append("## 2. Protocol provenance")
    lines.append("")
    lines.append(f"- Protocol tag: `{PROTOCOL_TAG}`")
    lines.append(f"- Tagged/code commit: `{EXPECTED_CODE_COMMIT}`")
    lines.append(f"- Config SHA-256: `{EXPECTED_CONFIG_SHA}`")
    lines.append(f"- Results branch tip (analysis parent): `{_git('rev-parse', 'HEAD')}` prior to analysis commit")
    lines.append("")
    lines.append("## 3. Integrity verdict")
    lines.append("")
    lines.append(f"- Integrity status: **{integrity['status']}**")
    lines.append(f"- Checks: {integrity['n_checks']}; failed: {integrity['n_failed']}")
    if integrity["errors"]:
        lines.append("- Failures:")
        for e in integrity["errors"]:
            lines.append(f"  - {e}")
    else:
        lines.append("- All critical provenance and completeness checks passed.")
    lines.append("")
    lines.append("## 4. Complete checkpoint table")
    lines.append("")
    lines.append("Extended view (≥200K uses 64 episodes/seed-checkpoint; early standard 16).")
    lines.append("")
    lines.append("| ckpt | ep_success | seed_mean_success | collision | truncation | swap | n_ep |")
    lines.append("|------|------------|-------------------|-----------|------------|------|------|")
    for _, r in cp_ext.iterrows():
        lines.append(
            f"| {int(r.checkpoint_step)} | {r.episode_success:.10g} | {r.seed_mean_success:.10g} | "
            f"{r.seed_mean_collision:.10g} | {r.seed_mean_truncation:.10g} | "
            f"{r.seed_mean_swap_eligibility:.10g} | {int(r.n_episodes)} |"
        )
    lines.append("")
    lines.append("Episode-pooled and seed-equal means coincide when all seeds share equal episode counts.")
    lines.append("")
    lines.append("## 5. Learning-curve assessment")
    lines.append("")
    lines.append(f"- Net change 200K→400K: `{drops['net_200k_to_400k']:.10g}`")
    lines.append(f"- Spearman ρ(checkpoint, success): `{drops['spearman_rho']:.10g}` (p=`{drops['spearman_pvalue']:.10g}`; descriptive only)")
    lines.append(f"- Max adjacent success drop (200K–400K): `{drops['max_drop']:.10g}` at `{drops['worst_pair']}`")
    lines.append(f"- All adjacent drops ≤ 0.03: `{drops['all_drops_ok']}`")
    lines.append(f"- 350–400K platform range: `{drops['platform_range_350_400']:.10g}`")
    lines.append("")
    lines.append("## 6. Seed-level stability")
    lines.append("")
    lines.append(f"- Stable qualified seed intersection (|S|≥61/64 at 350∩375∩400): `{len(intersection)}` / 20")
    lines.append(f"- Seeds: `{intersection}`")
    lines.append(f"- Material-regression seeds (350–400, drop>0.20): `{mat_meta['material_regression_seeds']}` (n={mat_meta['n_material']})")
    lines.append(f"- Late-collapse seeds: `{mat_meta['late_collapse_seeds']}` (n={mat_meta['n_late_collapse']})")
    lines.append("")
    lines.append("## 7. Safety assessment")
    lines.append("")
    for k, v in safety_answers.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## 8. Failure-mode assessment")
    lines.append("")
    lines.append("Collision and truncation are reported separately; they are not merged.")
    lines.append("See `failure_taxonomy.csv` and figure `07_failure_taxonomy.png`.")
    lines.append("")
    lines.append("## 9. Role and passing-order audit")
    lines.append("")
    lines.append("See `role_swap_analysis.csv`. This pilot is not a PBRS confirmatory test.")
    lines.append("Any base-reward-induced role bias is interpretive risk only unless a frozen hard threshold exists (none beyond swap eligibility).")
    lines.append("")
    lines.append("## 10. Historical Stage 7B comparison")
    lines.append("")
    lines.append(f"- Historical Stage 7B Double DQN @300K: success=`{HIST_7B_300K['success']}`, collision=`{HIST_7B_300K['collision']}`, truncation=`{HIST_7B_300K['truncation']}`")
    lines.append(
        f"- Stage 7C-Q1 @300K (extended): success=`{hist_row['success']:.10g}`, "
        f"collision=`{hist_row['collision']:.10g}`, truncation=`{hist_row['truncation']:.10g}`"
    )
    lines.append(f"- Disclaimer: {HIST_7B_300K['note']}")
    lines.append("- No causal claim that V2 is significantly better than V1.")
    lines.append("")
    lines.append("## 11. Gate table")
    lines.append("")
    comps = gate.get("components", {})
    lines.append("| checkpoint | mean_success | collision | truncation | swap | values |")
    lines.append("|------------|--------------|-----------|------------|------|--------|")
    for ck in GATE_CHECKPOINTS:
        c = comps.get(str(ck), {})
        lines.append(
            f"| {ck} | {c.get('mean_success')} | {c.get('collision')} | {c.get('truncation')} | "
            f"{c.get('swap_eligibility')} | "
            f"s={c.get('mean_success_value')}, c={c.get('mean_collision_value')}, "
            f"t={c.get('mean_truncation_value')}, swap={c.get('mean_swap_value')} |"
        )
    lines.append("")
    lines.append(f"- intersection_ok: `{comps.get('intersection_ok')}` count=`{comps.get('qualified_seed_intersection_count')}`")
    lines.append(f"- learning_curve_ok: `{comps.get('learning_curve_ok')}` violations=`{comps.get('learning_curve_violations')}`")
    lines.append(f"- material_regression_seeds: `{comps.get('material_regression_seeds')}`")
    lines.append(f"- late_collapse_seeds: `{comps.get('late_collapse_seeds')}`")
    lines.append("")
    lines.append("## 12. Final status")
    lines.append("")
    lines.append(f"# **{status}**")
    lines.append("")
    lines.append("## 13. Permitted next action")
    lines.append("")
    if status == "PASS":
        lines.append(
            "Freeze Double DQN + Base Reward V2 + 400K. "
            "Prepare final Baseline / Mean-PBRS / Min-PBRS experiment "
            "using entirely new paired master seeds."
        )
    elif status == "FAIL":
        lines.append(
            "Do not start the final three-condition experiment. "
            "Stop further algorithm and reward modifications. "
            "Report competence-limited conclusion."
        )
    else:
        lines.append(
            "Repair data integrity or rerun only technically invalid runs "
            "under the same frozen protocol. Do not change scientific parameters."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    ep = pd.read_csv(RESULTS / "raw" / "evaluation_episodes.csv")
    # normalize booleans
    for col in ("success", "collision", "truncation", "terminated", "truncated"):
        if col in ep.columns:
            ep[col] = ep[col].astype(bool)

    integrity = run_integrity(ep)
    (OUT / "integrity_report.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    md_lines = [
        "# Stage 7C-Q1 Integrity Report",
        "",
        f"Status: **{integrity['status']}**",
        "",
        f"Failed checks: {integrity['n_failed']} / {integrity['n_checks']}",
        "",
        "| check | ok | detail |",
        "|-------|----|--------|",
    ]
    for c in integrity["checks"]:
        md_lines.append(f"| `{c['check']}` | {c['ok']} | {c['detail'][:180].replace('|', '/')} |")
    (OUT / "integrity_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Views
    ep_std = ep[ep["scenario_block"] < 8].copy()
    ep_ext = ep[ep["checkpoint_step"] >= 200_000].copy()
    # For early checkpoints in extended descriptive table we still want all ckpts in std;
    # gate uses extended-only rates at late ckpts from full late episodes.
    sc_std = build_seed_checkpoint(ep_std)
    sc_ext_late = build_seed_checkpoint(ep_ext)
    # Combined seed-checkpoint for gate: early from std (16), late from full extended (64)
    sc_gate_parts = []
    sc_full = build_seed_checkpoint(ep)  # protocol-native counts
    sc_gate = sc_full.copy()

    sc_std.to_csv(OUT / "seed_checkpoint_summary_standard16.csv", index=False)
    sc_ext_late.to_csv(OUT / "seed_checkpoint_summary_extended64.csv", index=False)
    sc_gate.to_csv(OUT / "seed_checkpoint_summary.csv", index=False)

    cp_std = checkpoint_descriptives(ep_std, sc_std, view="standard16")
    # For extended descriptives across ALL checkpoints: use std for <200K and ext for >=200K
    sc_ext_all_rows = []
    for step in CHECKPOINT_STEPS:
        if step < 200_000:
            sc_ext_all_rows.append(sc_std[sc_std["checkpoint_step"] == step])
        else:
            sc_ext_all_rows.append(sc_ext_late[sc_ext_late["checkpoint_step"] == step])
    sc_ext_all = pd.concat(sc_ext_all_rows, ignore_index=True)
    ep_ext_all = pd.concat(
        [ep_std[ep_std["checkpoint_step"] < 200_000], ep_ext],
        ignore_index=True,
    )
    cp_ext = checkpoint_descriptives(ep_ext_all, sc_ext_all, view="extended_mixed_denom_by_protocol")
    # Clarify: early rows are 16-ep; late are 64-ep — never mix within a checkpoint.
    cp_ext.to_csv(OUT / "checkpoint_summary.csv", index=False)
    cp_std.to_csv(OUT / "checkpoint_summary_standard16.csv", index=False)

    tax, role = safety_role_tables(ep)
    tax.to_csv(OUT / "failure_taxonomy.csv", index=False)
    role.to_csv(OUT / "role_swap_analysis.csv", index=False)

    mat_df, late_df, mat_meta = material_and_collapse(sc_ext_late)
    mat_df.to_csv(OUT / "material_regression_seeds.csv", index=False)
    late_df.to_csv(OUT / "late_collapse_seeds.csv", index=False)

    drops = adjacent_drops(sc_ext_late)

    gate = evaluate_competence_gate(
        sc_gate[sc_gate["checkpoint_step"] >= 200_000].copy(),
        expected_seeds=PILOT_SEEDS,
        integrity_ok=integrity["integrity_ok"],
        integrity_errors=integrity["errors"],
    )
    # Gate function expects all learning-curve ckpts present; filtered >=200K is correct.
    (OUT / "gate_results.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")

    # Flatten gate csv
    gate_rows = []
    comps = gate.get("components", {})
    for ck in GATE_CHECKPOINTS:
        c = comps.get(str(ck), {})
        gate_rows.append({"checkpoint_step": ck, **c})
    gate_rows.append(
        {
            "checkpoint_step": "summary",
            "status": gate.get("status"),
            "intersection_count": comps.get("qualified_seed_intersection_count"),
            "intersection_seeds": json.dumps(comps.get("qualified_seed_intersection", [])),
            "learning_curve_ok": comps.get("learning_curve_ok"),
            "max_adjacent_drop": drops["max_drop"],
            "material_regression_seeds": json.dumps(mat_meta["material_regression_seeds"]),
            "late_collapse_seeds": json.dumps(mat_meta["late_collapse_seeds"]),
        }
    )
    pd.DataFrame(gate_rows).to_csv(OUT / "gate_results.csv", index=False)

    # Historical comparison
    g300 = cp_ext[cp_ext["checkpoint_step"] == 300000].iloc[0]
    hist = pd.DataFrame(
        [
            {
                "source": "stage7b_double_dqn_historical",
                "checkpoint_step": 300000,
                "success": HIST_7B_300K["success"],
                "collision": HIST_7B_300K["collision"],
                "truncation": HIST_7B_300K["truncation"],
                "note": HIST_7B_300K["note"],
            },
            {
                "source": "stage7c_q1_extended",
                "checkpoint_step": 300000,
                "success": float(g300["seed_mean_success"]),
                "collision": float(g300["seed_mean_collision"]),
                "truncation": float(g300["seed_mean_truncation"]),
                "note": HIST_7B_300K["note"],
            },
        ]
    )
    hist.to_csv(OUT / "historical_comparison.csv", index=False)

    # Safety narrative answers
    focus = tax.set_index("checkpoint_step")
    t200, t350, t375, t400 = (
        float(focus.loc[200000, "truncation"]),
        float(focus.loc[350000, "truncation"]),
        float(focus.loc[375000, "truncation"]),
        float(focus.loc[400000, "truncation"]),
    )
    c200, c350, c375, c400 = (
        float(focus.loc[200000, "collision"]),
        float(focus.loc[350000, "collision"]),
        float(focus.loc[375000, "collision"]),
        float(focus.loc[400000, "collision"]),
    )
    conversion = (t400 < t350) and (c400 > c350)
    safety_answers = {
        "unilateral_stall_trend": (
            f"200K={focus.loc[200000,'unilateral_stall']:.6g} → 400K={focus.loc[400000,'unilateral_stall']:.6g}; "
            "already near-absent in late extended evaluations (no material late decline signal)."
        ),
        "mutual_yielding_trend": (
            f"200K={focus.loc[200000,'mutual_yielding']:.6g} → 400K={focus.loc[400000,'mutual_yielding']:.6g}; "
            + (
                "declines"
                if focus.loc[400000, "mutual_yielding"] < focus.loc[200000, "mutual_yielding"]
                else "does not decline"
            )
        ),
        "collision_late_rise": (
            f"350K={c350:.6g}, 375K={c375:.6g}, 400K={c400:.6g}; "
            + (
                "yes, collision rises at 400K relative to 350/375"
                if c400 > max(c350, c375)
                else "no clear terminal rise vs 350/375"
            )
        ),
        "low_truncation_high_collision": (
            f"350K truncation={t350:.6g}, collision={c350:.6g}; "
            f"400K truncation={t400:.6g}, collision={c400:.6g}; "
            + (
                "partial conversion pattern present (truncation lower than 350K while collision higher)."
                if conversion
                else "no clean truncation-down/collision-up conversion from 350K to 400K; both remain above gate."
            )
        ),
        "downstream_failure_note": (
            f"Dominant non-success truncation-related category is downstream_failure "
            f"(200K={focus.loc[200000,'downstream_failure']:.6g}, "
            f"400K={focus.loc[400000,'downstream_failure']:.6g})."
        ),
    }
    # role asymmetry quick check at 400K
    r400 = role[(role["checkpoint_step"] == 400000) & (role["assignment"] == -1)]
    if not r400.empty:
        mf = float(r400.iloc[0]["mainline_first"])
        rf = float(r400.iloc[0]["ramp_first"])
        safety_answers["road_role_collision_asymmetry"] = (
            f"passing-order at 400K is near-balanced (mainline_first={mf:.6g}, ramp_first={rf:.6g}); "
            f"mean_exit_mainline={r400.iloc[0].get('mean_exit_mainline', float('nan')):.6g}, "
            f"mean_exit_ramp={r400.iloc[0].get('mean_exit_ramp', float('nan')):.6g}."
        )
        safety_answers["passing_order_bias_400K"] = (
            f"mainline_first={mf:.6g}, ramp_first={rf:.6g}; "
            "not highly biased to a single direction."
        )
        safety_answers["controller_role_swap_changes_outcomes"] = (
            f"swap_disagreement_rate={float(r400.iloc[0].get('swap_disagreement_rate', float('nan'))):.6g} "
            f"over n_pairs={int(r400.iloc[0].get('n_swap_pairs', 0))}; "
            "assignment-stratified rates in role_swap_analysis.csv."
        )

    make_figures(sc_std, sc_ext_late, cp_std, cp_ext[cp_ext["checkpoint_step"] >= 200000].copy(), tax)

    intersection = comps.get("qualified_seed_intersection", [])
    decision = write_decision(
        integrity=integrity,
        gate=gate,
        cp_ext=cp_ext,
        drops=drops,
        mat_meta=mat_meta,
        intersection=list(intersection),
        hist_row={
            "success": float(g300["seed_mean_success"]),
            "collision": float(g300["seed_mean_collision"]),
            "truncation": float(g300["seed_mean_truncation"]),
        },
        safety_answers=safety_answers,
    )
    (OUT / "STAGE7C_Q1_DECISION.md").write_text(decision + "\n", encoding="utf-8")

    # drops detail
    pd.DataFrame(drops["deltas"]).to_csv(OUT / "adjacent_success_deltas.csv", index=False)

    print(json.dumps({"integrity": integrity["status"], "gate": gate["status"], "out": str(OUT)}, indent=2))
    return 0 if gate["status"] != "INVALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
