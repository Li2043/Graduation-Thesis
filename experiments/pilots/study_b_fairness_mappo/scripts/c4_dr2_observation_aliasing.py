#!/usr/bin/env python3
"""RUNBOOK Diversity Recovery DR2 -- local-observation confusion/
aliasing audit (2026-08-17), scoped to the states involved in the
600K<->700K checkpoint outcome flips.

For every divergent (seed, scenario, vehicle) state already identified
by c4_checkpoint_action_flip_analysis.py, tags the observation with the
action that led to the eventual SUCCESS outcome for that trajectory
(the "required-correct" action for that physical situation), then
searches all pairs for near-duplicate observation vectors (small L2
distance) whose required actions differ -- the operational definition
of observation aliasing this audit uses.

Does not modify the observation space. Diagnostic only."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def load_divergence_states(action_flip_dir: Path) -> list[dict]:
    states = []
    for path in sorted(action_flip_dir.glob("FLIP_*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        dv = result["divergence_detail"]
        if dv is None:
            continue
        for vid, v in dv["vehicles"].items():
            if not v["differs"]:
                continue
            if result["outcome_a"] == "success" and result["outcome_b"] != "success":
                required_action = v["action_a_name"]
            elif result["outcome_b"] == "success" and result["outcome_a"] != "success":
                required_action = v["action_b_name"]
            else:
                required_action = None  # ambiguous (both or neither succeeded) -- excluded below
            states.append({
                "seed": result["seed"], "scenario_id": result["scenario_id"], "vehicle_id": vid,
                "t": dv["t"], "obs": np.array(v["obs_a"], dtype=float),
                "required_action": required_action,
                "outcome_a": result["outcome_a"], "outcome_b": result["outcome_b"],
            })
    return [s for s in states if s["required_action"] is not None]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--action-flip-dir", type=Path, required=True)
    p.add_argument("--near-duplicate-threshold", type=float, default=0.05,
                    help="L2 distance below which two observations are considered near-duplicates")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)

    states = load_divergence_states(args.action_flip_dir)
    n = len(states)
    print(f"[DR2] loaded {n} divergence-point observation states with a well-defined required action")

    pairwise_rows = []
    all_distances = []
    near_duplicates_conflicting = []
    near_duplicates_agreeing = []

    for i, j in combinations(range(n), 2):
        a, b = states[i], states[j]
        if a["scenario_id"] == b["scenario_id"]:
            # Same fixed scenario -> t=0 observation is a pure function of the
            # (seed-independent) initial physical state, so different seeds'
            # trajectories on the SAME scenario are trivially observation-
            # identical at the divergence point. Comparing them tests nothing
            # about aliasing (different network weights choosing differently
            # on an IDENTICAL physical situation is the already-characterized
            # Q-margin-instability finding, not an observation question) --
            # excluded here so DR2 only tests genuinely different physical
            # situations, per its own question.
            continue
        dist = float(np.linalg.norm(a["obs"] - b["obs"]))
        all_distances.append(dist)
        same_action = a["required_action"] == b["required_action"]
        row = {
            "i_seed": a["seed"], "i_scenario": a["scenario_id"], "i_vehicle": a["vehicle_id"], "i_required_action": a["required_action"],
            "j_seed": b["seed"], "j_scenario": b["scenario_id"], "j_vehicle": b["vehicle_id"], "j_required_action": b["required_action"],
            "l2_distance": dist, "same_required_action": same_action,
        }
        pairwise_rows.append(row)
        if dist < args.near_duplicate_threshold:
            if same_action:
                near_duplicates_agreeing.append(row)
            else:
                near_duplicates_conflicting.append(row)

    all_distances_sorted = sorted(all_distances)
    percentiles = {
        "min": all_distances_sorted[0], "p5": all_distances_sorted[int(0.05 * len(all_distances_sorted))],
        "p25": all_distances_sorted[int(0.25 * len(all_distances_sorted))],
        "median": all_distances_sorted[len(all_distances_sorted) // 2],
        "max": all_distances_sorted[-1],
    }

    if len(near_duplicates_conflicting) >= 3:
        classification = "STRONG_EVIDENCE_OF_ALIASING"
    elif len(near_duplicates_conflicting) >= 1:
        classification = "WEAK_EVIDENCE_OF_ALIASING"
    else:
        classification = "NO_EVIDENCE_OF_ALIASING"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "C4_DIVERSITY_DR2_OBSERVATION_ALIASING.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairwise_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pairwise_rows)

    report = {
        "classification": classification,
        "n_states": n, "n_pairs": len(pairwise_rows),
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "distance_percentiles": percentiles,
        "n_near_duplicate_pairs_conflicting_action": len(near_duplicates_conflicting),
        "n_near_duplicate_pairs_agreeing_action": len(near_duplicates_agreeing),
        "conflicting_pairs": near_duplicates_conflicting,
        "evidence_note": (
            "Observation aliasing here means: two states whose PHYSICAL situations require DIFFERENT "
            "actions to succeed produce a NEAR-IDENTICAL local observation vector, so the policy input "
            "cannot distinguish them. This is a DIFFERENT question from the checkpoint-to-checkpoint "
            "Q-margin instability already characterized separately (same state, same observation, "
            "different NETWORK WEIGHTS) -- that finding is not evidence of aliasing by itself."
        ),
    }
    (args.output_dir / "C4_DIVERSITY_DR2_OBSERVATION_ALIASING.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [f"# DR2 -- local-observation aliasing audit\n\n**Classification: {classification}**\n"]
    lines.append(f"n_states={n}, n_pairs={len(pairwise_rows)}, near_duplicate_threshold={args.near_duplicate_threshold}\n")
    lines.append(f"Observation L2 distance across all pairs: {percentiles}\n")
    lines.append(f"Near-duplicate pairs with CONFLICTING required action: {len(near_duplicates_conflicting)}\n")
    lines.append(f"Near-duplicate pairs with AGREEING required action: {len(near_duplicates_agreeing)}\n")
    if near_duplicates_conflicting:
        lines.append("\n## Conflicting near-duplicate pairs\n")
        for r in near_duplicates_conflicting:
            lines.append(f"- seed{r['i_seed']}/{r['i_scenario']}/{r['i_vehicle']} (needs {r['i_required_action']}) "
                          f"vs seed{r['j_seed']}/{r['j_scenario']}/{r['j_vehicle']} (needs {r['j_required_action']}) "
                          f"-- dist={r['l2_distance']:.4f}")
    (args.output_dir / "C4_DIVERSITY_DR2_OBSERVATION_ALIASING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[DR2] classification={classification}, conflicting_near_dup={len(near_duplicates_conflicting)}, "
          f"agreeing_near_dup={len(near_duplicates_agreeing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
