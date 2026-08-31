#!/usr/bin/env python3
"""One-off authoritative ensemble gate for C4 (R=50m) at K(700000).

Reuses stage_q_ensemble_gate.py's run_ensemble_gate_for_seed() unchanged
-- the only reason this is a separate script is that K(700000) =
{550000, 600000, 650000, 700000} spans TWO stage names for this run
(550000 was saved under stage "C4_R50" before the +100K uniform
extension; 600000/650000/700000 were saved under "C4_R50ext" after it),
so a single --checkpoint-stage-name CLI flag cannot express it.
run_ensemble_gate_for_seed already accepts a full per-step
expected_stage_by_step dict and per-step checkpoint_paths dict, so no
change to that function was needed."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_SRC = Path(__file__).resolve().parents[4] / "src"
for p in (REPO_SRC, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from stage_q_ensemble_gate import EPISODE_CSV_FIELDS, apply_original_gate, run_ensemble_gate_for_seed  # noqa: E402

WINDOW_STEPS = (550000, 600000, 650000, 700000)
STAGE_BY_STEP = {550000: "C4_R50", 600000: "C4_R50ext", 650000: "C4_R50ext", 700000: "C4_R50ext"}
CKPT_ROOT = Path("experiments/pilots/study_b_fairness_mappo/checkpoints/autonomous_highwayenv")
SCENARIO_BANK = Path("experiments/pilots/study_b_fairness_mappo/scenario_banks/C4.json")
SCENARIO_IDS = ["Q_00000", "Q_00016", "Q_00032", "Q_00048"]
OUTPUT_DIR = Path("output/c4_r50_diagnostics/Q_ENSEMBLE_GATE")
SEEDS = [900101, 900102, 900103, 900104]


def checkpoint_paths_for_seed(seed: int) -> dict[int, Path]:
    return {
        550000: CKPT_ROOT / f"C4_R50_{seed}" / f"seed_{seed}_C4_R50" / "ckpt_step_550000.pt",
        600000: CKPT_ROOT / f"C4_R50ext_{seed}" / f"seed_{seed}_C4_R50ext" / "ckpt_step_600000.pt",
        650000: CKPT_ROOT / f"C4_R50ext_{seed}" / f"seed_{seed}_C4_R50ext" / "ckpt_step_650000.pt",
        700000: CKPT_ROOT / f"C4_R50ext_{seed}" / f"seed_{seed}_C4_R50ext" / "ckpt_step_700000.pt",
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_episode_rows: list[dict] = []
    all_diagnostic_rows: list[dict] = []
    per_seed_summary: dict[str, dict] = {}

    for seed in SEEDS:
        checkpoint_paths = checkpoint_paths_for_seed(seed)
        episode_rows, diagnostic_rows = run_ensemble_gate_for_seed(
            seed=seed, checkpoint_paths=checkpoint_paths, window_steps=WINDOW_STEPS,
            expected_stage_by_step=STAGE_BY_STEP, scenario_bank_path=SCENARIO_BANK,
            scenario_ids=SCENARIO_IDS, episode_max_steps=200, device="cpu",
            local_sensing_range_m=50.0,
        )
        all_episode_rows.extend(episode_rows)
        all_diagnostic_rows.extend(diagnostic_rows)

        n = len(episode_rows)
        completion = sum(r["term_reason"] == "success" for r in episode_rows) / n
        collision = sum(r["term_reason"] == "collision" for r in episode_rows) / n
        timeout = sum(r["term_reason"] == "truncation" for r in episode_rows) / n
        gate = apply_original_gate(completion)
        per_seed_summary[str(seed)] = {
            "n_episodes": n, "completion": completion, "collision": collision, "timeout": timeout, "gate": gate,
            "per_scenario": {r["scenario_id"]: {"term_reason": r["term_reason"], "episode_length": r["episode_length"]} for r in episode_rows},
        }
        print(f"[C4_R50_Q_ENSEMBLE_GATE] seed={seed} n={n} completion={completion:.3f} "
              f"collision={collision:.3f} timeout={timeout:.3f} gate={gate}")

    ep_csv = OUTPUT_DIR / "C4_R50_Q_ENSEMBLE_GATE.csv"
    with open(ep_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_episode_rows)

    diag_csv = OUTPUT_DIR / "C4_R50_Q_ENSEMBLE_VS_COMPONENT_DIAGNOSTIC.csv"
    with open(diag_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_diagnostic_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_diagnostic_rows)

    gate_counts = {"PASS": 0, "SOFT_PASS": 0, "FAIL": 0}
    for s in per_seed_summary.values():
        gate_counts[s["gate"]] += 1
    n_strict_pass = gate_counts["PASS"]
    case = "CASE_A" if n_strict_pass >= 3 else "CASE_B_OR_C"
    classification = "QUALIFIED" if n_strict_pass >= 3 else "NOT_QUALIFIED"
    stage_status = "QUALIFIED_FOR_NEXT_CURRICULUM_STAGE" if n_strict_pass >= 3 else "NOT_QUALIFIED"

    report = {
        "label": "C4_R50_AUTHORITATIVE_ENSEMBLE_GATE",
        "stage_end_step": 700000, "window_steps": list(WINDOW_STEPS),
        "stage_by_step": STAGE_BY_STEP,
        "local_sensing_range_m": 50.0,
        "ensemble_definition": f"Q_ensemble(o,a) = (1/4) * sum over {WINDOW_STEPS} of Q_theta_k(o,a); a = argmax_a Q_ensemble(o,a); epsilon=0",
        "n_strict_pass": n_strict_pass, "gate_counts": gate_counts,
        "case": case, "stage_qualification": classification, "stage_status": stage_status,
        "per_seed": per_seed_summary,
        "original_gate_criteria": {"PASS": "completion>=0.90 and collision<=0.05 and timeout<=0.05", "SOFT_PASS": "0.75<=completion<0.90", "FAIL": "completion<0.75"},
        "note": "K(700000) spans two stage names (550000 under C4_R50, 600000/650000/700000 under C4_R50ext, from the uniform +100K extension) -- expressed via a full per-step expected_stage_by_step dict passed directly to run_ensemble_gate_for_seed, not via stage_q_ensemble_gate.py's single-stage-name CLI.",
    }
    (OUTPUT_DIR / "C4_R50_Q_ENSEMBLE_GATE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [f"# C4_R50_Q_ENSEMBLE_GATE -- authoritative epsilon=0 stage gate (K(700000), R=50m)\n"]
    lines.append(f"Window K(700000) = {WINDOW_STEPS}, stage_by_step={STAGE_BY_STEP}\n")
    lines.append(f"**Case: {case} -- C4_R50 = {stage_status}**\n")
    lines.append(f"Strict PASS count: {n_strict_pass}/4. Gate counts: {gate_counts}\n")
    lines.append("\n| seed | n | completion | collision | timeout | gate |")
    lines.append("|---|---|---|---|---|---|")
    for seed, s in per_seed_summary.items():
        lines.append(f"| {seed} | {s['n_episodes']} | {s['completion']:.3f} | {s['collision']:.3f} | {s['timeout']:.3f} | {s['gate']} |")
    (OUTPUT_DIR / "C4_R50_Q_ENSEMBLE_GATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {ep_csv}, {diag_csv}, and companion JSON/MD to {OUTPUT_DIR}")
    print(f"CASE={case} C4_R50_QUALIFICATION={classification} C4_R50={stage_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
