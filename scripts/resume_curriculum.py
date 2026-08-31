#!/usr/bin/env python3
"""Advance the R=50m task-only curriculum (M6_R50_audited -> C4_R50
[-> C4_R50ext] -> C16_R50 -> C64_R50) for the two new formal seeds
910101/910102, exactly reproducing the procedure already used for
900101-900104 (documented in configs/FROZEN_EXPERIMENT_CONFIG.md
Section 5 and experiment_records/RUNBOOK.md).

Idempotent and resumable: run it again any time, it inspects existing
checkpoints and does exactly the next needed step, then stops (one
stage-step per invocation, so a crash never loses more than the
in-flight training call). Calls the EXISTING, unmodified
train_curriculum_stage_highwayenv.py and stage_q_ensemble_gate.py --
does not reimplement training or gating logic.

900101-900104 are NOT touched by this script -- they already have a
qualified C64_R50 checkpoint (verify with verify_checkpoints.py)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CHECKPOINTS_ROOT, LOGS, PROJECT_ROOT, SB_SCRIPTS, SCENARIO_BANKS,
    find_latest_checkpoint, load_frozen_config, needs_user_decision, python_exe,
)

NEW_SEEDS = [910101, 910102]

C16_SCENARIO_IDS = ["Q_00000", "Q_00004", "Q_00008", "Q_00012", "Q_00016", "Q_00020", "Q_00024",
                     "Q_00028", "Q_00032", "Q_00036", "Q_00040", "Q_00044", "Q_00048", "Q_00052",
                     "Q_00056", "Q_00060"]
C4_SCENARIO_IDS = ["Q_00000", "Q_00016", "Q_00032", "Q_00048"]


def _all_scenario_ids(bank_path: Path) -> list[str]:
    data = json.loads(bank_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [s["scenario_id"] for s in data]
    return list(data.keys())


def _run_training(*, scenario_bank: Path, scenario_ids: list[str], stage_name: str, seed: int,
                   resume_from: Path | None, start_step: int, max_additional_steps: int,
                   checkpoint_every: int) -> int:
    ckpt_root = CHECKPOINTS_ROOT / "curriculum_910101_910102" / str(seed)
    out_root = ckpt_root  # manifests live alongside checkpoints for this bundle's layout
    cfg = load_frozen_config()
    cmd = [
        python_exe(), str(SB_SCRIPTS / "train_curriculum_stage_highwayenv.py"),
        "--scenario-bank", str(scenario_bank),
        "--scenario-ids", *scenario_ids,
        "--stage-name", stage_name,
        "--master-seed", str(seed),
        "--output-root", str(out_root / stage_name),
        "--checkpoint-root", str(ckpt_root / stage_name),
        "--start-step", str(start_step),
        "--max-additional-steps", str(max_additional_steps),
        "--episode-max-steps", str(cfg["environment"]["episode_max_steps"]),
        "--checkpoint-every", str(checkpoint_every),
        "--device", "cpu",
        "--replay-warmup", "512",
        "--eps-decay-steps-absolute", str(cfg["dqn"]["eps_decay_steps_absolute"]),
        "--lr-decay-steps-absolute", str(cfg["dqn"]["lr_decay_steps_absolute"]),
        "--welfare-lambda", "0.0",
        "--condition", "mean",
        "--action-representation", cfg["environment"]["action_representation"],
        "--local-sensing-range-m", str(cfg["observation"]["local_sensing_range_m"]),
    ]
    if resume_from is not None:
        cmd += ["--resume-from", str(resume_from)]
    log_path = LOGS / f"resume_curriculum_{stage_name}_{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[resume_curriculum] seed={seed} launching {stage_name}: {' '.join(cmd)}")
    with open(log_path, "a", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=f, stderr=subprocess.STDOUT)
    print(f"[resume_curriculum] seed={seed} {stage_name} exited with code {proc.returncode}; log: {log_path}")
    return proc.returncode


def _run_ensemble_gate(*, stage_label: str, stage_end_step: int, checkpoint_stage_name: str,
                        scenario_bank: Path, scenario_ids: list[str], seed: int, output_dir: Path) -> dict:
    cfg = load_frozen_config()
    # stage_q_ensemble_gate expects the directory that *directly contains*
    # ckpt_step_*.pt and whose name includes seed_{seed}_ (see q_ensemble.py).
    # Training writes to .../{stage}/seed_{seed}_{stage}/ckpt_step_*.pt.
    ckpt_root = (CHECKPOINTS_ROOT / "curriculum_910101_910102" / str(seed)
                 / checkpoint_stage_name / f"seed_{seed}_{checkpoint_stage_name}")
    cmd = [
        python_exe(), str(SB_SCRIPTS / "stage_q_ensemble_gate.py"),
        "--stage-label", f"{stage_label}_{seed}",
        "--stage-end-step", str(stage_end_step),
        "--checkpoint-stage-name", checkpoint_stage_name,
        "--scenario-bank", str(scenario_bank),
        "--scenario-ids", *scenario_ids,
        "--seed-checkpoint-roots", f"{seed}:{ckpt_root}",
        "--local-sensing-range-m", str(cfg["observation"]["local_sensing_range_m"]),
        "--output-dir", str(output_dir),
    ]
    print(f"[resume_curriculum] running ensemble gate: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    print(proc.stdout[-2000:])
    report_path = output_dir / f"{stage_label}_{seed}_Q_ENSEMBLE_GATE.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    return {"error": "gate report not found", "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}


def _training_window_completion(seed: int, stage_name: str) -> list[tuple[int, float]]:
    """Parse [stage] step=... completion=... lines from this stage's
    resume_curriculum log for the honest improving/declining check."""
    log_path = LOGS / f"resume_curriculum_{stage_name}_{seed}.log"
    if not log_path.exists():
        return []
    points = []
    pattern = re.compile(rf"\[{re.escape(stage_name)}\] step= *(\d+) +completion=([\d.]+)")
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.search(line)
        if m:
            points.append((int(m.group(1)), float(m.group(2))))
    return points


def advance_seed(seed: int) -> str:
    """Returns one of: 'advanced', 'stopped_for_decision', 'complete_qualified',
    'complete_failed_all_seeds_kept'."""
    cfg = load_frozen_config()
    root = CHECKPOINTS_ROOT / "curriculum_910101_910102" / str(seed)

    m6_dir = root / "M6_R50_audited" / f"seed_{seed}_M6_R50_audited"
    latest_m6 = find_latest_checkpoint(m6_dir)
    m6_step = latest_m6[0] if latest_m6 else 0

    if m6_step < 400000:
        remaining = 400000 - m6_step
        resume_from = latest_m6[1] if latest_m6 and m6_step > 0 else None
        rc = _run_training(scenario_bank=SCENARIO_BANKS / "Q.json", scenario_ids=["Q_00000"],
                            stage_name="M6_R50_audited", seed=seed, resume_from=resume_from,
                            start_step=m6_step, max_additional_steps=remaining, checkpoint_every=25000)
        if rc != 0:
            needs_user_decision(
                issue=f"M6_R50_audited training subprocess for seed {seed} exited with code {rc}.",
                evidence=f"See logs/resume_curriculum_M6_R50_audited_{seed}.log",
                options=["Inspect the log for a crash/exception and fix the environment, then rerun.",
                         "If this is a genuine technical failure (not a scientific result), rerun this script."],
                consequences="No checkpoint was silently treated as valid; nothing scientific was changed.",
                recommendation="Read the log tail first -- this is almost always an environment/dependency issue, not a training bug.")
        return "advanced"

    # M6 complete at 400000 -- classify (four-tier rule), same thresholds used for 900101-900104.
    points = _training_window_completion(seed, "M6_R50_audited")
    final_completion = points[-1][1] if points else None
    c4_dir = root / "C4_R50" / f"seed_{seed}_C4_R50"
    if find_latest_checkpoint(c4_dir) is None:
        if final_completion is None:
            needs_user_decision(
                issue=f"M6_R50_audited for seed {seed} reached 400000 steps but no training-window "
                      "completion could be parsed from the log to classify STRONG/LEARNABLE_WITH_VARIANCE/"
                      "INSUFFICIENT/FAIL.",
                evidence=f"logs/resume_curriculum_M6_R50_audited_{seed}.log",
                options=["Manually inspect the log and classify by hand, per RUNBOOK's four-tier table.",
                         "Re-run M6 if the log is missing/corrupted (genuine technical failure only)."],
                consequences="Cannot safely decide whether to proceed to C4 without this classification.",
                recommendation="Read the log; if completion is clearly >=0.60 by 400K with a sustained rising "
                                "trend and no collapse, LEARNABLE_WITH_VARIANCE or STRONG both proceed to C4 "
                                "automatically -- only INSUFFICIENT/FAIL requires a different path.")
        if final_completion < 0.60:
            needs_user_decision(
                issue=f"Seed {seed}'s M6_R50_audited final completion ({final_completion:.3f}) is below the "
                      "0.60 floor that would even count as INSUFFICIENT/FAIL-adjacent learnability.",
                evidence=f"logs/resume_curriculum_M6_R50_audited_{seed}.log, final completion={final_completion:.3f}",
                options=["Do not proceed to C4 for this seed -- this is the same situation the frozen "
                         "protocol's M6-R4 HARD STOP branch covers.",
                         "Consult RUNBOOK.md's M6 recovery ladder (M6-R1/R2/R3) before deciding."],
                consequences="Proceeding to C4 with a non-learnable base seed would waste significant compute "
                              "and produce meaningless curriculum results.",
                recommendation="Do NOT auto-proceed. This is a genuine scientific judgment call requiring the "
                                "same care given to 900101-900104's original M6 classification.")
        print(f"[resume_curriculum] seed={seed} M6 final completion={final_completion:.3f} "
              "-- proceeding to C4 (LEARNABLE_WITH_VARIANCE-or-better range).")
        rc = _run_training(scenario_bank=SCENARIO_BANKS / "C4.json", scenario_ids=C4_SCENARIO_IDS,
                            stage_name="C4_R50", seed=seed, resume_from=latest_m6[1],
                            start_step=400000, max_additional_steps=200000, checkpoint_every=50000)
        return "advanced" if rc == 0 else "stopped_for_decision"

    # C4 complete at 600000 (or already extended) -- run the gate.
    latest_c4 = find_latest_checkpoint(c4_dir)
    c4ext_dir = root / "C4_R50ext" / f"seed_{seed}_C4_R50ext"
    latest_c4ext = find_latest_checkpoint(c4ext_dir)
    c16_dir = root / "C16_R50" / f"seed_{seed}_C16_R50"

    if find_latest_checkpoint(c16_dir) is None:
        if latest_c4ext is not None and latest_c4ext[0] >= 700000:
            # NOTE: deliberately does NOT call stage_q_ensemble_gate.py here --
            # its single --checkpoint-stage-name flag cannot express a window
            # spanning two stage names (would raise inside load_ensemble_agents's
            # own stage-tag validation). Stop and hand this to a human/adapted script.
            needs_user_decision(
                issue=f"Seed {seed} finished the C4_R50 +100K extension to 700000. The gate window "
                      "K(700000)={550000,600000,650000,700000} spans TWO stage names (550000 under C4_R50, "
                      "600000-700000 under C4_R50ext) -- the same situation 900101-900104 hit. "
                      "stage_q_ensemble_gate.py's single --checkpoint-stage-name flag cannot express this.",
                evidence="See experiment_records/RUNBOOK.md Amendment 15 and "
                         "project/experiments/pilots/study_b_fairness_mappo/scripts/"
                         "c4_r50_mixed_stage_ensemble_gate.py (the one-off script used for 900101-900104).",
                options=["Adapt c4_r50_mixed_stage_ensemble_gate.py for this seed's checkpoint paths "
                         "(straightforward -- same pattern, different seed number).",
                         "Call run_ensemble_gate_for_seed() directly with a per-step expected_stage_by_step dict."],
                consequences="The gate result determines whether this seed proceeds to C16 -- do not guess.",
                recommendation="Reuse the exact pattern from c4_r50_mixed_stage_ensemble_gate.py; it is designed "
                                "to be adapted per-seed.")
            return "stopped_for_decision"

        # No extension yet -- check 600K training-window trend for the SOFT_PASS-and-improving rule.
        gate = _run_ensemble_gate(stage_label="C4_R50", stage_end_step=600000, checkpoint_stage_name="C4_R50",
                                   scenario_bank=SCENARIO_BANKS / "C4.json", scenario_ids=C4_SCENARIO_IDS,
                                   seed=seed, output_dir=root / "gate_C4_R50")
        per_seed = gate.get("per_seed", {}).get(str(seed), {})
        completion = per_seed.get("completion")
        if completion is None:
            needs_user_decision(
                issue=f"C4_R50 ensemble gate for seed {seed} did not produce a usable result.",
                evidence=json.dumps(gate)[:1500],
                options=["Inspect the gate script's stdout/stderr and checkpoint files directly."],
                consequences="Cannot decide PASS/SOFT_PASS/FAIL/extend without this.",
                recommendation="Check that all 4 window checkpoints (450000,500000,550000,600000) exist and "
                                "are tagged stage='C4_R50'.")
        if completion >= 0.90:
            # Frozen C16_R50 is 700000->950000 after C4ext; a C4 PASS at 600K
            # has no ext stage, so train C16_R50 from 600000 for 350000 steps
            # to reach the same 950000 end (window K(950K) still all C16_R50).
            print(f"[resume_curriculum] seed={seed} C4_R50 PASS ({completion:.3f}) -- proceeding to C16.")
            rc = _run_training(scenario_bank=SCENARIO_BANKS / "C16.json", scenario_ids=C16_SCENARIO_IDS,
                                stage_name="C16_R50", seed=seed, resume_from=latest_c4[1],
                                start_step=600000, max_additional_steps=350000, checkpoint_every=50000)
            return "advanced" if rc == 0 else "stopped_for_decision"
        if completion < 0.75:
            needs_user_decision(
                issue=f"Seed {seed}'s C4_R50 gate result ({completion:.3f}) is a FAIL (<0.75), matching "
                      "RUNBOOK sec 40's Diversity-Recovery (DR1-DR4) branch, not a routine continuation.",
                evidence=json.dumps(per_seed)[:1000],
                options=["Run the DR1-DR4 diagnostic sequence per RUNBOOK sec 40/Amendment 11, same as "
                         "900101-900104's history if this ever happened to them (it didn't -- all 4 were "
                         "SOFT_PASS or better at 600K).",
                         "If the frozen protocol's 'keep all seeds' rule already covers this outcome, "
                         "record it honestly and treat this seed's curriculum as its own scientific result "
                         "(consult the user)."],
                consequences="This is exactly the kind of result that must NOT be silently worked around.",
                recommendation="Do not auto-extend or auto-retry. Report to the user.")
        # SOFT_PASS band: check improving via the honest non-monotonic-oscillation standard.
        pts = _training_window_completion(seed, "C4_R50")
        recent = [c for s, c in pts if s >= 500000]
        clearly_improving = len(recent) >= 2 and recent[-1] > recent[0] and all(
            b - a >= -0.02 for a, b in zip(recent, recent[1:]))  # small negative wobble tolerated, net must rise
        if not clearly_improving:
            needs_user_decision(
                issue=f"Seed {seed}'s C4_R50 gate is SOFT_PASS ({completion:.3f}) and the training-window "
                      f"trend near 600K is NOT clearly improving (points: {recent}).",
                evidence=f"Training-window completion trajectory: {pts}",
                options=["Extend +100K anyway if you judge the noise pattern comparable to 900101-900104's "
                         "own (which all extended uniformly regardless of individual trend, per Amendment 8's "
                         "precedent of extending ALL 4 seeds together when the group is SOFT_PASS).",
                         "Do not extend; accept the SOFT_PASS result and stop this seed's curriculum here, "
                         "consulting RUNBOOK sec 40's FAIL/DR branch if that is more appropriate."],
                consequences="900101-900104 were extended as a GROUP without per-seed trend-checking "
                              "(Amendment 8) -- applying a stricter per-seed check here for 910101/910102 "
                              "would be an inconsistent standard between the four reused seeds and the two "
                              "new ones. This inconsistency is exactly why this script stops here instead of "
                              "deciding alone.",
                recommendation="Match the 900101-900104 precedent (extend uniformly), unless the user prefers "
                                "a stricter per-seed standard for the new seeds specifically.")
        print(f"[resume_curriculum] seed={seed} C4_R50 SOFT_PASS+improving -- extending +100K.")
        rc = _run_training(scenario_bank=SCENARIO_BANKS / "C4.json", scenario_ids=C4_SCENARIO_IDS,
                            stage_name="C4_R50ext", seed=seed, resume_from=latest_c4[1],
                            start_step=600000, max_additional_steps=100000, checkpoint_every=50000)
        return "advanced" if rc == 0 else "stopped_for_decision"

    # C16 stage (frozen end step 950000)
    latest_c16 = find_latest_checkpoint(c16_dir)
    c64_dir = root / "C64_R50" / f"seed_{seed}_C64_R50"
    if latest_c16 is None or latest_c16[0] < 950000:
        # Incomplete / not-yet-started C16: continue to the frozen 950K end.
        # (A prior orchestration bug launched PASS-path C16 with only +100K;
        # re-running this script must resume rather than stop for a decision.)
        if latest_c16 is None:
            needs_user_decision(
                issue=f"Seed {seed}'s C16_R50 has no checkpoint yet but C4 is complete -- "
                      "expected the PASS/ext path to have launched C16 already.",
                evidence=str(c16_dir),
                options=["Re-run after confirming C4 gate PASS/ext completed.",
                         "Manually launch C16_R50 from the C4 (or C4ext) checkpoint."],
                consequences="Cannot invent a C16 start without knowing which C4 outcome path applied.",
                recommendation="Inspect gate_C4_R50 / C4_R50ext state, then re-run this script.")
            return "stopped_for_decision"
        c16_start = latest_c16[0]
        remaining = 950000 - c16_start
        print(f"[resume_curriculum] seed={seed} C16_R50 incomplete at {c16_start} -- "
              f"continuing +{remaining} to 950000.")
        rc = _run_training(scenario_bank=SCENARIO_BANKS / "C16.json", scenario_ids=C16_SCENARIO_IDS,
                            stage_name="C16_R50", seed=seed, resume_from=latest_c16[1],
                            start_step=c16_start, max_additional_steps=remaining, checkpoint_every=50000)
        return "advanced" if rc == 0 else "stopped_for_decision"
    if find_latest_checkpoint(c64_dir) is None:
        gate = _run_ensemble_gate(stage_label="C16_R50", stage_end_step=950000, checkpoint_stage_name="C16_R50",
                                   scenario_bank=SCENARIO_BANKS / "C16.json", scenario_ids=C16_SCENARIO_IDS,
                                   seed=seed, output_dir=root / "gate_C16_R50")
        per_seed = gate.get("per_seed", {}).get(str(seed), {})
        n_strict_pass = 1 if per_seed.get("gate") == "PASS" else 0
        # Single-seed context: report honestly, do not force a 4-seed CASE_A/B label onto one seed.
        print(f"[resume_curriculum] seed={seed} C16_R50 gate: {per_seed}")
        if per_seed.get("completion", 0) < 0.60:
            needs_user_decision(
                issue=f"Seed {seed}'s C16_R50 gate result ({per_seed.get('completion')}) is very weak.",
                evidence=json.dumps(per_seed), options=["Consult RUNBOOK's C16 recovery guidance."],
                consequences="Proceeding to C64 with a very weak C16 result may waste compute.",
                recommendation="Compare against 900103's own precedent (C16_R50=0.625 FAIL, still proceeded "
                                "to C64 and was retained per the no-cherry-picking rule) before deciding "
                                "whether to proceed anyway.")
        print(f"[resume_curriculum] seed={seed} proceeding to C64_R50.")
        rc = _run_training(scenario_bank=SCENARIO_BANKS / "Q.json",
                            scenario_ids=_all_scenario_ids(SCENARIO_BANKS / "Q.json"),
                            stage_name="C64_R50", seed=seed, resume_from=latest_c16[1],
                            start_step=950000, max_additional_steps=250000, checkpoint_every=50000)
        return "advanced" if rc == 0 else "stopped_for_decision"

    # C64 stage: final gate.
    latest_c64 = find_latest_checkpoint(c64_dir)
    if latest_c64 is None or latest_c64[0] < 1200000:
        needs_user_decision(issue=f"Seed {seed}'s C64_R50 checkpoint state is unexpected.",
                             evidence=str(c64_dir), options=["Inspect manually."],
                             consequences="Cannot determine next action.", recommendation="Check for a crash.")
    gate = _run_ensemble_gate(stage_label="C64_R50", stage_end_step=1200000, checkpoint_stage_name="C64_R50",
                               scenario_bank=SCENARIO_BANKS / "Q.json",
                               scenario_ids=_all_scenario_ids(SCENARIO_BANKS / "Q.json"),
                               seed=seed, output_dir=root / "gate_C64_R50")
    per_seed = gate.get("per_seed", {}).get(str(seed), {})
    print(f"[resume_curriculum] seed={seed} FINAL C64_R50 gate: {per_seed}")
    print(f"[resume_curriculum] seed={seed} curriculum build COMPLETE (per the frozen 'keep all seeds' rule, "
          "this result is used regardless of PASS/SOFT_PASS/FAIL -- see FROZEN_EXPERIMENT_CONFIG.md sec 7).")
    return "complete_qualified" if per_seed.get("completion", 0) >= 0.75 else "complete_failed_all_seeds_kept"


def main() -> int:
    for seed in NEW_SEEDS:
        print(f"\n=== seed {seed} ===")
        status = advance_seed(seed)
        print(f"[resume_curriculum] seed={seed} status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
