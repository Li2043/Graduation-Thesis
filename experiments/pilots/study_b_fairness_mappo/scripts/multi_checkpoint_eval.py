#!/usr/bin/env python3
"""VDN_Conditional_Amendment_Protocol.md sec 5: evaluate a seed's saved
checkpoints across several training steps (not just the final one) and
classify the resulting completion/collision/timeout trend into one of
the document's four failure types:

  A -- never learns        (completion stays near-zero throughout)
  B -- learns then collapses (reuses analysis.competence.check_no_collapse)
  C -- frozen timeout attractor  (completion~0, collision~0, timeout~1)
  D -- aggressive collision attractor (completion~0, collision~1, timeout~0)

Read-only against already-finished checkpoints -- runs no new training,
does not touch anything a live training job depends on."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(REPO_SRC))
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_policy import run_eval, write_csv  # noqa: E402
from train_joint_dqn_diagnostic import run_eval_joint  # noqa: E402

from thesis.study_b.analysis.competence import check_no_collapse  # noqa: E402
from thesis.study_b.analysis.welfare import seed_level_summary  # noqa: E402

__all__ = ["classify_failure_type", "evaluate_checkpoints"]

DEFAULT_CHECKPOINT_STEPS = (200_000, 400_000, 600_000, 800_000)


def classify_failure_type(checkpoint_metrics: list[dict]) -> str | None:
    """``checkpoint_metrics``: ordered list of dicts, each with at least
    ``step``, ``completion_rate``, ``collision_rate``, ``timeout_rate``
    (one entry per evaluated checkpoint, ascending step order). Returns
    ``None`` if the FINAL checkpoint already qualifies (completion_rate
    >= 0.90 -- see ``check_qualification_gate``'s own default), else one
    of ``"A"``/``"B"``/``"C"``/``"D"``/``"mixed"`` per the document's
    taxonomy (sec 5). This is a coarse classifier meant to point a human
    at the right explanation, not a formal statistical test."""
    if not checkpoint_metrics:
        raise ValueError("checkpoint_metrics must be non-empty")
    final = checkpoint_metrics[-1]
    if final["completion_rate"] >= 0.90:
        return None

    # reach_threshold=0.5 (not check_no_collapse's own default of 0.90):
    # the document's own Type-B illustration peaks at 0.84 -- well short of
    # the formal 0.90 qualification bar -- then collapses to 0.05. Using
    # the qualification-gate's 0.90 here would wrongly call that "never
    # reached, so no collapse to flag". 0.5 is a "clearly learned
    # something" bar for THIS classifier's purpose only, distinct from the
    # 0.90 competence gate itself.
    no_collapse = check_no_collapse(
        [{"step": m["step"], "window": {"completion_rate": m["completion_rate"]}} for m in checkpoint_metrics],
        reach_threshold=0.5,
    )
    if not no_collapse["pass"]:
        return "B"  # learned meaningfully, then dropped after reaching the reach_threshold

    if final["timeout_rate"] >= 0.90 and final["collision_rate"] <= 0.10:
        return "C"
    if final["collision_rate"] >= 0.90 and final["timeout_rate"] <= 0.10:
        return "D"
    if all(m["completion_rate"] < 0.10 for m in checkpoint_metrics):
        return "A"
    return "mixed"


def evaluate_checkpoints(
    *, checkpoint_dir: Path, scenario_bank: Path, output_dir: Path,
    steps: tuple[int, ...] = DEFAULT_CHECKPOINT_STEPS,
    algorithm: str = "dqn", episode_max_steps: int = 200, device: str = "cpu",
) -> dict:
    """Evaluates ``checkpoint_dir/ckpt_step_<step>.pt`` for each ``step``,
    writes one CSV per checkpoint into ``output_dir``, and returns a
    report dict: per-checkpoint metrics (completion/collision/timeout,
    mean episode length, mean undiscounted return, min_U) plus the
    overall failure-type classification."""
    output_dir = Path(output_dir)
    checkpoint_metrics = []
    for step in steps:
        ckpt_path = Path(checkpoint_dir) / f"ckpt_step_{step}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        if algorithm == "joint_dqn":
            rows = run_eval_joint(
                checkpoint=ckpt_path, scenario_bank=scenario_bank,
                episode_max_steps=episode_max_steps, device=device,
            )
        else:
            rows = run_eval(
                algorithm=algorithm, checkpoint=ckpt_path, scenario_bank=scenario_bank,
                episode_max_steps=episode_max_steps, device=device,
            )
        csv_path = output_dir / f"eval_step_{step}.csv"
        write_csv(rows, csv_path)
        summary = seed_level_summary(csv_path)
        n = len(rows)
        checkpoint_metrics.append(
            {
                "step": step,
                "completion_rate": summary["completion_rate"],
                "collision_rate": summary["collision_rate"],
                "timeout_rate": summary["timeout_rate"],
                "mean_U": summary["mean_U"],
                "min_U": summary["min_U"],
                "mean_episode_length": sum(r["episode_length"] for r in rows) / n,
                "mean_undiscounted_return": sum(r["mean_undiscounted_return"] for r in rows) / n,
                "csv": str(csv_path),
            }
        )

    failure_type = classify_failure_type(checkpoint_metrics)
    return {"checkpoint_dir": str(checkpoint_dir), "checkpoints": checkpoint_metrics, "failure_type": failure_type}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, required=True, help="e.g. .../checkpoints/qualification_direct_welfare/seed_900101")
    p.add_argument("--scenario-bank", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_CHECKPOINT_STEPS))
    p.add_argument("--algorithm", type=str, default="dqn", choices=["mappo", "dqn", "joint_dqn"])
    p.add_argument("--episode-max-steps", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args(argv)

    report = evaluate_checkpoints(
        checkpoint_dir=args.checkpoint_dir, scenario_bank=args.scenario_bank, output_dir=args.output_dir,
        steps=tuple(args.steps), algorithm=args.algorithm, episode_max_steps=args.episode_max_steps, device=args.device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "multi_checkpoint_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"failure_type: {report['failure_type']}")
    for m in report["checkpoints"]:
        print(
            f"  step={m['step']:>7}  completion={m['completion_rate']:.3f}  "
            f"collision={m['collision_rate']:.3f}  timeout={m['timeout_rate']:.3f}  "
            f"mean_ep_len={m['mean_episode_length']:.1f}  mean_return={m['mean_undiscounted_return']:.3f}  "
            f"min_U={m['min_U']:.3f}"
        )
    print(f"report written -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
