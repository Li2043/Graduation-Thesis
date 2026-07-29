"""Load and validate frozen Stage 3A scripted traces for Stage 3B calibration."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


NOMINAL_SAFE = frozenset(
    {"safe_mainline_first", "safe_ramp_first", "safe_near_simultaneous"}
)
SLOW_SAFE = frozenset({"slow_safe_mainline_first", "slow_safe_ramp_first"})
HARD_BRAKING_SAFE = frozenset({"hard_braking_safe"})
NEGATIVE = frozenset(
    {
        "stall_at_start",
        "stall_after_partial_progress",
        "early_collision",
        "late_collision",
    }
)
FIXTURE_EXCLUDED = frozenset(
    {
        "fixture_collision_A",
        "fixture_collision_B",
        "fixture_collision_B_front",
        "fixture_collision_B_rear",
    }
)
# Threshold fitting uses these classes only
THRESHOLD_FIT_SCENARIOS = NOMINAL_SAFE | SLOW_SAFE | HARD_BRAKING_SAFE
# Behavioural eta ranking uses primary non-fixture scenarios
PRIMARY_RANKING_SCENARIOS = NOMINAL_SAFE | SLOW_SAFE | HARD_BRAKING_SAFE | NEGATIVE

REQUIRED_SOURCE_FILES = (
    "transition_trace.jsonl",
    "scenario_outcomes.jsonl",
    "matched_order_pairs.jsonl",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def scenario_class(scenario_id: str) -> str:
    if scenario_id in NOMINAL_SAFE:
        return "nominal_safe"
    if scenario_id in SLOW_SAFE:
        return "slow_safe"
    if scenario_id in HARD_BRAKING_SAFE:
        return "hard_braking_safe"
    if scenario_id in NEGATIVE:
        return "negative"
    if scenario_id in FIXTURE_EXCLUDED or scenario_id.startswith("fixture_"):
        return "fixture"
    if scenario_id in {"oscillation_closed_cycle", "reverse_then_recover"}:
        return "diagnostic"
    return "other"


@dataclass(frozen=True)
class SourceTraceManifest:
    stage3a_run_id: str
    stage3a_git_commit: str
    raw_dir: Path
    file_hashes: dict[str, str]
    summary_overall: str
    summary_git_dirty: bool
    policy_training_started: bool
    dt: float
    gamma: float
    n_transitions: int
    n_outcomes: int


@dataclass
class FilterStats:
    included: int = 0
    excluded_by_reason: dict[str, int] = field(default_factory=dict)

    def exclude(self, reason: str) -> None:
        self.excluded_by_reason[reason] = self.excluded_by_reason.get(reason, 0) + 1


def load_and_validate_stage3a_source(
    *,
    repo_root: Path,
    stage3a_run_id: str,
    expected_git_commit: str,
    dt: float = 0.2,
    gamma: float = 0.995,
) -> tuple[SourceTraceManifest, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load Stage 3A raw outputs; fail if missing, dirty, or not PASS."""
    raw_dir = (
        repo_root
        / "experiments"
        / "pre_impl"
        / "stage3a_scripted_base_outcome_audit"
        / "data"
        / "raw"
        / stage3a_run_id
    )
    reports_summary = (
        repo_root
        / "experiments"
        / "pre_impl"
        / "stage3a_scripted_base_outcome_audit"
        / "reports"
        / stage3a_run_id
        / "stage3a_summary.json"
    )
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Stage 3A raw directory missing: {raw_dir}")
    if not reports_summary.is_file():
        raise FileNotFoundError(f"Stage 3A summary missing: {reports_summary}")

    summary = json.loads(reports_summary.read_text(encoding="utf-8"))
    overall = str(summary.get("overall", summary.get("pass_fail_status", "")))
    git_dirty = bool(summary.get("git_dirty", True))
    policy_training = bool(summary.get("policy_training_started", True))
    if git_dirty:
        raise RuntimeError("Stage 3A source reports git_dirty=true")
    if overall != "PASS":
        raise RuntimeError(f"Stage 3A source overall/status is {overall!r}, expected PASS")
    if policy_training:
        raise RuntimeError("Stage 3A source reports policy_training_started!=false")

    file_hashes: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for name in REQUIRED_SOURCE_FILES:
        p = raw_dir / name
        if not p.is_file():
            raise FileNotFoundError(f"Required Stage 3A source missing: {p}")
        paths[name] = p
        file_hashes[name] = sha256_file(p)
    file_hashes["stage3a_summary.json"] = sha256_file(reports_summary)

    transitions = load_jsonl(paths["transition_trace.jsonl"])
    outcomes = load_jsonl(paths["scenario_outcomes.jsonl"])
    order_pairs = load_jsonl(paths["matched_order_pairs.jsonl"])

    manifest = SourceTraceManifest(
        stage3a_run_id=stage3a_run_id,
        stage3a_git_commit=expected_git_commit,
        raw_dir=raw_dir,
        file_hashes=file_hashes,
        summary_overall=overall,
        summary_git_dirty=git_dirty,
        policy_training_started=policy_training,
        dt=float(dt),
        gamma=float(gamma),
        n_transitions=len(transitions),
        n_outcomes=len(outcomes),
    )
    return manifest, transitions, outcomes, order_pairs


def is_already_completed(row: Mapping[str, Any]) -> bool:
    """True when the learner was already complete before / without this exit event."""
    aid = str(row["controller_id"])
    flags = row.get("completed_flags") or {}
    completed = bool(flags.get(aid, False))
    exit_event = float(row.get("exit_event", 0.0))
    return completed and exit_event < 1.0


def filter_active_calibration_transitions(
    transitions: Sequence[Mapping[str, Any]],
    *,
    approved_scenarios: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], FilterStats]:
    """Include physically active, non-fixture, finite-accel learner transitions."""
    approved = frozenset(approved_scenarios) if approved_scenarios is not None else THRESHOLD_FIT_SCENARIOS
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    stats = FilterStats()

    for row in transitions:
        r = dict(row)
        reason = None
        if bool(r.get("fixture_only", False)) or str(r.get("scenario_id")) in FIXTURE_EXCLUDED:
            reason = "fixture_only"
        elif str(r.get("scenario_id")) not in approved:
            reason = "scenario_not_approved_for_calibration"
        elif str(r.get("controller_id")) not in {"A", "B"}:
            reason = "absent_or_non_learner"
        elif is_already_completed(r):
            reason = "inactive_completed_controller"
        else:
            accel = r.get("realised_acceleration")
            try:
                a = float(accel)
                if not math.isfinite(a):
                    reason = "non_finite_acceleration"
            except (TypeError, ValueError):
                reason = "non_finite_acceleration"

        if reason is not None:
            stats.exclude(reason)
            r["exclude_reason"] = reason
            excluded.append(r)
            continue
        stats.included += 1
        r["scenario_class"] = scenario_class(str(r["scenario_id"]))
        included.append(r)

    return included, excluded, stats
