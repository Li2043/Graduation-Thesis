"""Stage 5C-0 prerequisite verification (locks + PASS artifacts).

Uses pilot *engineering integrity* only. Must never read
``pilot_behavioral_observations.csv`` or condition-comparative pilot outcomes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thesis.calibration.final_environment_trace_loader import sha256_file
from thesis.training.final_lock_loader import (
    EXPECTED_COMFORT_LOCK_SHA256,
    EXPECTED_ENVIRONMENT_LOCK_SHA256,
    FinalLockBlockedError,
    load_final_locks,
)

STAGE5A0_RUN_ID = "20260730T015222Z_05e9613c"
STAGE5B0_RUN_ID = "20260730T065218Z_33b6ebf9"

ENV_LOCK_PATH = Path(
    "experiments/pre_impl/stage4a_r1_final_environment_reselection/artifacts/"
    "20260730T003122Z_aee2d425/final_environment_lock.yaml"
)
COMFORT_LOCK_PATH = Path(
    "experiments/pre_impl/stage3b_r1_final_environment_comfort_calibration/artifacts/"
    "20260730T005639Z_c6992dd4/final_comfort_parameters.yaml"
)
STAGE5A0_SUMMARY = Path(
    "experiments/pre_impl/stage5a0_final_v3_end_to_end_integration/reports/"
    f"{STAGE5A0_RUN_ID}/stage5a0_summary.json"
)
STAGE5B0_SUMMARY = Path(
    "experiments/pilot/stage5b0_bounded_engineering_pilot/reports/"
    f"{STAGE5B0_RUN_ID}/stage5b0_summary.json"
)

# Forbidden for protocol construction (comparative / behavioural pilot outcomes).
FORBIDDEN_PILOT_BEHAVIORAL_PATH = Path(
    "experiments/pilot/stage5b0_bounded_engineering_pilot/data/processed/"
    f"{STAGE5B0_RUN_ID}/pilot_behavioral_observations.csv"
)


class ProtocolBlockedError(RuntimeError):
    """Raised when Stage 5C-0 prerequisites cannot be verified (BLOCKED)."""


@dataclass(frozen=True)
class PilotEngineeringIntegrity:
    """Engineering-only pilot facts admissible for protocol promotion."""

    run_id: str
    overall: str
    runs_completed: int
    resume_equivalence_errors: int
    nan_inf_counts: int
    illegal_actions: int
    evaluation_state_mutation_counts: int
    lock_hash_mismatch: int
    engineering_failures: int
    environment_lock_sha256: str
    comfort_lock_sha256: str
    pilot_config_hash: str

    def assert_admissible(self) -> None:
        if self.overall != "PASS":
            raise ProtocolBlockedError(f"Stage 5B-0 overall is not PASS: {self.overall}")
        if self.runs_completed != 6:
            raise ProtocolBlockedError(
                f"Stage 5B-0 runs_completed={self.runs_completed}, expected 6"
            )
        if self.resume_equivalence_errors != 0:
            raise ProtocolBlockedError("Stage 5B-0 resume equivalence failed")
        if self.nan_inf_counts != 0:
            raise ProtocolBlockedError("Stage 5B-0 reported NaN/inf")
        if self.illegal_actions != 0:
            raise ProtocolBlockedError("Stage 5B-0 reported illegal actions")
        if self.evaluation_state_mutation_counts != 0:
            raise ProtocolBlockedError("Stage 5B-0 evaluation mutated training state")
        if self.lock_hash_mismatch != 0 or self.engineering_failures != 0:
            raise ProtocolBlockedError("Stage 5B-0 integrity failures present")


@dataclass(frozen=True)
class Stage5C0Prerequisites:
    environment_lock_path: Path
    environment_lock_sha256: str
    comfort_lock_path: Path
    comfort_lock_sha256: str
    stage5a0_run_id: str
    stage5b0_run_id: str
    stage5a0_overall: str
    pilot_integrity: PilotEngineeringIntegrity
    candidate_id: str
    observation_dimension: int
    a_comfort: float
    a_hard: float
    eta_H: float


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProtocolBlockedError(f"missing prerequisite artifact: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ProtocolBlockedError(f"invalid JSON object: {path}")
    return data


def load_pilot_engineering_integrity(
    summary_path: Path | None = None,
) -> PilotEngineeringIntegrity:
    """Load admissible engineering fields from Stage 5B-0 summary only.

    Explicitly does **not** open ``pilot_behavioral_observations.csv``.
    """
    path = Path(summary_path) if summary_path is not None else STAGE5B0_SUMMARY
    data = _load_json(path)
    if str(data.get("run_id")) != STAGE5B0_RUN_ID:
        raise ProtocolBlockedError(
            f"Stage 5B-0 run_id mismatch: {data.get('run_id')!r} != {STAGE5B0_RUN_ID!r}"
        )
    integrity = data.get("integrity") or {}
    action_counts = data.get("action_replay_integrity_counts") or {}
    return PilotEngineeringIntegrity(
        run_id=str(data["run_id"]),
        overall=str(data.get("overall")),
        runs_completed=int(data.get("runs_completed", -1)),
        resume_equivalence_errors=int(data.get("resume_equivalence_errors", -1)),
        nan_inf_counts=int(data.get("nan_inf_counts", -1)),
        illegal_actions=int(action_counts.get("illegal_actions", -1)),
        evaluation_state_mutation_counts=int(
            data.get("evaluation_state_mutation_counts", -1)
        ),
        lock_hash_mismatch=int(integrity.get("lock_hash_mismatch", -1)),
        engineering_failures=int(integrity.get("engineering_failures", -1)),
        environment_lock_sha256=str(data.get("environment_lock_sha256_after", "")),
        comfort_lock_sha256=str(data.get("comfort_lock_sha256_after", "")),
        pilot_config_hash=str(data.get("pilot_config_hash", "")),
    )


def verify_stage5c0_prerequisites(
    *,
    repo_root: Path | None = None,
) -> Stage5C0Prerequisites:
    root = Path(repo_root) if repo_root is not None else Path(".")
    env_path = root / ENV_LOCK_PATH
    comfort_path = root / COMFORT_LOCK_PATH

    env_sha = sha256_file(env_path)
    comfort_sha = sha256_file(comfort_path)
    if env_sha != EXPECTED_ENVIRONMENT_LOCK_SHA256:
        raise ProtocolBlockedError(
            f"environment lock SHA-256 mismatch: {env_sha} != {EXPECTED_ENVIRONMENT_LOCK_SHA256}"
        )
    if comfort_sha != EXPECTED_COMFORT_LOCK_SHA256:
        raise ProtocolBlockedError(
            f"comfort lock SHA-256 mismatch: {comfort_sha} != {EXPECTED_COMFORT_LOCK_SHA256}"
        )

    try:
        bundle = load_final_locks(
            environment_lock_path=env_path,
            comfort_lock_path=comfort_path,
        )
    except FinalLockBlockedError as exc:
        raise ProtocolBlockedError(str(exc)) from exc

    stage5a0 = _load_json(root / STAGE5A0_SUMMARY)
    if str(stage5a0.get("run_id")) != STAGE5A0_RUN_ID:
        raise ProtocolBlockedError("Stage 5A-0 run_id mismatch")
    if str(stage5a0.get("overall")) != "PASS":
        raise ProtocolBlockedError(
            f"Stage 5A-0 overall is not PASS: {stage5a0.get('overall')}"
        )

    pilot = load_pilot_engineering_integrity(root / STAGE5B0_SUMMARY)
    pilot.assert_admissible()
    if pilot.environment_lock_sha256 != EXPECTED_ENVIRONMENT_LOCK_SHA256:
        raise ProtocolBlockedError("Stage 5B-0 environment lock hash mismatch")
    if pilot.comfort_lock_sha256 != EXPECTED_COMFORT_LOCK_SHA256:
        raise ProtocolBlockedError("Stage 5B-0 comfort lock hash mismatch")

    return Stage5C0Prerequisites(
        environment_lock_path=env_path.resolve(),
        environment_lock_sha256=env_sha,
        comfort_lock_path=comfort_path.resolve(),
        comfort_lock_sha256=comfort_sha,
        stage5a0_run_id=STAGE5A0_RUN_ID,
        stage5b0_run_id=STAGE5B0_RUN_ID,
        stage5a0_overall="PASS",
        pilot_integrity=pilot,
        candidate_id=str(bundle.candidate_id),
        observation_dimension=int(bundle.observation_dimension),
        a_comfort=float(bundle.comfort.a_comfort),
        a_hard=float(bundle.comfort.a_hard),
        eta_H=float(bundle.comfort.eta_H),
    )


__all__ = [
    "COMFORT_LOCK_PATH",
    "ENV_LOCK_PATH",
    "FORBIDDEN_PILOT_BEHAVIORAL_PATH",
    "ProtocolBlockedError",
    "PilotEngineeringIntegrity",
    "STAGE5A0_RUN_ID",
    "STAGE5A0_SUMMARY",
    "STAGE5B0_RUN_ID",
    "STAGE5B0_SUMMARY",
    "Stage5C0Prerequisites",
    "load_pilot_engineering_integrity",
    "verify_stage5c0_prerequisites",
]
