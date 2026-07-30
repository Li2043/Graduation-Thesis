"""Comfort / hard-braking calibration (Stage 3B and Stage 3B-R1; no policy training)."""

from thesis.calibration.comfort_calibration import (
    CalibrationSelection,
    ComfortCandidate,
    EtaMetrics,
    ThresholdMetrics,
    run_comfort_calibration,
)
from thesis.calibration.final_environment_trace_loader import load_final_environment_lock
from thesis.calibration.joint_comfort_calibration import run_joint_calibration
from thesis.calibration.trace_loader import SourceTraceManifest, load_and_validate_stage3a_source

__all__ = [
    "CalibrationSelection",
    "ComfortCandidate",
    "EtaMetrics",
    "SourceTraceManifest",
    "ThresholdMetrics",
    "load_and_validate_stage3a_source",
    "load_final_environment_lock",
    "run_comfort_calibration",
    "run_joint_calibration",
]
