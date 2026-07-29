"""Stage 3B comfort / hard-braking weight calibration (no policy training)."""

from thesis.calibration.comfort_calibration import (
    CalibrationSelection,
    ComfortCandidate,
    EtaMetrics,
    ThresholdMetrics,
    run_comfort_calibration,
)
from thesis.calibration.trace_loader import SourceTraceManifest, load_and_validate_stage3a_source

__all__ = [
    "CalibrationSelection",
    "ComfortCandidate",
    "EtaMetrics",
    "SourceTraceManifest",
    "ThresholdMetrics",
    "load_and_validate_stage3a_source",
    "run_comfort_calibration",
]
