"""Formal dissertation protocol locks (Stage 5C-0).

Freezes PBRS scales and the training protocol. Does not execute formal training.
"""

from thesis.protocol.final_pbrs_lock import (
    EXPECTED_LAMBDA_MEAN,
    EXPECTED_LAMBDA_MIN,
    build_final_pbrs_lock,
    write_final_pbrs_lock,
)
from thesis.protocol.final_training_protocol import (
    FORMAL_MASTER_SEEDS,
    build_formal_analysis_plan,
    build_formal_run_matrix,
    build_final_training_protocol,
    derive_formal_seeds,
    write_final_training_protocol,
)
from thesis.protocol.prerequisites import (
    STAGE5A0_RUN_ID,
    STAGE5B0_RUN_ID,
    ProtocolBlockedError,
    verify_stage5c0_prerequisites,
)

__all__ = [
    "EXPECTED_LAMBDA_MEAN",
    "EXPECTED_LAMBDA_MIN",
    "FORMAL_MASTER_SEEDS",
    "ProtocolBlockedError",
    "STAGE5A0_RUN_ID",
    "STAGE5B0_RUN_ID",
    "build_final_pbrs_lock",
    "build_final_training_protocol",
    "build_formal_analysis_plan",
    "build_formal_run_matrix",
    "derive_formal_seeds",
    "verify_stage5c0_prerequisites",
    "write_final_pbrs_lock",
    "write_final_training_protocol",
]
