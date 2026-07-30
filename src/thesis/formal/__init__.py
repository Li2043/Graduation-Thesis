"""Formal 100K training runtime (Stage 6A-0).

Infrastructure for independent formal jobs. Does not start retained 100K runs
by itself; callers must explicitly invoke training.
"""

from __future__ import annotations

from thesis.formal.formal_config import FormalConfig, derive_formal_job_seeds, epsilon_at_step
from thesis.formal.formal_schedule import FormalICSchedule, evaluation_episode_seed
from thesis.formal.publish import (
    MAX_ORDINARY_GIT_FILE_MIB,
    build_publish_manifest,
    is_publish_allowed,
    reject_oversized_git_files,
)

__all__ = [
    "FormalConfig",
    "FormalICSchedule",
    "MAX_ORDINARY_GIT_FILE_MIB",
    "build_publish_manifest",
    "derive_formal_job_seeds",
    "epsilon_at_step",
    "evaluation_episode_seed",
    "is_publish_allowed",
    "reject_oversized_git_files",
]
