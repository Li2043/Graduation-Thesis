"""Stage 11 formal confirmatory protocol -- frozen constants (DRAFT).

Companion to `STAGE11_PROTOCOL.md`
(`stage11_dyad_merge_pilot_v12_joint_network/STAGE11_PROTOCOL.md`). This
module holds only the constants that protocol document's proposed design
depends on -- it does not itself declare the protocol "frozen" (that is a
user decision, per the protocol doc's Status section).
"""

from __future__ import annotations

from thesis.pilots.stage11_dyad_merge_pilot_config import STAGE11_RESERVED_SEED_BLOCK

PROTOCOL_TAG = "stage11-confirmatory-v1"

# Proposed formal seed blocks (STAGE11_PROTOCOL.md Sec 4) -- contiguous,
# disjoint from STAGE11_RESERVED_SEED_BLOCK (69001-69120, all Stage 11
# pilot iterations v1-v12).
BASELINE_SEEDS: tuple[int, ...] = tuple(range(69121, 69129))
MEAN_PBRS_SEEDS: tuple[int, ...] = tuple(range(69129, 69137))
MIN_PBRS_SEEDS: tuple[int, ...] = tuple(range(69137, 69145))

ALL_CONFIRMATORY_SEEDS: tuple[int, ...] = BASELINE_SEEDS + MEAN_PBRS_SEEDS + MIN_PBRS_SEEDS

MAX_STEPS = 400_000
CHECKPOINT_STEPS: tuple[int, ...] = tuple(range(0, MAX_STEPS + 1, 10_000))
# Corrected 2026-08-10: 375_000 is not a multiple of the 10K checkpoint
# interval (CHECKPOINT_STEPS) and no such file was ever saved -- 370_000 is
# the nearest existing checkpoint. See STAGE11_PROTOCOL.md Sec 5 for the
# same correction note (documentation-bug fix caught before the held-out
# evaluation was run, not a post-hoc criterion change).
GATE_CHECKPOINTS: tuple[int, ...] = (350_000, 370_000, 400_000)

N_VALIDATION_BLOCKS = 8
ASSIGNMENTS_PER_BLOCK = 2  # ramp/mainline swapped
EPISODES_PER_CHECKPOINT = N_VALIDATION_BLOCKS * ASSIGNMENTS_PER_BLOCK  # 16

# Competence gate thresholds (STAGE11_PROTOCOL.md Sec 9.1) -- looser than
# the pilot's own observed baseline (99.0% completion) on purpose: this
# gate exists to catch a regression relative to the pilot, not to
# re-litigate whether 99% is "good enough."
GATE_COMPLETION_MIN = 0.90
GATE_COLLISION_MAX = 0.05
GATE_TRUNCATION_MAX = 0.05
GATE_SEED_INTERSECTION_MIN = 6  # of 8 per condition
GATE_MAX_ADJACENT_DROP = 0.05  # completion_rate, 300K->400K

__all__ = [
    "PROTOCOL_TAG",
    "BASELINE_SEEDS",
    "MEAN_PBRS_SEEDS",
    "MIN_PBRS_SEEDS",
    "ALL_CONFIRMATORY_SEEDS",
    "MAX_STEPS",
    "CHECKPOINT_STEPS",
    "GATE_CHECKPOINTS",
    "N_VALIDATION_BLOCKS",
    "ASSIGNMENTS_PER_BLOCK",
    "EPISODES_PER_CHECKPOINT",
    "GATE_COMPLETION_MIN",
    "GATE_COLLISION_MAX",
    "GATE_TRUNCATION_MAX",
    "GATE_SEED_INTERSECTION_MIN",
    "GATE_MAX_ADJACENT_DROP",
    "STAGE11_RESERVED_SEED_BLOCK",
]
