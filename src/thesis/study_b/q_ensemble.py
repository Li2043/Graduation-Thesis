"""Additive checkpoint-Q ensemble for the C4 pre-formal stabilization
amendment (RUNBOOK Amendment 12, 2026-08-17) and its frozen
continuation rule for C16/C64/Mean/GGI/Maximin (Amendment 13,
2026-08-17: ``output/highwayenv_migration/
C16_C64_CHECKPOINT_Q_ENSEMBLE_CONTINUATION_AMENDMENT_2026-08-17.md``).

Purely additive: does not modify any legacy single-checkpoint
evaluation path (``evaluate_policy_highwayenv.py``,
``c4_greedy_gate_eval.py``). Defines an equal-weight arithmetic-mean
ensemble over a FIXED set of four frozen checkpoints from the SAME
seed lineage, motivated by the diagnosed SMALL_Q_MARGIN_POLICY_
BOUNDARY_INSTABILITY mechanism
(``output/c4_diagnostics/C4_DIVERSITY_RECOVERY_SYNTHESIS.md``).

The C4 window (550K/600K/clean-650K/700K) is the default -- kept as
the module-level default so all of Amendment 12's existing tests and
callers are unaffected. Amendment 13 generalizes the SAME window width
and spacing (K(S) = {S-150K, S-100K, S-50K, S}) to any future stage
ending at absolute step S (C16: S=950000; C64 and later formal
training: to be filled in at each stage's own completion) --
``load_ensemble_agents`` accepts an explicit ``expected_steps`` /
``expected_stage_by_step`` pair for this; callers that omit them get
the original C4 window/stage-check behavior unchanged.

No training, no checkpoint selection based on outcome, no weighting
other than equal 1/4 per checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from thesis.study_b.local_observation import LOCAL_OBS_DIM
from thesis.study_b.shared_local_dqn import SharedLocalDQNAgent, build_study_b_dqn_config

EXPECTED_ENSEMBLE_STEPS: tuple[int, ...] = (550_000, 600_000, 650_000, 700_000)

# Each C4 ensemble step must come from exactly this training stage --
# this is what rejects the interrupted/superseded "C4ext" 650K
# checkpoint (which has stage=="C4ext", not "C4ext2") without relying
# on file-path string matching alone. This is the DEFAULT stage map
# (C4's own window); later stages pass their own via
# expected_stage_by_step.
_EXPECTED_STAGE_BY_STEP: dict[int, str] = {
    550_000: "C4",
    600_000: "C4ext2",
    650_000: "C4ext2",
    700_000: "C4ext2",
}


def ensemble_window_for_stage_end(final_step: int) -> tuple[int, ...]:
    """K(S) = {S-150K, S-100K, S-50K, S} -- the frozen Amendment-13
    continuation rule, same width/spacing as the successful C4 window."""
    return (final_step - 150_000, final_step - 100_000, final_step - 50_000, final_step)


class EnsembleValidationError(ValueError):
    pass


def load_ensemble_agents(
    *, seed: int, checkpoint_paths: dict[int, Path], device: str = "cpu",
    expected_steps: tuple[int, ...] = EXPECTED_ENSEMBLE_STEPS,
    expected_stage_by_step: dict[int, str] | None = None,
    obs_dim: int = LOCAL_OBS_DIM,
) -> dict[int, SharedLocalDQNAgent]:
    """Loads exactly the four expected checkpoints (test A), verifies
    each belongs to the given seed via its checkpoint directory naming
    convention `seed_{seed}_{stage}` (test B), and verifies each
    checkpoint's saved `stage` field matches the expected stage for
    its step -- which rejects the superseded C4ext 650K checkpoint at
    the C4 window (test C) and any other seed/stage/step mismatch
    defensively at any window. ``expected_steps``/``expected_
    stage_by_step`` default to the original C4 window so all existing
    callers/tests are unaffected; later stages (C16/C64/formal) pass
    their own via ``ensemble_window_for_stage_end`` and a stage-name
    map for that run.

    ``obs_dim`` (WSC addition): defaults to ``LOCAL_OBS_DIM`` (18), so
    every existing caller that does not pass it builds the network with
    EXACTLY the same input dimension as before. WSC evaluation passes
    ``obs_dim=LOCAL_OBS_DIM_WSC`` (22) to load 22D checkpoints instead.
    All four checkpoints in one call share the same ``obs_dim`` (an
    ensemble mixing 18D and 22D checkpoints is not a supported/meaningful
    configuration and is not attempted here)."""
    if expected_stage_by_step is None:
        expected_stage_by_step = _EXPECTED_STAGE_BY_STEP

    if set(checkpoint_paths.keys()) != set(expected_steps):
        raise EnsembleValidationError(
            f"ensemble requires exactly steps {expected_steps}, got {sorted(checkpoint_paths)}"
        )

    agents: dict[int, SharedLocalDQNAgent] = {}
    for step, path in checkpoint_paths.items():
        expected_dir_prefix = f"seed_{seed}_"
        if expected_dir_prefix not in path.parent.name:
            raise EnsembleValidationError(
                f"checkpoint at {path} (dir={path.parent.name}) does not belong to seed {seed} "
                f"(expected directory name to contain '{expected_dir_prefix}')"
            )

        ckpt = torch.load(path, map_location=device)
        if int(ckpt["step"]) != step:
            raise EnsembleValidationError(f"checkpoint step mismatch: expected {step}, file says {ckpt['step']} ({path})")
        expected_stage = expected_stage_by_step[step]
        if ckpt.get("stage") != expected_stage:
            raise EnsembleValidationError(
                f"checkpoint at step {step} has stage={ckpt.get('stage')!r}, expected {expected_stage!r} "
                f"({path}) -- this is the check that rejects a superseded/wrong-lineage checkpoint"
            )

        dqn_config = build_study_b_dqn_config(device=device, obs_dim=obs_dim)
        agent = SharedLocalDQNAgent(dqn_config, seed=0)  # irrelevant: ensemble action selection never touches the RNG
        agent.learner.online.load_state_dict(ckpt["online"])
        agents[step] = agent

    return agents


def q_component_values(agents: dict[int, SharedLocalDQNAgent], obs: np.ndarray) -> dict[int, np.ndarray]:
    """Each component checkpoint's own raw Q-value vector (diagnostic use)."""
    return {step: agent.learner.q_values(obs, network="online") for step, agent in agents.items()}


def q_ensemble_values(agents: dict[int, SharedLocalDQNAgent], obs: np.ndarray) -> np.ndarray:
    """Q_ensemble(o,a) = (1/N) * sum_k Q_theta_k(o,a) -- exact equal-weight
    arithmetic mean over whatever checkpoints `agents` holds (4 for
    every stage under the frozen Amendment-13 rule), no per-checkpoint
    weighting, no normalization. Iterates agents.keys() directly (not
    the C4-specific module constant) so this works unchanged for any
    stage's window."""
    components = q_component_values(agents, obs)
    stacked = np.stack([components[step] for step in sorted(agents.keys())])
    return stacked.mean(axis=0)


def select_ensemble_action(agents: dict[int, SharedLocalDQNAgent], obs: np.ndarray) -> int:
    """argmax over the ensemble-averaged Q-values. Ties broken by
    np.argmax's existing deterministic lowest-index behavior -- no
    hysteresis, margin threshold, or action prior is introduced."""
    return int(np.argmax(q_ensemble_values(agents, obs)))


def select_ensemble_actions(agents: dict[int, SharedLocalDQNAgent], obs: dict[str, np.ndarray]) -> dict[str, int]:
    return {vid: select_ensemble_action(agents, o) for vid, o in obs.items()}
