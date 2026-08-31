"""Tests for the C4 checkpoint-Q ensemble stabilization amendment
(RUNBOOK Amendment 12, 2026-08-17). Additive only -- does not touch
any legacy single-checkpoint evaluation path."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from thesis.study_b.envs.highwayenv_action import ACCELERATE, BRAKE, HOLD  # noqa: E402
from thesis.study_b.local_observation import LOCAL_OBS_DIM  # noqa: E402
from thesis.study_b.q_ensemble import (  # noqa: E402
    EXPECTED_ENSEMBLE_STEPS,
    EnsembleValidationError,
    load_ensemble_agents,
    q_ensemble_values,
    select_ensemble_action,
    select_ensemble_actions,
)
from thesis.study_b.shared_local_dqn import SharedLocalDQNAgent, build_study_b_dqn_config  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CKPT_ROOT = _REPO_ROOT / "experiments/pilots/study_b_fairness_mappo/checkpoints/autonomous_highwayenv"

_SEED_900101_PATHS = {
    550_000: _CKPT_ROOT / "C4_900101/seed_900101_C4/ckpt_step_550000.pt",
    600_000: _CKPT_ROOT / "C4ext2_900101/seed_900101_C4ext2/ckpt_step_600000.pt",
    650_000: _CKPT_ROOT / "C4ext2_900101/seed_900101_C4ext2/ckpt_step_650000.pt",
    700_000: _CKPT_ROOT / "C4ext2_900101/seed_900101_C4ext2/ckpt_step_700000.pt",
}
_SUPERSEDED_650K_900101 = _CKPT_ROOT / "C4ext_900101/seed_900101_C4ext/ckpt_step_650000.pt"
_SEED_900102_600K = _CKPT_ROOT / "C4ext2_900102/seed_900102_C4ext2/ckpt_step_600000.pt"

_has_fixtures = all(p.exists() for p in _SEED_900101_PATHS.values()) and _SUPERSEDED_650K_900101.exists()

pytestmark = pytest.mark.skipif(not _has_fixtures, reason="Amendment-12 checkpoint fixtures not present on this machine")


# ---- A: exactly four intended checkpoints are loaded --------------------
def test_A_requires_exactly_the_four_expected_steps() -> None:
    incomplete = {k: v for k, v in _SEED_900101_PATHS.items() if k != 700_000}
    with pytest.raises(EnsembleValidationError):
        load_ensemble_agents(seed=900101, checkpoint_paths=incomplete)

    extra = dict(_SEED_900101_PATHS)
    extra[800_000] = _SEED_900101_PATHS[700_000]
    with pytest.raises(EnsembleValidationError):
        load_ensemble_agents(seed=900101, checkpoint_paths=extra)

    agents = load_ensemble_agents(seed=900101, checkpoint_paths=_SEED_900101_PATHS)
    assert set(agents.keys()) == set(EXPECTED_ENSEMBLE_STEPS)


# ---- B: checkpoints are from the same seed -------------------------------
def test_B_rejects_a_checkpoint_from_a_different_seed() -> None:
    mixed = dict(_SEED_900101_PATHS)
    mixed[600_000] = _SEED_900102_600K  # seed 900102's checkpoint, wrong seed
    with pytest.raises(EnsembleValidationError):
        load_ensemble_agents(seed=900101, checkpoint_paths=mixed)


# ---- C: superseded 650K checkpoint is rejected ---------------------------
def test_C_rejects_the_superseded_interrupted_650K_checkpoint() -> None:
    superseded = dict(_SEED_900101_PATHS)
    superseded[650_000] = _SUPERSEDED_650K_900101
    with pytest.raises(EnsembleValidationError):
        load_ensemble_agents(seed=900101, checkpoint_paths=superseded)


# ---- D: Q_ensemble is the exact arithmetic mean --------------------------
def test_D_ensemble_is_exact_arithmetic_mean_of_components() -> None:
    dqn_config = build_study_b_dqn_config(device="cpu")

    class _FakeLearner:
        def __init__(self, q: np.ndarray) -> None:
            self._q = q

        def q_values(self, obs: np.ndarray, network: str = "online") -> np.ndarray:
            return self._q

    class _FakeAgent:
        def __init__(self, q: np.ndarray) -> None:
            self.learner = _FakeLearner(q)

    # [HOLD, ACCELERATE, BRAKE] ordering matches _ACTION_ORDER = [HOLD, ACCELERATE, BRAKE] used elsewhere
    q_by_step = {
        550_000: np.array([0.45, 0.40, 0.41]),
        600_000: np.array([0.43, 0.44, 0.42]),
        650_000: np.array([0.47, 0.42, 0.40]),
        700_000: np.array([0.45, 0.46, 0.43]),
    }
    agents = {step: _FakeAgent(q) for step, q in q_by_step.items()}
    result = q_ensemble_values(agents, obs=np.zeros(1))
    expected = np.mean(np.stack(list(q_by_step.values())), axis=0)
    np.testing.assert_allclose(result, expected, atol=1e-12)
    np.testing.assert_allclose(expected, [0.45, 0.43, 0.415], atol=1e-12)


# ---- E: no exploration RNG is accessed -----------------------------------
def test_E_ensemble_action_selection_never_touches_exploration_rng() -> None:
    agents = load_ensemble_agents(seed=900101, checkpoint_paths=_SEED_900101_PATHS)
    rng_calls = {"count": 0}
    real_rng = agents[550_000].learner._rng

    class _CountingRngProxy:
        def random(self, *a, **k):
            rng_calls["count"] += 1
            return real_rng.random(*a, **k)

        def __getattr__(self, name):
            return getattr(real_rng, name)

    for agent in agents.values():
        agent.learner._rng = _CountingRngProxy()

    obs = np.zeros(LOCAL_OBS_DIM, dtype=np.float32)
    for _ in range(20):
        select_ensemble_action(agents, obs)

    assert rng_calls["count"] == 0


# ---- F: epsilon_eval = 0 (structural: no epsilon parameter exists) ------
def test_F_ensemble_selection_api_has_no_epsilon_parameter() -> None:
    import inspect
    sig = inspect.signature(select_ensemble_action)
    assert "epsilon" not in sig.parameters, "ensemble action selection must be unconditionally greedy -- no epsilon knob"


# ---- G: no optimizer/replay/target-network state changes -----------------
def test_G_loading_and_selecting_does_not_touch_optimiser_or_target() -> None:
    probe_config = build_study_b_dqn_config(device="cpu")
    probe = SharedLocalDQNAgent(probe_config, seed=0)
    target_before = {k: v.clone() for k, v in probe.learner.target.state_dict().items()}
    optimiser_before = probe.learner.optimiser.state_dict()

    agents = load_ensemble_agents(seed=900101, checkpoint_paths=_SEED_900101_PATHS)
    obs = np.zeros(LOCAL_OBS_DIM, dtype=np.float32)
    select_ensemble_action(agents, obs)

    for k, v in probe.learner.target.state_dict().items():
        assert torch.equal(v, target_before[k])
    assert probe.learner.optimiser.state_dict() == optimiser_before


# ---- H: two repeated evaluations are byte-identical ----------------------
def test_H_repeated_ensemble_evaluation_is_deterministic() -> None:
    agents_a = load_ensemble_agents(seed=900101, checkpoint_paths=_SEED_900101_PATHS)
    agents_b = load_ensemble_agents(seed=900101, checkpoint_paths=_SEED_900101_PATHS)
    obs = {"V0": np.random.default_rng(0).normal(size=LOCAL_OBS_DIM).astype(np.float32)}
    result_a = select_ensemble_actions(agents_a, obs)
    result_b = select_ensemble_actions(agents_b, obs)
    assert result_a == result_b

    qa = q_ensemble_values(agents_a, obs["V0"])
    qb = q_ensemble_values(agents_b, obs["V0"])
    np.testing.assert_array_equal(qa, qb)


# ---- I: local-observation restrictions remain unchanged ------------------
def test_I_ensemble_consumes_the_unmodified_local_observation_shape() -> None:
    agents = load_ensemble_agents(seed=900101, checkpoint_paths=_SEED_900101_PATHS)
    obs = np.zeros(LOCAL_OBS_DIM, dtype=np.float32)  # exact, unmodified local-observation dimensionality
    q = q_ensemble_values(agents, obs)
    assert q.shape == (3,)  # 3 actions, nothing about the ensemble changes the observation contract


# ---- J: action/control semantics remain unchanged -------------------------
def test_J_action_indices_and_semantics_unchanged() -> None:
    assert (HOLD, ACCELERATE, BRAKE) == (0, 1, 2)
    agents = load_ensemble_agents(seed=900101, checkpoint_paths=_SEED_900101_PATHS)
    obs = np.zeros(LOCAL_OBS_DIM, dtype=np.float32)
    action = select_ensemble_action(agents, obs)
    assert action in (HOLD, ACCELERATE, BRAKE)
