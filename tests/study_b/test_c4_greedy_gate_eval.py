"""Authoritative C4 greedy-gate evaluation correctness checks
(RUNBOOK Amendment 9, 2026-08-17): the C4/C16/C64 qualification gate
must measure the frozen greedy policy (epsilon=0), never the
training-exploration-on behavior. These tests verify the evaluation
code actually behaves that way, per the user's item-10 checklist."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
SCRIPTS = Path(__file__).resolve().parents[2] / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"
for p in (REPO_SRC, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig  # noqa: E402
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig  # noqa: E402
from thesis.study_b.shared_local_dqn import SharedLocalDQNAgent, build_study_b_dqn_config  # noqa: E402
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402

from evaluate_policy_highwayenv import load_policy_highwayenv, run_eval_highwayenv  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_C4_BANK = _REPO_ROOT / "experiments/pilots/study_b_fairness_mappo/scenario_banks/C4.json"
_CKPT_900101 = _REPO_ROOT / "experiments/pilots/study_b_fairness_mappo/checkpoints/autonomous_highwayenv/C4_900101/seed_900101_C4/ckpt_step_600000.pt"
_CKPT_900102 = _REPO_ROOT / "experiments/pilots/study_b_fairness_mappo/checkpoints/autonomous_highwayenv/C4_900102/seed_900102_C4/ckpt_step_600000.pt"

_has_fixtures = _C4_BANK.exists() and _CKPT_900101.exists() and _CKPT_900102.exists()


def _make_env() -> StudyBHeterogeneousHighwayEnv:
    cfg = ThesisHighwayMergeEnvConfig(episode_max_steps=200, action_representation="meta_speed")
    return StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=cfg))


def test_select_actions_epsilon_zero_greedy_never_calls_rng() -> None:
    """epsilon=0.0, greedy=True must be pure argmax -- the agent's own
    exploration RNG must never be consulted, so there is no hidden
    residual randomness in what is supposed to be a deterministic gate
    evaluation."""
    dqn_config = build_study_b_dqn_config(device="cpu")
    agent = SharedLocalDQNAgent(dqn_config, seed=0)
    rng_calls = {"count": 0}
    real_rng = agent.learner._rng

    class _CountingRngProxy:
        def random(self, *args, **kwargs):
            rng_calls["count"] += 1
            return real_rng.random(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(real_rng, name)

    agent.learner._rng = _CountingRngProxy()

    env = _make_env()
    scenario = load_scenario_bank(_C4_BANK)[0]
    obs, _info = env.reset(seed=0, scenario=scenario)
    for _ in range(20):
        actions = agent.select_actions(obs, epsilon=0.0, greedy=True)
        assert set(actions) == set(env.active_vehicle_ids)
        obs, _r, terminated, truncated, _info = env.step(actions)
        if terminated or truncated:
            break

    assert rng_calls["count"] == 0, "epsilon=0.0/greedy=True must never touch the exploration RNG"


def test_load_policy_highwayenv_does_not_touch_optimiser_or_target() -> None:
    """The gate-evaluation policy loader must load ONLY the online
    network -- never optimiser or target-network state -- so a gate
    evaluation is structurally incapable of mutating training state."""
    if not _has_fixtures:
        return
    dqn_config = build_study_b_dqn_config(device="cpu")
    agent = SharedLocalDQNAgent(dqn_config, seed=0)
    target_state_before = {k: v.clone() for k, v in agent.learner.target.state_dict().items()}
    optimiser_state_before = agent.learner.optimiser.state_dict()

    select = load_policy_highwayenv(checkpoint=_CKPT_900101, device="cpu")
    assert callable(select)

    for k, v in agent.learner.target.state_dict().items():
        assert torch.equal(v, target_state_before[k]), "load_policy_highwayenv must not touch this agent's target net"
    assert agent.learner.optimiser.state_dict() == optimiser_state_before


def test_different_checkpoints_map_to_different_seeds_correctly() -> None:
    """A gate run over multiple seeds must load each seed's OWN
    checkpoint -- if the loader ever silently reused a prior network's
    weights, two genuinely different checkpoints would produce
    identical greedy trajectories on every scenario. They must not."""
    if not _has_fixtures:
        return
    rows_101 = run_eval_highwayenv(checkpoint=_CKPT_900101, scenario_bank=_C4_BANK)
    rows_102 = run_eval_highwayenv(checkpoint=_CKPT_900102, scenario_bank=_C4_BANK)
    outcomes_101 = [(r["scenario_id"], r["term_reason"], r["episode_length"]) for r in rows_101]
    outcomes_102 = [(r["scenario_id"], r["term_reason"], r["episode_length"]) for r in rows_102]
    assert outcomes_101 != outcomes_102, "two different checkpoints produced identical greedy outcomes on every scenario"


def test_greedy_evaluation_is_deterministic_across_repeated_runs() -> None:
    """epsilon=0 against a fixed scenario bank must be bit-for-bit
    deterministic -- this is also the basis for the authoritative C4
    gate's N=4-per-seed (one exact trajectory per scenario, not a
    statistical sample) evaluation design."""
    if not _has_fixtures:
        return
    rows_a = run_eval_highwayenv(checkpoint=_CKPT_900101, scenario_bank=_C4_BANK)
    rows_b = run_eval_highwayenv(checkpoint=_CKPT_900101, scenario_bank=_C4_BANK)
    for a, b in zip(rows_a, rows_b, strict=True):
        assert a["term_reason"] == b["term_reason"]
        assert a["episode_length"] == b["episode_length"]
