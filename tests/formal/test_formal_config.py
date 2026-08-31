"""FormalConfig frozen settings."""

from thesis.formal.formal_config import FormalConfig, epsilon_at_step


def test_formal_config_defaults():
    cfg = FormalConfig()
    cfg.validate()
    assert cfg.duration.environment_steps_per_run == 100_000
    assert cfg.exploration.epsilon_decay_environment_steps == 50_000
    assert cfg.num_parallel_training_envs_per_run == 1
    assert cfg.vectorized_training is False
    assert cfg.formal_training_started is False
    assert cfg.dqn.hidden_sizes == (64, 64)
    eps = epsilon_at_step(50_000, cfg.exploration)
    assert eps == 0.10
    assert epsilon_at_step(0, cfg.exploration) == 1.0
    assert abs(epsilon_at_step(25_000, cfg.exploration) - 0.55) < 1e-9
