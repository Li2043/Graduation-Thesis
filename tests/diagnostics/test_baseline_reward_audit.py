"""Reward audit script registry."""

from thesis.diagnostics.stage7a0_reward_audit import SCRIPTS


def test_script_registry_contains_stall_and_success_attempts():
    assert "maintain_only" in SCRIPTS
    assert "mutual_yield_decel" in SCRIPTS
    assert "mainline_bias_success_attempt" in SCRIPTS
    assert len(SCRIPTS["maintain_only"](10)) == 10
