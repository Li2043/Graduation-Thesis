"""Rawlsian DQN policy (V1 core contribution).

Structurally identical to the egoistic baseline: same network, replay buffer,
epsilon-greedy action selection, and TD update. The Rawlsian fairness objective
is *not* an algorithm change; it enters only through the scalar reward injected
by the training loop (see ``v1/rewards/rawlsian_reward.py``).

Consequently this policy is decoupled from the environment state and uses the
inherited, standardised interface:

    remember(state, action, reward, next_state, done)

The reward passed in is expected to be R_rawls = min_i E_i, but the policy
itself neither knows nor cares how the scalar was produced.
"""

from __future__ import annotations

from v1.policies.egoistic_dqn import EgoisticDQN


class RawlsianDQN(EgoisticDQN):
    """DQN trained on an externally supplied Rawlsian scalar reward."""
