"""Stage 11 pilot (E30) v12 -- joint (centralised) DQN, two independent
per-role heads sharing one trunk and one optimiser.

v1-v11 all used ONE learner per role (either one shared network for both
roles, v1-v5, or two fully independent per-role networks, v6-v11) -- both
variants train each role's Q-function against a target that depends on the
OTHER role's (co-evolving) policy, making the effective transition process
each learner sees non-stationary during training (exactly the theoretical
framing this project's own theory chapter uses, ``chap02.tex`` Section 2.1:
"each learner treats the other's changing policy as part of its
environment... the effective transition process each controller sees is
non-stationary during training"). Eleven rounds of algorithmic patches on
top of that two-independent-learner premise (role-split architecture,
periodic resets, visit-count exploration, hysteretic TD-error weighting,
a symmetry-breaking observation signal) did not reliably resolve it.

v12 abandons the two-independent-learner premise instead of patching it:
ONE network sees the JOINT observation (both vehicles' local observations,
concatenated in ROLE order -- ramp first, then mainline -- not physical
vehicle-id order, since role is randomised per episode but must map to a
consistent output head) and outputs two per-role action-value heads. There
is only ever one set of parameters being updated by one optimiser -- from
this network's own perspective there is no "other learner" whose policy is
silently drifting underneath it; the two-agent coordination problem is
re-expressed as a single-agent problem with a two-headed action space.

This deliberately gives up the "two independent, persistent, identity-
swappable controllers" premise Stage 9's own theory chapter (``chap02.tex``
Section 2.1, the controller-identity / D_swap machinery) is built around --
an explicit, user-directed trade-off in favour of task success rate, not an
oversight. See the plan at the time of this change
(``stage11_dyad_merge_pilot_config.py``'s v12 docstring block) for the full
rationale.

Reuses ``independent_dqn_v2.QNetwork``'s exact MLP-layer construction style
and ``dqn_bootstrap``'s masking/Double-DQN bootstrap machinery -- no new
hyperparameter-search machinery invented here. The replay buffer is a new,
JOINT-shaped structure (one row per environment STEP, not per vehicle) since
the existing ``replay_buffer_v2.ReplayBuffer`` is single-vehicle-shaped.

Prioritized Experience Replay + N-step returns (this round's addition):
two of Rainbow DQN's six components (Hessel et al. 2018, "Rainbow:
Combining Improvements in Deep Reinforcement Learning", AAAI -- read
directly from the paper's own LaTeX source this session, not a paraphrase,
so the citations below can be made with real confidence). Chosen
specifically because Rainbow's OWN ablation study identifies these two as
the most load-bearing of the six: removing either caused a large, broadly
consistent drop in median performance across 53 of 57 Atari games, while
the other four components (Dueling, Distributional RL, Noisy Nets, and to a
lesser, confounded extent Double Q-learning, which this project already
uses) showed smaller or mixed per-game effects in that same ablation.
Two things are this project's OWN adaptation, not a literal port of
Schaul et al. (2015)'s single-agent PER or the standard single-agent
n-step formulation, and are flagged as such at the exact point they are
decided, below:
  (a) this is a TWO-HEADED joint network, not a single agent, so a single
      transition has TWO TD-errors (one per head) -- there is no canonical
      single-agent answer for how to combine them into one priority;
  (b) n-step windows can be truncated by an episode boundary before a full
      n-step accumulates, which the standard fixed-n formulation does not
      have to handle, requiring a per-transition (not global) step count.

N-vehicle generalisation (Study B pilot, see
``experiments/pilots/stage11_dyad_merge_pilot_v12_joint_network/STAGE11B_MULTI_VEHICLE_PILOT_DESIGN.md``
for the full design): every class in this module now accepts
``n_vehicles`` in ``{2, 4, 6}`` (default 2, byte-identical to the original
two-role-one-vehicle-each design when left at the default). ``per_role =
n_vehicles // 2`` vehicles per role; the network gains one output head per
VEHICLE SLOT (not per role) -- ``n_vehicles`` heads sharing one trunk --
so the "one network, one optimiser, one gradient step" property that
motivated v12 in the first place is unchanged at any N. Slot order is
role-major (all ramp slots, then all mainline slots), matching
``stage10_symmetric_merge_env.py``'s own spawn-queue ordering rule within
each role, so slot ``(role, i)`` has a stable rule-based meaning every
episode even though the physical vehicle occupying it varies (roles are
re-randomised every reset, same as the original N=2 design already did).
At ``n_vehicles=2`` the network's parameter names (``head_ramp``/
``head_mainline``) are preserved EXACTLY (not folded into a generic
``ModuleList``) specifically so Study A's already-trained, already-PASSED
checkpoints keep loading via plain ``state_dict`` matching -- this is a
deliberate backward-compatibility carve-out, not an oversight; see
``JointQNetwork.__init__``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from thesis.agents.action_masking import (
    masked_argmax,
    masked_random_action,
    validate_action_mask,
)
from thesis.agents.dqn_bootstrap import DQNTargetMode, compute_bootstrap_values

ROLE_ORDER: tuple[str, ...] = ("ramp", "mainline")


def beta_at_step(step: int, *, total_steps: int, beta_start: float, beta_end: float) -> float:
    """Linear anneal of PER's importance-sampling exponent beta from
    ``beta_start`` to ``beta_end`` over ``total_steps`` -- Hessel et al.
    (2018)'s own published schedule (0.4 -> 1.0), read directly from the
    paper's hyperparameter table this session. Deliberately takes
    ``total_steps`` as an explicit argument rather than importing Stage 11's
    own ``MAX_STEPS`` -- this module stays agnostic of any one caller's
    training-budget constant; the runner supplies the real schedule."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    frac = min(1.0, max(0.0, float(step) / float(total_steps)))
    return float(beta_start + frac * (beta_end - beta_start))


@dataclass
class JointDQNConfig:
    """Joint-network configuration. ``per_vehicle_obs_dim`` is ONE vehicle's
    own observation width (13 today, role/zone features included, same as
    v1-v11's ``OBS_DIM``) -- the network's actual input width is
    ``2 * per_vehicle_obs_dim`` (see ``JointQNetwork``), computed internally
    so every other caller of this config only ever has to think in
    per-vehicle terms, matching how every prior version's ``DQNConfig`` is
    already used.

    ``prioritized_replay``/``per_alpha``/``per_beta_start``/``per_beta_end``
    (default off -- every existing caller unaffected): Prioritized
    Experience Replay, Hessel et al. (2018)'s own published defaults
    (alpha=0.5, beta 0.4->1.0), see module docstring.

    ``n_vehicles`` (default 2, every existing caller unaffected): total
    vehicle count, ``per_role = n_vehicles // 2`` per role. Determines
    both ``joint_obs_dim`` (below) and the number of output heads (see
    ``JointQNetwork``)."""

    per_vehicle_obs_dim: int
    n_actions: int
    hidden_sizes: tuple[int, ...]
    learning_rate: float
    gamma: float
    epsilon: float
    replay_capacity: int
    batch_size: int
    device: str = "cpu"
    target_mode: DQNTargetMode | str = DQNTargetMode.DOUBLE
    prioritized_replay: bool = False
    per_alpha: float = 0.5
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    n_vehicles: int = 2

    def validate(self) -> None:
        if self.per_vehicle_obs_dim <= 0:
            raise ValueError("per_vehicle_obs_dim must be > 0")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be > 0")
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must satisfy 0 <= gamma < 1, got {self.gamma}")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.replay_capacity <= 0:
            raise ValueError("replay_capacity must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.per_alpha < 0.0:
            raise ValueError("per_alpha must be >= 0")
        if not (0.0 <= self.per_beta_start <= 1.0) or not (0.0 <= self.per_beta_end <= 1.0):
            raise ValueError("per_beta_start/per_beta_end must be in [0, 1]")
        if self.n_vehicles not in (2, 4, 6):
            raise ValueError(f"n_vehicles must be 2, 4, or 6, got {self.n_vehicles}")
        self.target_mode = DQNTargetMode(self.target_mode)

    @property
    def joint_obs_dim(self) -> int:
        return self.n_vehicles * self.per_vehicle_obs_dim


class JointQNetwork(nn.Module):
    """Single MLP trunk (``Linear -> ReLU`` per hidden layer, identical shape
    convention to ``independent_dqn_v2.QNetwork``), one output head per
    VEHICLE SLOT (``n_actions`` wide each) instead of ``QNetwork``'s single
    output layer. ``forward`` returns an ``n_vehicles``-tuple of Q-value
    tensors, in slot order (role-major: all ramp slots, then all mainline
    slots -- see module docstring).

    ``n_vehicles=2`` (default) is a special case, not merely the first
    entry of a generic list: the two heads are registered under their
    original attribute names, ``self.head_ramp``/``self.head_mainline``,
    so the resulting ``state_dict`` keys are byte-identical to every
    checkpoint already saved under the pre-generalisation code (Study A's
    24 archived confirmatory checkpoints, and every pilot round before
    it) -- ``load_state_dict`` keeps working unchanged. At ``n_vehicles>2``
    there are no legacy keys to preserve, so heads are a plain
    ``nn.ModuleList`` instead. Either way, ``self._heads`` is the uniform,
    order-preserving accessor every other method in this module uses --
    callers never need to branch on which case they're in."""

    def __init__(self, per_vehicle_obs_dim: int, n_actions: int, hidden_sizes: Sequence[int], n_vehicles: int = 2):
        super().__init__()
        self.n_vehicles = int(n_vehicles)
        joint_obs_dim = per_vehicle_obs_dim * self.n_vehicles
        layers: list[nn.Module] = []
        prev = joint_obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.trunk = nn.Sequential(*layers)
        if self.n_vehicles == 2:
            self.head_ramp = nn.Linear(prev, n_actions)
            self.head_mainline = nn.Linear(prev, n_actions)
            self._heads: list[nn.Module] = [self.head_ramp, self.head_mainline]
        else:
            self.heads = nn.ModuleList(nn.Linear(prev, n_actions) for _ in range(self.n_vehicles))
            self._heads = list(self.heads)

    def forward(self, joint_obs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h = self.trunk(joint_obs)
        return tuple(head(h) for head in self._heads)


class _JointHeadView(nn.Module):
    """Thin, parameter-sharing wrapper exposing ONE head of a
    ``JointQNetwork`` as a plain single-head module (``forward`` returns one
    ``[B, n_actions]`` tensor), so ``dqn_bootstrap.compute_bootstrap_values``
    -- written for a module whose forward returns exactly one Q-value tensor
    -- can be reused unchanged for Double-DQN bootstrap computation, called
    once per head, instead of duplicating that logic. Registers the wrapped
    network as a submodule (so its parameters are the SAME tensors, not a
    copy) but is only ever used transiently inside ``torch.no_grad()``
    blocks -- it is never itself attached to an optimiser."""

    def __init__(self, joint_net: JointQNetwork, head_index: int):
        """``head_index`` ranges over ``range(joint_net.n_vehicles)`` (was
        ``{0, 1}`` before the N-vehicle generalisation) -- indexes into
        ``forward()``'s returned tuple, so any valid slot index works
        unchanged regardless of ``n_vehicles``."""
        super().__init__()
        self.joint_net = joint_net
        self.head_index = head_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.joint_net(x)[self.head_index]


@dataclass
class JointReplayTransition:
    """One row per environment STEP (not per vehicle) -- the joint learner
    sees the whole N-vehicle interaction as a single decision each step.

    ``actions``/``rewards``/``action_masks``/``next_action_masks`` are
    fixed-length-``n_vehicles`` tuples in SLOT order (role-major -- see
    module docstring), generalising the original scalar
    ``action_ramp``/``action_mainline`` (etc.) pair -- at ``n_vehicles=2``
    a 2-tuple in that same order is exactly equivalent to the original
    fields.

    ``n_steps`` (default 1, byte-identical to pre-n-step behaviour): the
    number of REAL environment steps this transition's reward actually
    spans and the bootstrap horizon (``gamma ** n_steps``) -- see module
    docstring point (b). A transition emitted near a true episode boundary
    can have ``n_steps`` shorter than the caller's configured window; this
    is why it lives on the transition, not as a fixed buffer-wide constant."""

    joint_observation: np.ndarray
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    next_joint_observation: np.ndarray | None
    terminated: bool
    truncated: bool
    action_masks: tuple[np.ndarray, ...]
    next_action_masks: tuple[np.ndarray, ...] | None
    controller_terminal: bool = False
    n_steps: int = 1

    def validate(self, *, n_actions: int, joint_obs_dim: int, n_slots: int) -> None:
        terminated = bool(self.terminated)
        truncated = bool(self.truncated)
        cterm = bool(self.controller_terminal)
        if terminated and truncated:
            raise ValueError("invalid transition: terminated=True and truncated=True simultaneously")
        if terminated and not cterm:
            raise ValueError("terminated=True requires controller_terminal=True")
        if int(self.n_steps) < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
        self.n_steps = int(self.n_steps)

        obs = np.asarray(self.joint_observation, dtype=np.float64)
        if obs.shape != (joint_obs_dim,):
            raise ValueError(f"joint_observation dim {obs.shape} != expected {(joint_obs_dim,)}")
        if not np.all(np.isfinite(obs)):
            raise ValueError(f"non-finite joint_observation: {obs}")
        self.joint_observation = obs

        if len(self.actions) != n_slots or len(self.rewards) != n_slots or len(self.action_masks) != n_slots:
            raise ValueError(
                f"actions/rewards/action_masks must each have length n_slots={n_slots}, got "
                f"{len(self.actions)}/{len(self.rewards)}/{len(self.action_masks)}"
            )

        for r in self.rewards:
            v = float(r)
            if not math.isfinite(v):
                raise ValueError(f"reward must be finite, got {v}")

        validated_masks = []
        for slot, (action, mask) in enumerate(zip(self.actions, self.action_masks)):
            m = validate_action_mask(mask, n_actions)
            a = int(action)
            if a < 0 or a >= n_actions:
                raise ValueError(f"actions[{slot}] {a} out of range for n_actions={n_actions}")
            if not bool(m[a]):
                raise ValueError(f"illegal stored actions[{slot}]={a} under its own action mask")
            validated_masks.append(m)
        self.actions = tuple(int(a) for a in self.actions)
        self.rewards = tuple(float(r) for r in self.rewards)
        self.action_masks = tuple(validated_masks)

        if cterm:
            self.next_joint_observation = None
            self.next_action_masks = None
            return

        if self.next_joint_observation is None:
            raise ValueError("controller_terminal=False requires next_joint_observation")
        if self.next_action_masks is None:
            raise ValueError("controller_terminal=False requires next_action_masks")
        if len(self.next_action_masks) != n_slots:
            raise ValueError(
                f"next_action_masks must have length n_slots={n_slots}, got {len(self.next_action_masks)}"
            )
        nobs = np.asarray(self.next_joint_observation, dtype=np.float64)
        if nobs.shape != (joint_obs_dim,):
            raise ValueError(f"next_joint_observation dim {nobs.shape} != expected {(joint_obs_dim,)}")
        if not np.all(np.isfinite(nobs)):
            raise ValueError(f"non-finite next_joint_observation: {nobs}")
        self.next_joint_observation = nobs
        self.next_action_masks = tuple(validate_action_mask(m, n_actions) for m in self.next_action_masks)


@dataclass
class JointReplayBatch:
    joint_observations: np.ndarray
    actions: np.ndarray  # [B, n_slots], int64
    rewards: np.ndarray  # [B, n_slots], float64
    next_joint_observations: list[np.ndarray | None]
    controller_terminal: np.ndarray
    action_masks: np.ndarray  # [B, n_slots, n_actions], bool
    next_action_masks: list[np.ndarray | None]  # each None or [n_slots, n_actions]
    n_steps: np.ndarray
    # PER-only fields: all-ones / arange respectively when prioritized
    # replay is off, so downstream code (JointDQNLearner.update) can always
    # multiply by importance_weights unconditionally without branching --
    # weight 1.0 exactly reproduces the pre-PER unweighted-mean loss.
    importance_weights: np.ndarray
    buffer_indices: np.ndarray
    transitions: list[JointReplayTransition] = field(default_factory=list)

    @property
    def bootstrap_indices(self) -> np.ndarray:
        return np.flatnonzero(~self.controller_terminal)


class JointReplayBuffer:
    """Circular FIFO replay buffer for joint (per-step, not per-vehicle)
    transitions -- same overwrite-oldest-when-full contract as
    ``replay_buffer_v2.ReplayBuffer``, implemented fresh (not imported)
    because the row shape is genuinely different (one joint row per step,
    two actions/rewards/masks per row, not one row per vehicle).

    ``prioritized`` (default False, byte-identical to pre-PER behaviour when
    off): Prioritized Experience Replay (Schaul et al. 2015; Rainbow's own
    published hyper-parameters used by the caller, see
    ``JointDQNConfig``). When on, ``sample`` draws WITH replacement,
    proportional to ``priority ** alpha`` (a deliberate change from the
    non-prioritized path's uniform-without-replacement sampling -- standard
    PER practice, since sampling is now proportional rather than uniform,
    not a bug)."""

    def __init__(
        self,
        capacity: int,
        *,
        joint_obs_dim: int,
        n_actions: int,
        n_slots: int = 2,
        seed: int = 0,
        prioritized: bool = False,
        per_epsilon: float = 1e-6,
    ):
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.capacity = int(capacity)
        self.joint_obs_dim = int(joint_obs_dim)
        self.n_actions = int(n_actions)
        self.n_slots = int(n_slots)
        self._storage: list[JointReplayTransition | None] = [None] * self.capacity
        self._size = 0
        self._write = 0
        self._rng = np.random.default_rng(int(seed))
        self.seed = int(seed)

        self.prioritized = bool(prioritized)
        self.per_epsilon = float(per_epsilon)
        if self.per_epsilon <= 0.0:
            raise ValueError(f"per_epsilon must be > 0, got {self.per_epsilon}")
        # Priorities are tracked unconditionally (cheap, fixed-size array)
        # even when prioritized=False, so append()'s bookkeeping doesn't
        # need to branch -- they are simply never read by sample() unless
        # prioritized=True.
        self._priorities = np.full(self.capacity, self.per_epsilon, dtype=np.float64)

    def __len__(self) -> int:
        return self._size

    def append(self, transition: JointReplayTransition) -> None:
        transition.validate(n_actions=self.n_actions, joint_obs_dim=self.joint_obs_dim, n_slots=self.n_slots)
        self._storage[self._write] = transition
        if self.prioritized:
            # New transitions enter at the current max priority (standard
            # PER convention) so every transition is guaranteed to be
            # sampled at least once before its real TD-error is known --
            # otherwise a never-yet-scored transition could starve.
            current_max = float(self._priorities[: self._size].max()) if self._size > 0 else self.per_epsilon
            self._priorities[self._write] = max(current_max, self.per_epsilon)
        self._write = (self._write + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, *, alpha: float | None = None, beta: float | None = None) -> JointReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.prioritized:
            # Sampling is WITH replacement here (standard PER practice --
            # proportional sampling is a different regime from the uniform
            # branch's without-replacement draw below), so batch_size is not
            # bounded by buffer size -- only non-emptiness is required.
            if self._size < 1:
                raise ValueError("cannot sample from an empty buffer")
        else:
            if self._size < batch_size:
                raise ValueError(f"cannot sample {batch_size} from buffer of size {self._size}")

        if self.prioritized:
            a = 0.5 if alpha is None else float(alpha)
            b = 1.0 if beta is None else float(beta)
            valid_priorities = self._priorities[: self._size]
            scaled = valid_priorities**a
            probs = scaled / scaled.sum()
            indices = self._rng.choice(self._size, size=batch_size, replace=True, p=probs)
            sample_probs = probs[indices]
            weights = (self._size * sample_probs) ** (-b)
            weights = weights / weights.max()  # normalise to [0, 1], standard PER stabiliser
        else:
            indices = self._rng.choice(self._size, size=batch_size, replace=False)
            weights = np.ones(batch_size, dtype=np.float64)

        start = (self._write - self._size) % self.capacity
        physical = np.asarray([(start + int(i)) % self.capacity for i in indices], dtype=np.int64)
        transitions = [self._storage[p] for p in physical]
        assert all(t is not None for t in transitions)
        return self._stack(transitions, physical, weights)  # type: ignore[arg-type]

    def update_priorities(self, buffer_indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Write back new priorities for the given PHYSICAL buffer indices
        (as returned in ``JointReplayBatch.buffer_indices``), using
        ``priority = |td_error| + per_epsilon``. ``td_errors`` here is
        expected to already be the project's own combined (max-of-both-
        heads) quantity -- see ``JointDQNLearner.update``'s docstring for
        why this project uses max rather than an average across the two
        heads (a design choice this two-headed joint network needs that
        Schaul et al. (2015)'s single-agent formulation does not)."""
        if not self.prioritized:
            raise RuntimeError("update_priorities called on a non-prioritized buffer")
        idx = np.asarray(buffer_indices, dtype=np.int64)
        errs = np.asarray(td_errors, dtype=np.float64)
        if idx.shape != errs.shape:
            raise ValueError(f"buffer_indices shape {idx.shape} != td_errors shape {errs.shape}")
        if not np.all(np.isfinite(errs)):
            raise ValueError("non-finite td_errors passed to update_priorities")
        self._priorities[idx] = np.abs(errs) + self.per_epsilon

    def _stack(
        self, transitions: Sequence[JointReplayTransition], buffer_indices: np.ndarray, weights: np.ndarray
    ) -> JointReplayBatch:
        return JointReplayBatch(
            joint_observations=np.stack([t.joint_observation for t in transitions]),
            actions=np.asarray([t.actions for t in transitions], dtype=np.int64),
            rewards=np.asarray([t.rewards for t in transitions], dtype=np.float64),
            next_joint_observations=[t.next_joint_observation for t in transitions],
            controller_terminal=np.asarray([t.controller_terminal for t in transitions], dtype=bool),
            action_masks=np.stack([np.stack(t.action_masks) for t in transitions]),
            next_action_masks=[
                None if t.next_action_masks is None else np.stack(t.next_action_masks) for t in transitions
            ],
            n_steps=np.asarray([t.n_steps for t in transitions], dtype=np.int64),
            importance_weights=np.asarray(weights, dtype=np.float64),
            buffer_indices=np.asarray(buffer_indices, dtype=np.int64),
            transitions=list(transitions),
        )


class JointDQNLearner:
    """One centralised Double-DQN stack: a single ``JointQNetwork`` (online +
    target), a single Adam optimiser over ALL its parameters (shared trunk +
    both heads), and a single ``JointReplayBuffer``. There is no "other
    learner" from this object's point of view -- both roles' actions are
    chosen and trained by the same parameters in the same optimiser step."""

    def __init__(self, config: JointDQNConfig, *, seed: int, replay_seed: int | None = None):
        config.validate()
        self.config = config
        self.seed = int(seed)
        self.replay_seed = int(self.seed + 17 if replay_seed is None else replay_seed)
        self.device = torch.device(config.device)

        self._rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        self.online = JointQNetwork(
            config.per_vehicle_obs_dim, config.n_actions, config.hidden_sizes, n_vehicles=config.n_vehicles
        ).to(self.device)
        self.target = JointQNetwork(
            config.per_vehicle_obs_dim, config.n_actions, config.hidden_sizes, n_vehicles=config.n_vehicles
        ).to(self.device)
        self.hard_sync_target()
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimiser = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = JointReplayBuffer(
            config.replay_capacity,
            joint_obs_dim=config.joint_obs_dim,
            n_actions=config.n_actions,
            n_slots=config.n_vehicles,
            seed=self.replay_seed,
            prioritized=config.prioritized_replay,
        )
        self._update_count = 0

    def hard_sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def set_learning_rate(self, lr: float) -> None:
        for group in self.optimiser.param_groups:
            group["lr"] = float(lr)

    @torch.no_grad()
    def q_values(self, joint_observation: np.ndarray, *, network: str = "online") -> tuple[np.ndarray, ...]:
        obs = np.asarray(joint_observation, dtype=np.float64)
        if obs.shape != (self.config.joint_obs_dim,):
            raise ValueError(f"joint_observation dim {obs.shape} != expected {(self.config.joint_obs_dim,)}")
        if not np.all(np.isfinite(obs)):
            raise ValueError(f"non-finite joint_observation: {obs}")
        net = self.online if network == "online" else self.target
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_all = net(t)
        result = []
        for q in q_all:
            q_np = q.squeeze(0).cpu().numpy().astype(np.float64)
            if not np.all(np.isfinite(q_np)):
                raise ValueError(f"non-finite Q-values: {q_np}")
            result.append(q_np)
        return tuple(result)

    def select_action(
        self,
        joint_observation: np.ndarray,
        action_masks: Sequence[Sequence[bool] | np.ndarray],
        *,
        epsilon: float | None = None,
        greedy: bool = False,
    ) -> tuple[int, ...]:
        """One forward pass through the joint network; each slot's action is
        then chosen INDEPENDENTLY (its own mask, its own epsilon-greedy draw)
        -- matching how v1-v11's per-role epsilon-greedy selection already
        worked, just against the same shared Q-values rather than separately
        -computed ones. ``action_masks`` is a length-``n_vehicles`` sequence
        in slot order (was two positional ``action_mask_ramp``/
        ``action_mask_mainline`` args before the N-vehicle generalisation --
        at ``n_vehicles=2``, ``(mask_ramp, mask_mainline)`` is the exact
        equivalent). Each slot draws its OWN independent epsilon coin (not
        one shared draw for the whole step) -- unchanged semantics from the
        original two-slot design, just looped."""
        eps = 0.0 if greedy else float(self.config.epsilon if epsilon is None else epsilon)
        masks = [validate_action_mask(m, self.config.n_actions) for m in action_masks]
        if len(masks) != self.config.n_vehicles:
            raise ValueError(f"expected {self.config.n_vehicles} action masks, got {len(masks)}")
        q_all = self.q_values(joint_observation, network="online")

        actions = []
        for slot in range(self.config.n_vehicles):
            if greedy or self._rng.random() >= eps:
                a = masked_argmax(q_all[slot], masks[slot], n_actions=self.config.n_actions)
            else:
                a = masked_random_action(masks[slot], self.config.n_actions, self._rng)
            actions.append(int(a))
        return tuple(actions)

    def store_transition(self, transition: JointReplayTransition) -> None:
        self.replay.append(transition)

    def update(
        self,
        *,
        batch: JointReplayBatch | None = None,
        current_step: int | None = None,
        total_steps: int | None = None,
    ) -> dict[str, Any]:
        """One optimiser step: Double-DQN bootstrap targets computed
        SEPARATELY for each head (via ``compute_bootstrap_values``, reused
        unchanged through the ``_JointHeadView`` wrapper), MSE loss per head
        against its own target, the two losses SUMMED into one scalar and
        backpropagated in a single ``backward()``/``optimiser.step()`` --
        this is what makes the trunk's parameters genuinely shared/joint
        learning, not two independent updates that happen to share storage.

        N-step (default: every transition has ``n_steps=1``, byte-identical
        to pre-n-step behaviour): the bootstrap discount is
        ``gamma ** n_steps`` PER ROW, not a single scalar ``gamma`` --
        transitions emitted near an episode boundary can have a shorter
        real n-step span than the caller's configured window (see
        ``JointReplayTransition``'s docstring).

        Prioritized replay (default off, byte-identical to today's plain
        ``mse_loss`` when ``self.config.prioritized_replay`` is False):
        when on, ``current_step``/``total_steps`` are required (anneal
        PER's beta via ``beta_at_step``), each head's per-sample squared
        error is weighted by the sampled batch's importance weights before
        averaging (NOT ``nn.functional.mse_loss``'s built-in reduction,
        which would ignore the weights), and after the loss is computed the
        RAW per-sample |TD-error| for each slot's head is combined via
        an N-way ``max`` across all ``n_vehicles`` slots (generalising the
        original two-headed ``max(|ramp error|, |mainline error|)``) --
        this project's own design choice, not a literal Rainbow result
        (see module docstring). Known limitation at N>2, not silently
        inherited: the probability that at least one of N per-slot
        TD-errors is currently large grows with N, so this increasingly
        biases replay priority toward "any one slot is off" as N grows;
        kept as the smallest-change default for the N=4/6 pilot anyway --
        see ``STAGE11B_MULTI_VEHICLE_PILOT_DESIGN.md`` Sec 2(c) for the
        full tradeoff discussion and the rejected alternatives (mean,
        top-k-mean) -- and written back into the replay buffer's
        priorities."""
        if batch is None:
            if self.config.prioritized_replay:
                if current_step is None or total_steps is None:
                    raise ValueError("prioritized_replay=True requires current_step and total_steps")
                beta = beta_at_step(
                    current_step,
                    total_steps=total_steps,
                    beta_start=self.config.per_beta_start,
                    beta_end=self.config.per_beta_end,
                )
                batch = self.replay.sample(self.config.batch_size, alpha=self.config.per_alpha, beta=beta)
            else:
                batch = self.replay.sample(self.config.batch_size)

        n_slots = self.config.n_vehicles
        joint_obs = torch.as_tensor(batch.joint_observations, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device)  # [B, n_slots]
        n = len(batch.actions)
        targets = [np.asarray(batch.rewards[:, s], dtype=np.float64).copy() for s in range(n_slots)]
        mode = DQNTargetMode(self.config.target_mode)
        gamma = float(self.config.gamma)
        n_steps_arr = np.asarray(batch.n_steps, dtype=np.int64)

        online_views = [_JointHeadView(self.online, s) for s in range(n_slots)]
        target_views = [_JointHeadView(self.target, s) for s in range(n_slots)]

        with torch.no_grad():
            boot_idx = [int(i) for i in batch.bootstrap_indices.tolist()]
            if boot_idx:
                next_obs_list = []
                next_masks_by_slot: list[list[np.ndarray]] = [[] for _ in range(n_slots)]
                for i in boot_idx:
                    nobs = batch.next_joint_observations[i]
                    nmasks = batch.next_action_masks[i]
                    if nobs is None or nmasks is None:
                        raise ValueError(f"bootstrap row {i} missing next-state data")
                    next_obs_list.append(nobs)
                    for s in range(n_slots):
                        next_masks_by_slot[s].append(np.asarray(nmasks[s], dtype=bool))
                next_obs_t = torch.as_tensor(np.stack(next_obs_list), dtype=torch.float32, device=self.device)
                next_values = []
                for s in range(n_slots):
                    next_mask_t = torch.as_tensor(np.stack(next_masks_by_slot[s]), dtype=torch.bool, device=self.device)
                    next_values.append(
                        compute_bootstrap_values(
                            online_network=online_views[s],
                            target_network=target_views[s],
                            next_observations=next_obs_t,
                            next_action_masks=next_mask_t,
                            mode=mode,
                        )
                    )
                for j, i in enumerate(boot_idx):
                    if bool(batch.controller_terminal[i]):
                        raise ValueError(f"bootstrap index {i} marked controller_terminal")
                    row_gamma_n = gamma ** int(n_steps_arr[i])
                    for s in range(n_slots):
                        nv = float(next_values[s][j].item())
                        if not math.isfinite(nv):
                            raise ValueError(f"non-finite bootstrap value at row {i}, slot {s}: {nv}")
                        targets[s][i] = float(batch.rewards[i, s]) + row_gamma_n * nv

            for i in range(n):
                if bool(batch.controller_terminal[i]):
                    for s in range(n_slots):
                        targets[s][i] = float(batch.rewards[i, s])

        for s in range(n_slots):
            if not np.all(np.isfinite(targets[s])):
                raise ValueError(f"non-finite targets in batch, slot {s}")
        target_t = [torch.as_tensor(targets[s], dtype=torch.float32, device=self.device) for s in range(n_slots)]
        weights_t = torch.as_tensor(batch.importance_weights, dtype=torch.float32, device=self.device)

        q_all = self.online(joint_obs)
        if not all(torch.isfinite(q).all() for q in q_all):
            raise ValueError("non-finite online Q-values")
        q_sa = [q_all[s].gather(1, actions_t[:, s : s + 1]).squeeze(1) for s in range(n_slots)]

        # Per-sample squared error, weighted, then averaged -- with
        # weights_t all exactly 1.0 (the non-PER default) this is
        # numerically identical to nn.functional.mse_loss's mean reduction.
        losses = [(weights_t * (q_sa[s] - target_t[s]) ** 2).mean() for s in range(n_slots)]
        loss = sum(losses)
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss: {loss.item()}")

        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        for p in self.online.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise ValueError("non-finite gradients")
        self.optimiser.step()
        self._update_count += 1

        if self.config.prioritized_replay:
            with torch.no_grad():
                td_by_slot = [(target_t[s] - q_sa[s]).detach().cpu().numpy() for s in range(n_slots)]
            combined_td = np.maximum.reduce([np.abs(td) for td in td_by_slot])
            self.replay.update_priorities(batch.buffer_indices, combined_td)

        return {
            "loss": float(loss.item()),
            "losses": [float(l.item()) for l in losses],
            "update_count": self._update_count,
            "n_bootstrap_rows": int(len(boot_idx)),
            "n_terminal_rows": int(n - len(boot_idx)),
        }


__all__ = [
    "ROLE_ORDER",
    "beta_at_step",
    "JointDQNConfig",
    "JointQNetwork",
    "JointReplayTransition",
    "JointReplayBatch",
    "JointReplayBuffer",
    "JointDQNLearner",
]
