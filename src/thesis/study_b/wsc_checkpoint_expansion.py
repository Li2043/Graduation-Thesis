"""WSC (Welfare-State Communication) checkpoint expansion -- read-only
against the source checkpoint, additive only. Expands a real 18D C64
checkpoint's first-layer (``net.0.weight``) input dimension from 18 to 22,
for BOTH the online and target networks independently, and expands the
corresponding Adam optimizer moment tensors (``exp_avg``/``exp_avg_sq``)
the same way. Every other parameter/optimizer-state tensor is copied
verbatim, unchanged.

CRITICAL CORRECTNESS NOTE (post-incident fix #2 -- see
wsc_formal_campaign_incident_diagnosis.md addendum): the WSC 22D layout
built by ``local_observation.build_local_observation`` does NOT append
the 4 new M features as a contiguous suffix. It INTERLEAVES them: M_i is
inserted right after the self block's 6 original features (so it lands at
index 6, immediately before neighbour features start), and each
neighbour's M_j is inserted at the END of that neighbour's own 5-field
block (indices 11, 16, 21). Consequently the 18 ORIGINAL features do NOT
occupy a contiguous [0:18] prefix of the 22D vector -- from index 6
onward, every original feature is shifted by +1, +2, or +3 positions
depending on which neighbour block it falls in.

An earlier version of this module assumed a simple prefix/suffix split
(``new[:, :18] = old; new[:, 18:] = 0``). That assumption was WRONG and
caused every formal WSC run (v1) to silently feed each old weight column
the WRONG input feature from the very first forward pass (e.g. the
weight column trained to interpret "neighbour_0 presence" was being fed
M_i instead, and so on for every feature from index 6 onward) -- a
data-corruption bug, not a gradient-magnitude/optimizer-stability issue.
This was diagnosed by first ruling out two other hypotheses (Adam
step-count inheritance; a graduated gradient-warmup fix) which had NO
effect on the observed collapse, which correctly pointed away from
"how much do the new weights change" and toward "what are the OLD weights
even being multiplied by."

The mapping below is derived PROGRAMMATICALLY from
``local_observation``'s own layout constants (never hand-written indices
a second time), with a self-test at import time.

Mathematical guarantee this expansion provides (re-verified after the
fix using REAL environment-generated observation pairs, not synthetic
concatenation -- see wsc_validation tests): for the SAME underlying
vehicle/neighbour state, ``Q_new(build_local_observation(..., True), a)
== Q_old(build_local_observation(..., False), a)`` for every action ``a``,
at initialization.

Does NOT modify the source checkpoint file. Does NOT write anywhere by
itself -- callers decide the output path, which must never be a formal
checkpoint directory.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

from thesis.study_b.local_observation import (
    LOCAL_OBS_DIM,
    LOCAL_OBS_DIM_WSC,
    NEIGHBOUR_OBS_DIM,
    NEIGHBOUR_OBS_DIM_WSC,
    NEIGHBOUR_SLOTS,
    SELF_OBS_DIM,
    SELF_OBS_DIM_WSC,
)

__all__ = [
    "WSC_OBS_DIM",
    "ORIGINAL_OBS_DIM",
    "N_NEW_COLUMNS",
    "OLD_TO_NEW_COLUMN_MAP",
    "NEW_ONLY_COLUMNS",
    "expand_state_dict",
    "expand_optimizer_state",
    "expand_checkpoint",
]

ORIGINAL_OBS_DIM = LOCAL_OBS_DIM  # 18
WSC_OBS_DIM = LOCAL_OBS_DIM_WSC  # 22
N_NEW_COLUMNS = WSC_OBS_DIM - ORIGINAL_OBS_DIM
FIRST_LAYER_WEIGHT_KEY = "net.0.weight"  # the only parameter whose shape changes


def _build_column_map() -> tuple[list[int], list[int]]:
    """Derive the OLD-index -> NEW-index mapping directly from
    local_observation's own block-size constants, mirroring
    build_local_observation's own construction order exactly: self block
    first, then NEIGHBOUR_SLOTS neighbour blocks in order. Returns
    (old_to_new, new_only) where old_to_new[i] is the NEW-vector index
    that OLD-vector index i maps to, and new_only lists the NEW-vector
    indices that have no OLD counterpart (must be zero-initialized)."""
    old_to_new: list[int] = []
    new_only: list[int] = []

    # self block: old indices [0, SELF_OBS_DIM) map directly to new indices
    # [0, SELF_OBS_DIM) -- the new M_i feature is appended AFTER them, at
    # new index SELF_OBS_DIM (i.e. SELF_OBS_DIM_WSC - 1).
    for i in range(SELF_OBS_DIM):
        old_to_new.append(i)
    new_only.append(SELF_OBS_DIM)  # M_i

    old_base = SELF_OBS_DIM
    new_base = SELF_OBS_DIM_WSC
    for _slot in range(NEIGHBOUR_SLOTS):
        # each neighbour's original NEIGHBOUR_OBS_DIM fields map directly,
        # offset by the (growing) new_base; that neighbour's M_j is
        # appended after them, at new_base + NEIGHBOUR_OBS_DIM.
        for j in range(NEIGHBOUR_OBS_DIM):
            old_to_new.append(new_base + j)
        new_only.append(new_base + NEIGHBOUR_OBS_DIM)
        old_base += NEIGHBOUR_OBS_DIM
        new_base += NEIGHBOUR_OBS_DIM_WSC

    assert old_base == LOCAL_OBS_DIM, (old_base, LOCAL_OBS_DIM)
    assert new_base == LOCAL_OBS_DIM_WSC, (new_base, LOCAL_OBS_DIM_WSC)
    return old_to_new, new_only


OLD_TO_NEW_COLUMN_MAP, NEW_ONLY_COLUMNS = _build_column_map()

# Self-test at import time: every old index must appear exactly once, every
# new-only index must appear exactly once, together they must exactly
# cover range(WSC_OBS_DIM), and the sizes must match the module's own
# ORIGINAL_OBS_DIM/N_NEW_COLUMNS constants.
assert len(OLD_TO_NEW_COLUMN_MAP) == ORIGINAL_OBS_DIM, OLD_TO_NEW_COLUMN_MAP
assert len(NEW_ONLY_COLUMNS) == N_NEW_COLUMNS, NEW_ONLY_COLUMNS
assert len(set(OLD_TO_NEW_COLUMN_MAP)) == ORIGINAL_OBS_DIM, "duplicate target column in OLD_TO_NEW_COLUMN_MAP"
assert set(OLD_TO_NEW_COLUMN_MAP) | set(NEW_ONLY_COLUMNS) == set(range(WSC_OBS_DIM)), (
    "OLD_TO_NEW_COLUMN_MAP + NEW_ONLY_COLUMNS must exactly cover every WSC column"
)
# Matches the hand-derived mapping documented in the incident diagnosis addendum:
assert OLD_TO_NEW_COLUMN_MAP == [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20], OLD_TO_NEW_COLUMN_MAP
assert NEW_ONLY_COLUMNS == [6, 11, 16, 21], NEW_ONLY_COLUMNS


def expand_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Expand a QNetwork state dict (online OR target -- called
    independently for each, never derives one from the other) from 18D to
    22D input using the CORRECT interleaved column mapping above. Only
    ``net.0.weight`` changes shape; every other tensor is a deep copy of
    the original, unchanged."""
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key == FIRST_LAYER_WEIGHT_KEY:
            old = tensor.detach().clone()
            if old.shape[1] != ORIGINAL_OBS_DIM:
                raise ValueError(
                    f"{FIRST_LAYER_WEIGHT_KEY} has input dim {old.shape[1]}, expected {ORIGINAL_OBS_DIM} "
                    "-- refusing to expand a checkpoint that is not a real 18D Original checkpoint"
                )
            new = torch.zeros((old.shape[0], WSC_OBS_DIM), dtype=old.dtype)
            new[:, OLD_TO_NEW_COLUMN_MAP] = old
            # NEW_ONLY_COLUMNS left at their torch.zeros() default -- explicit for clarity:
            new[:, NEW_ONLY_COLUMNS] = 0.0
            out[key] = new
        else:
            out[key] = tensor.detach().clone()
    return out


def _expand_moment_tensor(name_hint: str, old: torch.Tensor) -> torch.Tensor:
    """Expand ONE Adam moment tensor (exp_avg or exp_avg_sq) that belongs
    to net.0.weight's optimizer slot, using the SAME interleaved mapping."""
    if old.dim() != 2 or old.shape[1] != ORIGINAL_OBS_DIM:
        raise ValueError(f"{name_hint}: expected shape (*, {ORIGINAL_OBS_DIM}), got {tuple(old.shape)}")
    new = torch.zeros((old.shape[0], WSC_OBS_DIM), dtype=old.dtype)
    new[:, OLD_TO_NEW_COLUMN_MAP] = old
    new[:, NEW_ONLY_COLUMNS] = 0.0
    return new


def expand_optimizer_state(
    optimiser_state_dict: dict[str, Any], *, online_state_dict: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Expand an Adam ``optimizer.state_dict()`` so its per-parameter
    moment tensors match the expanded (22D) online network's parameter
    shapes.

    PyTorch's ``Adam.state_dict()`` keys ``state`` by INTEGER parameter
    index (matching ``param_groups[0]["params"]``'s order), not by name --
    this function does NOT assume which integer index corresponds to
    ``net.0.weight``; it determines it by walking ``online_state_dict``
    (an ``nn.Module.state_dict()``, which IS ordered by registration order,
    identical to ``model.parameters()``'s order for a plain
    ``nn.Sequential``) and finding that key's position directly.

    Any state key associated with a parameter OTHER than net.0.weight is
    copied verbatim (deep copy), including any additional per-parameter
    state fields beyond exp_avg/exp_avg_sq/step (e.g. AMSGrad's
    max_exp_avg_sq, if ever enabled) -- handled generically by copying
    every key in that parameter's state dict unchanged, and only
    special-casing exp_avg/exp_avg_sq for the identified net.0.weight
    index. ``step`` and ``param_groups`` (hyperparameters: lr, betas, eps,
    weight_decay, amsgrad) are preserved unchanged throughout."""
    ordered_param_names = list(online_state_dict.keys())
    if FIRST_LAYER_WEIGHT_KEY not in ordered_param_names:
        raise ValueError(f"{FIRST_LAYER_WEIGHT_KEY} not found in online_state_dict keys: {ordered_param_names}")
    net0_weight_index = ordered_param_names.index(FIRST_LAYER_WEIGHT_KEY)

    new_state_dict = copy.deepcopy(optimiser_state_dict)
    state = new_state_dict["state"]
    if net0_weight_index not in state:
        raise ValueError(
            f"optimizer state has no entry for param index {net0_weight_index} "
            f"(expected to correspond to {FIRST_LAYER_WEIGHT_KEY})"
        )

    param_state = state[net0_weight_index]
    handled_keys = set()
    for moment_key in ("exp_avg", "exp_avg_sq"):
        if moment_key in param_state:
            param_state[moment_key] = _expand_moment_tensor(
                f"optimiser.state[{net0_weight_index}].{moment_key}", param_state[moment_key]
            )
            handled_keys.add(moment_key)
    unexpected_unhandled = []
    for key, value in list(param_state.items()):
        if key in handled_keys:
            continue
        if isinstance(value, torch.Tensor) and value.dim() == 2 and value.shape[1] == ORIGINAL_OBS_DIM:
            param_state[key] = _expand_moment_tensor(f"optimiser.state[{net0_weight_index}].{key}", value)
            unexpected_unhandled.append(key)
    if unexpected_unhandled:
        import warnings
        warnings.warn(
            f"wsc_checkpoint_expansion: expanded additional, non-exp_avg/exp_avg_sq optimizer "
            f"state tensor(s) for {FIRST_LAYER_WEIGHT_KEY}: {unexpected_unhandled} "
            f"(e.g. AMSGrad max_exp_avg_sq) -- verify this is expected for this checkpoint.",
            stacklevel=2,
        )
    return new_state_dict


def expand_checkpoint(source_checkpoint_path: Path, *, device: str = "cpu") -> dict[str, Any]:
    """Load a real C64/formal checkpoint (READ-ONLY, source file never
    written to) and return a NEW in-memory checkpoint dict with:
    - ``online``/``target`` expanded to 22D input (independently, target
      NOT derived from online -- see expand_state_dict, called twice),
      using the CORRECT interleaved column mapping;
    - ``optimiser`` expanded to match the expanded online network's shapes;
    - ``update_count`` preserved unchanged;
    - additive WSC provenance metadata (does not remove any existing key).

    Caller is responsible for choosing a non-formal output path if/when
    saving this dict (this function never calls torch.save itself)."""
    ckpt = torch.load(source_checkpoint_path, map_location=device)

    online_old = ckpt["online"]
    target_old = ckpt["target"]
    online_new = expand_state_dict(online_old)
    target_new = expand_state_dict(target_old)  # independent expansion, NOT target_new = online_new

    optimiser_new = expand_optimizer_state(ckpt["optimiser"], online_state_dict=online_old)

    expanded = dict(ckpt)  # preserve every existing key (step, stage, scenario_ids, update_count, replay_size, ...)
    expanded["online"] = online_new
    expanded["target"] = target_new
    expanded["optimiser"] = optimiser_new
    expanded["wsc_obs_dim"] = WSC_OBS_DIM
    expanded["welfare_state_communication"] = True
    expanded["wsc_definition"] = "M_i(t)=running_active_attainment(trace); absolute M values; neutral=1.0"
    expanded["source_checkpoint"] = str(source_checkpoint_path)
    expanded["source_checkpoint_step"] = int(ckpt["step"])
    expanded["source_obs_dim"] = ORIGINAL_OBS_DIM
    return expanded
