"""Fix for the WSC formal-campaign collapse documented in
``wsc_formal_campaign_incident_diagnosis.md``.

Root cause (confirmed via three diagnostics: Adam step-count A/B test --
refuted; Original-18D harness control -- ruled out a test-harness bug;
Q-value margin analysis -- confirmed): the C64 checkpoint's decisions ride
on razor-thin Q-value margins (median 0.0042 across 2,000 sampled
observations; 91.7% below 0.01). The 4 new WSC input columns in
``net.0.weight`` are exactly zero at initialization (an exact Q-equivalence
no-op, verified in ``wsc_implementation_validation.md`` TEST 2-4), but
their GRADIENT is not zero -- it is proportional to the (generally
non-zero) M-feature input value and the ordinary backpropagated error
signal. The very first real optimizer update therefore moves these new
weights by an ordinary Adam step size, which is large enough relative to
the measured margins to flip a large fraction of decisions across all
four vehicles within a few thousand steps, before any useful training can
occur.

Fix: linearly ramp the GRADIENT (not the weights, not the learning-rate
schedule shared with the old columns) for the new input columns from 0 to
1 over a configurable number of steps, counted from the start of THIS
continuation (local step, not absolute environment step -- separate from
the existing epsilon/LR schedules, which remain untouched and still
operate on absolute step as before). The old 18 columns' gradient is
NEVER touched (scale is always exactly 1.0 for them) -- this fix only
slows down how quickly the NEW pathway can influence the network, it does
not change how the pre-existing, well-converged pathway continues to
train, and it does not touch the frozen learning-rate/epsilon schedules,
reward formulas, or any other scientific constant.

This is implemented as a backward hook on ``net.0.weight`` (shape
``(hidden, 22)``), not an architecture change -- no new parameters, no
change to hidden sizes, activation, or output dimension.
"""
from __future__ import annotations

import torch

__all__ = ["NewColumnGradientRamp"]


class NewColumnGradientRamp:
    """Scales the gradient of ``net.0.weight[:, n_old_cols:]`` linearly
    from 0 to 1 over ``warmup_steps`` LOCAL steps (steps since this
    continuation began, advanced explicitly via ``advance()`` -- the
    caller is responsible for calling ``advance()`` once per environment
    step, mirroring how ``epsilon_at_step_v12``/``lr_at_step_v12`` already
    use the absolute step counter). Columns ``[:n_old_cols]`` are never
    scaled (always gradient-scale 1.0).

    Register via ``tensor.register_hook(ramp.hook)`` on the relevant
    weight tensor; call ``ramp.advance()`` once per environment step in
    the training loop.
    """

    def __init__(self, *, n_old_cols: int, warmup_steps: int):
        if n_old_cols <= 0:
            raise ValueError(f"n_old_cols must be > 0, got {n_old_cols}")
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps must be > 0, got {warmup_steps}")
        self.n_old_cols = int(n_old_cols)
        self.warmup_steps = int(warmup_steps)
        self.local_step = 0

    def scale(self) -> float:
        return min(1.0, self.local_step / self.warmup_steps)

    def hook(self, grad: torch.Tensor) -> torch.Tensor:
        s = self.scale()
        if s >= 1.0:
            return grad
        new_grad = grad.clone()
        new_grad[:, self.n_old_cols:] *= s
        return new_grad

    def advance(self) -> None:
        self.local_step += 1
