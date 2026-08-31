"""Unit tests for thesis.study_b.dense_shaping (Dense Reward Study scaffold).

Covers: OFF-path is a strict no-op, ON-path requires explicit
magnitude/epsilon (no invented defaults), discrete +c/0/-c boundary
behaviour, and the "active-set / exit artifact" guarantee -- a vehicle
exiting must not change how many terms enter welfare_fn, so its frozen
running_active_attainment value stays in the aggregate rather than being
dropped.
"""

from __future__ import annotations

import pytest

from thesis.study_b.dense_shaping import (
    NEUTRAL_PHI,
    DenseShapingConfig,
    dense_shaping_term,
    welfare_objective_snapshot,
)
from thesis.study_b.utility import EpisodeVehicleTrace
from thesis.study_b.welfare_reward import GGI, MAXIMIN, MEAN


def _trace(vehicle_id: str, speeds: list[float], active_flags: list[bool], target_speed: float = 20.0) -> EpisodeVehicleTrace:
    return EpisodeVehicleTrace(
        vehicle_id=vehicle_id, target_speed=target_speed, speeds=speeds, active_flags=active_flags,
    )


class TestDenseShapingConfigValidation:
    def test_disabled_by_default(self) -> None:
        cfg = DenseShapingConfig()
        assert cfg.enabled is False

    def test_disabled_accepts_missing_magnitude_epsilon(self) -> None:
        # Must not raise -- this is the state every run uses before a protocol freeze.
        DenseShapingConfig(enabled=False)

    def test_enabled_without_magnitude_raises(self) -> None:
        with pytest.raises(ValueError, match="explicit, pre-frozen"):
            DenseShapingConfig(enabled=True, epsilon=0.01)

    def test_enabled_without_epsilon_raises(self) -> None:
        with pytest.raises(ValueError, match="explicit, pre-frozen"):
            DenseShapingConfig(enabled=True, magnitude=0.1)

    def test_enabled_with_nonpositive_magnitude_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            DenseShapingConfig(enabled=True, magnitude=0.0, epsilon=0.01)

    def test_enabled_with_negative_epsilon_raises(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            DenseShapingConfig(enabled=True, magnitude=0.1, epsilon=-0.01)

    def test_enabled_with_unimplemented_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="not implemented"):
            DenseShapingConfig(enabled=True, mode="continuous", magnitude=0.1, epsilon=0.01)

    def test_enabled_with_valid_params_does_not_raise(self) -> None:
        DenseShapingConfig(enabled=True, magnitude=0.1, epsilon=0.01)


class TestDenseShapingTermOffPath:
    @pytest.mark.parametrize("delta_phi", [-1.0, -0.01, 0.0, 0.01, 1.0])
    def test_disabled_always_returns_zero(self, delta_phi: float) -> None:
        cfg = DenseShapingConfig(enabled=False)
        assert dense_shaping_term(delta_phi, cfg) == 0.0


class TestDenseShapingTermOnPath:
    def test_positive_delta_above_epsilon_gives_plus_magnitude(self) -> None:
        cfg = DenseShapingConfig(enabled=True, magnitude=0.2, epsilon=0.01)
        assert dense_shaping_term(0.05, cfg) == 0.2

    def test_negative_delta_below_negative_epsilon_gives_minus_magnitude(self) -> None:
        cfg = DenseShapingConfig(enabled=True, magnitude=0.2, epsilon=0.01)
        assert dense_shaping_term(-0.05, cfg) == -0.2

    def test_delta_within_epsilon_dead_zone_gives_zero(self) -> None:
        cfg = DenseShapingConfig(enabled=True, magnitude=0.2, epsilon=0.01)
        assert dense_shaping_term(0.005, cfg) == 0.0
        assert dense_shaping_term(-0.005, cfg) == 0.0
        assert dense_shaping_term(0.0, cfg) == 0.0

    def test_delta_exactly_at_epsilon_boundary_is_dead_zone(self) -> None:
        # Strict inequality in the design spec ("> epsilon" / "< -epsilon"):
        # exactly at the boundary must NOT shape.
        cfg = DenseShapingConfig(enabled=True, magnitude=0.2, epsilon=0.01)
        assert dense_shaping_term(0.01, cfg) == 0.0
        assert dense_shaping_term(-0.01, cfg) == 0.0


class TestWelfareObjectiveSnapshot:
    def test_all_neutral_at_episode_start(self) -> None:
        vehicle_ids = ["V0", "V1", "V2", "V3"]
        traces = {vid: _trace(vid, speeds=[], active_flags=[]) for vid in vehicle_ids}
        for condition in (MEAN, GGI, MAXIMIN):
            phi = welfare_objective_snapshot(traces, vehicle_ids, condition)
            assert phi == pytest.approx(NEUTRAL_PHI)

    def test_exit_freezes_contribution_instead_of_dropping_it(self) -> None:
        """The core active-set/exit-artifact guarantee: a vehicle that exits
        (stops receiving new samples) keeps its LAST computed
        running_active_attainment value in the aggregate -- welfare_fn is
        always called over all 4 ids, never a shrinking subset. If this
        guarantee were violated (aggregate computed only over "still
        active" vehicles), the worst-off vehicle exiting would cause a
        spurious jump in MAXIMIN's Phi the moment it leaves the pool."""
        vehicle_ids = ["V0", "V1", "V2", "V3"]
        # V0 is the clear worst-off vehicle (attainment 0.2 while active),
        # then exits (no further samples appended, active_flags stops
        # growing) while the other three continue at attainment 1.0.
        traces = {
            "V0": _trace("V0", speeds=[4.0], active_flags=[True], target_speed=20.0),  # attainment 0.2, then exits
            "V1": _trace("V1", speeds=[20.0, 20.0], active_flags=[True, True], target_speed=20.0),
            "V2": _trace("V2", speeds=[20.0, 20.0], active_flags=[True, True], target_speed=20.0),
            "V3": _trace("V3", speeds=[20.0, 20.0], active_flags=[True, True], target_speed=20.0),
        }
        phi_maximin = welfare_objective_snapshot(traces, vehicle_ids, MAXIMIN)
        # V0's frozen M_i = 0.2 is still the minimum -- Phi must reflect it,
        # NOT jump to 1.0 as it would if V0 were dropped from the pool for
        # having stopped accumulating new samples.
        assert phi_maximin == pytest.approx(0.2)

    def test_vehicle_ids_order_does_not_matter_for_welfare_fn(self) -> None:
        vehicle_ids_a = ["V0", "V1", "V2", "V3"]
        vehicle_ids_b = ["V3", "V2", "V1", "V0"]
        traces = {
            "V0": _trace("V0", speeds=[10.0], active_flags=[True], target_speed=20.0),
            "V1": _trace("V1", speeds=[15.0], active_flags=[True], target_speed=20.0),
            "V2": _trace("V2", speeds=[20.0], active_flags=[True], target_speed=20.0),
            "V3": _trace("V3", speeds=[5.0], active_flags=[True], target_speed=20.0),
        }
        for condition in (MEAN, GGI, MAXIMIN):
            phi_a = welfare_objective_snapshot(traces, vehicle_ids_a, condition)
            phi_b = welfare_objective_snapshot(traces, vehicle_ids_b, condition)
            assert phi_a == pytest.approx(phi_b)
