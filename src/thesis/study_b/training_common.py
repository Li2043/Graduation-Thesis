"""Shared bookkeeping for Study B's training scripts (``scripts/train_mappo.py``,
``scripts/train_dqn_fallback.py``) -- a rolling episode-window stats
accumulator (mirrors this project's own established
``stage11_dyad_merge_runner.EpisodeWindowStats`` convention: reset after
every checkpoint save) and scenario-bank JSON load/save, kept here so both
training scripts and ``scripts/evaluate_policy.py`` share one
implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from thesis.study_b.scenario_generator import ScenarioSpec, VehicleSpawnSpec
from thesis.study_b.utility import gini_coefficient, generalized_gini_welfare, mean_welfare, min_welfare

__all__ = [
    "StudyBEpisodeWindowStats",
    "save_scenario_bank",
    "load_scenario_bank",
]


@dataclass
class StudyBEpisodeWindowStats:
    """Accumulates outcomes for episodes completed since the previous
    checkpoint -- call ``record_episode`` once per finished episode,
    ``as_dict()`` at checkpoint time, then ``reset()``."""

    episodes: int = 0
    completions: int = 0
    collisions: int = 0
    truncations: int = 0
    mean_U_sum: float = 0.0
    min_U_sum: float = 0.0
    ggi_sum: float = 0.0
    gini_sum: float = 0.0
    gini_count: int = 0
    all_zero_count: int = 0
    C_max_sum: float = 0.0
    C_mean_sum: float = 0.0

    def record_episode(
        self, *, term_reason: str, utilities: dict[str, float], burdens: dict[str, float]
    ) -> None:
        self.episodes += 1
        if term_reason == "success":
            self.completions += 1
        elif term_reason == "collision":
            self.collisions += 1
        elif term_reason == "truncation":
            self.truncations += 1

        u_values = list(utilities.values())
        self.mean_U_sum += mean_welfare(u_values)
        self.min_U_sum += min_welfare(u_values)
        self.ggi_sum += generalized_gini_welfare(u_values)
        gini = gini_coefficient(u_values)
        if gini is None:
            self.all_zero_count += 1
        else:
            self.gini_sum += gini
            self.gini_count += 1

        c_values = list(burdens.values())
        self.C_max_sum += max(c_values)
        self.C_mean_sum += sum(c_values) / len(c_values)

    def as_dict(self) -> dict[str, Any]:
        n = max(self.episodes, 1)
        return {
            "episodes": self.episodes,
            "completion_rate": self.completions / n,
            "collision_rate": self.collisions / n,
            "truncation_rate": self.truncations / n,
            "mean_U_mean": self.mean_U_sum / n,
            "min_U_mean": self.min_U_sum / n,
            "ggi_mean": self.ggi_sum / n,
            # None (JSON null), NOT 0.0, when every episode this window was
            # all-zero-utility -- see utility.gini_coefficient's docstring.
            "gini_mean": (self.gini_sum / self.gini_count) if self.gini_count > 0 else None,
            "all_zero_utility_rate": self.all_zero_count / n,
            "C_max_mean": self.C_max_sum / n,
            "C_mean_mean": self.C_mean_sum / n,
        }

    def reset(self) -> None:
        self.episodes = 0
        self.completions = 0
        self.collisions = 0
        self.truncations = 0
        self.mean_U_sum = 0.0
        self.min_U_sum = 0.0
        self.ggi_sum = 0.0
        self.gini_sum = 0.0
        self.gini_count = 0
        self.all_zero_count = 0
        self.C_max_sum = 0.0
        self.C_mean_sum = 0.0


def save_scenario_bank(scenarios: list[ScenarioSpec], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([s.to_json_dict() for s in scenarios], indent=2),
        encoding="utf-8",
    )


def load_scenario_bank(path: Path) -> list[ScenarioSpec]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    scenarios: list[ScenarioSpec] = []
    for entry in raw:
        vehicles = {
            a["vehicle_id"]: VehicleSpawnSpec(
                vehicle_id=a["vehicle_id"],
                role=a["role"],
                speed_class=a["speed_class"],
                ttc_slot=a["ttc_slot"],
                target_speed=a["target_speed"],
                spawn_speed=a["spawn_speed"],
                route_position=a["route_position"],
                nominal_ttc=a["nominal_ttc"],
            )
            for a in entry["agents"]
        }
        scenarios.append(
            ScenarioSpec(
                scenario_id=entry["scenario_id"],
                episode_seed=entry["episode_seed"],
                traffic_type=entry["traffic_type"],
                vehicles=vehicles,
            )
        )
    return scenarios
