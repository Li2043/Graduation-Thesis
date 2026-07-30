"""Publication Matplotlib style for Stage 6C formal result figures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib as mpl
from matplotlib import font_manager


CONDITION_ORDER: tuple[str, ...] = ("baseline", "mean_pbrs", "min_pbrs")

CONDITION_STYLE: dict[str, dict[str, str]] = {
    "baseline": {"color": "#4D4D4D", "marker": "o", "linestyle": "-"},
    "mean_pbrs": {"color": "#0072B2", "marker": "s", "linestyle": "--"},
    "min_pbrs": {"color": "#D55E00", "marker": "^", "linestyle": "-."},
}

CONDITION_DISPLAY: dict[str, str] = {
    "baseline": "Baseline",
    "mean_pbrs": "Mean-PBRS",
    "min_pbrs": "Min-PBRS",
}

ENDPOINT_KEYS: dict[str, str] = {
    "success_rate": "evaluation_success_rate",
    "collision_rate": "stakeholder_collision_rate",
    "mean_stakeholder_utility": "mean_stakeholder_episode_utility",
    "min_stakeholder_utility": "minimum_stakeholder_episode_utility",
    "convention_consistency": "convention_consistency",
}

ENDPOINT_DISPLAY: dict[str, str] = {
    "evaluation_success_rate": "Success rate",
    "stakeholder_collision_rate": "Stakeholder-collision rate",
    "mean_stakeholder_episode_utility": "Mean stakeholder utility",
    "minimum_stakeholder_episode_utility": "Minimum stakeholder utility",
    "convention_consistency": "Convention consistency",
}

PRIMARY_NON_CONVENTION: tuple[str, ...] = (
    "evaluation_success_rate",
    "stakeholder_collision_rate",
    "mean_stakeholder_episode_utility",
    "minimum_stakeholder_episode_utility",
)

CONTRAST_ORDER: tuple[str, ...] = (
    "mean_pbrs - baseline",
    "min_pbrs - baseline",
    "min_pbrs - mean_pbrs",
)

CONTRAST_DISPLAY: dict[str, str] = {
    "mean_pbrs - baseline": "Mean-PBRS − Baseline",
    "min_pbrs - baseline": "Min-PBRS − Baseline",
    "min_pbrs - mean_pbrs": "Min-PBRS − Mean-PBRS",
}

FORMAL_SEEDS: tuple[int, ...] = tuple(range(61001, 61011))
PRIMARY_STEP = 100_000

WIDTH_SINGLE = 3.35
WIDTH_ONE_HALF = 5.2
WIDTH_FULL = 6.8

PROB_YTICKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class StyleManifest:
    resolved_font: str
    matplotlib_version: str
    base_fontsize: float
    condition_styles: dict[str, dict[str, str]]


def preferred_serif_fonts() -> tuple[str, ...]:
    return ("Times New Roman", "Liberation Serif", "DejaVu Serif")


def resolve_serif_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred_serif_fonts():
        if name in available:
            return name
    return "DejaVu Serif"


def apply_publication_style() -> StyleManifest:
    font = resolve_serif_font()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font, "DejaVu Serif", "serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.5,
            "errorbar.capsize": 3,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return StyleManifest(
        resolved_font=font,
        matplotlib_version=mpl.__version__,
        base_fontsize=9.0,
        condition_styles=dict(CONDITION_STYLE),
    )


def style_manifest_dict(manifest: StyleManifest) -> dict[str, Any]:
    return {
        "resolved_font": manifest.resolved_font,
        "matplotlib_version": manifest.matplotlib_version,
        "base_fontsize": manifest.base_fontsize,
        "condition_styles": manifest.condition_styles,
        "widths_inches": {
            "single": WIDTH_SINGLE,
            "one_half": WIDTH_ONE_HALF,
            "full": WIDTH_FULL,
        },
        "probability_yticks": list(PROB_YTICKS),
    }


__all__ = [
    "CONDITION_DISPLAY",
    "CONDITION_ORDER",
    "CONDITION_STYLE",
    "CONTRAST_DISPLAY",
    "CONTRAST_ORDER",
    "ENDPOINT_DISPLAY",
    "ENDPOINT_KEYS",
    "FORMAL_SEEDS",
    "PRIMARY_NON_CONVENTION",
    "PRIMARY_STEP",
    "PROB_YTICKS",
    "StyleManifest",
    "WIDTH_FULL",
    "WIDTH_ONE_HALF",
    "WIDTH_SINGLE",
    "apply_publication_style",
    "resolve_serif_font",
    "style_manifest_dict",
]
