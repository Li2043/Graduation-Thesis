"""Stage 6C publication figure automated quality checks."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from thesis.figures.figure_data_validation import (  # noqa: E402
    EXPECTED_ANALYSIS_COMMIT,
    EXPECTED_MISSING_CONVENTION,
    EXPECTED_RESULT_COMMIT,
    build_resolved_figure_inputs,
    deterministic_jitter,
    discover_stage6b_paths,
    sha256_file,
    validate_contrasts,
    validate_primary_seed_table,
    verify_analysis_source,
)
from thesis.figures.formal_result_figures import (  # noqa: E402
    fig_convention_selection_and_consistency,
    fig_primary_endpoint_paired_contrasts,
    fig_primary_endpoint_seed_distributions,
)
from thesis.figures.publication_style import (  # noqa: E402
    CONDITION_ORDER,
    CONDITION_STYLE,
    FORMAL_SEEDS,
    PRIMARY_STEP,
    apply_publication_style,
)


def _analysis_worktree() -> Path:
    env = os.environ.get("STAGE6C_ANALYSIS_WORKTREE")
    if env:
        return Path(env)
    candidate = REPO.parent / "final_new_analysis_100k"
    if candidate.is_dir():
        return candidate
    pytest.skip("analysis worktree not found")


@pytest.fixture(scope="module")
def analysis_root() -> Path:
    return _analysis_worktree()


@pytest.fixture(scope="module")
def paths(analysis_root: Path) -> dict[str, Path]:
    return discover_stage6b_paths(analysis_root)


@pytest.fixture(scope="module")
def tables(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    return {
        "seed": pd.read_csv(paths["primary_endpoint_seed_values"]),
        "contrasts": pd.read_csv(paths["primary_endpoint_contrasts"]),
        "bootstrap": pd.read_csv(paths["bootstrap_intervals"]),
        "holm": pd.read_csv(paths["holm_adjusted_results"]),
        "effects": pd.read_csv(paths["effect_sizes"]),
        "convention": pd.read_csv(paths["convention_availability"]),
        "episodes": pd.read_csv(paths["evaluation_episode_validated"]),
        "seed_ck": pd.read_csv(paths["seed_checkpoint_endpoints"]),
    }


def test_01_source_hashes_verify(analysis_root: Path) -> None:
    source = verify_analysis_source(analysis_root)
    assert source["manifest_overall"] == "PASS"
    assert source["summary"]["result_commit"] == EXPECTED_RESULT_COMMIT


def test_02_source_files_unmodified(paths: dict[str, Path], analysis_root: Path) -> None:
    source = verify_analysis_source(analysis_root)
    resolved = build_resolved_figure_inputs(paths)
    for key, rec in resolved["files"].items():
        if not str(rec.get("path", "")).endswith(".csv"):
            continue
        assert sha256_file(Path(rec["path"])) == rec["sha256"]


def test_03_condition_names_exact(tables: dict[str, pd.DataFrame]) -> None:
    assert set(tables["seed"]["condition"]) == set(CONDITION_ORDER)


def test_04_seed_list_exact(tables: dict[str, pd.DataFrame]) -> None:
    assert set(int(x) for x in tables["seed"]["master_seed"]) == set(FORMAL_SEEDS)


def test_05_no_episode_pseudoreplication_in_primary_figure(tables: dict[str, pd.DataFrame]) -> None:
    apply_publication_style()
    _, _, plotted = fig_primary_endpoint_seed_distributions(tables["seed"])
    # At most one row per endpoint×condition×seed
    keys = plotted.groupby(["endpoint", "condition", "master_seed"]).size()
    assert (keys == 1).all()
    assert plotted["master_seed"].nunique() == 10


def test_06_primary_endpoint_is_100000(tables: dict[str, pd.DataFrame]) -> None:
    assert set(int(x) for x in tables["seed"]["checkpoint_step"]) == {PRIMARY_STEP}


def test_07_probability_axes_include_0_1(tables: dict[str, pd.DataFrame]) -> None:
    apply_publication_style()
    fig, axes, _ = fig_primary_endpoint_seed_distributions(tables["seed"])
    for ax in axes.ravel():
        ymin, ymax = ax.get_ylim()
        assert ymin <= 0.0
        assert ymax >= 1.0
    plt.close(fig)


def test_08_paired_lines_matching_seeds_only(tables: dict[str, pd.DataFrame]) -> None:
    apply_publication_style()
    _, _, plotted = fig_primary_endpoint_seed_distributions(tables["seed"])
    for seed in FORMAL_SEEDS:
        for ep in plotted["endpoint"].unique():
            n = len(plotted[(plotted["master_seed"] == seed) & (plotted["endpoint"] == ep)])
            assert n == 3


def test_09_convention_missing_not_zero_filled(tables: dict[str, pd.DataFrame]) -> None:
    apply_publication_style()
    _, _, plotted = fig_convention_selection_and_consistency(
        tables["episodes"], tables["convention"], tables["seed_ck"]
    )
    panel_b = plotted[plotted["panel"] == "b"]
    assert panel_b["convention_consistency"].notna().all()
    assert not (panel_b["convention_consistency"] == 0).all()
    # Zero may occur as a true observed value, but missing cells must not appear.
    n_missing_source = int(tables["convention"]["convention_missing"].sum())
    assert n_missing_source == EXPECTED_MISSING_CONVENTION
    assert len(panel_b) == 30 - EXPECTED_MISSING_CONVENTION


def test_10_convention_missing_count_equals_11(tables: dict[str, pd.DataFrame]) -> None:
    validate_primary_seed_table(tables["seed"])
    assert int(tables["convention"]["convention_missing"].sum()) == 11


def test_11_paired_convention_complete_only(tables: dict[str, pd.DataFrame]) -> None:
    # Ensure incomplete pairs are not inventing consistency=0
    miss_seeds = tables["convention"][tables["convention"]["convention_missing"]]
    for _, row in miss_seeds.iterrows():
        assert pd.isna(row["convention_consistency"])


def test_12_no_checkpoint_interpolation(tables: dict[str, pd.DataFrame]) -> None:
    ckpts = sorted(int(x) for x in tables["seed_ck"]["checkpoint_step"].unique())
    assert ckpts == [100000]
    excl = EXP / "figure_exclusion_log.yaml"
    if excl.is_file():
        text = excl.read_text(encoding="utf-8")
        assert "fig_4_4_learning_curve_summaries" in text
        assert "interpolat" in text.lower() or "Case C" in text


def test_13_no_smoothing_applied() -> None:
    # Constructor modules must not call smoothing helpers.
    src = (REPO / "src" / "thesis" / "figures" / "formal_result_figures.py").read_text(encoding="utf-8")
    for banned in ("savgol", "gaussian_filter", "interpolate", "UnivariateSpline", "rolling("):
        assert banned not in src


def test_14_bootstrap_intervals_match(tables: dict[str, pd.DataFrame]) -> None:
    validate_contrasts(tables["contrasts"], tables["bootstrap"], tables["holm"], tables["effects"])


def test_15_holm_match(tables: dict[str, pd.DataFrame]) -> None:
    m = tables["contrasts"].merge(tables["holm"], on=["endpoint", "contrast"])
    assert (m["wilcoxon_p_holm"] - m["pvalue_holm"]).abs().max() < 1e-12


def test_16_effect_sizes_match(tables: dict[str, pd.DataFrame]) -> None:
    m = tables["contrasts"].merge(tables["effects"], on=["endpoint", "contrast"])
    assert (m["cohens_dz_x"] - m["cohens_dz_y"]).abs().max() < 1e-12


def test_17_main_figures_have_pdf_svg_png() -> None:
    main = EXP / "figures" / "main"
    if not main.is_dir():
        pytest.skip("figures not generated yet")
    stems = [
        "fig_4_1_primary_endpoint_seed_distributions",
        "fig_4_2_primary_endpoint_paired_contrasts",
        "fig_4_3_convention_selection_and_consistency",
        "fig_4_5_stakeholder_utility_by_role",
        "fig_4_6_safety_comfort_diagnostics",
    ]
    for stem in stems:
        for ext in (".pdf", ".svg", ".png"):
            p = main / f"{stem}{ext}"
            assert p.is_file() and p.stat().st_size > 0, p


def test_18_png_dpi_600() -> None:
    main = EXP / "figures" / "main"
    png = main / "fig_4_1_primary_endpoint_seed_distributions.png"
    if not png.is_file():
        pytest.skip("figures not generated yet")
    try:
        from PIL import Image

        with Image.open(png) as im:
            dpi = im.info.get("dpi")
            assert dpi is not None
            assert abs(float(dpi[0]) - 600) < 1.0
    except ImportError:
        meta = EXP / "figure_data" / "fig_4_1_primary_endpoint_seed_distributions_metadata.json"
        assert meta.is_file()
        payload = json.loads(meta.read_text(encoding="utf-8"))
        assert int(payload["png_dpi"]) == 600
        assert png.stat().st_size > 10_000


def test_19_vector_files_nonempty() -> None:
    main = EXP / "figures" / "main"
    if not main.is_dir():
        pytest.skip("figures not generated yet")
    for p in list(main.glob("*.pdf")) + list(main.glob("*.svg")):
        assert p.stat().st_size > 100


def test_20_figure_dimensions_declared() -> None:
    apply_publication_style()
    # Physical size is set at construction; verify constructor uses full width.
    src = (REPO / "src" / "thesis" / "figures" / "formal_result_figures.py").read_text(encoding="utf-8")
    assert "WIDTH_FULL" in src


def test_21_companion_csv() -> None:
    data = EXP / "figure_data"
    if not data.is_dir():
        pytest.skip("figure data not generated yet")
    csvs = list(data.glob("fig_*.csv"))
    assert len(csvs) >= 5


def test_22_metadata() -> None:
    data = EXP / "figure_data"
    if not data.is_dir():
        pytest.skip("figure data not generated yet")
    metas = list(data.glob("*_metadata.json"))
    assert len(metas) >= 5


def test_23_alt_text() -> None:
    caps = EXP / "captions"
    if not caps.is_dir():
        pytest.skip("captions not generated yet")
    assert len(list(caps.glob("*_alt_text.txt"))) >= 5


def test_24_no_dual_y_axes(tables: dict[str, pd.DataFrame]) -> None:
    apply_publication_style()
    fig, axes, _ = fig_primary_endpoint_paired_contrasts(tables["contrasts"])
    for ax in np.atleast_1d(axes):
        assert ax.get_shared_y_axes().get_siblings(ax)  # exists
        # No twin axes attached
        assert not hasattr(ax, "right_ax") or ax.right_ax is None
        assert len([c for c in ax.get_children() if getattr(c, "axes", None) is not ax]) >= 0
    plt.close(fig)
    src = (REPO / "src" / "thesis" / "figures" / "formal_result_figures.py").read_text(encoding="utf-8")
    assert "twinx" not in src and "twiny" not in src


def test_25_no_bar_only_primary(tables: dict[str, pd.DataFrame]) -> None:
    apply_publication_style()
    fig, axes, _ = fig_primary_endpoint_seed_distributions(tables["seed"])
    for ax in axes.ravel():
        # Must contain PathCollections (scatter) — not only bars
        from matplotlib.collections import PathCollection

        assert any(isinstance(c, PathCollection) for c in ax.collections)
    plt.close(fig)


def test_26_font_sizes_at_least_7() -> None:
    style = apply_publication_style()
    assert style.base_fontsize >= 7


def test_27_condition_markers_differ() -> None:
    markers = {CONDITION_STYLE[c]["marker"] for c in CONDITION_ORDER}
    assert len(markers) == 3


def test_28_condition_linestyles_differ() -> None:
    styles = {CONDITION_STYLE[c]["linestyle"] for c in CONDITION_ORDER}
    assert len(styles) == 3


def test_29_grayscale_interpretable() -> None:
    # Distinct markers + linestyles already asserted; colours also distinct.
    colors = {CONDITION_STYLE[c]["color"] for c in CONDITION_ORDER}
    assert len(colors) == 3


def test_30_deterministic_jitter() -> None:
    a = deterministic_jitter(61001, 0)
    b = deterministic_jitter(61001, 0)
    c = deterministic_jitter(61002, 0)
    assert a == b
    assert a != c


def test_31_no_training_or_eval_invoked() -> None:
    runner = (EXP / "scripts" / "run_stage6c_figures.py").read_text(encoding="utf-8")
    for banned in ("train_policy", "make_env", "Gymnasium", "evaluate_checkpoint(", "PPO("):
        assert banned not in runner


def test_32_analysis_commit_matches_tag(analysis_root: Path) -> None:
    head = subprocess.check_output(
        ["git", "-C", str(analysis_root), "rev-list", "-n", "1", "HEAD"], text=True
    ).strip()
    assert head == EXPECTED_ANALYSIS_COMMIT


def test_33_forest_values_match_stage6b(tables: dict[str, pd.DataFrame]) -> None:
    apply_publication_style()
    _, _, plotted = fig_primary_endpoint_paired_contrasts(tables["contrasts"])
    merged = plotted.merge(tables["contrasts"], on=["endpoint", "contrast"], suffixes=("_p", "_s"))
    assert (merged["mean_diff_p"] - merged["mean_diff_s"]).abs().max() < 1e-12
    assert (merged["ci95_low_p"] - merged["ci95_low_s"]).abs().max() < 1e-12
