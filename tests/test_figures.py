"""The suite figure layer, driven entirely by synthetic schema-conforming tables.

Every test here renders a real figure into ``tmp_path`` with the Agg backend
and then asserts on the *result*: the three files exist, the CSV holds exactly
the points that were drawn, the legend says how each method was trained, the
infinite beta is not on the finite log axis, and the tau figures say that tau
is the learner radius while the evaluation radius stays at 0.5.

The tables come from
:mod:`empirical_testing.experiments.figure_fixtures`, which builds
schema-conforming tables with invented numbers to exercise the figure contract
independently of production results. Nothing here reads or writes a run, and none of
these numbers is a result.
"""

from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from empirical_testing.experiments import figures as figures  # noqa: E402
from empirical_testing.experiments import figure_fixtures as fixtures  # noqa: E402
from empirical_testing.experiments import schema  # noqa: E402
from empirical_testing.experiments.suite import (  # noqa: E402
    ACTIVE_BETAS,
    BETA_GRID,
    SUITES,
    TAU_GRID,
)

#: A label is only acceptable if it says how the method was trained.
PROTOCOL_PHRASES = ("passive-trained", "active-trained", "not fitted")


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return fixtures.synthetic_tables()


@pytest.fixture
def drawn(tables):
    """Draw one figure by name, and close it afterwards."""

    made = []

    def _draw(name: str):
        figures.configure_style()
        artifact = figures.DRAWERS[name](tables)
        made.append(artifact)
        return artifact

    yield _draw
    for artifact in made:
        plt.close(artifact.figure)


def _series_points(artifact) -> dict[str, set[tuple[float, float]]]:
    """Every (x, y) actually on the axes, keyed by series id."""

    points: dict[str, set[tuple[float, float]]] = {}
    for ax in artifact.figure.axes:
        for line in ax.get_lines():
            gid = line.get_gid() or ""
            if not gid.startswith(figures.SERIES_GID_PREFIX):
                continue
            key = gid[len(figures.SERIES_GID_PREFIX):]
            bucket = points.setdefault(key, set())
            for x, y in zip(line.get_xdata(), line.get_ydata()):
                bucket.add((round(float(x), 9), round(float(y), 9)))
    return points


def _csv_points(frame: pd.DataFrame) -> dict[str, set[tuple[float, float]]]:
    points: dict[str, set[tuple[float, float]]] = {}
    for _, row in frame.iterrows():
        points.setdefault(str(row["series"]), set()).add(
            (round(float(row["x"]), 9), round(float(row["y"]), 9))
        )
    return points


# ------------------------------------------------------------ the fixtures

def test_fixture_tables_conform_to_the_schema(tables):
    from empirical_testing.experiments.figure_fixtures import SYNTHETIC_MARKER_COLUMN

    for name, frame in tables.items():
        # The schema columns, in order, plus the synthetic marker the fixtures
        # append so anything derived from them can identify itself.
        assert list(frame.columns) == [*schema.TABLES[name], SYNTHETIC_MARKER_COLUMN], name
        assert frame[SYNTHETIC_MARKER_COLUMN].all(), name
        assert not frame.empty, name
        # The marker is the fixtures' own addition, never part of the schema a
        # real reduction writes.
        assert SYNTHETIC_MARKER_COLUMN not in schema.TABLES[name]


def test_fixture_covers_both_conditions_and_the_active_overlay(tables):
    levels = tables["method_levels"]
    beta = levels[levels["condition"] == "beta"]
    assert {str(b) for b in beta["beta"]} == {
        "inf" if np.isinf(b) else f"{b:g}" for b in BETA_GRID
    }
    active = beta[beta["protocol"] == "active"]
    assert {str(b) for b in active["beta"]} == {
        "inf" if np.isinf(b) else f"{b:g}" for b in ACTIVE_BETAS
    }
    tau = levels[levels["condition"] == "tau"]
    assert sorted({float(t) for t in tau["learner_tau"]}) == sorted(TAU_GRID)
    # The evaluation radius never moves, in either condition.
    assert set(levels["evaluation_tau"]) == {0.5}


# ------------------------------------------------------------- the contract

def test_every_schema_figure_has_a_drawer():
    assert set(figures.DRAWERS) == set(schema.FIGURES)


@pytest.mark.parametrize("name", sorted(schema.FIGURES))
def test_writes_pdf_png_and_csv_for_every_figure(tables, tmp_path, name):
    written = figures.write_all_figures(tables, tmp_path, dataset="coffee", figures=[name])
    assert set(written[name]) == {"pdf", "png", "csv"}
    for kind, path in written[name].items():
        assert path.is_file(), (name, kind)
        assert path.stat().st_size > 0, (name, kind)
        assert path.parent == tmp_path


def test_write_all_figures_writes_the_whole_set(tables, tmp_path):
    written = figures.write_all_figures(tables, tmp_path, dataset="coffee")
    assert set(written) == set(schema.FIGURES)
    assert len(list(tmp_path.glob("*.pdf"))) == len(schema.FIGURES)
    assert len(list(tmp_path.glob("*.png"))) == len(schema.FIGURES)
    assert len(list(tmp_path.glob("*.csv"))) == len(schema.FIGURES)


def test_output_directory_is_explicit_and_created(tables, tmp_path):
    target = tmp_path / "nested" / "figures"
    figures.write_all_figures(tables, target, figures=["tau_curves"])
    # The evaluation regime is part of every file name, so a moving-tau file
    # can never be mistaken for the fixed-tau comparison.
    assert (target / "tau_curves_fixed_0.5.pdf").is_file()


def test_suite_figure_directory_is_repository_relative(tmp_path):
    path = figures.suite_figure_directory(tmp_path, SUITES["coffee"])
    assert path == tmp_path / "figures" / "coffee"
    assert figures.suite_figure_directory(tmp_path, SUITES["persona"]).name == "persona"


def test_unknown_figure_name_is_refused(tables, tmp_path):
    with pytest.raises(KeyError, match="unknown suite figure"):
        figures.write_all_figures(tables, tmp_path, figures=["beta_curves", "nope"])


# ------------------------------------------------- loud failure on bad input

@pytest.mark.parametrize(
    "name, column, figure_name",
    [
        ("method_levels", "worst_group_distortion", "beta_curves"),
        # ``training_protocol`` is a constant of the suite rather than data, and
        # the real reduction does not write it, so it is derived rather than
        # required.  ``learner_tau`` is genuinely drawn and stays required.
        ("method_levels", "learner_tau", "tau_curves"),
        ("method_contrasts", "ci_low", "beta_gaps"),
        ("method_group_levels", "group", "tau_group_panels"),
        ("protocol_contrasts", "estimate", "protocol_contrasts"),
    ],
)
def test_missing_column_fails_loudly(tables, tmp_path, name, column, figure_name):
    broken = dict(tables)
    broken[name] = tables[name].drop(columns=[column])
    with pytest.raises(figures.MissingColumnsError) as excinfo:
        figures.write_all_figures(broken, tmp_path, figures=[figure_name])
    message = str(excinfo.value)
    assert column in message
    assert name in message
    assert figure_name in message
    # Nothing is written when the input is rejected.
    assert list(tmp_path.glob("*")) == []


def test_missing_table_fails_loudly(tables, tmp_path):
    broken = {k: v for k, v in tables.items() if k != "method_contrasts"}
    with pytest.raises(figures.MissingTableError) as excinfo:
        figures.write_all_figures(broken, tmp_path, figures=["tau_gaps"])
    assert "method_contrasts" in str(excinfo.value)
    assert "tau_gaps" in str(excinfo.value)


def test_required_columns_are_satisfiable_by_the_schema():
    """Every required field has at least one spelling the schema declares.

    The requirement is per *field*, not per column name: the real reduction
    spells a level estimate ``estimate``/``corrected_se`` where the schema says
    ``worst_group_distortion``/``worst_group_se``, and the figure layer accepts
    either -- but never neither.
    """

    for name, required in figures.REQUIRED_COLUMNS.items():
        declared = set(schema.TABLES[name])
        for field, accepted in required.items():
            assert set(accepted) & declared, (name, field, accepted)


# --------------------------------------------- the plotted CSV is the figure

@pytest.mark.parametrize("name", sorted(schema.FIGURES))
def test_plotted_csv_holds_exactly_the_drawn_points(tables, tmp_path, name, drawn):
    artifact = drawn(name)
    paths = figures.write_figure(artifact, tmp_path, close=False)
    frame = pd.read_csv(paths["csv"])
    assert list(frame.columns) == list(figures.PLOTTED_COLUMNS)
    assert _csv_points(artifact.frame) == _series_points(artifact), name
    # And the file on disk is the same table, not a re-derived one.
    assert _csv_points(frame) == _series_points(artifact), name
    assert len(frame) == len(artifact.frame)


@pytest.mark.parametrize("name", sorted(schema.FIGURES))
def test_plotted_csv_names_the_interval_and_the_cell(tables, tmp_path, name, drawn):
    artifact = drawn(name)
    frame = artifact.frame
    assert set(frame["interval"]) == {figures.INTERVAL_KIND}
    assert set(frame["figure"]) == {name}
    assert (frame["evaluation_tau"] == 0.5).all()
    # Every drawn row names the method and how it was trained.
    assert not frame["series_label"].eq("").any()
    assert not frame["training_protocol"].eq("").any()


# ----------------------------------------------------- training protocols

@pytest.mark.parametrize("name", sorted(schema.FIGURES))
def test_every_legend_entry_states_a_training_protocol(tables, name, drawn):
    artifact = drawn(name)
    assert artifact.legend_labels
    for label in artifact.legend_labels:
        assert any(phrase in label for phrase in PROTOCOL_PHRASES), (name, label)


@pytest.mark.parametrize("name", ["tau_curves", "tau_gaps", "tau_group_panels"])
def test_nlhf_in_an_active_condition_is_the_passive_trained_benchmark(name, drawn):
    """The active conditions show a passive fit; it must not read as active NLHF."""

    artifact = drawn(name)
    nlhf = [label for label in artifact.legend_labels if "NLHF" in label]
    assert nlhf, name
    for label in nlhf:
        assert "passive-trained benchmark" in label, label
        assert "active-trained" not in label.split("$-$")[0], label
    text = artifact.text
    assert not re.search(r"active[ -]trained\s+NLHF", text, flags=re.I)
    assert not re.search(r"\bactive NLHF\b", text, flags=re.I)
    assert figures.NLHF_NOTE in text


def test_nlhf_series_rows_are_labelled_passive_everywhere(tables, drawn):
    for name in ("tau_curves", "tau_group_panels", "beta_curves"):
        frame = drawn(name).frame
        nlhf = frame[frame["method"] == "pooled_nlhf"]
        assert not nlhf.empty, name
        assert set(nlhf["protocol_trained"]) == {"passive"}, name
        assert nlhf["series_label"].str.contains("passive-trained benchmark").all(), name


def test_active_glhf_series_are_labelled_active(drawn):
    frame = drawn("tau_curves").frame
    glhf = frame[frame["method"].isin(["multigroup", "pooled"])]
    assert set(glhf["protocol_trained"]) == {"active"}
    assert glhf["series_label"].str.contains("active-trained").all()


def test_uniform_is_labelled_as_never_fitted(drawn):
    labels = drawn("beta_curves").legend_labels
    uniform = [label for label in labels if "Uniform reference" in label]
    assert uniform and all("not fitted" in label for label in uniform)


def test_beta_curves_draw_active_glhf_only_where_it_was_run(drawn):
    frame = drawn("beta_curves").frame
    active = frame[frame["protocol_trained"] == "active"]
    assert set(active["beta"]) == {"4", "inf"}
    assert set(active["method"]) == {"multigroup", "pooled"}
    # The shared NLHF benchmark is drawn once, not once per protocol position.
    nlhf = frame[frame["method"] == "pooled_nlhf"]
    assert len(nlhf) == len(set(nlhf["beta"]))


# ---------------------------------------------- infinity off the log axis

def _axes_by_label(artifact, label):
    return [ax for ax in artifact.figure.axes if ax.get_label() == label]


@pytest.mark.parametrize("name", ["beta_curves", "beta_gaps"])
def test_infinity_is_drawn_in_a_separate_axes_with_a_break(name, drawn):
    artifact = drawn(name)
    finite = _axes_by_label(artifact, figures.FINITE_AXES_LABEL)
    infinite = _axes_by_label(artifact, figures.INF_AXES_LABEL)
    assert len(finite) == 1 and len(infinite) == 1
    ax_finite, ax_inf = finite[0], infinite[0]
    assert ax_finite is not ax_inf

    # The finite axis is a log axis carrying exactly the finite betas.
    assert ax_finite.get_xscale() == "log"
    ticks = sorted(float(t) for t in ax_finite.get_xticks())
    expected = sorted(b for b in BETA_GRID if np.isfinite(b))
    assert ticks == pytest.approx(expected)
    assert all(np.isfinite(ticks))

    # Infinity is on neither the tick list nor any point of the finite axis.
    inf_x = [float(x) for line in ax_finite.get_lines()
             for x in np.atleast_1d(line.get_xdata())]
    assert all(np.isfinite(x) for x in inf_x)
    assert max(ticks) < np.inf
    frame = artifact.frame
    finite_rows = frame[frame["axis_segment"] == "finite"]
    inf_rows = frame[frame["axis_segment"] == "infinity"]
    assert not inf_rows.empty
    assert set(finite_rows["beta"]).isdisjoint({"inf"})
    assert set(inf_rows["beta"]) == {"inf"}
    # The infinity points live in the separate slot's own coordinates, inside
    # its limits, and that slot carries no finite beta tick at all.
    low, high = ax_inf.get_xlim()
    assert all(low <= float(x) <= high for x in inf_rows["x"])
    assert [float(t) for t in ax_inf.get_xticks()] == [0.0]
    assert not any(np.isclose(float(t), expected).any() for t in ax_inf.get_xticks())

    # There is a visible break between the two axes.
    marks = [line for ax in (ax_finite, ax_inf) for line in ax.get_lines()
             if line.get_gid() == figures.BREAK_MARK_GID]
    assert marks, "the broken axis carries no break marks"
    assert r"$\infty$" in [t.get_text() for t in ax_inf.get_xticklabels()]


@pytest.mark.parametrize("name", ["beta_curves", "beta_gaps"])
def test_no_line_joins_a_finite_beta_to_infinity(name, drawn):
    """A join would assert a continuity the x axis does not have."""

    artifact = drawn(name)
    ax_inf = _axes_by_label(artifact, figures.INF_AXES_LABEL)[0]
    low, high = ax_inf.get_xlim()
    for line in ax_inf.get_lines():
        if not (line.get_gid() or "").startswith(figures.SERIES_GID_PREFIX):
            continue
        xs = np.atleast_1d(line.get_xdata()).astype(float)
        assert len(xs) == 1, "the infinity slot holds one point per series"
        assert low <= xs[0] <= high


def test_protocol_contrast_beta_is_categorical_not_a_log_axis(drawn):
    artifact = drawn("protocol_contrasts")
    ax = artifact.figure.axes[0]
    assert ax.get_xscale() == "linear"
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert r"$\beta=\infty$" in labels
    assert set(artifact.frame["axis_segment"]) == {"categorical"}
    assert set(artifact.frame["beta"]) == {"4", "inf"}
    assert "no log axis" in ax.get_xlabel()


# ------------------------------------------------------- the tau statement

@pytest.mark.parametrize("name", ["tau_curves", "tau_gaps", "tau_group_panels"])
def test_tau_figures_say_tau_is_the_learner_radius(name, drawn):
    artifact = drawn(name)
    text = artifact.text
    assert figures.TAU_CAPTION in text
    assert "LEARNER radius" in text
    assert "evaluation radius is held fixed at tau = 0.5" in text
    # ... and the axis label says it too, for a reader who crops the caption.
    axis_labels = [ax.get_xlabel() for ax in artifact.figure.axes if ax.get_xlabel()]
    assert axis_labels
    assert any("Learner radius" in label and "evaluation radius fixed" in label
               for label in axis_labels)


def test_tau_axis_carries_the_whole_learner_grid(drawn):
    artifact = drawn("tau_curves")
    ax = artifact.figure.axes[0]
    assert sorted(float(t) for t in ax.get_xticks()) == pytest.approx(sorted(TAU_GRID))
    assert sorted({float(t) for t in artifact.frame["learner_tau"]}) == pytest.approx(
        sorted(TAU_GRID))


def test_tau_group_panels_have_one_panel_per_group(drawn):
    artifact = drawn("tau_group_panels")
    panels = [ax for ax in artifact.figure.axes
              if ax.get_label().startswith("tau_group_panel:")]
    assert len(panels) == len(fixtures.GROUPS)
    assert set(artifact.frame["group"]) == set(fixtures.GROUPS)
    # A dollar sign in a group name must not be read as mathtext.
    titles = [ax.get_title(loc="left") for ax in panels]
    assert any(r"\$" in title for title in titles)


# ----------------------------------------------------- what the bands are

@pytest.mark.parametrize("name", sorted(schema.FIGURES))
def test_captions_call_the_intervals_marginal_and_nothing_else(name, drawn):
    artifact = drawn(name)
    text = artifact.text.lower()
    if "interval" in text and "nadeau" in text:
        assert "marginal" in text
    # No caption may claim a simultaneous band or a bootstrap band; where those
    # words appear at all they must be denied, not asserted.
    for phrase in ("simultaneous", "bootstrap"):
        for match in re.finditer(phrase, text):
            prefix = text[max(0, match.start() - 24):match.start()]
            assert "not " in prefix or "would read as a " in prefix, (name, phrase, prefix)


def test_interval_caption_is_explicit_about_what_it_is_not():
    caption = figures.INTERVAL_CAPTION
    assert "Nadeau-Bengio" in caption
    assert "marginal per cell" in caption
    assert "not simultaneous over the grid" in caption
    assert "not respondent bootstrap intervals" in caption


def test_level_intervals_come_from_the_corrected_standard_error(tables, drawn):
    """Levels carry a corrected SE, so the drawn endpoints must match it."""

    artifact = drawn("tau_curves")
    levels = tables["method_levels"]
    levels = levels[(levels["condition"] == "tau") & (levels["method"] == "multigroup")]
    levels = levels.sort_values("learner_tau")
    drawn_rows = artifact.frame[artifact.frame["method"] == "multigroup"]
    drawn_rows = drawn_rows.sort_values("learner_tau")
    expected_low = (levels["worst_group_distortion"]
                    - figures.T_CRITICAL_DF99 * levels["worst_group_se"]).to_numpy()
    assert drawn_rows["ci_low"].to_numpy() == pytest.approx(expected_low)


def test_contrast_intervals_are_drawn_as_saved(tables, drawn):
    artifact = drawn("tau_gaps")
    saved = tables["method_contrasts"]
    saved = saved[(saved["condition"] == "tau") & (saved["subtrahend"] == "multigroup")
                  & (saved["minuend"] == "uniform")].sort_values("learner_tau")
    got = artifact.frame[artifact.frame["method"] == "uniform"].sort_values("learner_tau")
    assert got["y"].to_numpy() == pytest.approx(saved["estimate"].to_numpy())
    assert got["ci_low"].to_numpy() == pytest.approx(saved["ci_low"].to_numpy())
    assert got["ci_high"].to_numpy() == pytest.approx(saved["ci_high"].to_numpy())


# ------------------------------------------------------------- legibility

@pytest.mark.parametrize("name", sorted(schema.FIGURES))
def test_figures_are_not_empty_and_fit_a_page(tables, tmp_path, name, drawn):
    artifact = drawn(name)
    assert len(artifact.frame) > 0
    assert _series_points(artifact), "nothing was drawn"
    figures.write_figure(artifact, tmp_path, close=False)
    artifact.figure.canvas.draw()
    box = artifact.figure.get_tightbbox(artifact.figure.canvas.get_renderer())
    assert box.width <= 9.5, (name, box.width)
    assert box.height <= 9.5, (name, box.height)


def test_fixture_writer_round_trips_through_csv(tmp_path):
    written = fixtures.write_synthetic_tables(tmp_path / "reduce")
    loaded = {name: pd.read_csv(path) for name, path in written.items()}
    paths = figures.write_all_figures(loaded, tmp_path / "figures", dataset="coffee")
    assert len(paths) == len(schema.FIGURES)
    for group in paths.values():
        assert all(path.is_file() for path in group.values())



# ------------------------------------------- synthetic output identifies itself

def test_synthetic_figures_are_watermarked(tmp_path, tables):
    """These fixtures are shaped to look like the real figures, so they must
    never be mistakable for them."""

    import matplotlib.pyplot as plt

    from empirical_testing.experiments import figures

    figures.write_all_figures(tables, tmp_path, dataset="coffee", close=False)
    stamped = 0
    for number in plt.get_fignums():
        figure = plt.figure(number)
        for artist in figure.texts:
            if artist.get_gid() == "synthetic_watermark":
                assert "NOT RESULTS" in artist.get_text()
                stamped += 1
    plt.close("all")
    assert stamped >= len(schema.FIGURES), f"only {stamped} figures were watermarked"


def test_real_tables_are_not_watermarked(tmp_path, tables):
    import matplotlib.pyplot as plt

    from empirical_testing.experiments import figures
    from empirical_testing.experiments.figure_fixtures import SYNTHETIC_MARKER_COLUMN

    # A reduction's own tables carry no marker.
    real = {name: frame.drop(columns=[SYNTHETIC_MARKER_COLUMN])
            for name, frame in tables.items()}
    figures.write_all_figures(real, tmp_path, dataset="coffee", close=False)
    for number in plt.get_fignums():
        for artist in plt.figure(number).texts:
            assert artist.get_gid() != "synthetic_watermark"
    plt.close("all")


def test_synthetic_figures_refuse_a_reportable_figure_directory(tmp_path, tables):
    from empirical_testing.experiments import figures

    reportable = tmp_path / "figures" / "coffee"
    with pytest.raises(ValueError, match="not results"):
        figures.write_all_figures(tables, reportable, dataset="coffee")
    assert not list(reportable.glob("*")) if reportable.exists() else True


# ===================================================================
# The real reduction's tables, and the two evaluation regimes
# ===================================================================
#
# ``reduce`` writes tables that carry more columns than the figures draw,
# stack worst-group and per-group quantities under ``quantity``, spell a level
# estimate ``estimate``/``corrected_se``, spell interval endpoints
# ``ci_lower``/``ci_upper``, record the *cell's* protocol rather than the
# training one, and write the tau condition twice -- once per evaluation
# regime, into separate directories.  The helper below reshapes the synthetic
# fixtures into exactly that shape, so these tests exercise the real contract
# without reading a run.  The numbers are still the fixtures' invented ones.

#: Columns the real reduction does not write, which the figure layer must
#: therefore derive (or do without) rather than require.
_ABSENT_FROM_THE_REAL_REDUCTION = (
    "display_name", "training_protocol", "displayed_beside_protocol", "shared_with",
    "worst_group_distortion", "worst_group_se", "distortion", "distortion_se",
    "ci_low", "ci_high", "df", "t", "p_value", "welfare", "welfare_optimum",
    "n_test_group", "mean_test_train_ratio", "minuend_training_protocol",
    "subtrahend_training_protocol", "n_seeds",
)

_REAL_DF = 9


def reduction_shaped(tables, regime: str) -> dict[str, pd.DataFrame]:
    """The fixtures, rewritten in the shape ``reduce`` actually writes."""

    out: dict[str, pd.DataFrame] = {}
    for name, frame in tables.items():
        frame = frame.drop(columns=[figures_fixtures_marker()], errors="ignore").copy()
        if name in ("method_levels", "method_group_levels"):
            value = "worst_group_distortion" if name == "method_levels" else "distortion"
            frame["estimate"] = frame[value].astype(float)
            frame["corrected_se"] = frame[f"{value.replace('worst_group_distortion', 'worst_group')}_se"
                                          if name == "method_levels" else "distortion_se"].astype(float)
            half = 2.262 * frame["corrected_se"]
            frame["ci_lower"] = frame["estimate"] - half
            frame["ci_upper"] = frame["estimate"] + half
        else:
            frame["ci_lower"] = frame["ci_low"].astype(float)
            frame["ci_upper"] = frame["ci_high"].astype(float)
        frame["quantity"] = figures.TABLE_QUANTITY[name]
        frame["status"] = "ok"
        frame["degrees_of_freedom"] = _REAL_DF
        frame["evaluation_regime"] = regime
        frame["cell"] = [
            f"{c}@{b}" for c, b in zip(frame["condition"], frame["beta"])
        ]
        is_tau = frame["condition"] == "tau"
        if regime == "moving":
            frame.loc[is_tau, "evaluation_tau"] = frame.loc[is_tau, "learner_tau"]
        if "displayed_beside_protocol" in frame.columns:
            # The reduction records where a row is *displayed*, not how its
            # method was trained: a passive-trained benchmark sitting in an
            # active cell is written with the active cell's protocol.
            frame["protocol"] = np.where(
                frame["displayed_beside_protocol"].astype(str) == "active",
                "active", frame["protocol"])
        if name == "protocol_contrasts":
            frame = frame.drop(columns=["protocol", "p_holm"], errors="ignore")
        out[name] = frame.drop(
            columns=[c for c in _ABSENT_FROM_THE_REAL_REDUCTION if c in frame.columns])
    return out


def figures_fixtures_marker() -> str:
    from empirical_testing.experiments.figure_fixtures import SYNTHETIC_MARKER_COLUMN

    return SYNTHETIC_MARKER_COLUMN


def split_by_condition(real: dict[str, pd.DataFrame], condition: str) -> dict[str, pd.DataFrame]:
    """One condition's rows, the way the reduction files them by directory."""

    return {name: frame[frame["condition"] == condition].copy()
            for name, frame in real.items()}


@pytest.fixture(scope="module")
def real_fixed(tables):
    return split_by_condition(reduction_shaped(tables, figures.REGIME_FIXED), "tau")


@pytest.fixture(scope="module")
def real_moving(tables):
    return split_by_condition(reduction_shaped(tables, figures.REGIME_MOVING), "tau")


@pytest.fixture(scope="module")
def real_beta(tables):
    return split_by_condition(reduction_shaped(tables, figures.REGIME_FIXED), "beta")


def _write_run(tmp_path, real_beta, real_fixed, real_moving):
    """A run directory shaped like a reduction's: reduce/{beta,fixed_0.5,moving}."""

    run = tmp_path / "run"
    for sub, group in (("beta", real_beta),
                       (figures.REGIME_FIXED, real_fixed),
                       (figures.REGIME_MOVING, real_moving)):
        directory = run / "reduce" / sub
        directory.mkdir(parents=True, exist_ok=True)
        for name, frame in group.items():
            frame.to_csv(directory / f"{name}.csv", index=False)
    return run


# ------------------------------------------- the real columns are what is read

@pytest.mark.parametrize("name", ["tau_curves", "tau_gaps", "tau_group_panels"])
def test_real_reduction_tables_draw_every_tau_figure(real_fixed, tmp_path, name):
    written = figures.write_all_figures(real_fixed, tmp_path, dataset="coffee",
                                        figures=[name], regime=figures.REGIME_FIXED)
    assert set(written[name]) == {"pdf", "png", "csv"}
    frame = pd.read_csv(written[name]["csv"])
    assert not frame.empty
    assert list(frame.columns) == list(figures.PLOTTED_COLUMNS)


@pytest.mark.parametrize("name", ["beta_curves", "beta_gaps", "protocol_contrasts"])
def test_real_reduction_tables_draw_every_beta_figure(real_beta, tmp_path, name):
    written = figures.write_all_figures(real_beta, tmp_path, dataset="coffee", figures=[name])
    frame = pd.read_csv(written[name]["csv"])
    assert not frame.empty
    assert set(frame["evaluation_regime"]) == {figures.REGIME_FIXED}


def test_derived_labels_come_from_the_suite_not_from_the_table(real_fixed):
    """The real tables carry no display name or training protocol; both are
    constants of the suite, and the labels must still be right."""

    levels = real_fixed["method_levels"]
    for column in ("display_name", "training_protocol", "displayed_beside_protocol"):
        assert column not in levels.columns
    frame = figures.draw_tau_curves(real_fixed, regime=figures.REGIME_FIXED).frame
    assert not frame["display_name"].eq("").any()
    assert not frame["training_protocol"].eq("").any()
    plt.close("all")


def test_worst_group_figure_ignores_per_group_rows(real_fixed):
    """The reduction stacks quantities under ``quantity``; a worst-group figure
    must filter on it rather than draw a per-group row by accident."""

    polluted = dict(real_fixed)
    decoy = real_fixed["method_levels"].copy()
    decoy["quantity"] = "group_distortion"
    decoy["estimate"] = decoy["estimate"] + 5.0
    polluted["method_levels"] = pd.concat([real_fixed["method_levels"], decoy],
                                          ignore_index=True)
    frame = figures.draw_tau_curves(polluted, regime=figures.REGIME_FIXED).frame
    assert float(frame["y"].max()) < 5.0
    plt.close("all")


def test_level_intervals_are_drawn_as_the_reduction_saved_them(real_fixed):
    """A pilot run's df is not 99; its saved endpoints are drawn, not recomputed."""

    artifact = figures.draw_tau_curves(real_fixed, regime=figures.REGIME_FIXED)
    saved = real_fixed["method_levels"]
    saved = saved[saved["method"] == "multigroup"].sort_values("learner_tau")
    got = artifact.frame[artifact.frame["method"] == "multigroup"].sort_values("learner_tau")
    assert got["ci_low"].to_numpy() == pytest.approx(saved["ci_lower"].to_numpy())
    assert got["ci_high"].to_numpy() == pytest.approx(saved["ci_upper"].to_numpy())
    # ... and the caption quotes the df the table states, not an assumed one.
    assert f"df = {_REAL_DF}" in artifact.text
    plt.close("all")


def test_nlhf_in_a_real_active_tau_cell_is_still_the_passive_benchmark(real_fixed):
    """The reduction files NLHF under the *cell's* active protocol; the figure
    must not turn that into an active-trained NLHF."""

    assert set(real_fixed["method_levels"]["protocol"]) == {"active"}
    artifact = figures.draw_tau_curves(real_fixed, regime=figures.REGIME_FIXED)
    nlhf = artifact.frame[artifact.frame["method"] == "pooled_nlhf"]
    assert set(nlhf["protocol_trained"]) == {"passive"}
    assert nlhf["series_label"].str.contains("passive-trained benchmark").all()
    glhf = artifact.frame[artifact.frame["method"].isin(["multigroup", "pooled"])]
    assert set(glhf["protocol_trained"]) == {"active"}
    assert not re.search(r"\bactive NLHF\b", artifact.text, flags=re.I)
    plt.close("all")


def test_missing_drawn_column_in_a_real_table_still_fails_loudly(real_fixed, tmp_path):
    """Neither spelling of the drawn quantity present: refuse, naming both."""

    dropped, spelling = "estimate", "worst_group_distortion"
    broken = dict(real_fixed)
    broken["method_levels"] = real_fixed["method_levels"].drop(columns=[dropped])
    with pytest.raises(figures.MissingColumnsError) as excinfo:
        figures.write_all_figures(broken, tmp_path, figures=["tau_curves"],
                                  regime=figures.REGIME_FIXED)
    message = str(excinfo.value)
    assert dropped in message and spelling in message
    assert "method_levels" in message and "tau_curves" in message
    assert list(tmp_path.glob("*")) == []


def test_a_real_level_table_without_endpoints_falls_back_to_the_standard_error(real_fixed):
    """Endpoints are optional on a level row: the corrected SE still implies one."""

    trimmed = dict(real_fixed)
    trimmed["method_levels"] = real_fixed["method_levels"].drop(
        columns=["ci_lower", "ci_upper"])
    frame = figures.draw_tau_curves(trimmed, regime=figures.REGIME_FIXED).frame
    assert frame["ci_low"].notna().all() and frame["ci_high"].notna().all()
    assert (frame["ci_low"] < frame["y"]).all()
    plt.close("all")


def test_missing_contrast_endpoints_in_a_real_table_fail_loudly(real_fixed, tmp_path):
    broken = dict(real_fixed)
    broken["method_contrasts"] = real_fixed["method_contrasts"].drop(columns=["ci_lower"])
    with pytest.raises(figures.MissingColumnsError) as excinfo:
        figures.write_all_figures(broken, tmp_path, figures=["tau_gaps"],
                                  regime=figures.REGIME_FIXED)
    assert "ci_low" in str(excinfo.value) and "ci_lower" in str(excinfo.value)
    assert list(tmp_path.glob("*")) == []


# ------------------------------------------------- the regimes never mix

@pytest.mark.parametrize("name", ["tau_curves", "tau_gaps", "tau_group_panels"])
def test_a_moving_figure_holds_no_fixed_row_and_the_reverse(real_fixed, real_moving,
                                                            tmp_path, name):
    moving = figures.write_all_figures(real_moving, tmp_path / "moving", dataset="coffee",
                                       figures=[name], regime=figures.REGIME_MOVING)
    fixed = figures.write_all_figures(real_fixed, tmp_path / "fixed", dataset="coffee",
                                      figures=[name], regime=figures.REGIME_FIXED)
    moving_rows = pd.read_csv(moving[name]["csv"])
    fixed_rows = pd.read_csv(fixed[name]["csv"])
    assert set(moving_rows["evaluation_regime"]) == {figures.REGIME_MOVING}
    assert figures.REGIME_FIXED not in set(moving_rows["evaluation_regime"])
    assert set(fixed_rows["evaluation_regime"]) == {figures.REGIME_FIXED}
    assert figures.REGIME_MOVING not in set(fixed_rows["evaluation_regime"])
    # The estimands differ in the one way that matters: where they were scored.
    assert (fixed_rows["evaluation_tau"] == 0.5).all()
    assert (moving_rows["evaluation_tau"].to_numpy()
            == pytest.approx(moving_rows["learner_tau"].to_numpy()))


def test_one_table_that_mixes_the_regimes_is_refused(real_fixed, real_moving, tmp_path):
    mixed = dict(real_fixed)
    mixed["method_levels"] = pd.concat(
        [real_fixed["method_levels"], real_moving["method_levels"]], ignore_index=True)
    with pytest.raises(figures.RegimeMismatchError) as excinfo:
        figures.write_all_figures(mixed, tmp_path, figures=["tau_curves"])
    assert "moving" in str(excinfo.value) and "fixed_0.5" in str(excinfo.value)
    assert list(tmp_path.glob("*")) == []


def test_asking_the_other_regime_for_a_tables_rows_is_refused(real_moving, tmp_path):
    with pytest.raises(figures.RegimeMismatchError) as excinfo:
        figures.write_all_figures(real_moving, tmp_path, figures=["tau_curves"],
                                  regime=figures.REGIME_FIXED)
    message = str(excinfo.value)
    assert "moving" in message and "fixed_0.5" in message
    assert list(tmp_path.glob("*")) == []


def test_an_unknown_regime_is_refused(real_fixed, tmp_path):
    with pytest.raises(figures.RegimeMismatchError, match="unknown evaluation regime"):
        figures.write_all_figures(real_fixed, tmp_path, figures=["tau_curves"],
                                  regime="whatever_0.5")


# ------------------------------------ the regime is on the figure and the file

@pytest.mark.parametrize("regime", sorted(figures.REGIMES))
@pytest.mark.parametrize("name", ["tau_curves", "tau_gaps", "tau_group_panels"])
def test_the_regime_is_in_the_file_name_and_in_the_rendered_text(
        real_fixed, real_moving, tmp_path, regime, name):
    source = real_fixed if regime == figures.REGIME_FIXED else real_moving
    figures.configure_style()
    artifact = figures.DRAWERS[name](source, regime=regime)
    paths = figures.write_figure(artifact, tmp_path, dataset="coffee", close=False)
    for kind, path in paths.items():
        assert path.name == f"coffee_{name}_{regime}.{kind}", kind
        assert path.is_file()
    text = artifact.text
    assert regime in text, "the regime string itself must be on the figure"
    assert figures.REGIME_MEANING[regime] in text, "and so must what it means"
    other = figures.REGIME_MOVING if regime == figures.REGIME_FIXED else figures.REGIME_FIXED
    assert other not in text, "the other regime is never named on this figure"
    plt.close(artifact.figure)


@pytest.mark.parametrize("name", ["tau_curves", "tau_gaps", "tau_group_panels"])
def test_the_moving_caption_never_claims_a_fixed_evaluation_radius(real_moving, name):
    figures.configure_style()
    artifact = figures.DRAWERS[name](real_moving, regime=figures.REGIME_MOVING)
    text = artifact.text.lower()
    assert "fixed" not in text, "a moving-tau figure must not use the word at all"
    assert "moves with the learner radius" in text
    assert "evaluation tau = learner tau" in text
    # The axis label says it too, for a reader who crops the caption.
    labels = [ax.get_xlabel() for ax in artifact.figure.axes if ax.get_xlabel()]
    assert any("moves with it" in label for label in labels)
    assert figures.TAU_CAPTION not in artifact.text
    plt.close(artifact.figure)


def test_the_fixed_caption_still_says_the_radius_is_held_fixed(real_fixed):
    figures.configure_style()
    artifact = figures.draw_tau_curves(real_fixed, regime=figures.REGIME_FIXED)
    assert figures.TAU_CAPTION in artifact.text
    assert "evaluation radius is held fixed at tau = 0.5" in artifact.text
    plt.close(artifact.figure)


def test_beta_figures_name_their_regime_too(real_beta):
    figures.configure_style()
    artifact = figures.draw_beta_curves(real_beta)
    assert figures.REGIME_FIXED in artifact.text
    assert figures.REGIME_MEANING[figures.REGIME_FIXED] in artifact.text
    plt.close(artifact.figure)


def test_infinity_stays_off_the_finite_axis_with_real_tables(real_beta):
    """The existing broken-axis guarantee, checked against the real schema."""

    figures.configure_style()
    artifact = figures.draw_beta_curves(real_beta)
    finite = [ax for ax in artifact.figure.axes
              if ax.get_label() == figures.FINITE_AXES_LABEL]
    infinite = [ax for ax in artifact.figure.axes
                if ax.get_label() == figures.INF_AXES_LABEL]
    assert len(finite) == 1 and len(infinite) == 1
    assert finite[0].get_xscale() == "log"
    assert all(np.isfinite(float(t)) for t in finite[0].get_xticks())
    frame = artifact.frame
    assert set(frame[frame["axis_segment"] == "finite"]["beta"]).isdisjoint({"inf"})
    assert set(frame[frame["axis_segment"] == "infinity"]["beta"]) == {"inf"}
    marks = [line for ax in (finite[0], infinite[0]) for line in ax.get_lines()
             if line.get_gid() == figures.BREAK_MARK_GID]
    assert marks
    plt.close(artifact.figure)


# --------------------------------------------------- the run-level entry points

def test_write_regime_figures_reads_one_regimes_directory(tmp_path, real_beta,
                                                          real_fixed, real_moving):
    run = _write_run(tmp_path, real_beta, real_fixed, real_moving)
    out = tmp_path / "figures"
    written = figures.write_regime_figures(run, out, dataset="coffee",
                                           regime=figures.REGIME_MOVING)
    assert set(written) == set(figures.TAU_FIGURES)
    for name, paths in written.items():
        assert paths["png"].name == f"coffee_{name}_moving.png"
        frame = pd.read_csv(paths["csv"])
        assert set(frame["evaluation_regime"]) == {figures.REGIME_MOVING}
    # Nothing of the other regime was written into the same directory.
    assert not list(out.glob("*_fixed_0.5.*"))


def test_write_beta_figures_reads_the_beta_directory(tmp_path, real_beta,
                                                     real_fixed, real_moving):
    run = _write_run(tmp_path, real_beta, real_fixed, real_moving)
    written = figures.write_beta_figures(run, tmp_path / "figures", dataset="persona")
    assert set(written) == set(figures.BETA_FIGURES)
    for name, paths in written.items():
        assert paths["pdf"].name == f"persona_{name}_fixed_0.5.pdf"


def test_write_run_figures_writes_both_regimes_and_the_beta_condition(
        tmp_path, real_beta, real_fixed, real_moving):
    run = _write_run(tmp_path, real_beta, real_fixed, real_moving)
    out = tmp_path / "figures"
    written = figures.write_run_figures(run, out, dataset="coffee")
    assert set(written) == (set(figures.BETA_FIGURES)
                            | {f"{n}:{r}" for n in figures.TAU_FIGURES
                               for r in figures.REGIMES})
    names = {path.name for path in out.glob("*.png")}
    assert "coffee_tau_curves_fixed_0.5.png" in names
    assert "coffee_tau_curves_moving.png" in names
    # Every tau figure exists once per regime, and the two never share a file.
    assert len(names) == len(figures.BETA_FIGURES) + 2 * len(figures.TAU_FIGURES)


def test_a_missing_reduction_directory_fails_loudly(tmp_path):
    with pytest.raises(figures.MissingTableError, match="no suite reduction directory"):
        figures.write_regime_figures(tmp_path / "nowhere", tmp_path / "out",
                                     regime=figures.REGIME_FIXED)
