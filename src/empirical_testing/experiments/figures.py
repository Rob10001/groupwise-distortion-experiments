"""The six suite figure families, drawn from the suite reduction tables.

This module is the figure half of the contract in
:mod:`empirical_testing.experiments.schema`: it consumes the tables that
module defines (``method_levels``, ``method_group_levels``,
``method_contrasts``, ``protocol_contrasts``) and writes, for each figure named
in :data:`schema.FIGURES`, a **PDF, a PNG and the CSV that was actually
plotted**.  The CSV carries one row per drawn point, with the plotted ``x`` and
``y`` beside the cell identity, so a reader can check the picture against the
numbers rather than trusting it.

Four commitments this module makes, each of which is tested:

**Training protocols are named in every series label.**  No series is called
just "NLHF" or just "Multigroup GLHF".  NLHF is passive-trained in every
condition of this study, the active ones included, so in the active conditions
it is drawn as *the passive-trained benchmark* and never as an "active NLHF"
(see ``TRAINING_PROTOCOL`` in :mod:`empirical_testing.experiments.suite`).
The uniform reference is labelled as not fitted at all.

**Infinity is kept off the finite log axis.**  ``beta = inf`` has no finite log
coordinate.  The beta figures therefore put the finite grid on a log axis and
``inf`` in a **separate axes** to its right, marked with a break, with nothing
joining the two.  ``inf`` is never given a made-up numeric position.

**The tau figures say what tau is, and which evaluation regime they show.**
Tau is the *learner* radius.  The evaluation radius is a **separate** choice,
and suite reduces two of them as two different estimands: ``fixed_0.5`` scores
every policy at evaluation tau = 0.50, and ``moving`` scores each policy at an
evaluation tau equal to its own learner tau.  The two are never mixed in one
table or one figure.  Every tau figure names its regime in the **title**, in
the **caption** and in the **file name**, so a moving-tau curve can never be
mistaken for the fixed-tau comparison; the fixed-tau caption is the only one
that says the evaluation radius is held fixed.

**Intervals are marginal.**  Every interval drawn here is a Nadeau-Bengio
corrected repeated-CV interval, marginal per cell.  It is not a simultaneous
band over the grid and not a respondent bootstrap band, and the caption says
so.  Where a reduction saves interval endpoints they are drawn as saved; where
a level table carries only a corrected standard error the endpoints are formed
here as ``estimate +/- t(0.975, df) * corrected_se``.  The degrees of freedom
the caption quotes come from the table when it states them.

Output goes to an explicit directory.  Production callers pass the repository's
``figures/coffee`` or ``figures/persona`` (``Suite.figure_directory``
carries it, repository-relative); tests pass a temporary directory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from empirical_testing.experiments import schema
from empirical_testing.experiments.engine import (
    REGIME_FIXED,
    REGIME_MOVING,
    REGIMES,
)
from empirical_testing.experiments.suite import (
    ACTIVE,
    ACTIVE_BETAS,
    DISPLAY_NAMES,
    METHODS,
    PASSIVE,
    TRAINING_PROTOCOL,
    Suite,
)

__all__ = [
    "MissingTableError",
    "MissingColumnsError",
    "RegimeMismatchError",
    "FigureArtifact",
    "FIGURE_INPUTS",
    "REQUIRED_COLUMNS",
    "INTERVAL_CAPTION",
    "TAU_CAPTION",
    "TAU_AXIS_LABEL",
    "REGIME_FIXED",
    "REGIME_MOVING",
    "REGIMES",
    "REGIME_MEANING",
    "TAU_FIGURES",
    "BETA_FIGURES",
    "interval_caption",
    "tau_caption",
    "tau_axis_label",
    "regime_title",
    "resolve_regime",
    "normalize_table",
    "read_reduction_tables",
    "write_regime_figures",
    "write_beta_figures",
    "write_run_figures",
    "series_label",
    "draw_beta_curves",
    "draw_beta_gaps",
    "draw_protocol_contrasts",
    "draw_tau_curves",
    "draw_tau_gaps",
    "draw_tau_group_panels",
    "write_figure",
    "write_all_figures",
    "escape_mathtext",
    "suite_figure_directory",
]


# --------------------------------------------------------------- house style

# Okabe-Ito, the same per-method assignment the Coffee and Persona figures use.
# Identity never rests on hue alone: every method also has its own marker.
METHOD_STYLE: Mapping[str, dict[str, Any]] = {
    "multigroup": dict(color="#D55E00", marker="s"),
    "pooled": dict(color="#0072B2", marker="o"),
    "pooled_nlhf": dict(color="#7A5195", marker="^"),
    "uniform": dict(color="#5B6470", marker="D"),
}
#: The training protocol is carried by line style as well as by the label, so
#: a reader who is looking rather than reading still sees two protocols.
PROTOCOL_LINESTYLE: Mapping[str, Any] = {
    PASSIVE: "-",
    ACTIVE: (0, (4.0, 1.8)),
}
#: The two regimes are also told apart without reading: the title sits in a
#: differently coloured badge in each.  The words remain the guarantee; this
#: only makes a mix-up visible at a glance.
REGIME_BADGE: Mapping[str, dict[str, Any]] = {
    REGIME_FIXED: dict(facecolor="#EAF2FA", edgecolor="#8FB4D6"),
    REGIME_MOVING: dict(facecolor="#FDF1E3", edgecolor="#DFA45B"),
}
GRID = "#E3E3E3"
TEXT = "#222222"
RULE = "#8A929E"
ZERO = "#8A929E"

#: Set as the matplotlib ``gid`` of every drawn data series, so a test can pair
#: what is on the axes with what is in the CSV.
SERIES_GID_PREFIX = "series:"
BAND_GID_PREFIX = "band:"
#: The ``gid`` of the diagonal marks that show the finite axis is broken.
BREAK_MARK_GID = "beta_axis_break"
#: ``Axes.get_label()`` values for the two halves of a broken beta axis.
FINITE_AXES_LABEL = "beta_finite_log_axis"
INF_AXES_LABEL = "beta_infinity_slot"

def interval_caption(df: Any = 99) -> str:
    """The interval sentence, quoting the degrees of freedom actually used."""

    return (
        f"Intervals are Nadeau-Bengio corrected repeated-CV 95% intervals (df = {df}), "
        "marginal per cell. They are not simultaneous over the grid and are not "
        "respondent bootstrap intervals."
    )


INTERVAL_CAPTION = interval_caption(99)

#: What each evaluation regime *means*, in the words the figures must carry.
#: The two regimes are different estimands; a figure never mixes their rows and
#: never describes one in the other's words.
REGIME_MEANING: Mapping[str, str] = {
    REGIME_FIXED: "evaluation radius fixed at tau = 0.50 at every learner radius",
    REGIME_MOVING: (
        "evaluation radius moves with the learner radius (evaluation tau = learner tau)"
    ),
}
assert set(REGIME_MEANING) == set(REGIMES)

#: The tau caption, per regime.  The moving caption never says the evaluation
#: radius is fixed, because in that regime it is not.
TAU_CAPTION = (
    "tau is the LEARNER radius; the evaluation radius is held fixed at tau = 0.5 "
    "at every point, so the vertical axis means the same thing throughout."
)
TAU_CAPTION_MOVING = (
    "tau is the LEARNER radius, and in this regime the evaluation radius MOVES with "
    "it (evaluation tau = learner tau), so each point is a different estimand: this "
    "panel is a diagnostic, not the cross-radius comparison."
)
TAU_CAPTIONS: Mapping[str, str] = {
    REGIME_FIXED: TAU_CAPTION,
    REGIME_MOVING: TAU_CAPTION_MOVING,
}

TAU_AXIS_LABEL = r"Learner radius $\tau$ (evaluation radius fixed at $\tau = 0.5$)"
TAU_AXIS_LABEL_MOVING = (
    r"Learner radius $\tau$ (evaluation radius moves with it: "
    r"evaluation $\tau$ = learner $\tau$)"
)
TAU_AXIS_LABELS: Mapping[str, str] = {
    REGIME_FIXED: TAU_AXIS_LABEL,
    REGIME_MOVING: TAU_AXIS_LABEL_MOVING,
}
BETA_AXIS_LABEL = r"Label parameter $\beta$ (log scale; $\infty$ in its own panel)"


def tau_caption(regime: str) -> str:
    _check_regime(regime)
    return TAU_CAPTIONS[regime]


def tau_axis_label(regime: str) -> str:
    _check_regime(regime)
    return TAU_AXIS_LABELS[regime]


def regime_title(headline: str, regime: str) -> str:
    """The two-line title every figure carries: what it is, and which regime.

    The regime *string* and the regime's *meaning* are both on the figure, so a
    reader who never sees the file name still cannot mistake one estimand for
    the other.
    """

    _check_regime(regime)
    return f"{headline}  |  evaluation regime: {regime}\n{regime}: {REGIME_MEANING[regime]}"


def _check_regime(regime: str) -> None:
    if regime not in REGIME_MEANING:
        raise RegimeMismatchError(
            f"unknown evaluation regime {regime!r}; suite reduces exactly {list(REGIMES)}, "
            "and their rows are separate estimands that are never mixed"
        )
NLHF_NOTE = (
    "NLHF is passive-trained in every condition of this study and is drawn here as "
    "the passive-trained benchmark, never as an active learner."
)

#: t(0.975, 99): the multiplier that turns a corrected standard error into the
#: marginal interval the level tables imply.
T_CRITICAL_DF99 = 1.9842


def configure_style() -> None:
    """The manuscript's house style, matching the Coffee and Persona scripts."""

    mpl.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.titleweight": "normal",
        "axes.edgecolor": TEXT,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
    })


# ------------------------------------------------------------- input contract

class MissingTableError(KeyError):
    """A figure was asked for without the reduction table it reads."""


class MissingColumnsError(ValueError):
    """A reduction table is missing columns the figure layer requires."""


class RegimeMismatchError(ValueError):
    """Rows of one evaluation regime were handed to the other's figure."""


#: What each figure actually **uses**, as *field name -> the column names that
#: may carry it*.  This is deliberately not equality with ``schema.TABLES``:
#: the real reduction writes many more columns (``status``, ``p_value``,
#: ``variance_factor``, provenance), and it names some of the drawn ones
#: differently (``estimate``/``corrected_se`` on a level row rather than
#: ``worst_group_distortion``; ``ci_lower``/``ci_upper`` rather than
#: ``ci_low``/``ci_high``).  A figure must tolerate the extra columns and accept
#: either spelling of a drawn one -- and must still fail loudly, naming every
#: acceptable spelling, when the quantity it draws is absent altogether.
#:
#: Fields that can be *derived* from the method identity (display name,
#: training protocol) are not listed: they are constants of the suite, not data.
REQUIRED_COLUMNS: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "method_levels": {
        "condition": ("condition",),
        "protocol": ("protocol",),
        "beta": ("beta",),
        "learner_tau": ("learner_tau",),
        "evaluation_tau": ("evaluation_tau",),
        "method": ("method",),
        "estimate": ("estimate", "worst_group_distortion"),
        "corrected_se": ("corrected_se", "worst_group_se"),
    },
    "method_group_levels": {
        "condition": ("condition",),
        "protocol": ("protocol",),
        "beta": ("beta",),
        "learner_tau": ("learner_tau",),
        "evaluation_tau": ("evaluation_tau",),
        "method": ("method",),
        "group": ("group",),
        "estimate": ("estimate", "distortion"),
        "corrected_se": ("corrected_se", "distortion_se"),
    },
    "method_contrasts": {
        "condition": ("condition",),
        "protocol": ("protocol",),
        "beta": ("beta",),
        "learner_tau": ("learner_tau",),
        "evaluation_tau": ("evaluation_tau",),
        "contrast": ("contrast",),
        "minuend": ("minuend",),
        "subtrahend": ("subtrahend",),
        "estimate": ("estimate",),
        "ci_low": ("ci_low", "ci_lower"),
        "ci_high": ("ci_high", "ci_upper"),
    },
    "protocol_contrasts": {
        "condition": ("condition",),
        "beta": ("beta",),
        "learner_tau": ("learner_tau",),
        "evaluation_tau": ("evaluation_tau",),
        "method": ("method",),
        "contrast": ("contrast",),
        "estimate": ("estimate",),
        "ci_low": ("ci_low", "ci_lower"),
        "ci_high": ("ci_high", "ci_upper"),
    },
}

#: The ``quantity`` a table's rows must carry to be the one a figure draws.
#: The real reduction stacks worst-group and per-group rows in sibling tables
#: that share a schema, so the figure filters rather than assuming.
TABLE_QUANTITY: Mapping[str, str] = {
    "method_levels": "worst_group_distortion",
    "method_group_levels": "group_distortion",
    "method_contrasts": "worst_group_distortion",
    "method_group_contrasts": "group_distortion",
    "protocol_contrasts": "worst_group_distortion",
}

#: Which table each figure reads.
FIGURE_INPUTS: Mapping[str, tuple[str, ...]] = {
    "beta_curves": ("method_levels",),
    "beta_gaps": ("method_contrasts",),
    "protocol_contrasts": ("protocol_contrasts",),
    "tau_curves": ("method_levels",),
    "tau_gaps": ("method_contrasts",),
    "tau_group_panels": ("method_group_levels",),
}

#: The plotted-CSV columns, identical for every figure so the six CSVs can be
#: read with one reader.  ``x`` and ``y`` are the coordinates actually drawn.
PLOTTED_COLUMNS: tuple[str, ...] = (
    "figure", "panel", "series", "series_label", "method", "display_name",
    "training_protocol", "protocol_trained", "displayed_beside_protocol",
    "condition", "evaluation_regime", "beta", "learner_tau", "evaluation_tau", "group",
    "axis_segment", "x", "y", "ci_low", "ci_high", "p_holm", "interval",
)

INTERVAL_KIND = "nadeau_bengio_corrected_repeated_cv_95_marginal_not_simultaneous"


def require_table(tables: Mapping[str, pd.DataFrame], name: str, *, figure: str,
                  regime: str | None = None) -> pd.DataFrame:
    """The named table, normalized, or a loud failure naming the figure.

    ``regime``, when given, is enforced: a table whose rows belong to the other
    evaluation regime is refused rather than drawn under the wrong caption.
    """

    if name not in tables or tables[name] is None:
        raise MissingTableError(
            f"figure {figure!r} reads the suite reduction table {name!r}, which was not "
            f"supplied; available tables: {sorted(tables)}"
        )
    frame = tables[name]
    require_columns(frame, name, figure=figure)
    normalized = normalize_table(frame, name)
    if regime is not None:
        _reject_other_regime(normalized, name, figure=figure, regime=regime)
    return normalized


def require_columns(frame: pd.DataFrame, name: str, *, figure: str) -> None:
    """Fail loudly, naming every missing column, before anything is drawn.

    A field is satisfied by *any* of its accepted spellings; the message names
    all of them, so a caller whose table uses neither knows what to supply.
    """

    required = REQUIRED_COLUMNS.get(name)
    if required is None:
        raise MissingTableError(
            f"{name!r} is not a suite reduction table the figure layer reads; "
            f"expected one of {sorted(REQUIRED_COLUMNS)}"
        )
    present = set(frame.columns)
    missing = [
        accepted[0] if len(accepted) == 1 else f"{accepted[0]} (or {' / '.join(accepted[1:])})"
        for accepted in required.values()
        if not present.intersection(accepted)
    ]
    if missing:
        raise MissingColumnsError(
            f"suite table {name!r} is missing {len(missing)} column(s) required by figure "
            f"{figure!r}: {missing}. The suite figure layer draws exactly the columns it "
            f"names here and refuses to guess or substitute them; columns present: "
            f"{sorted(frame.columns)}"
        )


# ------------------------------------------------- the real reduction's rows

#: Level and contrast columns the figure layer draws, and the spellings the
#: reduction may have written them under.  Applied after ``require_columns``.
_ALIASES: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "method_levels": {
        "worst_group_distortion": ("estimate",),
        "worst_group_se": ("corrected_se",),
        "ci_low": ("ci_lower",),
        "ci_high": ("ci_upper",),
    },
    "method_group_levels": {
        "distortion": ("estimate",),
        "distortion_se": ("corrected_se",),
        "ci_low": ("ci_lower",),
        "ci_high": ("ci_upper",),
    },
    "method_contrasts": {"ci_low": ("ci_lower",), "ci_high": ("ci_upper",)},
    "method_group_contrasts": {"ci_low": ("ci_lower",), "ci_high": ("ci_upper",)},
    "protocol_contrasts": {"ci_low": ("ci_lower",), "ci_high": ("ci_upper",)},
}

_LEVEL_TABLES = ("method_levels", "method_group_levels")

#: The only methods this study ever trains actively.  NLHF is the
#: passive-trained benchmark everywhere and the uniform reference is never
#: fitted, so a row of either sitting in an active cell is displayed beside the
#: active protocol rather than trained under it.
ACTIVELY_TRAINED_METHODS: tuple[str, ...] = ("multigroup", "pooled")


def normalize_table(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """The reduction's rows in the shape the drawers read.

    The real reduction tables carry more columns than the figures draw, stack
    the worst-group and per-group quantities under a ``quantity`` column, spell
    the interval endpoints ``ci_lower``/``ci_upper``, and record the *cell's*
    protocol rather than a display name or a training protocol.  Nothing is
    invented here: display names and training protocols are constants of the
    suite, and which methods are ever trained actively is a fact of the design.
    """

    out = frame.copy()

    # One quantity per figure: a worst-group figure never draws a per-group row.
    quantity = TABLE_QUANTITY.get(name)
    if quantity is not None and "quantity" in out.columns:
        out = out[out["quantity"].astype(str) == quantity]
    # Rows the reduction could not estimate carry no drawable number.
    if "status" in out.columns:
        out = out[out["status"].astype(str) == "ok"]
    out = out.copy()

    for canonical, alternatives in _ALIASES.get(name, {}).items():
        if canonical in out.columns:
            continue
        for alternative in alternatives:
            if alternative in out.columns:
                out[canonical] = out[alternative]
                break

    if "evaluation_regime" not in out.columns:
        out["evaluation_regime"] = ""

    if name in _LEVEL_TABLES:
        methods = out["method"].astype(str)
        if "display_name" not in out.columns:
            out["display_name"] = [DISPLAY_NAMES.get(m, m) for m in methods]
        if "training_protocol" not in out.columns:
            out["training_protocol"] = [TRAINING_PROTOCOL.get(m, "") for m in methods]
        if "displayed_beside_protocol" not in out.columns:
            # The real tables carry the *cell's* protocol, which is where a row
            # is displayed, not necessarily how its method was trained.  NLHF
            # is the passive-trained benchmark in every condition of this study
            # and the uniform reference is never fitted at all, so a row of
            # either standing in an active cell is passive-trained and
            # *displayed beside* the active protocol.  ``provenance`` says
            # where a number came from, not how it was trained, and is not used
            # for this: a shared active GLHF row is still active-trained.
            passive_only = ~methods.isin(ACTIVELY_TRAINED_METHODS)
            protocols = out["protocol"].astype(str)
            out["displayed_beside_protocol"] = np.where(
                passive_only & (protocols == ACTIVE), ACTIVE, "")
            out["protocol"] = np.where(passive_only, PASSIVE, protocols)
    elif name in ("method_contrasts", "method_group_contrasts"):
        if "minuend_training_protocol" not in out.columns:
            out["minuend_training_protocol"] = [
                TRAINING_PROTOCOL.get(str(m), "") for m in out["minuend"]]
        if "subtrahend_training_protocol" not in out.columns:
            out["subtrahend_training_protocol"] = [
                TRAINING_PROTOCOL.get(str(m), "") for m in out["subtrahend"]]
    elif name == "protocol_contrasts":
        if "display_name" not in out.columns:
            out["display_name"] = [DISPLAY_NAMES.get(str(m), str(m)) for m in out["method"]]

    if "p_holm" not in out.columns:
        # Holm adjusts decisions within a stated family; where the reduction
        # ran no family there is no adjusted p, and the figure says nothing.
        out["p_holm"] = math.nan
    return out


def table_degrees_of_freedom(frame: pd.DataFrame) -> Any:
    """The df the table states, for the interval caption; ``99`` if it is mute."""

    for column in ("degrees_of_freedom", "df"):
        if column in frame.columns:
            values = sorted({v for v in frame[column].tolist() if not _blank(v)})
            if len(values) == 1:
                value = values[0]
                return int(value) if float(value).is_integer() else value
            if values:
                return "varies by cell: " + ", ".join(f"{v:g}" for v in map(float, values))
    return 99


# ------------------------------------------------------ the evaluation regime

def frame_regimes(frame: pd.DataFrame) -> set[str]:
    if "evaluation_regime" not in frame.columns:
        return set()
    return {str(v) for v in frame["evaluation_regime"].tolist() if not _blank(v)}


def _reject_other_regime(frame: pd.DataFrame, name: str, *, figure: str, regime: str) -> None:
    _check_regime(regime)
    found = frame_regimes(frame)
    wrong = sorted(found - {regime})
    if wrong:
        raise RegimeMismatchError(
            f"figure {figure!r} was asked for the {regime!r} evaluation regime, but the "
            f"suite table {name!r} carries rows of regime(s) {wrong}. The regimes are "
            f"different estimands ({REGIME_MEANING[regime]} vs the other), and suite never "
            f"mixes their rows in one table or one figure."
        )


def resolve_regime(frame: pd.DataFrame, *, regime: str | None = None,
                   figure: str = "") -> str:
    """Which evaluation regime a frame's rows belong to.

    An explicit ``regime`` wins and is checked against the rows.  Otherwise the
    rows' own ``evaluation_regime`` decides; a table that states none is read
    off its radii, which is unambiguous: an evaluation radius equal to 0.5
    everywhere is the fixed regime, one equal to the learner radius is moving.
    """

    if regime is not None:
        _check_regime(regime)
        _reject_other_regime(frame, "<frame>", figure=figure or "<figure>", regime=regime)
        return regime
    stated = frame_regimes(frame)
    if len(stated) > 1:
        raise RegimeMismatchError(
            f"a suite table handed to figure {figure!r} mixes evaluation regimes "
            f"{sorted(stated)}; they are different estimands and are reduced, tabulated "
            "and drawn separately"
        )
    if len(stated) == 1:
        found = stated.pop()
        _check_regime(found)
        return found
    learner = np.asarray([float(t) for t in frame["learner_tau"]], dtype=float)
    evaluation = np.asarray([float(t) for t in frame["evaluation_tau"]], dtype=float)
    if len(evaluation) and np.allclose(evaluation, 0.5):
        return REGIME_FIXED
    if len(evaluation) and np.allclose(evaluation, learner):
        return REGIME_MOVING
    raise RegimeMismatchError(
        f"the table handed to figure {figure!r} states no evaluation_regime and its "
        "evaluation radii match neither regime (fixed at 0.5, or equal to the learner "
        "radius); refusing to guess which estimand it is"
    )


# -------------------------------------------------------------------- labels

def series_label(method: str, protocol_trained: str, *,
                 displayed_beside: str | None = None) -> str:
    """A legend entry that says how the method was **trained**.

    Every label carries its training protocol.  NLHF's says *benchmark* and
    *passive-trained* wherever it appears, because it is the same passive fit
    in the active conditions as in the passive ones; the uniform reference's
    says it is not fitted at all.
    """

    display = DISPLAY_NAMES[method]
    if method == "uniform":
        return f"{display} (not fitted)"
    if method == "pooled_nlhf":
        if displayed_beside == ACTIVE:
            return f"{display} (passive-trained benchmark, shown beside active)"
        return f"{display} (passive-trained benchmark)"
    if protocol_trained == ACTIVE:
        return f"{display} (active-trained)"
    return f"{display} (passive-trained)"


def contrast_label(minuend: str, subtrahend: str, protocol_trained: str) -> str:
    """A gap's legend entry, with both arms' training protocols spelled out."""

    right = series_label(subtrahend, protocol_trained)
    if minuend in ("multigroup", "pooled"):
        # Both arms are GLHF trained the same way; saying so once is shorter
        # than saying it twice, and no less explicit.
        both = "active-trained" if protocol_trained == ACTIVE else "passive-trained"
        return (f"{DISPLAY_NAMES[minuend]} $-$ {DISPLAY_NAMES[subtrahend]} "
                f"(both {both})")
    left = series_label(minuend, protocol_trained)
    return f"{left} $-$ {right}"


# ------------------------------------------------------------------ plumbing

@dataclass
class FigureArtifact:
    """One drawn figure, its plotted table and the text it carries."""

    name: str
    figure: plt.Figure
    frame: pd.DataFrame
    caption: str
    legend_labels: tuple[str, ...] = ()
    paths: dict[str, Path] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Every string the rendered figure carries, for checking captions."""

        return "\n".join(rendered_text(self.figure) + list(self.legend_labels))


def rendered_text(figure: plt.Figure) -> list[str]:
    """All text on a rendered figure: titles, labels, ticks, legends, notes."""

    out: list[str] = []
    for text in figure.texts:
        out.append(text.get_text())
    for ax in figure.axes:
        out.append(ax.get_title())
        out.append(ax.get_title(loc="left"))
        out.append(ax.get_xlabel())
        out.append(ax.get_ylabel())
        out.extend(t.get_text() for t in ax.texts)
        out.extend(t.get_text() for t in ax.get_xticklabels())
        out.extend(t.get_text() for t in ax.get_yticklabels())
        legend = ax.get_legend()
        if legend is not None:
            out.extend(t.get_text() for t in legend.get_texts())
    for legend in figure.legends:
        out.extend(t.get_text() for t in legend.get_texts())
    return [t for t in out if t]


def _beta_text(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in {"inf", "infinity", "+inf"}:
        return "inf"
    return f"{float(text):g}"


def _beta_sort_key(text: str) -> float:
    return math.inf if text == "inf" else float(text)


def _finite_betas(frame: pd.DataFrame) -> list[str]:
    labels = {_beta_text(b) for b in frame["beta"]}
    return sorted((b for b in labels if b != "inf"), key=float)


def _row(**kwargs: Any) -> dict[str, Any]:
    row = {column: kwargs.get(column, "") for column in PLOTTED_COLUMNS}
    row["interval"] = INTERVAL_KIND
    return row


def _interval(estimate: float, se: float) -> tuple[float, float]:
    """The marginal interval a corrected standard error implies at df = 99."""

    if se is None or (isinstance(se, float) and math.isnan(se)):
        return math.nan, math.nan
    half = T_CRITICAL_DF99 * float(se)
    return estimate - half, estimate + half


def _level_intervals(block: pd.DataFrame, estimates: np.ndarray,
                     ses: np.ndarray) -> tuple[list[float], list[float]]:
    """Saved endpoints where the reduction wrote them, else formed from the SE.

    The reduction knows its own degrees of freedom (a pilot run has far fewer
    than a production one), so its endpoints are drawn as saved rather than
    recomputed against an assumed df.
    """

    if "ci_low" in block.columns and "ci_high" in block.columns:
        low = block["ci_low"].astype(float).to_numpy()
        high = block["ci_high"].astype(float).to_numpy()
        if not (np.isnan(low).any() or np.isnan(high).any()):
            return list(low), list(high)
    pairs = [_interval(e, s) for e, s in zip(estimates, ses)]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def escape_mathtext(text: Any) -> str:
    """A label that matplotlib will not read as mathtext.

    Group names carry dollar signs (``spend:$40-$60``); left alone, matplotlib
    treats the pair as a math span and silently reformats the label.
    """

    return str(text).replace("$", r"\$")


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


# --------------------------------------------------------- the broken x axis

def make_beta_axes(fig: plt.Figure, subspec=None, *, finite: Sequence[str],
                   xlabel: str = BETA_AXIS_LABEL, slot_note: bool = True):
    """A log axis for the finite betas plus a **separate** axes for infinity.

    Infinity is not a point on a log axis, so it does not get one.  The finite
    grid is drawn on a log scale; ``beta = inf`` is drawn in its own axes to
    the right, the two are separated by diagonal break marks, and no line ever
    joins a finite point to the infinite one.
    """

    if not finite:
        raise ValueError("a beta figure needs at least one finite beta on the grid")
    if subspec is None:
        inner = fig.add_gridspec(1, 2, width_ratios=(7.0, 1.0), wspace=0.07)
    else:
        inner = subspec.subgridspec(1, 2, width_ratios=(7.0, 1.0), wspace=0.07)
    ax_finite = fig.add_subplot(inner[0, 0])
    ax_inf = fig.add_subplot(inner[0, 1], sharey=ax_finite)
    ax_finite.set_label(FINITE_AXES_LABEL)
    ax_inf.set_label(INF_AXES_LABEL)

    ticks = [float(b) for b in finite]
    ax_finite.set_xscale("log")
    ax_finite.set_xticks(ticks)
    ax_finite.set_xticks([], minor=True)
    ax_finite.set_xticklabels([f"{t:g}" for t in ticks])
    ax_finite.set_xlim(min(ticks) / 1.45, max(ticks) * 1.45)
    ax_finite.set_xlabel(xlabel)
    ax_finite.spines["right"].set_visible(False)

    ax_inf.set_xlim(-1.0, 1.0)
    ax_inf.set_xticks([0.0])
    ax_inf.set_xticklabels([r"$\infty$"])
    ax_inf.spines["left"].set_visible(False)
    ax_inf.tick_params(axis="y", which="both", left=False, labelleft=False)
    if slot_note:
        ax_inf.set_title(r"$\beta=\infty$", loc="center", pad=4, fontsize=7.5, color=RULE)

    for ax in (ax_finite, ax_inf):
        ax.grid(True, color=GRID, linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)
    _break_marks(ax_finite, ax_inf)
    return ax_finite, ax_inf


def _break_marks(ax_finite, ax_inf) -> None:
    """The diagonal marks that say the axis is cut, not continuous."""

    kwargs = dict(color=TEXT, linewidth=0.8, clip_on=False, linestyle="none",
                  marker=[(-1, -0.6), (1, 0.6)], markersize=5,
                  markeredgewidth=0.8)
    right = ax_finite.plot([1, 1], [0, 1], transform=ax_finite.transAxes, **kwargs)
    left = ax_inf.plot([0, 0], [0, 1], transform=ax_inf.transAxes, **kwargs)
    for line in list(right) + list(left):
        line.set_gid(BREAK_MARK_GID)


def _inf_offsets(count: int) -> list[float]:
    """Where several series sit inside the infinity slot, so none overlaps."""

    if count <= 1:
        return [0.0]
    return list(np.linspace(-0.5, 0.5, count))


# ------------------------------------------------------------- drawing atoms

def _draw_curve(ax, xs: Sequence[float], ys: Sequence[float], *, style: Mapping[str, Any],
                linestyle: Any, gid: str, band: tuple[Sequence[float], Sequence[float]] | None,
                filled: bool) -> None:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    face = style["color"] if filled else "white"
    line, = ax.plot(xs, ys, color=style["color"], marker=style["marker"], markersize=3.8,
                    markerfacecolor=face, markeredgewidth=1.1, linewidth=1.3,
                    linestyle=linestyle if len(xs) > 1 else "none", zorder=3)
    line.set_gid(f"{SERIES_GID_PREFIX}{gid}")
    if band is None:
        return
    low, high = np.asarray(band[0], dtype=float), np.asarray(band[1], dtype=float)
    if len(xs) > 1:
        patch = ax.fill_between(xs, low, high, color=style["color"], alpha=0.10, linewidth=0)
    else:
        patch = ax.vlines(xs, low, high, color=style["color"], linewidth=1.1, alpha=0.55)
    patch.set_gid(f"{BAND_GID_PREFIX}{gid}")


def _legend(fig, handles: Sequence[Line2D], *, ncol: int, y: float = 0.93) -> tuple[str, ...]:
    fig.legend(handles=list(handles), loc="lower center", bbox_to_anchor=(0.5, y),
               ncol=ncol, columnspacing=1.2, handlelength=2.2)
    return tuple(h.get_label() for h in handles)


def _handle(method: str, protocol: str, label: str) -> Line2D:
    style = METHOD_STYLE[method]
    return Line2D([0], [0], color=style["color"], marker=style["marker"], markersize=3.8,
                  linewidth=1.3, linestyle=PROTOCOL_LINESTYLE[protocol],
                  markerfacecolor=style["color"] if protocol == PASSIVE else "white",
                  markeredgewidth=1.1, label=label)


def _caption(fig, text: str, *, y: float = -0.02) -> None:
    fig.text(0.5, y, text, ha="center", va="top", fontsize=6.6, color=RULE, wrap=True)


def _title(fig, headline: str, regime: str, *, y: float | None = None) -> None:
    """The headline and the evaluation regime, at the top of the canvas.

    Placed above whatever legend the figure already carries, measured rather
    than guessed, so a long legend pushes the title up instead of being
    overprinted by it.  ``savefig(bbox="tight")`` then grows the canvas to fit.
    """

    if y is None:
        y = 1.0
        if fig.legends:
            fig.canvas.draw()
            box = fig.legends[-1].get_window_extent(fig.canvas.get_renderer())
            y = max(y, box.transformed(fig.transFigure.inverted()).y1 + 0.035)
    fig.text(0.5, y, regime_title(headline, regime), ha="center", va="bottom",
             fontsize=8.6, color=TEXT, linespacing=1.35, gid="figure_title",
             bbox=dict(boxstyle="round,pad=0.35", linewidth=0.7,
                       **REGIME_BADGE[regime]))


def _finish(name: str, fig, rows: Sequence[Mapping[str, Any]], caption: str,
            labels: Sequence[str], regime: str) -> "FigureArtifact":
    """The artifact, with the regime stamped on every plotted row."""

    frame = pd.DataFrame(list(rows), columns=list(PLOTTED_COLUMNS))
    frame["evaluation_regime"] = regime
    return FigureArtifact(name, fig, frame, caption, tuple(labels))


# ----------------------------------------------------------- (1) beta curves

def draw_beta_curves(tables: Mapping[str, pd.DataFrame], *,
                     regime: str | None = None) -> FigureArtifact:
    """Worst-group distortion against beta, for the four passive methods.

    Active GLHF is overlaid at ``beta = 4`` and ``inf`` only -- the two betas
    at which it was run -- as dashed, open-marker series.  The NLHF column
    beside the active protocol is the *same* passive fit and is not drawn
    twice.
    """

    figure_name = "beta_curves"
    levels = require_table(tables, "method_levels", figure=figure_name, regime=regime)
    frame = levels[levels["condition"] == "beta"].copy()
    regime = resolve_regime(frame, regime=regime, figure=figure_name)
    frame["beta_text"] = [_beta_text(b) for b in frame["beta"]]
    finite = _finite_betas(frame)

    fig = plt.figure(figsize=(6.1, 3.2))
    ax_finite, ax_inf = make_beta_axes(fig, finite=finite)
    ax_finite.set_ylabel("Worst-group distortion")

    rows: list[dict[str, Any]] = []
    # Passive rows displayed in their own protocol's position; the shared NLHF
    # row carried beside the active protocol is skipped so it is drawn once.
    passive = frame[(frame["protocol"] == PASSIVE)
                    & frame["displayed_beside_protocol"].map(_blank)]
    active = frame[frame["protocol"] == ACTIVE]
    plans = [(method, PASSIVE, passive) for method in METHODS]
    plans += [(method, ACTIVE, active) for method in ("multigroup", "pooled")]

    inf_series = [plan for plan in plans
                  if not plan[2][(plan[2]["method"] == plan[0])
                                 & (plan[2]["beta_text"] == "inf")].empty]
    offsets = dict(zip([(p[0], p[1]) for p in inf_series], _inf_offsets(len(inf_series))))

    for method, protocol, source in plans:
        block = source[source["method"] == method]
        if block.empty:
            continue
        block = block.assign(_sort=[_beta_sort_key(b) for b in block["beta_text"]])
        block = block.sort_values("_sort")
        gid = f"{method}|{protocol}"
        label = series_label(method, protocol)
        style = METHOD_STYLE[method]
        for segment in ("finite", "infinity"):
            part = block[(block["beta_text"] != "inf") if segment == "finite"
                         else (block["beta_text"] == "inf")]
            if part.empty:
                continue
            estimates = part["worst_group_distortion"].astype(float).to_numpy()
            ses = part["worst_group_se"].astype(float).to_numpy()
            low, high = _level_intervals(part, estimates, ses)
            if segment == "finite":
                xs = np.array([float(b) for b in part["beta_text"]], dtype=float)
                ax = ax_finite
            else:
                xs = np.array([offsets[(method, protocol)]], dtype=float)
                ax = ax_inf
            _draw_curve(ax, xs, estimates, style=style,
                        linestyle=PROTOCOL_LINESTYLE[protocol], gid=gid,
                        band=(low, high), filled=protocol == PASSIVE)
            for x, y, lo, hi, (_, record) in zip(xs, estimates, low, high, part.iterrows()):
                rows.append(_row(
                    figure=figure_name, panel=segment, series=gid, series_label=label,
                    method=method, display_name=record["display_name"],
                    training_protocol=record["training_protocol"],
                    protocol_trained=record["protocol"],
                    displayed_beside_protocol=record["displayed_beside_protocol"],
                    condition="beta", beta=record["beta_text"],
                    learner_tau=record["learner_tau"],
                    evaluation_tau=record["evaluation_tau"],
                    axis_segment=segment, x=float(x), y=float(y), ci_low=lo, ci_high=hi,
                ))

    for ax in (ax_finite, ax_inf):
        ax.axhline(1.0, color=RULE, linewidth=0.7, zorder=1)

    handles = [_handle(method, PASSIVE, series_label(method, PASSIVE)) for method in METHODS]
    handles += [_handle(method, ACTIVE, series_label(method, ACTIVE))
                for method in ("multigroup", "pooled")]
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.17, top=0.80)
    labels = _legend(fig, handles, ncol=3, y=0.83)
    _title(fig, "Label-parameter condition: worst-group distortion", regime)
    caption = (
        f"{interval_caption(table_degrees_of_freedom(frame))} {NLHF_NOTE} Active GLHF was run at "
        + " and ".join(r"$\beta=\infty$" if _beta_text(b) == "inf" else rf"$\beta={_beta_text(b)}$"
                       for b in ACTIVE_BETAS)
        + " only. "
        r"$\beta=\infty$ has no finite log coordinate and is drawn in its own panel."
    )
    _caption(fig, caption, y=0.02)
    return _finish(figure_name, fig, rows, caption, labels, regime)


# ------------------------------------------------------------- (2) beta gaps

def draw_beta_gaps(tables: Mapping[str, pd.DataFrame], *,
                   regime: str | None = None) -> FigureArtifact:
    """Competitor minus multigroup GLHF, across the beta grid.

    Solid series are the passive-trained comparison; dashed series are the two
    betas at which the GLHF arms were also trained actively.  In both the NLHF
    arm is the passive-trained benchmark, which the legend says.
    """

    figure_name = "beta_gaps"
    contrasts = require_table(tables, "method_contrasts", figure=figure_name, regime=regime)
    frame = contrasts[(contrasts["condition"] == "beta")
                      & (contrasts["subtrahend"] == "multigroup")].copy()
    regime = resolve_regime(frame, regime=regime, figure=figure_name)
    frame["beta_text"] = [_beta_text(b) for b in frame["beta"]]
    finite = _finite_betas(frame)

    fig = plt.figure(figsize=(6.1, 3.2))
    ax_finite, ax_inf = make_beta_axes(fig, finite=finite)
    ax_finite.set_ylabel("Competitor $-$ Multigroup GLHF")

    competitors = [m for m in METHODS if m != "multigroup"]
    plans = [(m, p) for p in (PASSIVE, ACTIVE) for m in competitors]
    inf_present = [
        plan for plan in plans
        if not frame[(frame["minuend"] == plan[0]) & (frame["protocol"] == plan[1])
                     & (frame["beta_text"] == "inf")].empty
    ]
    offsets = dict(zip(inf_present, _inf_offsets(len(inf_present))))

    rows: list[dict[str, Any]] = []
    for method, protocol in plans:
        block = frame[(frame["minuend"] == method) & (frame["protocol"] == protocol)]
        if block.empty:
            continue
        block = block.assign(_sort=[_beta_sort_key(b) for b in block["beta_text"]])
        block = block.sort_values("_sort")
        gid = f"{method}_minus_multigroup|{protocol}"
        label = contrast_label(method, "multigroup", protocol)
        for segment in ("finite", "infinity"):
            part = block[(block["beta_text"] != "inf") if segment == "finite"
                         else (block["beta_text"] == "inf")]
            if part.empty:
                continue
            estimates = part["estimate"].astype(float).to_numpy()
            low = part["ci_low"].astype(float).to_numpy()
            high = part["ci_high"].astype(float).to_numpy()
            if segment == "finite":
                xs = np.array([float(b) for b in part["beta_text"]], dtype=float)
                ax = ax_finite
            else:
                xs = np.array([offsets[(method, protocol)]], dtype=float)
                ax = ax_inf
            _draw_curve(ax, xs, estimates, style=METHOD_STYLE[method],
                        linestyle=PROTOCOL_LINESTYLE[protocol], gid=gid,
                        band=(low, high), filled=protocol == PASSIVE)
            for x, y, lo, hi, (_, record) in zip(xs, estimates, low, high, part.iterrows()):
                rows.append(_row(
                    figure=figure_name, panel=segment, series=gid, series_label=label,
                    method=method, display_name=DISPLAY_NAMES[method],
                    training_protocol=record["minuend_training_protocol"],
                    protocol_trained=record["protocol"],
                    condition="beta", beta=record["beta_text"],
                    learner_tau=record["learner_tau"],
                    evaluation_tau=record["evaluation_tau"],
                    axis_segment=segment, x=float(x), y=float(y), ci_low=lo, ci_high=hi,
                    p_holm=record["p_holm"],
                ))

    for ax in (ax_finite, ax_inf):
        ax.axhline(0.0, color=ZERO, linewidth=0.7, zorder=1)
    ax_finite.annotate("positive favors Multigroup GLHF", xy=(0.015, 0.97),
                       xycoords="axes fraction", ha="left", va="top",
                       fontsize=7, color=RULE)

    handles = [_handle(m, p, contrast_label(m, "multigroup", p))
               for p in (PASSIVE, ACTIVE) for m in competitors
               if not frame[(frame["minuend"] == m) & (frame["protocol"] == p)].empty]
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.17, top=0.74)
    labels = _legend(fig, handles, ncol=1, y=0.77)
    _title(fig, "Label-parameter condition: competitor gaps", regime)
    caption = f"{interval_caption(table_degrees_of_freedom(frame))} {NLHF_NOTE}"
    _caption(fig, caption, y=0.02)
    return _finish(figure_name, fig, rows, caption, labels, regime)


# ----------------------------------------------------- (3) protocol contrasts

def draw_protocol_contrasts(tables: Mapping[str, pd.DataFrame], *,
                            regime: str | None = None) -> FigureArtifact:
    """Active minus passive GLHF, at beta 4 and infinity, for both GLHF arms.

    Beta is categorical here -- there are two of them and no curve to draw --
    so there is no log axis and infinity is one of two labelled slots rather
    than a numeric position.
    """

    figure_name = "protocol_contrasts"
    frame = require_table(tables, "protocol_contrasts", figure=figure_name,
                          regime=regime).copy()
    regime = resolve_regime(frame, regime=regime, figure=figure_name)
    frame["beta_text"] = [_beta_text(b) for b in frame["beta"]]
    betas = sorted(set(frame["beta_text"]), key=_beta_sort_key)
    if not betas:
        raise ValueError(
            "the protocol_contrasts table is empty: no beta has both an active and a "
            "passive GLHF fit, so no active-minus-passive contrast exists"
        )

    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.set_label("protocol_contrast_axis")
    methods = [m for m in ("multigroup", "pooled") if m in set(frame["method"])]
    dodge = dict(zip(methods, _inf_offsets(len(methods))))
    rows: list[dict[str, Any]] = []
    for method in methods:
        block = frame[frame["method"] == method]
        gid = f"active_minus_passive|{method}"
        label = (f"{DISPLAY_NAMES[method]}: active-trained $-$ passive-trained "
                 f"(same learner, same data budget)")
        xs, ys = [], []
        for position, beta in enumerate(betas):
            record = block[block["beta_text"] == beta]
            if record.empty:
                continue
            record = record.iloc[0]
            x = position + 0.16 * dodge[method] * 2.0
            estimate = float(record["estimate"])
            lo, hi = float(record["ci_low"]), float(record["ci_high"])
            ax.vlines([x], lo, hi, color=METHOD_STYLE[method]["color"], linewidth=1.2,
                      alpha=0.65, zorder=2)
            xs.append(x)
            ys.append(estimate)
            rows.append(_row(
                figure=figure_name, panel="worst_group", series=gid, series_label=label,
                method=method, display_name=record["display_name"],
                training_protocol=TRAINING_PROTOCOL[method],
                protocol_trained="active_minus_passive",
                condition=record["condition"], beta=beta,
                learner_tau=record["learner_tau"], evaluation_tau=record["evaluation_tau"],
                axis_segment="categorical", x=float(x), y=estimate,
                ci_low=lo, ci_high=hi, p_holm=record["p_holm"],
            ))
        line, = ax.plot(xs, ys, color=METHOD_STYLE[method]["color"],
                        marker=METHOD_STYLE[method]["marker"], markersize=4.6,
                        markerfacecolor="white", markeredgewidth=1.2, linestyle="none",
                        zorder=3)
        line.set_gid(f"{SERIES_GID_PREFIX}{gid}")

    ax.axhline(0.0, color=ZERO, linewidth=0.7, zorder=1)
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([r"$\beta=\infty$" if b == "inf" else rf"$\beta={b}$" for b in betas])
    ax.set_xlim(-0.6, len(betas) - 0.4)
    ax.set_xlabel(r"Label parameter $\beta$ (categorical: no log axis, "
                  r"$\infty$ is a labelled slot)")
    ax.set_ylabel("Active $-$ passive worst-group distortion")
    ax.annotate("negative = active sampling helps", xy=(0.015, 0.97),
                xycoords="axes fraction", ha="left", va="top", fontsize=7, color=RULE)
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)

    handles = [
        Line2D([0], [0], color=METHOD_STYLE[m]["color"], marker=METHOD_STYLE[m]["marker"],
               markersize=4.2, linestyle="none", markerfacecolor="white",
               markeredgewidth=1.2,
               label=f"{DISPLAY_NAMES[m]}: active-trained $-$ passive-trained")
        for m in methods
    ]
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.86)
    labels = _legend(fig, handles, ncol=2, y=0.89)
    _title(fig, "Sampling-protocol contrast: active $-$ passive GLHF", regime)
    caption = (
        f"{interval_caption(table_degrees_of_freedom(frame))} Both arms are GLHF; NLHF "
        "has no active counterpart in this study and therefore no protocol contrast."
    )
    _caption(fig, caption, y=0.02)
    return _finish(figure_name, fig, rows, caption, labels, regime)


# ------------------------------------------------------------ (4) tau curves

def draw_tau_curves(tables: Mapping[str, pd.DataFrame], *,
                    regime: str | None = None) -> FigureArtifact:
    """Active GLHF against the learner radius, with the passive benchmarks.

    The GLHF arms are active-trained here; NLHF is the passive-trained
    benchmark at the same radius and the uniform reference is not fitted at
    all.  Tau is the learner radius only: the evaluation radius stays at 0.5,
    which the axis label and the caption both say.
    """

    figure_name = "tau_curves"
    levels = require_table(tables, "method_levels", figure=figure_name, regime=regime)
    frame = levels[levels["condition"] == "tau"].copy()
    regime = resolve_regime(frame, regime=regime, figure=figure_name)

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.set_label("tau_curve_axis")
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        block = frame[frame["method"] == method].sort_values("learner_tau")
        if block.empty:
            continue
        protocol = str(block.iloc[0]["protocol"])
        beside = str(block.iloc[0]["displayed_beside_protocol"] or "")
        label = series_label(method, protocol, displayed_beside=beside or None)
        xs = block["learner_tau"].astype(float).to_numpy()
        ys = block["worst_group_distortion"].astype(float).to_numpy()
        ses = block["worst_group_se"].astype(float).to_numpy()
        low, high = _level_intervals(block, ys, ses)
        gid = f"{method}|{protocol}"
        _draw_curve(ax, xs, ys, style=METHOD_STYLE[method],
                    linestyle=PROTOCOL_LINESTYLE[protocol], gid=gid,
                    band=(low, high), filled=protocol == PASSIVE)
        for x, y, lo, hi, (_, record) in zip(xs, ys, low, high, block.iterrows()):
            rows.append(_row(
                figure=figure_name, panel="worst_group", series=gid, series_label=label,
                method=method, display_name=record["display_name"],
                training_protocol=record["training_protocol"],
                protocol_trained=record["protocol"],
                displayed_beside_protocol=record["displayed_beside_protocol"],
                condition="tau", beta=_beta_text(record["beta"]),
                learner_tau=float(record["learner_tau"]),
                evaluation_tau=record["evaluation_tau"],
                axis_segment="linear", x=float(x), y=float(y), ci_low=lo, ci_high=hi,
            ))

    taus = sorted({float(t) for t in frame["learner_tau"]})
    ax.axhline(1.0, color=RULE, linewidth=0.7, zorder=1)
    ax.set_xticks(taus)
    ax.set_xticklabels([f"{t:g}" for t in taus])
    ax.set_xlim(min(taus) - 0.04, max(taus) + 0.04)
    ax.set_xlabel(tau_axis_label(regime))
    ax.set_ylabel("Worst-group distortion")
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)

    handles = []
    for method in METHODS:
        block = frame[frame["method"] == method]
        if block.empty:
            continue
        protocol = str(block.iloc[0]["protocol"])
        beside = str(block.iloc[0]["displayed_beside_protocol"] or "")
        handles.append(_handle(method, protocol,
                               series_label(method, protocol, displayed_beside=beside or None)))
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.19, top=0.80)
    labels = _legend(fig, handles, ncol=2, y=0.83)
    _title(fig, "Learner-radius ablation: worst-group distortion", regime)
    caption = (f"{tau_caption(regime)} "
               f"{interval_caption(table_degrees_of_freedom(frame))} {NLHF_NOTE}")
    _caption(fig, caption, y=0.02)
    return _finish(figure_name, fig, rows, caption, labels, regime)


# -------------------------------------------------------------- (5) tau gaps

def draw_tau_gaps(tables: Mapping[str, pd.DataFrame], *,
                  regime: str | None = None) -> FigureArtifact:
    """Competitor minus multigroup GLHF, across the learner-radius grid."""

    figure_name = "tau_gaps"
    contrasts = require_table(tables, "method_contrasts", figure=figure_name, regime=regime)
    frame = contrasts[(contrasts["condition"] == "tau")
                      & (contrasts["subtrahend"] == "multigroup")].copy()
    regime = resolve_regime(frame, regime=regime, figure=figure_name)

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.set_label("tau_gap_axis")
    rows: list[dict[str, Any]] = []
    for method in [m for m in METHODS if m != "multigroup"]:
        block = frame[frame["minuend"] == method].sort_values("learner_tau")
        if block.empty:
            continue
        protocol = str(block.iloc[0]["protocol"])
        label = contrast_label(method, "multigroup", ACTIVE)
        xs = block["learner_tau"].astype(float).to_numpy()
        ys = block["estimate"].astype(float).to_numpy()
        low = block["ci_low"].astype(float).to_numpy()
        high = block["ci_high"].astype(float).to_numpy()
        gid = f"{method}_minus_multigroup|tau"
        _draw_curve(ax, xs, ys, style=METHOD_STYLE[method],
                    linestyle=PROTOCOL_LINESTYLE[ACTIVE], gid=gid,
                    band=(low, high), filled=False)
        for x, y, lo, hi, (_, record) in zip(xs, ys, low, high, block.iterrows()):
            rows.append(_row(
                figure=figure_name, panel="worst_group", series=gid, series_label=label,
                method=method, display_name=DISPLAY_NAMES[method],
                training_protocol=record["minuend_training_protocol"],
                protocol_trained=record["protocol"],
                condition="tau", beta=_beta_text(record["beta"]),
                learner_tau=float(record["learner_tau"]),
                evaluation_tau=record["evaluation_tau"],
                axis_segment="linear", x=float(x), y=float(y), ci_low=lo, ci_high=hi,
                p_holm=record["p_holm"],
            ))

    taus = sorted({float(t) for t in frame["learner_tau"]})
    ax.axhline(0.0, color=ZERO, linewidth=0.7, zorder=1)
    ax.set_xticks(taus)
    ax.set_xticklabels([f"{t:g}" for t in taus])
    ax.set_xlim(min(taus) - 0.04, max(taus) + 0.04)
    ax.set_xlabel(tau_axis_label(regime))
    ax.set_ylabel("Competitor $-$ Multigroup GLHF")
    ax.annotate("positive favors Multigroup GLHF", xy=(0.015, 0.97),
                xycoords="axes fraction", ha="left", va="top", fontsize=7, color=RULE)
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)

    handles = [
        Line2D([0], [0], color=METHOD_STYLE[m]["color"], marker=METHOD_STYLE[m]["marker"],
               markersize=3.8, linewidth=1.3, linestyle=PROTOCOL_LINESTYLE[ACTIVE],
               markerfacecolor="white", markeredgewidth=1.1,
               label=contrast_label(m, "multigroup", ACTIVE))
        for m in [m for m in METHODS if m != "multigroup"]
        if not frame[frame["minuend"] == m].empty
    ]
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.19, top=0.78)
    labels = _legend(fig, handles, ncol=1, y=0.81)
    _title(fig, "Learner-radius ablation: competitor gaps", regime)
    caption = (f"{tau_caption(regime)} "
               f"{interval_caption(table_degrees_of_freedom(frame))} {NLHF_NOTE}")
    _caption(fig, caption, y=0.02)
    return _finish(figure_name, fig, rows, caption, labels, regime)


# ------------------------------------------------------ (6) tau group panels

def draw_tau_group_panels(tables: Mapping[str, pd.DataFrame], *,
                          regime: str | None = None) -> FigureArtifact:
    """One small multiple per group, all four methods, across the tau grid.

    Bands are omitted: several dozen marginal intervals on one sheet would
    read as a simultaneous statement the numbers do not support.  The levels
    are the same ones the worst-group panel maximizes over.
    """

    figure_name = "tau_group_panels"
    levels = require_table(tables, "method_group_levels", figure=figure_name, regime=regime)
    frame = levels[levels["condition"] == "tau"].copy()
    regime = resolve_regime(frame, regime=regime, figure=figure_name)
    groups = sorted(set(frame["group"]))
    if not groups:
        raise ValueError("the tau condition carries no per-group rows to panel")

    columns = min(3, len(groups))
    nrows = int(math.ceil(len(groups) / columns))
    fig, axes = plt.subplots(nrows, columns, sharex=True, sharey=True,
                             figsize=(2.1 * columns + 0.9, 1.95 * nrows + 1.0))
    flat = np.atleast_1d(axes).ravel()
    rows: list[dict[str, Any]] = []
    for position, group in enumerate(groups):
        ax = flat[position]
        ax.set_label(f"tau_group_panel:{group}")
        block_group = frame[frame["group"] == group]
        for method in METHODS:
            block = block_group[block_group["method"] == method].sort_values("learner_tau")
            if block.empty:
                continue
            protocol = str(block.iloc[0]["protocol"])
            beside = str(block.iloc[0]["displayed_beside_protocol"] or "")
            label = series_label(method, protocol, displayed_beside=beside or None)
            xs = block["learner_tau"].astype(float).to_numpy()
            ys = block["distortion"].astype(float).to_numpy()
            gid = f"{group}|{method}|{protocol}"
            _draw_curve(ax, xs, ys, style=METHOD_STYLE[method],
                        linestyle=PROTOCOL_LINESTYLE[protocol], gid=gid, band=None,
                        filled=protocol == PASSIVE)
            for x, y, (_, record) in zip(xs, ys, block.iterrows()):
                rows.append(_row(
                    figure=figure_name, panel=group, series=gid, series_label=label,
                    method=method, display_name=record["display_name"],
                    training_protocol=record["training_protocol"],
                    protocol_trained=record["protocol"],
                    displayed_beside_protocol=record["displayed_beside_protocol"],
                    condition="tau", beta=_beta_text(record["beta"]),
                    learner_tau=float(record["learner_tau"]),
                    evaluation_tau=record["evaluation_tau"], group=group,
                    axis_segment="linear", x=float(x), y=float(y),
                ))
        ax.axhline(1.0, color=RULE, linewidth=0.7, zorder=1)
        ax.set_title(escape_mathtext(group), loc="left", pad=4, fontsize=8)
        ax.grid(True, color=GRID, linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)

    taus = sorted({float(t) for t in frame["learner_tau"]})
    for ax in flat[:len(groups)]:
        ax.set_xticks(taus)
        ax.set_xticklabels([f"{t:g}" for t in taus], fontsize=7)
        ax.set_xlim(min(taus) - 0.04, max(taus) + 0.04)
    for extra in flat[len(groups):]:
        extra.set_visible(False)
    # The axis label says what tau is and which radius the evaluation used, so
    # it is long.  One copy, under the middle of the bottom row, rather than
    # one per column overprinting its neighbours.
    bottom = list(range(max(len(groups) - columns, 0), len(groups)))
    middle = min(bottom, key=lambda index: abs((index % columns) - (columns - 1) / 2.0))
    flat[middle].set_xlabel(tau_axis_label(regime), fontsize=7.5)
    for index in range(0, len(groups), columns):
        flat[index].set_ylabel(r"Group distortion $\widehat{D}_k$")

    handles = []
    for method in METHODS:
        block = frame[frame["method"] == method]
        if block.empty:
            continue
        protocol = str(block.iloc[0]["protocol"])
        beside = str(block.iloc[0]["displayed_beside_protocol"] or "")
        handles.append(_handle(method, protocol,
                               series_label(method, protocol, displayed_beside=beside or None)))
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.90))
    labels = _legend(fig, handles, ncol=2, y=0.92)
    _title(fig, "Learner-radius ablation: per-group levels", regime)
    caption = (
        f"{tau_caption(regime)} {NLHF_NOTE} Per-group levels are drawn without intervals: "
        "many marginal intervals on one sheet would read as a simultaneous statement, "
        "and these intervals are marginal per cell."
    )
    _caption(fig, caption, y=0.045)
    return _finish(figure_name, fig, rows, caption, labels, regime)


# ------------------------------------------------------------------- writing

DRAWERS: Mapping[str, Any] = {
    "beta_curves": draw_beta_curves,
    "beta_gaps": draw_beta_gaps,
    "protocol_contrasts": draw_protocol_contrasts,
    "tau_curves": draw_tau_curves,
    "tau_gaps": draw_tau_gaps,
    "tau_group_panels": draw_tau_group_panels,
}
assert set(DRAWERS) == set(schema.FIGURES), (
    "the figure layer must draw exactly the figures schema declares"
)

#: Figure filenames use the dataset, figure family, and evaluation regime.
STEM_PREFIX = ""

# Preserve the frozen beta PDFs' page bounds when display names change.
BETA_PDF_BOUNDS = (
    -0.0020876736111105613, -0.3968055555555555,
    6.104175347222221, 3.9593055555555553,
)


#: The two tau figures' condition, and the beta condition's, named so a caller
#: can ask for one regime's figures without listing them.
TAU_FIGURES: tuple[str, ...] = ("tau_curves", "tau_gaps", "tau_group_panels")
BETA_FIGURES: tuple[str, ...] = ("beta_curves", "beta_gaps", "protocol_contrasts")


def stem_for(name: str, *, dataset: str | None = None,
             regime: str | None = None) -> str:
    """The file stem.  The regime is part of the name, never only the caption."""

    if regime is not None:
        _check_regime(regime)
    return "_".join(part for part in (STEM_PREFIX, dataset, name, regime) if part)


def write_figure(artifact: FigureArtifact, out_dir: Path, *,
                 dataset: str | None = None, regime: str | None = None,
                 close: bool = True) -> dict[str, Path]:
    """Write one artifact as PDF, PNG and the CSV that was actually plotted.

    ``regime`` defaults to the regime the artifact's own plotted rows carry, so
    a file written from moving-tau rows cannot be named as the fixed one.
    """

    if regime is None:
        regime = _artifact_regime(artifact)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = stem_for(artifact.name, dataset=dataset, regime=regime)
    paths = {
        "pdf": out / f"{stem}.pdf",
        "png": out / f"{stem}.png",
        "csv": out / f"{stem}.csv",
    }
    pdf_options = {}
    if artifact.name == "beta_curves" and dataset in ("coffee", "persona") and regime == REGIME_FIXED:
        pdf_options["bbox_inches"] = Bbox.from_bounds(*BETA_PDF_BOUNDS)
    artifact.figure.savefig(paths["pdf"], **pdf_options)
    artifact.figure.savefig(paths["png"], facecolor="white")
    artifact.frame.to_csv(paths["csv"], index=False)
    artifact.paths = paths
    if close:
        plt.close(artifact.figure)
    return paths


def _artifact_regime(artifact: FigureArtifact) -> str | None:
    """The regime an artifact's plotted rows carry, if they carry exactly one."""

    found = frame_regimes(artifact.frame)
    if len(found) > 1:
        raise RegimeMismatchError(
            f"the plotted rows of figure {artifact.name!r} carry more than one "
            f"evaluation regime {sorted(found)}; they are different estimands"
        )
    return found.pop() if found else None


def _watermark(fig) -> None:
    """Stamp a figure drawn from synthetic fixtures so it cannot be mistaken.

    Placed across the middle of the canvas, above the axes, because the whole
    hazard is that these figures are shaped to look like the real ones.
    """

    fig.text(
        0.5, 0.5, "SYNTHETIC FIXTURE\nNOT RESULTS",
        ha="center", va="center", fontsize=34, color="#d62728",
        alpha=0.18, rotation=24, zorder=1000, fontweight="bold",
        linespacing=1.1, gid="synthetic_watermark",
    )


def write_all_figures(tables: Mapping[str, pd.DataFrame], out_dir: Path, *,
                      dataset: str | None = None,
                      figures: Iterable[str] | None = None,
                      regime: str | None = None,
                      close: bool = True) -> dict[str, dict[str, Path]]:
    """Draw and write the requested figures (all six by default).

    ``out_dir`` is explicit and mandatory: production passes the suite's
    ``figures/<dataset>``, tests pass a temporary directory, and neither
    path is hard-coded here.
    """

    synthetic = any(
        "synthetic_fixture" in getattr(frame, "columns", ())
        for frame in tables.values()
    )
    if synthetic:
        # Drawn from the synthetic fixtures.  These figures are deliberately
        # shaped to look like the real ones, which is exactly why they must not
        # be mistaken for them: refuse the real figure directories outright,
        # and watermark whatever is drawn.
        resolved = Path(out_dir).resolve()
        if resolved.name in ("coffee", "persona") and resolved.parts[-2:-1] == ("figures",):
            raise ValueError(
                f"refusing to write synthetic-fixture figures into {resolved}: that is a "
                f"reportable suite figure directory. These numbers are not results."
            )

    requested = list(figures) if figures is not None else list(schema.FIGURES)
    unknown = [name for name in requested if name not in DRAWERS]
    if unknown:
        raise KeyError(
            f"unknown suite figure(s) {unknown}; expected some of {sorted(DRAWERS)}"
        )
    configure_style()
    written: dict[str, dict[str, Path]] = {}
    for name in requested:
        artifact = DRAWERS[name](tables, regime=regime)
        if synthetic:
            _watermark(artifact.figure)
        written[name] = write_figure(artifact, out_dir, dataset=dataset, close=close)
    return written


# ------------------------------------------------- reading a reduction's run

#: The reduction tables the figure layer reads out of a ``reduce/`` directory.
READABLE_TABLES: tuple[str, ...] = (
    "method_levels", "method_group_levels", "method_contrasts",
    "method_group_contrasts", "protocol_contrasts", "protocol_group_contrasts",
)


def read_reduction_tables(directory: Path) -> dict[str, pd.DataFrame]:
    """Every reduction table present in one ``reduce/<...>`` directory."""

    source = Path(directory)
    if not source.is_dir():
        raise MissingTableError(
            f"no suite reduction directory at {source}: a figure run reads the CSVs a "
            f"reduction wrote, and refuses to draw anything without them"
        )
    tables = {name: pd.read_csv(source / f"{name}.csv")
              for name in READABLE_TABLES if (source / f"{name}.csv").is_file()}
    if not tables:
        raise MissingTableError(
            f"{source} holds none of the suite reduction tables {list(READABLE_TABLES)}"
        )
    return tables


def write_regime_figures(run_dir: Path, out_dir: Path, *, dataset: str | None = None,
                         regime: str, close: bool = True) -> dict[str, dict[str, Path]]:
    """The tau figures of one evaluation regime, from ``<run>/reduce/<regime>/``.

    The regime is read from the directory the rows came out of, checked against
    the rows themselves, drawn onto every figure, and written into every file
    name.  The two regimes are different estimands and never share a figure or
    a file.
    """

    _check_regime(regime)
    tables = read_reduction_tables(Path(run_dir) / "reduce" / regime)
    return write_all_figures(tables, out_dir, dataset=dataset,
                             figures=TAU_FIGURES, regime=regime, close=close)


def write_beta_figures(run_dir: Path, out_dir: Path, *, dataset: str | None = None,
                       close: bool = True) -> dict[str, dict[str, Path]]:
    """The beta-condition figures, from ``<run>/reduce/beta/``.

    The beta condition is reduced in one regime only -- every policy scored at
    evaluation tau = 0.50 -- and these figures say so in the same words the tau
    figures use, rather than leaving the reader to assume it.
    """

    tables = read_reduction_tables(Path(run_dir) / "reduce" / "beta")
    return write_all_figures(tables, out_dir, dataset=dataset,
                             figures=BETA_FIGURES, close=close)


def write_run_figures(run_dir: Path, out_dir: Path, *, dataset: str | None = None,
                      close: bool = True) -> dict[str, dict[str, Path]]:
    """Every suite figure of one run: the beta condition and both tau regimes."""

    written = dict(write_beta_figures(run_dir, out_dir, dataset=dataset, close=close))
    for regime in REGIMES:
        for name, paths in write_regime_figures(
                run_dir, out_dir, dataset=dataset, regime=regime, close=close).items():
            written[f"{name}:{regime}"] = paths
    return written


def suite_figure_directory(project_root: Path, suite: Suite) -> Path:
    """The repository-relative figure directory this suite writes into."""

    return Path(project_root) / suite.figure_directory
