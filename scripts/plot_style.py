"""Shared manuscript styles for the manuscript figure generator.

This module contains presentation constants and layout helpers only. It does
not import historical experiment generators or load experiment artifacts.
"""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt


MG = "multigroup"


NLHF = "pooled_nlhf"


POOLED = "pooled"


UNIFORM = "uniform"


METHOD_COLOR = {MG: "#D55E00", NLHF: "#7B52AB", POOLED: "#0072B2", UNIFORM: "#6B7280"}


METHOD_MARKER = {MG: "s", NLHF: "^", POOLED: "o", UNIFORM: "D"}


DISPLAY = {
    "coffee": (
        ("age:45+", "Age 45+", 483),
        ("expertise:1-4", "Expertise 1–4", 815),
        ("gender:female", "Female", 759),
        ("children:reported-1+", "Children 1+", 767),
        ("work:from-home", "Works from home", 1424),
        ("universal", "Universal", 3369),
    ),
    "persona": (
        ("technical_background_not_technical", "Not technical", 750),
        ("writing_expressive", "Expressive wording", 747),
        ("writing_direct", "Direct wording", 750),
        ("technical_background_very_technical", "Very technical", 747),
        ("education_university", "University education", 748),
        ("universal", "Universal", 1497),
    ),
}


COLOR = METHOD_COLOR


MARKER = METHOD_MARKER


LABELLED_TICKS = (0, 1, 3, 5)


TEXT = "#222222"


GRID = "#E3E3E3"


WHISKER = "#333333"


LABEL_FONTSIZE = 7.5


GROUP_PALETTE = ("#0072B2", "#CC3311", "#E69F00", "#009E73", "#7A5195", "#8C6D31")


SHORT_LABEL = {
    "age:45+": "Age 45+",
    "expertise:1-4": "Expertise 1–4",
    "gender:female": "Female",
    "children:reported-1+": "Children 1+",
    "work:from-home": "From home",
    "universal": "Universal",
    "technical_background_not_technical": "Not technical",
    "writing_expressive": "Expressive",
    "writing_direct": "Direct",
    "technical_background_very_technical": "Very technical",
    "education_university": "University",
}


def wrap_tick(label: str, limit: int = 13) -> str:
    """Break a long cluster name at its most central space.

    Seven clusters across 5.5 in leaves about 0.75 in per tick, which two of
    the Persona names and one Coffee name overrun on a single line.
    """

    if "\n" in label or len(label) <= limit or " " not in label:
        return label
    spaces = [i for i, ch in enumerate(label) if ch == " "]
    cut = min(spaces, key=lambda i: abs(i - len(label) / 2))
    return label[:cut] + "\n" + label[cut + 1:]


def configure_style() -> None:
    """The manuscript's house style: Times-compatible STIX at 5.5 in width."""

    mpl.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.5,
        "axes.titleweight": "normal",
        "axes.edgecolor": TEXT,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
    })


def _min_separation(ax: plt.Axes, fontsize: float) -> float:
    lower, upper = ax.get_ylim()
    height_inches = ax.get_window_extent().height / ax.figure.dpi
    if height_inches <= 0:
        return 0.0
    return (fontsize * 1.28 / 72.0) / height_inches * (upper - lower)


def labels_at_right(ax: plt.Axes, x: float, entries: Sequence[tuple[float, str, str]],
                    fontsize: float = LABEL_FONTSIZE) -> list[Any]:
    """Endpoint tags pushed apart just enough not to collide (shared manuscript style)."""

    if not entries:
        return []
    drawn: list[Any] = []
    step = _min_separation(ax, fontsize)
    placed: list[tuple[float, str, str]] = []
    previous = None
    for y, text, colour in sorted(entries, key=lambda item: -item[0]):
        position = y if previous is None else min(y, previous - step)
        placed.append((position, text, colour))
        previous = position
    # Pushing strictly downwards drifts the whole block off the curves it
    # labels, so it is re-centred on the values and only then clamped.
    drift = (sum(y for y, _, _ in entries) / len(entries)
             - sum(position for position, _, _ in placed) / len(placed))
    placed = [(position + drift, text, colour) for position, text, colour in placed]
    lower, upper = ax.get_ylim()
    overflow = lower + 0.5 * step - placed[-1][0]
    if overflow > 0:
        placed = [(position + overflow, text, colour) for position, text, colour in placed]
    excess = placed[0][0] - (upper - 0.5 * step)
    if excess > 0:
        placed = [(position - excess, text, colour) for position, text, colour in placed]
    for position, text, colour in placed:
        drawn.append(ax.annotate(
            text, xy=(x, position), xytext=(2.5, 0), textcoords="offset points",
            color=colour, fontsize=fontsize, va="center", ha="left",
            annotation_clip=False))
    return drawn


def fit_right_margin(ax: plt.Axes, x_last: float, labels: Sequence[Any]) -> None:
    """Widen the x range until the endpoint tags fit inside the axes.

    Guessing a padding in data units overflows the 5.5 in canvas as soon as a
    dataset has a long group name, so the padding is measured from the tags
    that were actually drawn.
    """

    if not labels:
        return
    renderer = ax.figure.canvas.get_renderer()
    needed = max(label.get_window_extent(renderer=renderer).width for label in labels) + 4.0
    width = ax.bbox.width
    if width <= needed:
        return
    lower = ax.get_xlim()[0]
    ax.set_xlim(lower, lower + (x_last - lower) * width / (width - needed))


def tidy(ax: plt.Axes, *, axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=axis, color=GRID, linewidth=0.55, linestyle=":", alpha=0.9)
    ax.set_axisbelow(True)


def axis_break_glyph(ax: plt.Axes) -> None:
    """Two slashes across the bottom of the y spine: the axis is cut."""

    ax.plot([0, 0], [0.004, 0.030], transform=ax.transAxes, color="white", linewidth=3.2,
            clip_on=False, zorder=5)
    for dy in (0.006, 0.017):
        ax.plot([-0.011, 0.011], [dy, dy + 0.012], transform=ax.transAxes, color=TEXT,
                linewidth=0.8, clip_on=False, zorder=6)
