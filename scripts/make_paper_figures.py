#!/usr/bin/env python3
"""Render manuscript figures from completed reductions only.

No learner, evaluator, or experiment runner is imported. Every plotted number
is copied from a reduction CSV and checked against that row before output is saved.
"""

from __future__ import annotations

import csv
import argparse
import os
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import plot_style as house


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results"
OUT = ROOT / "figures/paper"
SMOKE = False
METHODS = ("multigroup", "pooled", "pooled_nlhf", "uniform")
LABELS = {
    "multigroup": "Multigroup GLHF (active)",
    "pooled": "Pooled GLHF (active)",
    "pooled_nlhf": "Batch NLHF (passive)",
    "uniform": "Uniform reference",
}
COLOR = house.COLOR
TAUS = ("0.05", "0.2", "0.35", "0.5", "0.75", "1")
NUMERIC = ("estimate", "corrected_se", "ci_lower", "ci_upper")


def comparison_run(dataset: str) -> Path:
    path = RUNS / dataset / "comparison"
    if not path.exists():
        path = RUNS / f"{dataset}-comparison-suite-v10"
    return path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select(rows: list[dict[str, str]], **filters: str) -> dict[str, str]:
    matches = [r for r in rows if all(r.get(k) == str(v) for k, v in filters.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one reduction row for {filters}; got {len(matches)}")
    row = matches[0]
    if row["status"] != "ok" or int(row["n_splits"]) != (2 if SMOKE else 100):
        raise ValueError(f"Incomplete reduction: {filters}")
    if not float(row["ci_lower"]) <= float(row["estimate"]) <= float(row["ci_upper"]):
        raise ValueError(f"Interval does not contain estimate: {filters}")
    return row


def val(row: dict[str, str], key: str) -> float:
    return float(row[key])


def portable(path: Path) -> str:
    return Path(os.path.relpath(path, ROOT)).as_posix()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output_row(figure: str, panel: str, source: Path, row: dict[str, str]) -> dict[str, str]:
    return {"figure": figure, "panel": panel, "source_file": portable(source), **row}


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_rows(rows: list[dict[str, str]]) -> int:
    """Exact CSV string comparison avoids silently rounded export values."""
    cache: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        source = row["source_file"]
        cache.setdefault(source, read(ROOT / source))
        raw = {k: v for k, v in row.items() if k not in ("figure", "panel", "source_file")}
        if raw not in cache[source]:
            raise AssertionError(f"Plotted row differs from its source: {source}")
    return len(rows)


def save(fig: plt.Figure, stem: str) -> list[Path]:
    if SMOKE:
        fig.text(.5, .5, "SMOKE — NOT RESULTS", ha="center", alpha=.4, rotation=20)
    paths = [OUT / f"{stem}.pdf", OUT / f"{stem}.png"]
    if not SMOKE:
        house.check_text_overlap(fig)
    fig.savefig(paths[0], bbox_inches="tight", pad_inches=0.04)
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.04, dpi=400)
    plt.close(fig)
    return paths


def group_order(rows: list[dict[str, str]]) -> list[str]:
    return list(dict.fromkeys(r["group"] for r in rows if r["method"] == "multigroup"
                              and r["cell"] == "active@inf"))


def group_label(key: str) -> str:
    label = house.SHORT_LABEL.get(key, key.replace("_", " ").replace(":", " "))
    return house.wrap_tick(label, limit=12)


def bar_figure(dataset: str, run: Path) -> tuple[list[Path], list[dict[str, str]]]:
    base = run / "reduce/beta"
    gp, wp = base / "method_group_levels.csv", base / "method_levels.csv"
    groups, worst = read(gp), read(wp)
    keys = [key for key, _, _ in house.DISPLAY[dataset]]
    assert len(keys) == 6, keys
    labels = [house.wrap_tick(label) for _, label, _ in house.DISPLAY[dataset]]
    labels += ["Worst group\n(maximum)"]
    chosen: dict[tuple[int, str], dict[str, str]] = {}
    exported: list[dict[str, str]] = []
    for gi, key in enumerate([*keys, None]):
        for method in METHODS:
            # The passive NLHF is displayed beside the active GLHF, never
            # relabelled as an active-trained fit. Uniform is also untrained.
            cell = "passive@inf" if method in ("pooled_nlhf", "uniform") else "active@inf"
            src = gp if key is not None else wp
            filters = {"cell": cell, "method": method}
            if key is not None:
                filters["group"] = key
            row = select(groups if key is not None else worst, **filters)
            chosen[(gi, method)] = row
            exported.append(output_row(f"{dataset}_group_bars", "bars", src, row))

    fig, ax = plt.subplots(figsize=(5.5, 1.85))
    x = np.arange(7, dtype=float)
    x[-1] += 0.35
    width = 0.19
    lows, highs = [], []
    for slot, method in enumerate(METHODS):
        offset = (slot - 1.5) * width
        estimates = np.array([val(chosen[(i, method)], "estimate") for i in range(7)])
        lower = np.array([val(chosen[(i, method)], "ci_lower") for i in range(7)])
        upper = np.array([val(chosen[(i, method)], "ci_upper") for i in range(7)])
        lows.extend(lower)
        highs.extend(upper)
        ax.bar(x + offset, estimates - 1, width=width * 0.9, bottom=1,
               color=COLOR[method], alpha=0.55 if method == "uniform" else 0.92,
               linewidth=0, zorder=2)
        ax.errorbar(x + offset, estimates, yerr=np.vstack((estimates - lower, upper - estimates)),
                    fmt="none", ecolor=house.WHISKER, elinewidth=0.8, capsize=2, zorder=3)
    ax.set_ylim(min(min(lows) - 0.02, 0.935), max(highs) + 0.012)
    ax.set_xlim(-0.55, x[-1] + 0.55)
    ax.axhline(1, color=house.TEXT, linewidth=1.1, zorder=4)
    divider = (x[-2] + x[-1]) / 2
    ax.axvspan(divider, x[-1] + 0.55, color="#F1F1F1", linewidth=0, zorder=0)
    ax.axvline(divider, color="#B4B4B4", linewidth=0.8, zorder=1)
    ax.set_xticks(x, labels, fontsize=7.2)
    ax.set_ylabel("Empirical distortion")
    ax.yaxis.grid(True, color=house.GRID, linewidth=0.55, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=COLOR[m], alpha=0.55 if m == "uniform" else 0.92,
                             label=LABELS[m]) for m in METHODS],
              loc="lower center", bbox_to_anchor=(0.5, 1.003), ncol=4,
              fontsize=6.3, columnspacing=0.7, handlelength=1.2, handletextpad=0.4)
    house.axis_break_glyph(ax)
    fig.subplots_adjust(left=0.098, right=0.995, bottom=0.16, top=0.905)
    return save(fig, f"{dataset}_group_bars"), exported


def radius_figure(dataset: str, run: Path) -> tuple[list[Path], list[dict[str, str]]]:
    base = run / "reduce/fixed_0.5"
    wp, gp = base / "method_levels.csv", base / "method_group_levels.csv"
    worst, groups = read(wp), read(gp)
    keys = list(dict.fromkeys(r["group"] for r in groups if r["method"] == "multigroup"
                              and r["cell"] == "tau@0.05"))
    assert len(keys) == 6, keys
    exported: list[dict[str, str]] = []
    xs = np.array([float(tau) for tau in TAUS])
    # Compact panels include the space beneath endpoint labels.
    fig = plt.figure(figsize=(7.2, 2.45))
    panel_width = 1.9
    panel_height = 1.35
    axes = [fig.add_axes([left / 7.2, 0.58 / 2.45,
                          panel_width / 7.2, panel_height / 2.45])
            for left in (0.55, 2.95, 5.25)]
    for ax in axes:
        ax.set_box_aspect(panel_height / panel_width)
    axw, axn, axm = axes
    for method in ("pooled", "pooled_nlhf", "multigroup"):
        rows = [select(worst, cell=f"tau@{tau}", method=method) for tau in TAUS]
        exported += [output_row(f"{dataset}_radius", "a", wp, r) for r in rows]
        y = np.array([val(r, "estimate") for r in rows])
        lo = np.array([val(r, "ci_lower") for r in rows])
        hi = np.array([val(r, "ci_upper") for r in rows])
        axw.plot(xs, y, color=COLOR[method], marker=house.MARKER[method],
                 markersize=3.2, linewidth=1.0 if method == "pooled" else 1.6)
        if method != "pooled":
            axw.fill_between(xs, lo, hi, color=COLOR[method], alpha=0.10, linewidth=0)
    uniform = select(worst, cell="tau@0.5", method="uniform")
    exported.append(output_row(f"{dataset}_radius", "a", wp, uniform))
    axw.axhline(val(uniform, "estimate"), color=COLOR["uniform"],
                linewidth=0.9, linestyle=(0, (1.5, 1.5)), zorder=0.8)
    axw.axhline(1, color="#8A929E", linewidth=0.7, zorder=0.8)
    axw.set_title("(a) Worst group", loc="left", pad=5)
    axw.set_ylabel("Worst-group distortion")
    tables = []
    for method, ax, panel in (("pooled_nlhf", axn, "b"), ("multigroup", axm, "c")):
        table = np.zeros((6, 6))
        for gi, key in enumerate(keys):
            rows = [select(groups, cell=f"tau@{tau}", method=method, group=key) for tau in TAUS]
            exported += [output_row(f"{dataset}_radius", panel, gp, r) for r in rows]
            table[:, gi] = [val(r, "estimate") for r in rows]
            ax.plot(xs, table[:, gi], color=house.GROUP_PALETTE[gi], marker="o",
                    markersize=2.5, linewidth=1.1)
        ax.axhline(1, color="#8A929E", linewidth=0.7, zorder=0.8)
        ax.set_title("(b) Batch NLHF" if panel == "b" else
                     "(c) Multigroup GLHF", loc="left", pad=5,
                     fontsize=7.6)
        tables.append(table)
    axn.set_ylabel("Group distortion")
    low, high = min(t.min() for t in tables), max(t.max() for t in tables)
    spread = max(high - low, 1e-3)
    for ax in (axn, axm):
        ax.set_ylim(min(low - 0.08 * spread, 1 - 0.04 * spread),
                    max(high + 0.08 * spread, 1 + 0.04 * spread))
    for ax in axes:
        ax.set_xticks(xs[[0, 3, 5]], [TAUS[i] for i in (0, 3, 5)])
        ax.axvline(0.5, color="#9A9A9A", linewidth=0.7, linestyle="--", zorder=0.8)
        house.tidy(ax)
    fig.supxlabel(r"Learner KL radius $\tau_L$ (evaluation KL radius $\tau_E=0.5$)",
                  fontsize=8, y=0.035)
    fig.legend(handles=[Line2D([0], [0], color=COLOR[m],
                               marker=None if m == "uniform" else house.MARKER[m],
                               linestyle=(0, (1.5, 1.5)) if m == "uniform" else "-",
                               markersize=3, label=LABELS[m])
                        for m in ("multigroup", "pooled_nlhf", "pooled", "uniform")],
               loc="lower center", bbox_to_anchor=(0.5, 0.955), ncol=4,
               fontsize=6.2, columnspacing=0.7, handlelength=2.0)
    fig.canvas.draw()
    for ax, table in ((axn, tables[0]), (axm, tables[1])):
        labels = house.labels_at_right(ax, xs[-1], [
            (table[-1, gi], house.SHORT_LABEL[key], house.GROUP_PALETTE[gi])
            for gi, key in enumerate(keys)
        ], fontsize=7.1)
        house.fit_right_margin(ax, xs[-1], labels)
    for ax in axes:
        # Bottom axes and gray horizontal guides stop at tau_L = 1.
        left, right = ax.get_xlim()
        ax.spines["bottom"].set_bounds(left, xs[-1])
        guide_end = (xs[-1] - left) / (right - left)
        for gridline in ax.yaxis.get_gridlines():
            gridline.set_xdata([0, guide_end])
        for line in ax.lines:
            if line.get_color() in ("#8A929E", COLOR["uniform"]):
                line.set_xdata([0, guide_end])
                line.set_zorder(0.8)
    return save(fig, f"{dataset}_radius"), exported


def beta_figure() -> tuple[list[Path], list[dict[str, str]]]:
    """Compact categorical beta comparison: passive grid and two active GLHF points."""
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3), sharex=True)
    methods = ("multigroup", "pooled_nlhf", "pooled", "uniform")
    exported: list[dict[str, str]] = []
    beta_values = ("0.5", "1", "2", "4", "inf")
    xs = np.arange(5)
    for ax, dataset in zip(axes, ("coffee", "persona")):
        source = comparison_run(dataset) / "reduce/beta/method_levels.csv"
        rows = read(source)
        for method in methods:
            selected = [select(rows, cell=f"passive@{beta}", method=method)
                        for beta in beta_values]
            exported += [output_row("beta_curves", dataset, source, r) for r in selected]
            y = np.array([val(r, "estimate") for r in selected])
            lower = np.array([val(r, "ci_lower") for r in selected])
            upper = np.array([val(r, "ci_upper") for r in selected])
            ax.plot(xs, y, color=COLOR[method], marker=house.MARKER[method],
                    markersize=3.2, linewidth=1.3 if method != "uniform" else 0.9)
            if method != "uniform":
                ax.fill_between(xs, lower, upper, color=COLOR[method], alpha=0.08,
                                linewidth=0)
        for method in ("multigroup", "pooled"):
            selected = [select(rows, cell=f"active@{beta}", method=method)
                        for beta in ("4", "inf")]
            exported += [output_row("beta_curves", dataset, source, r) for r in selected]
            y = np.array([val(r, "estimate") for r in selected])
            lower = np.array([val(r, "ci_lower") for r in selected])
            upper = np.array([val(r, "ci_upper") for r in selected])
            ax.plot(xs[-2:], y, color=COLOR[method], linestyle="--", marker="o",
                    markerfacecolor="white", markeredgecolor=COLOR[method],
                    markersize=4.2, linewidth=1.25, zorder=5)
            ax.fill_between(xs[-2:], lower, upper, color=COLOR[method], alpha=0.12,
                            linewidth=0, zorder=1)
        ax.set_xticks(xs, ["0.5", "1", "2", "4", r"$\infty$"])
        ax.set_xlim(-0.15, 4.15)
        ax.set_title("(a) Coffee-rating study" if dataset == "coffee" else "(b) Completion-rating study",
                     loc="left", fontsize=8.2)
        ax.axhline(1, color="#8A929E", linewidth=0.7)
        house.tidy(ax)
    axes[0].set_ylabel("Worst-group distortion")
    fig.supxlabel(r"Bradley–Terry label parameter $\beta$", fontsize=8, y=0.025)
    passive_labels = {"multigroup": "Multigroup GLHF (passive)", "pooled_nlhf": "Batch NLHF (passive)",
                      "pooled": "Pooled GLHF (passive)", "uniform": "Uniform reference"}
    handles = [Line2D([0], [0], color=COLOR[m], marker=house.MARKER[m], markersize=3,
                      linewidth=1.2, label=passive_labels[m]) for m in methods]
    handles += [Line2D([0], [0], color=COLOR[m], linestyle="--", marker="o",
                       markerfacecolor="white", markeredgecolor=COLOR[m], markersize=4,
                       linewidth=1.2, label=("Multigroup GLHF" if m == "multigroup" else "Pooled GLHF") + " (active)")
                for m in ("multigroup", "pooled")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.90),
               ncol=3, fontsize=6.5, columnspacing=1.1, handlelength=1.8)
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.23, top=0.72, wspace=0.23)
    return save(fig, "beta_curves"), exported


def gap_figure() -> tuple[list[Path], list[dict[str, str]]]:
    """The appendix's paired active-pooled minus active-multigroup contrast."""
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8), sharey=False)
    exported: list[dict[str, str]] = []
    for ax, dataset in zip(axes, ("coffee", "persona")):
        source = comparison_run(dataset) / "reduce/beta/method_group_contrasts.csv"
        rows = read(source)
        display = house.DISPLAY[dataset]
        chosen = [select(rows, cell="active@inf", contrast="pooled-multigroup", group=key)
                  for key, _, _ in display]
        x = np.arange(6)
        y = np.array([val(r, "estimate") for r in chosen])
        lo = np.array([val(r, "ci_lower") for r in chosen])
        hi = np.array([val(r, "ci_upper") for r in chosen])
        ax.axvline(0, color=house.TEXT, linewidth=0.9)
        ax.barh(x, y, color=COLOR["multigroup"], height=0.68, alpha=0.88)
        ax.errorbar(y, x, xerr=np.vstack((y - lo, hi - y)), fmt="none",
                    ecolor=house.WHISKER, capsize=2, elinewidth=0.8)
        ax.set_yticks(x, [house.SHORT_LABEL.get(key, label) for key, label, _ in display],
                      fontsize=6.3)
        ax.invert_yaxis()
        ax.set_xlabel("Pooled GLHF − Multigroup GLHF", fontsize=7.2)
        ax.set_title("(a) Coffee-rating study" if dataset == "coffee" else "(b) Completion-rating study",
                     loc="left", fontsize=8.3)
        house.tidy(ax, axis="x")
        exported += [output_row("per_group_paired_gaps", dataset, source, r) for r in chosen]
    fig.suptitle(r"Paired group gaps, active GLHF at $\beta=\infty$, $\tau_L=\tau_E=0.5$",
                 fontsize=8.5, y=0.99)
    fig.subplots_adjust(left=0.16, right=0.995, bottom=0.17, top=0.83, wspace=1.05)
    return save(fig, "per_group_paired_gaps"), exported


def main() -> int:
    global RUNS, OUT, SMOKE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RUNS)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    RUNS, OUT, SMOKE = args.results_root.resolve(), args.output.resolve(), args.smoke
    house.configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    source_rows: list[dict[str, str]] = []
    sources: dict[str, str] = {}
    for dataset in ("coffee", "persona"):
        run = comparison_run(dataset)
        info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
        if info["status"] != "completed" or bool(info.get("smoke")) != SMOKE:
            raise ValueError(f"{dataset}: comparison suite is not completed production")
        for render in (bar_figure, radius_figure):
            files, rows = render(dataset, run)
            outputs += files
            source_rows += rows
    files, rows = gap_figure()
    outputs += files
    source_rows += rows
    files, rows = beta_figure()
    outputs += files
    source_rows += rows
    checks = verify_rows(source_rows)
    sidecar = OUT / "plotted_rows.csv"
    write_rows(sidecar, source_rows)
    outputs.append(sidecar)
    for row in source_rows:
        path = ROOT / row["source_file"]
        sources[portable(path)] = sha(path)
    captions = OUT / "CAPTIONS.md"
    captions.write_text(
        "# Empirical figures\n\n" +
        ("SMOKE CHECK — NOT REPORTABLE RESULTS.\n\n" if SMOKE else "") +
        "Production uses reductions from 100 CV splits and "
        "50 learner seeds per split. Group bars compare active-trained multigroup and "
        "pooled GLHF at β=∞, τL=τE=0.5 with passive-trained batch NLHF and the untrained "
        "uniform reference. The rightmost cluster is the mean of the within-run "
        "maximum over selected groups. Bars extend from one, where welfare equals that of "
        "each group's best fixed feasible lottery; lower distortion means higher welfare. "
        "Whiskers show approximate 95% confidence intervals for each mean, corrected "
        "for overlapping cross-validation folds (Nadeau–Bengio over split-level scores).\n\n"
        "Radius panels use active-trained GLHF over six learner radii, passive-trained "
        "batch NLHF at each radius, and a fixed evaluation radius τE=0.5. The paired group "
        "gap figure shows active pooled minus active Multigroup GLHF at the anchor; "
        "positive values favor Multigroup GLHF. The combined beta figure plots the "
        "five passive settings for all four methods and active GLHF at β=4 and ∞ only; "
        "the x-axis is categorical. Shaded bands are marginal corrected 95% intervals.\n\n"
        "The plotted_rows.csv sidecar contains the exact source rows used for new "
        "plots. This script reads completed results; it performs no training.\n",
        encoding="utf-8",
    )
    outputs.append(captions)
    manifest = {
        "script": "scripts/make_paper_figures.py",
        "inputs": sources,
        "checks": {"plotted_rows_exactly_match_source_tables": checks,
                   "source_runs_completed": ["coffee", "persona"]},
        "outputs": {portable(p): sha(p) for p in outputs},
        "training": "none",
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(outputs) + 1} files; verified {checks} plotted rows against reduction CSVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
