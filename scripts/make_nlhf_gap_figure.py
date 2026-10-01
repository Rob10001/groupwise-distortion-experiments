#!/usr/bin/env python3
"""Render the NLHF-minus-multigroup per-group paired gap figure from reductions.

A companion to the pooled-minus-multigroup panel in make_paper_figures.py:
same layout, same anchor cell, different contrast. No learner or evaluator is
imported; every plotted number is copied from a reduction CSV row and rechecked
against that row before the output is saved.
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

import plot_style as house


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures/paper"
CONTRAST = "pooled_nlhf-multigroup"
CELL = "active@inf"
STEM = "per_group_paired_gaps_nlhf"
RUNS = ROOT / "results"
SMOKE = False


def contrast_csv(dataset: str) -> Path:
    path = RUNS / dataset / "comparison/reduce/beta/method_group_contrasts.csv"
    if not path.is_file():
        raise FileNotFoundError(f"No contrast reduction for {dataset}")
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
    if row["quantity"] != "group_distortion" or row["positive_favors"] != "multigroup":
        raise ValueError(f"Unexpected estimand or sign convention: {filters}")
    return row


def portable(path: Path) -> str:
    return Path(os.path.relpath(path, ROOT)).as_posix()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gap_figure() -> tuple[list[Path], list[dict[str, str]]]:
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8), sharey=False)
    exported: list[dict[str, str]] = []
    for ax, dataset in zip(axes, ("coffee", "persona")):
        source = contrast_csv(dataset)
        rows = read(source)
        display = house.DISPLAY[dataset]
        chosen = [select(rows, cell=CELL, contrast=CONTRAST, group=key)
                  for key, _, _ in display]
        x = np.arange(6)
        y = np.array([float(r["estimate"]) for r in chosen])
        lo = np.array([float(r["ci_lower"]) for r in chosen])
        hi = np.array([float(r["ci_upper"]) for r in chosen])
        ax.axvline(0, color=house.TEXT, linewidth=0.9)
        ax.barh(x, y, color=house.COLOR["pooled_nlhf"], height=0.68, alpha=0.88)
        ax.errorbar(y, x, xerr=np.vstack((y - lo, hi - y)), fmt="none",
                    ecolor=house.WHISKER, capsize=2, elinewidth=0.8)
        ax.set_yticks(x, [house.SHORT_LABEL.get(key, label) for key, label, _ in display],
                      fontsize=6.3)
        ax.invert_yaxis()
        ax.set_xlabel("Batch NLHF − Multigroup GLHF", fontsize=7.2)
        ax.set_title("(a) Coffee-rating study" if dataset == "coffee" else "(b) Completion-rating study",
                     loc="left", fontsize=8.3)
        house.tidy(ax, axis="x")
        exported += [{"figure": STEM, "panel": dataset,
                      "source_file": portable(source), **r} for r in chosen]
    fig.suptitle(r"Paired group gaps, passive batch NLHF minus active GLHF at "
                 r"$\beta=\infty$, $\tau_L=\tau_E=0.5$", fontsize=8.5, y=0.99)
    fig.subplots_adjust(left=0.16, right=0.995, bottom=0.17, top=0.83, wspace=0.70)
    paths = [OUT / f"{STEM}.pdf", OUT / f"{STEM}.png"]
    if SMOKE:
        fig.text(.5, .5, "SMOKE — NOT RESULTS", ha="center", alpha=.4, rotation=20)
    fig.savefig(paths[0], bbox_inches="tight", pad_inches=0.04)
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.04, dpi=400)
    plt.close(fig)
    return paths, exported


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
    outputs, source_rows = gap_figure()
    sidecar = OUT / f"plotted_{STEM}_rows.csv"
    fields = list(dict.fromkeys(key for row in source_rows for key in row))
    with sidecar.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(source_rows)
    outputs.append(sidecar)

    # Re-read every plotted row from its CSV and confirm the figure's numbers.
    checked = 0
    for row in source_rows:
        origin = read(ROOT / row["source_file"])
        match = select(origin, cell=CELL, contrast=CONTRAST, group=row["group"])
        for key in ("estimate", "ci_lower", "ci_upper", "corrected_se", "p_holm"):
            if match[key] != row[key]:
                raise ValueError(f"Plotted {key} disagrees with source for {row['group']}")
        checked += 1

    sources = {portable(ROOT / r["source_file"]): sha(ROOT / r["source_file"])
               for r in source_rows}
    manifest = OUT / f"{STEM}_manifest.json"
    manifest.write_text(json.dumps({
        "script": "scripts/make_nlhf_gap_figure.py",
        "contrast": CONTRAST,
        "cell": CELL,
        "inputs": sources,
        "checks": {"plotted_rows_exactly_match_source_tables": checked},
        "outputs": {portable(p): sha(p) for p in outputs},
        "training": "none",
    }, indent=2) + "\n", encoding="utf-8")
    for path in outputs + [manifest]:
        print(portable(path))
    print(f"verified {checked} plotted rows against reduction CSVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
