"""Draw the two-dataset protocol trajectories from completed reconstructions.

Run from EmpiricalTesting/ after each dataset's learning-curves command.
This command reads saved reductions only and never fits a learner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAPTION = (
    "Active and passive Multigroup GLHF training trajectories at beta=infinity "
    "and learner/evaluation KL radii 0.5. Each checkpoint evaluates the uniform "
    "average of the first t pre-round policy maps from the original full-horizon "
    "fit on the same held-out fold; the learning-rate horizon is not retuned. "
    "One comparison is collected per training respondent. The x coordinate is "
    "the mean comparison count across folds at a fixed fraction of training; "
    "fold sizes differ slightly. Worst-group distortion maximizes over the six "
    "selected groups within each seed before seeds are averaged within each "
    "split. Curves average 100 split scores, each based on 50 learner seeds. "
    "Shading shows approximate 95% pointwise confidence intervals, corrected "
    "for overlapping cross-validation folds. These describe each method's mean "
    "at a checkpoint; protocol comparisons use the separately saved paired differences. "
    "The horizontal line at one marks the fixed-lottery benchmark. Bands in the "
    "synthetic panel are narrow at the displayed scale. The datasets use "
    "different vertical scales. The protocol comparison includes each "
    "sampler's native score normalization."
)


def draw(frames: dict[str, pd.DataFrame], output: Path, *, smoke: bool = False) -> list[Path]:
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.05), layout="constrained")
    for ax, (dataset, data) in zip(axes, frames.items(), strict=True):
        for protocol, color, style in (
            ("active", "#D9650B", "-"),
            ("passive", "#546477", "--"),
        ):
            rows = data[data.protocol == protocol].sort_values("mean_comparisons")
            ax.plot(
                rows.mean_comparisons,
                rows.estimate,
                color=color,
                linestyle=style,
                linewidth=1.6,
                label=protocol.title(),
            )
            ax.fill_between(
                rows.mean_comparisons,
                rows.ci_lower,
                rows.ci_upper,
                color=color,
                alpha=0.20,
                linewidth=0,
            )
        ax.axhline(1, color="#777777", linewidth=0.7)
        ax.set_title(
            "Coffee-rating study" if dataset == "coffee" else "Completion-rating study",
            loc="left",
            fontsize=9,
        )
        ax.set_xlabel("Training comparisons", fontsize=8)
        ax.set_xlim(left=0)
        step = 500 if dataset == "coffee" else 250
        ax.set_xticks(range(0, int(data.mean_comparisons.max()) + 1, step))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Mean worst-group distortion", fontsize=8)
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle(r"Multigroup GLHF: $\beta=\infty$, KL radius $0.5$", fontsize=9)
    if smoke:
        fig.text(.5, .5, "SMOKE — NOT RESULTS", ha="center", alpha=.4, rotation=20)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        path = output.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=220)
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for dataset in ("coffee", "persona"):
        parser.add_argument(
            f"--{dataset}-run",
            type=Path,
            default=ROOT / f"results/{dataset}/learning-curves",
        )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figures/learning-curves/active_passive",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    frames, sources = {}, {}
    for dataset in ("coffee", "persona"):
        run = getattr(args, f"{dataset}_run")
        info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
        if (
            info.get("status") != "completed"
            or bool(info.get("smoke")) != args.smoke
            or (not args.smoke and not info.get("reportable"))
            or not info.get("endpoint_verified")
            or info.get("n_splits") != (2 if args.smoke else 100)
            or info.get("n_seeds") != (2 if args.smoke else 50)
        ):
            raise ValueError(
                f"Completed, endpoint-verified production curves required: {run}"
            )
        table = run / "reduce/method_levels.csv"
        frames[dataset] = pd.read_csv(table)
        sources[dataset] = {
            "run": Path(os.path.relpath(run.resolve(), ROOT)).as_posix(),
            "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        }
    paths = draw(frames, args.output, smoke=args.smoke)
    caption = ("SMOKE CHECK — NOT REPORTABLE RESULTS.\n\n" if args.smoke else "") + CAPTION
    pd.concat(frames.values(), ignore_index=True).to_csv(
        args.output.with_suffix(".csv"), index=False
    )
    args.output.with_suffix(".caption.md").write_text(caption + "\n", encoding="utf-8")
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "sources": sources,
                "caption": caption,
                "outputs": [Path(os.path.relpath(p.resolve(), ROOT)).as_posix() for p in paths],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
