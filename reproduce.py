"""Run every paper experiment and render its figures from newly computed results."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def commands(output: Path, workers: int, smoke: bool) -> list[list[str]]:
    common = ["--workers", str(workers)] + (["--smoke"] if smoke else [])
    figure_flags = ["--smoke"] if smoke else []
    runs = [
        [f"run_{dataset}.py", "comparison-suite", "--execution", "fresh",
         "--with-learning-curves", "--output", str(output / dataset / "comparison"), *common]
        for dataset in ("coffee", "persona")
    ]
    runs.append(["run_persona.py", "panel-permutations", "--output",
                 str(output / "persona/panel-permutations"), *common])
    for script in ("make_paper_figures.py", "make_nlhf_gap_figure.py"):
        runs.append([f"scripts/{script}", "--results-root", str(output),
                     "--output", str(output / "figures/paper"), *figure_flags])
    runs.append(["scripts/make_learning_curve_figure.py",
                 "--coffee-run", str(output / "coffee/comparison/learning_curves"),
                 "--persona-run", str(output / "persona/comparison/learning_curves"),
                 "--output", str(output / "figures/learning-curves/active_passive"), *figure_flags])
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="Two splits; not reportable results")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = (args.output or ROOT / ("results/reproduction-smoke" if args.smoke
                                    else "results/reproduced")).resolve()
    for command in commands(output, args.workers, args.smoke):
        print(subprocess.list2cmdline([sys.executable, *command]), flush=True)
        if not args.dry_run:
            subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
    if not args.dry_run:
        print(f"Completed {'smoke check' if args.smoke else 'reproduction'}: {output}")


if __name__ == "__main__":
    main()
