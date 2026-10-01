"""Run the Coffee comparison suite on the bundled survey and frozen analysis design.

    python run_coffee.py comparison-suite
    python run_coffee.py comparison-suite --smoke --output results/coffee-smoke
    python run_coffee.py learning-curves
    python run_coffee.py protocol-beta --dry-run

Use comparison-suite --execution fresh to train from ratings, or omit that
option to reuse bundled beta results. protocol-beta trains the GLHF beta grid.
See README.md for setup, inputs, outputs, and the interpretation of smoke runs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
for entry in (PROJECT_ROOT / "src",):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from empirical_testing.experiments.cli import (  # noqa: E402
    add_arguments, run_mode, validate_arguments,
)
from empirical_testing.experiments.suite import MODES  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=MODES)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="output directory (default: the matching coffee results directory)",
    )
    parser.add_argument(
        "--workers", type=int, default=min(4, os.cpu_count() or 1),
        help="worker processes; results do not depend on this setting",
    )
    add_arguments(parser)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    validate_arguments(parser, args)
    return run_mode("coffee", args, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
