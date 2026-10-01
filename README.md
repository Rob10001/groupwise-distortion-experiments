# Coffee and Persona experiments

Code, input data, and reference results for the paper's empirical studies.
Clone or download this repository and run commands from its root directory.
After installing dependencies, reproduction needs no network access or API key.

## Setup

Use Python **3.12**:

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or
`source .venv/bin/activate` on macOS/Linux. Then install dependencies:

```sh
python -m pip install -r requirements.txt
```

## Reproduce all experiments and figures

```sh
python reproduce.py --workers 4
```

This runs both datasets' beta/protocol sweeps, GLHF and NLHF comparisons,
learner-radius sweeps, active/passive learning curves, all five Persona panel
alignments, and the paper figure generators. It trains from the bundled input
data using `configs/coffee.json` and `configs/persona.json`.
Full experiments use 100 cross-validation splits and 50 learner seeds per split.

Outputs go to `results/reproduced/`: dataset results under `coffee/` and
`persona/`, figures under `figures/`, and panel estimates in
`persona/panel-permutations/panel_summary.csv`. Use `--output DIR` to choose
another location. Use a new output directory after changing code or settings.

For an end-to-end smoke check:

```sh
python reproduce.py --smoke --workers 2
```

This uses two original splits and writes to `results/reproduction-smoke/`.
Beta/comparison runs keep 50 seeds; curves and panel checks use two seeds.
Smoke results are not paper results. `--dry-run` prints the commands without
running them. To run the automated tests, use `python -m pytest`.

## Recreate figures from bundled results

```sh
python scripts/make_paper_figures.py
python scripts/make_nlhf_gap_figure.py
python scripts/make_learning_curve_figure.py
```

These commands require no training and write to `figures/paper/` and
`figures/learning-curves/`.

## Individual experiments

Replace `run_coffee.py` with `run_persona.py` for Persona:

```sh
python run_coffee.py comparison-suite --execution fresh --with-learning-curves --workers 4 --output results/coffee-rerun
python run_coffee.py protocol-beta --workers 4 --output results/coffee-beta-rerun
python run_coffee.py learning-curves --workers 4 --output results/coffee-curves-rerun
python run_persona.py panel-permutations --workers 4 --output results/panels-rerun
```

`comparison-suite` includes the beta grid, radius sweep, and passive NLHF.
Without `--execution fresh`, it reuses bundled beta results; `--beta-source DIR`
selects your own completed beta run. `protocol-beta` runs the GLHF beta
and uniform-reference cells. Standalone `learning-curves` uses bundled beta
results; `--curve-source DIR` selects another source, such as
`results/coffee-rerun/beta-source`. Each runner's `--help` lists its options.

## Inputs and saved results

Both complete datasets are included; see [data/README.md](data/README.md) for
sources and preprocessing. `src/` contains the implementation, `scripts/` the
figure generators, and `tests/` the automated checks.

Bundled `results/{coffee,persona}/` directories contain reference comparisons,
beta sources, and learning-curve reductions. Persona also includes the five
panel runs. Within comparison results, `reduce/beta/` contains protocol/beta
tables, `reduce/fixed_0.5/` the paper's radius comparisons, and `reduce/moving/`
a diagnostic using a changing evaluation radius. `method_levels.csv` reports
worst-group means; `method_group_levels.csv` reports group means; corresponding
`*contrasts.csv` files report paired comparisons. `run_info.json` records run
settings and completion status.

The approximately 86 MB ZIP omits bulky comparison logs and per-split curve
arrays. Reproduction regenerates them. Saved comparison fits used Python 3.13
on macOS; the package is tested with Python 3.12 on Windows. Small floating-point
differences across platforms are expected.

Rebuild the ZIP with `python build_submission.py`. Local reruns, environments,
caches, and console logs are excluded.
