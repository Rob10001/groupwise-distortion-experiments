# Groupwise Distortion Guarantees for Preference-Based Alignment

**Jacob Brodkey\*, Roberto Tamez\*, and Aaron Roth**

University of Pennsylvania

\*Equal contribution; co-first authors.

Code, input data, and saved results for the Coffee and Persona experiments in
our paper. GLHF learns a lottery over alternatives conditional on a user's
possibly overlapping group memberships, using one pairwise comparison per
training user. These experiments compare it with learners that serve the same
lottery to everyone.

The manuscript is not yet on arXiv; its identifier and link will be added here
when available. See [Citation](#citation) for the current manuscript citation.
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

## Start with the algorithm

For a small example that takes seconds rather than running the experimental
sweeps:

```sh
python examples/minimal_glhf.py
```

The example trains GLHF on a uniform passive comparison log for three
alternatives and three overlapping groups. It supplies group memberships,
queried pairs, winners, and query probabilities, but no cardinal utilities.
It then prints the learned lotteries for four membership patterns. Replace
the toy log with your own comparisons to use the same interface.

The returned policy averages the complete pre-round policies; `predict(groups)`
evaluates that average for new users. Each row is a lottery within the specified
KL radius of the reference lottery. The example uses the Python backend for
readability; the reproduction runs use the tested Numba backend.

### Code guide

- [Minimal example](examples/minimal_glhf.py): input shapes, winner encoding,
  and prediction on new group memberships.
- [GLHF implementation](src/empirical_testing/algorithms/span_adaptive.py):
  adaptive rates, specialist updates, and the returned policy.
- [Independent reference](src/empirical_testing/reference/span_adaptive_reference.py):
  the same algorithm written with explicit NumPy/SciPy operations.
- [Batch NLHF](src/empirical_testing/algorithms/nlhf.py): the pooled comparison
  baseline, solved numerically within a KL ball.
- [Evaluation](src/empirical_testing/reporting/metrics.py): groupwise welfare,
  best feasible fixed-lottery benchmarks, and distortion.

In the GLHF code, `eta` denotes the paper's outer rate `alpha`, `theta` its
inner rate `eta`, and `schedule.complexity` its logarithmic factor `Lambda_T`.
The active-sampling experiment interface takes ratings to simulate a queried
user's comparison; the GLHF updates themselves use only the resulting pair and
winner. This interface is an experiment simulator, not a live human-feedback
service.

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

### Runtime and storage

The minimal example and figure-only commands are quick checks. The full
command performs hundreds of thousands of fits; it is a batch experiment,
not a quick-start example. As a scale reference, the saved comparison stages
took about 44 minutes for Coffee and 28 minutes for Persona with six workers
on macOS (`results/*/comparison/run_info.json`). Those stages reused the
GLHF beta grid: these times exclude fresh beta fitting, learning curves, and
panel checks, so they are **not** timings for the full reproduction command.
Runtime depends on the machine and worker count; the first Numba call also
compiles kernels.

The checkout contains roughly 240 MB of data and saved results before Git
history. Fresh runs also save comparison logs and learning-curve arrays omitted
from the bundled results, so allow additional disk space. The smoke command
still uses 50 seeds for its beta/comparison stages; use the minimal example
for a fast installation check.

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

Build a compact archive with `python build_submission.py`. Local reruns,
environments, caches, and console logs are excluded. The builder strips local
paths and historical source-revision metadata from the archived result files;
it retains the public authorship, citation, and license information.

## Citation

Until the arXiv version is available, please cite the manuscript as:

```bibtex
@unpublished{brodkey2026groupwise,
  title = {Groupwise Distortion Guarantees for Preference-Based Alignment},
  author = {Brodkey, Jacob and Tamez, Roberto and Roth, Aaron},
  year = {2026},
  note = {Manuscript. Jacob Brodkey and Roberto Tamez contributed equally}
}
```

[CITATION.cff](CITATION.cff) supplies machine-readable metadata. When reporting
a reproduction, also record the repository commit (`git rev-parse HEAD`) and
the command used. [RELEASE.md](RELEASE.md) identifies the original public
snapshot and describes how manuscript-linked releases are frozen.

## Licenses

Original code and accompanying software documentation are under the
[MIT license](LICENSE). Author-created Persona materials, saved results, and
figures are under [CC BY 4.0](LICENSE-DATA.md). The Coffee survey retains its
upstream terms; see [data/README.md](data/README.md) for source attribution and
the scope of each license.
