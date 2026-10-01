"""CLI tests for execution modes, defaults, and option validation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from empirical_testing.experiments import cli as cli
from empirical_testing.experiments.suite import (
    MODES, SUITES, ExecutionPolicyError, ReuseBlockedError,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = {"coffee": ROOT / "run_coffee.py", "persona": ROOT / "run_persona.py"}


def _run(runner: Path, *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(runner), *argv],
        cwd=ROOT, capture_output=True, text=True,
    )


def _args(mode, **overrides):
    base = dict(mode=mode, execution="reuse-required", dry_run=True, smoke=False, output=None)
    base.update(overrides)
    return SimpleNamespace(**base)


# ------------------------------------------------------------------ the modes

@pytest.mark.parametrize("dataset", sorted(RUNNERS))
@pytest.mark.parametrize("mode", MODES)
def test_both_runners_offer_every_suite_mode(dataset, mode):
    if dataset == "coffee" and mode == "panel-permutations":
        pytest.skip("Panel construction applies only to Persona")
    result = _run(RUNNERS[dataset], mode, "--dry-run")
    assert "invalid choice" not in result.stderr, result.stderr
    assert result.returncode in (0, 1), result.stderr


@pytest.mark.parametrize("dataset", sorted(RUNNERS))
def test_the_suite_options_are_accepted_by_both_runners(dataset):
    result = _run(RUNNERS[dataset], "comparison-suite", "--dry-run", "--smoke",
                  "--execution", "fresh", "--workers", "2")
    assert result.returncode == 0, result.stderr
    assert "SMOKE" in result.stdout


@pytest.mark.parametrize("dataset", sorted(RUNNERS))
def test_retired_modes_are_refused(dataset):
    other = "crossfit"
    result = _run(RUNNERS[dataset], other, "--dry-run")
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_an_unknown_execution_policy_is_refused_at_the_command_line():
    result = _run(RUNNERS["coffee"], "comparison-suite", "--execution", "cheapest")
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


# ----------------------------------------------------------- the resolved plan

@pytest.mark.parametrize("dataset", sorted(SUITES))
def test_reuse_required_is_the_default_policy(dataset):
    result = _run(RUNNERS[dataset], "comparison-suite", "--dry-run")
    assert "execution reuse-required" in result.stdout


@pytest.mark.parametrize("dataset", sorted(SUITES))
@pytest.mark.parametrize("mode", MODES)
def test_no_plan_ever_schedules_a_beta_glhf_fit_under_reuse_required(dataset, mode):
    if dataset == "coffee" and mode == "panel-permutations":
        pytest.skip("Panel construction applies only to Persona")
    plan = cli.build_run_plan(dataset, mode, project_root=ROOT, execution="reuse-required")
    assert plan["scheduled_beta_glhf_fits"] == 0
    assert plan["plan"]["beta_glhf_fits"] == 0


@pytest.mark.parametrize("dataset", sorted(SUITES))
def test_the_comparison_mode_schedules_the_specified_fits(dataset):
    plan = cli.build_run_plan(dataset, "comparison-suite", project_root=ROOT)
    # Five of six radii are fitted; tau = 0.5 is shared with the beta grid.
    assert plan["scheduled_fits"]["glhf"] == 50_000
    assert plan["scheduled_fits"]["nlhf"] == 50_000


@pytest.mark.parametrize("dataset", sorted(SUITES))
def test_the_complete_suite_covers_every_stage(dataset):
    plan = cli.build_run_plan(dataset, "comparison-suite", project_root=ROOT)
    assert plan["stages"] == list(cli.SUITE_STAGES)
    for stage in ("validate_inputs", "import_reuse", "missing_fits", "evaluate",
                  "reduce", "coverage", "summary"):
        assert stage in plan["stages"]
    assert "figures" not in plan["stages"]  # Paper figures use the separate plotting script.


@pytest.mark.parametrize("dataset", sorted(SUITES))
def test_the_complete_suite_schedules_no_auxiliary_experiment(dataset):
    """No budget curves, trace replays, stability studies or older routes."""

    plan = cli.build_run_plan(dataset, "comparison-suite", project_root=ROOT)
    scheduled = json.dumps(plan["cells_in_scope"]) + json.dumps(plan["stages"])
    for excluded in cli.EXCLUDED_EXPERIMENTS:
        assert excluded not in scheduled, f"the suite scheduled {excluded!r}"


def test_smoke_is_a_reduced_workload_and_is_not_reportable():
    plan = cli.build_run_plan("coffee", "comparison-suite", project_root=ROOT, smoke=True)
    assert plan["splits"] == cli.SMOKE_SPLITS and plan["seeds"] == cli.SMOKE_SEEDS
    assert plan["reportable"] is False
    assert "not results" in plan["smoke_note"]
    full = cli.build_run_plan("coffee", "comparison-suite", project_root=ROOT)
    assert plan["scheduled_fits"]["glhf"] < full["scheduled_fits"]["glhf"]


def test_an_unknown_mode_or_dataset_is_refused():
    with pytest.raises(ExecutionPolicyError, match="unknown suite mode"):
        cli.conditions_for_mode("comparison-suite-v9")
    with pytest.raises(ExecutionPolicyError, match="unknown dataset"):
        cli.build_run_plan("tea", "comparison-suite", project_root=ROOT)


# ------------------------------------------------------------------ the blocker

@pytest.mark.parametrize("dataset", sorted(SUITES))
def test_a_missing_import_blocks_the_beta_condition_instead_of_retraining(dataset, tmp_path):
    """The blocker is reported; the beta grid is never retrained to repair it."""

    plan = cli.build_run_plan(dataset, "protocol-beta", project_root=tmp_path)
    assert plan["blocked"] is True
    assert plan["blockers"], "blocked without saying why"
    assert plan["scheduled_beta_glhf_fits"] == 0, "a blocked run scheduled beta GLHF fits"


@pytest.mark.parametrize("dataset", sorted(SUITES))
def test_a_blocked_non_dry_run_raises_rather_than_falling_back(dataset, tmp_path):
    with pytest.raises(ReuseBlockedError):
        cli.run_mode(dataset, _args("comparison-suite", dry_run=False),
                         project_root=tmp_path, log=lambda _m: None)


@pytest.mark.parametrize("dataset", sorted(SUITES))
def test_fresh_comparison_needs_no_import(dataset, tmp_path):
    """A fresh run must not require a bundled reference run."""

    plan = cli.build_run_plan(dataset, "comparison-suite", project_root=tmp_path, execution="fresh")
    assert not plan.get("blocked")


def test_a_blocked_dry_run_reports_the_blocker_and_exits_nonzero(tmp_path, capsys):
    code = cli.run_mode("coffee", _args("protocol-beta", output=tmp_path),
                            project_root=tmp_path, log=lambda _m: None)
    assert code == 1
    written = json.loads((tmp_path / "plan.json").read_text())
    assert written["blocked"] is True


def test_a_dry_run_writes_its_plan_and_does_no_work(tmp_path):
    code = cli.run_mode("coffee", _args("comparison-suite", output=tmp_path, execution="fresh"),
                            project_root=tmp_path, log=lambda _m: None)
    assert code == 0
    written = json.loads((tmp_path / "plan.json").read_text())
    assert written["scheduled_beta_glhf_fits"] == 70_000
    # Nothing but the plan was produced.
    assert [p.name for p in tmp_path.iterdir()] == ["plan.json"]


@pytest.mark.parametrize("dataset", sorted(RUNNERS))
def test_fresh_execution_dispatches_without_bundled_results(dataset, tmp_path, monkeypatch):
    from empirical_testing.experiments import run
    calls = []
    def execute(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(exit_code=0)
    monkeypatch.setattr(run, "run_suite", execute)
    output = tmp_path / "fresh"
    assert cli.run_mode(dataset, _args("comparison-suite", execution="fresh", dry_run=False,
                                      output=output), project_root=tmp_path,
                        log=lambda _: None) == 0
    assert calls[0]["execution"] == "fresh"
    assert calls[0]["imported_source"] == output / "beta-source"


@pytest.mark.parametrize("dataset", sorted(SUITES))
def test_fresh_dry_run_counts_required_beta_training(dataset, tmp_path):
    plan = cli.build_run_plan(dataset, "comparison-suite", project_root=tmp_path, execution="fresh")
    assert plan["scheduled_beta_glhf_fits"] == 70_000
    assert plan["unsupported_execution"] is False
    assert plan["execution_note"] is None


def test_supplied_beta_source_is_used_without_bundled_results(tmp_path, monkeypatch):
    from empirical_testing.experiments import run
    calls = []
    def execute(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(exit_code=0)
    monkeypatch.setattr(run, "run_suite", execute)
    source = tmp_path / "my-beta-run"
    assert cli.run_mode("persona", _args("comparison-suite", dry_run=False,
                        beta_source=source), project_root=tmp_path, log=lambda _: None) == 0
    assert calls[0]["imported_source"] == source
    assert calls[0]["supplied_beta_source"] is True


@pytest.mark.parametrize("dataset", sorted(SUITES))
def test_standalone_beta_dispatches_shared_raw_input_driver(dataset, tmp_path, monkeypatch):
    from empirical_testing.experiments import fresh
    calls = []
    def execute(label, **kwargs):
        calls.append((label, kwargs))
        return SimpleNamespace(exit_code=0)
    monkeypatch.setattr(fresh, "run_beta", execute)
    assert cli.run_mode(dataset, _args("protocol-beta", execution="fresh", dry_run=False,
                        smoke=True), project_root=tmp_path, log=lambda _: None) == 0
    assert calls[0][0] == dataset
    assert calls[0][1]["smoke"] is True


@pytest.mark.parametrize("dataset", sorted(SUITES))
def test_archive_source_is_shared_by_reuse_validation_runner_and_optional_curves(dataset, tmp_path, monkeypatch):
    from empirical_testing.experiments import learning_curves, provenance, run

    source = tmp_path / "results" / dataset / "beta-source" / "source"
    source.mkdir(parents=True)
    (source / "run_info.json").write_text('{"reportable": true, "smoke": false}', encoding="utf-8")
    (source.parent / provenance.IMPORT_MANIFEST).write_text("{}", encoding="utf-8")
    calls = []

    def verify(path, spec):
        calls.append(("validation", path))
        return provenance.VerificationReport(spec.dataset, str(path))

    def fake_runner(label):
        def invoke(*_args, **kwargs):
            calls.append((label, kwargs["imported_source"]))
            return SimpleNamespace(exit_code=0)
        return invoke

    monkeypatch.setattr(provenance, "verify_source", verify)
    monkeypatch.setattr(run, "run_suite", fake_runner("suite"))
    monkeypatch.setattr(learning_curves, "run_learning_curves", fake_runner("curves"))
    code = cli.run_mode(dataset, _args("comparison-suite", dry_run=False, with_learning_curves=True),
                            project_root=tmp_path, log=lambda _m: None)
    assert code == 0
    assert calls == [("validation", source), ("suite", source), ("curves", source)]
    assert learning_curves.resolve_source(tmp_path, dataset, None) == source
