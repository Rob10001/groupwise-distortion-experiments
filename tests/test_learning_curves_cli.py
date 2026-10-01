"""The optional suite curve command stays separate from the comparison workload."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from empirical_testing.experiments import cli

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dataset", ("coffee", "persona"))
def test_curve_plan_counts_only_groupwise_reconstructions(dataset):
    plan = cli.build_run_plan(dataset, "learning-curves", project_root=ROOT)
    assert plan["groupwise_reconstructions"] == 10_000
    assert plan["scheduled_fits"] == {"glhf": 10_000, "nlhf": 0}
    assert plan["pooled_fits"] == plan["nlhf_fits"] == 0
    assert plan["protocols"] == ["active", "passive"]
    assert plan["beta"] == "inf" and plan["learner_tau"] == 0.5
    assert plan["training_horizon"] == {"coffee": "3032-3033", "persona": "1347-1348"}[dataset]
    assert "rounds" not in plan
    smoke = cli.build_run_plan(dataset, "learning-curves", project_root=ROOT, smoke=True)
    assert smoke["groupwise_reconstructions"] == 8
    assert smoke["reportable"] is False


@pytest.mark.parametrize("dataset", ("coffee", "persona"))
def test_both_runners_offer_standalone_curve_dry_run(dataset):
    result = subprocess.run(
        [sys.executable, str(ROOT / f"run_{dataset}.py"), "learning-curves", "--dry-run"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "10,000 groupwise" in result.stdout
    assert "0 pooled; 0 NLHF" in result.stdout


def test_curve_source_is_standalone_only():
    result = subprocess.run(
        [sys.executable, str(ROOT / "run_coffee.py"), "comparison-suite",
         "--dry-run", "--curve-source", "x"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "--curve-source applies only" in result.stderr


def test_standalone_curves_refuse_fresh_execution():
    result = subprocess.run(
        [sys.executable, str(ROOT / "run_coffee.py"), "learning-curves",
         "--execution", "fresh", "--dry-run"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "requires completed beta source artifacts" in result.stderr


def test_suite_opt_in_plan_keeps_base_workload():
    base = cli.build_run_plan("coffee", "comparison-suite", project_root=ROOT)
    extended = cli.build_run_plan(
        "coffee", "comparison-suite", project_root=ROOT, with_learning_curves=True,
    )
    assert base["scheduled_fits"] == extended["scheduled_fits"]
    assert "optional_learning_curves" not in base
    assert extended["optional_learning_curves"]["groupwise_reconstructions"] == 10_000
