"""Shared CLI for beta sweeps, comparisons, and learning curves.

Persona additionally supports the five panel-construction runs.
Use reproduce.py to run the complete workflow."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from empirical_testing.experiments.suite import (
    EXECUTION_POLICIES,
    MODES,
    SUITES,
    ExecutionPolicyError,
    ReuseBlockedError,
    dry_run_inventory,
    resolve_beta_source,
)

#: Smoke runs exercise two original folds.
SMOKE_SPLITS = 2
#: Comparisons keep the full seed axis to join with saved beta results.
SMOKE_SEEDS = 50

#: Workflows outside the paper's comparison suite.
EXCLUDED_EXPERIMENTS = (
    "budget", "trace_replay", "rate_trace_replay", "group_stability",
    "fixed_rate", "adaptive_rate", "utility_map_sensitivity", "bootstrap",
    "expanded_suite", "primary", "crossfit-v1", "crossfit-v2", "crossfit-v3",
    "tau_sweep_passive", "beta_sweep_v3",
)


def add_arguments(parser: Any) -> None:
    """Add the options every suite mode accepts, to an existing parser."""

    parser.add_argument(
        "--execution", choices=EXECUTION_POLICIES, default=None,
        help="reuse saved beta results or train from ratings; default: fresh for "
             "protocol-beta/panel-permutations, reuse-required otherwise",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="show the run plan without fitting",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=f"check {SMOKE_SPLITS} splits; beta/comparisons keep {SMOKE_SEEDS} seeds, "
             "curves/panels use two; not paper results",
    )
    parser.add_argument(
        "--with-learning-curves", action="store_true",
        help="comparison-suite only: reconstruct the full training trajectories for active/passive "
             "MULTIGROUP learning curves after the suite completes",
    )
    parser.add_argument(
        "--curve-source", type=Path, default=None,
        help="learning-curves only: completed beta source directory to validate and reuse",
    )
    parser.add_argument("--beta-source", type=Path,
                        help="comparison-suite: reuse a completed standalone beta run")


def validate_arguments(parser: Any, args: Any) -> None:
    """Validate the supported commands and their option boundaries."""

    if getattr(args, "execution", None) is None:
        args.execution = "fresh" if args.mode in ("protocol-beta", "panel-permutations") else "reuse-required"
    if getattr(args, "beta_source", None) is not None and (
        args.mode != "comparison-suite" or args.execution != "reuse-required"
    ):
        parser.error("--beta-source requires comparison-suite --execution reuse-required")
    is_suite_mode = getattr(args, "mode", None) in MODES
    if not is_suite_mode:
        for option, flag in (("--dry-run", "dry_run"), ("--smoke", "smoke")):
            if getattr(args, flag, False):
                parser.error(f"{option} applies only to the suite modes {MODES}")
        if getattr(args, "execution", "reuse-required") != "reuse-required":
            parser.error(f"--execution applies only to the suite modes {MODES}")
    if getattr(args, "with_learning_curves", False) and getattr(args, "mode", None) != "comparison-suite":
        parser.error("--with-learning-curves applies only to comparison-suite")
    if getattr(args, "curve_source", None) is not None and getattr(args, "mode", None) != "learning-curves":
        parser.error("--curve-source applies only to learning-curves")
    if getattr(args, "mode", None) == "learning-curves" and getattr(args, "execution", "reuse-required") == "fresh":
        parser.error("learning-curves requires completed beta source artifacts; --execution fresh does not apply")
    if is_suite_mode and not getattr(args, "dry_run", False):
        reason = unsupported_execution_reason(args.mode, getattr(args, "execution", "reuse-required"))
        if reason:
            parser.error(reason)


def unsupported_execution_reason(mode: str, execution: str) -> str | None:
    """Explain planning surfaces the production driver cannot execute."""

    if mode in ("protocol-beta", "panel-permutations") and execution != "fresh":
        return f"{mode} trains from ratings; use --execution fresh"
    return None


def plan_for(args: Any) -> tuple[int, int]:
    """``(splits, seeds)`` for this invocation."""

    if getattr(args, "smoke", False):
        return SMOKE_SPLITS, SMOKE_SEEDS
    return 100, 50


def conditions_for_mode(mode: str) -> tuple[str, ...]:
    """Which conditions a mode covers."""

    if mode == "protocol-beta":
        return ("beta",)
    if mode == "panel-permutations":
        return ("panel_permutations",)
    if mode == "comparison-suite":
        return ("beta", "tau")
    if mode == "learning-curves":
        return ("learning_curves",)
    raise ExecutionPolicyError(f"unknown suite mode {mode!r}; expected one of {MODES}")


#: The stages the complete-suite command covers, in order.
SUITE_STAGES = (
    "validate_inputs",
    "import_reuse",
    "missing_fits",
    "evaluate",
    "reduce",
    "coverage",
    "summary",
)

# Comparisons in each frozen production training log, not CV split counts.
CURVE_TRAINING_HORIZON: Mapping[str, str] = {"coffee": "3032-3033", "persona": "1347-1348"}


def build_run_plan(dataset: str, mode: str, *, project_root: Path,
                   execution: str = "reuse-required", smoke: bool = False,
                   with_learning_curves: bool = False,
                   beta_source: Path | None = None) -> dict[str, Any]:
    """Everything a run would do, resolved but not started.

    Under ``reuse-required`` the reuse gate is evaluated here, so a missing or
    inconsistent import is a blocker reported before any work begins rather
    than a surprise partway through.
    """

    if dataset not in SUITES:
        raise ExecutionPolicyError(f"unknown dataset {dataset!r}; expected one of {sorted(SUITES)}")
    suite = SUITES[dataset]
    conditions = conditions_for_mode(mode)
    splits, seeds = (SMOKE_SPLITS, SMOKE_SEEDS) if smoke else (100, 50)

    if mode == "panel-permutations":
        if dataset != "persona":
            raise ExecutionPolicyError("panel-permutations is only available for Persona")
        return dict(mode=mode, dataset=dataset, execution="fresh", conditions=list(conditions),
                    smoke=smoke, reportable=not smoke, splits=splits, seeds=2 if smoke else seeds,
                    method_inventory=["pooled", "multigroup"], panel_seeds=list(range(20260916, 20260921)),
                    scheduled_fits={"glhf": 5 * splits * (2 if smoke else seeds) * 2, "nlhf": 0},
                    scheduled_beta_glhf_fits=0, plan={"beta_glhf_fits": 0})

    if mode == "learning-curves":
        if execution == "fresh":
            raise ExecutionPolicyError("learning-curves requires completed beta source artifacts")
        curve_seeds = 2 if smoke else seeds
        return {
            "mode": mode, "dataset": dataset, "execution": execution,
            "conditions": list(conditions), "smoke": bool(smoke),
            "reportable": not smoke, "splits": splits, "seeds": curve_seeds,
            "training_horizon": CURVE_TRAINING_HORIZON[dataset],
            "method_inventory": ["multigroup"],
            "protocols": ["active", "passive"], "beta": "inf",
            "learner_tau": 0.5, "evaluation_tau": 0.5,
            "groupwise_reconstructions": splits * curve_seeds * 2,
            "scheduled_fits": {"glhf": splits * curve_seeds * 2, "nlhf": 0},
            "scheduled_beta_glhf_fits": 0, "plan": {"beta_glhf_fits": 0},
            "pooled_fits": 0, "nlhf_fits": 0,
            "stages": ["validate_inputs", "validate_beta_source", "reconstruct", "evaluate", "reduce", "summary"],
            "smoke_note": "Smoke numbers are not results." if smoke else None,
        }

    inventory = dry_run_inventory(suite, policy=execution, n_splits=splits, n_seeds=seeds)
    cells = [c for c in inventory["plan"]["cells"] if c["condition"] in conditions]
    if mode == "protocol-beta":
        cells = [c for c in cells if c["method"] != "pooled_nlhf"]
    scheduled = {"glhf": 0, "nlhf": 0}
    units = splits * seeds
    for cell in cells:
        if cell["provenance"] != "fit":
            continue
        scheduled["nlhf" if cell["method"] == "pooled_nlhf" else "glhf"] += units

    plan: dict[str, Any] = {
        **inventory,
        "mode": mode,
        "unsupported_execution": bool(unsupported_execution_reason(mode, execution)),
        "execution_note": unsupported_execution_reason(mode, execution),
        "conditions": list(conditions),
        "smoke": bool(smoke),
        "splits": splits,
        "seeds": seeds,
        "stages": list(SUITE_STAGES) if mode == "comparison-suite" else [
            s for s in SUITE_STAGES if s != "import_reuse" or "beta" in conditions
        ],
        "cells_in_scope": cells,
        "scheduled_fits": scheduled,
        "scheduled_beta_glhf_fits": sum(
            units for c in cells
            if c["provenance"] == "fit" and c["condition"] == "beta"
            and c["method"] != "pooled_nlhf"
        ),
        "excluded_experiments": list(EXCLUDED_EXPERIMENTS),
        "excluded_note": (
            "The comparison-suite command schedules none of these; it does not "
            "depend on the historical expanded experiment having been run."
        ),
    }
    if mode == "protocol-beta":
        plan["method_inventory"] = ["multigroup", "pooled", "uniform"]
    if smoke:
        plan["reportable"] = False
        plan["smoke_note"] = "Smoke numbers are not results."
    if with_learning_curves:
        if mode != "comparison-suite":
            raise ExecutionPolicyError("learning curves can only extend comparison-suite")
        plan["optional_learning_curves"] = {
            "after_suite": True, "training_horizon": CURVE_TRAINING_HORIZON[dataset],
            "method": "multigroup",
            "protocols": ["active", "passive"], "beta": "inf", "tau": 0.5,
            "groupwise_reconstructions": (8 if smoke else splits * seeds * 2),
            "pooled_fits": 0, "nlhf_fits": 0,
            "reportable": not smoke,
        }

    # The reuse gate: only the beta condition needs imported artifacts.
    if execution == "reuse-required" and "beta" in conditions and beta_source is None:
        from empirical_testing.experiments.suite import check_reuse_available

        status = check_reuse_available(project_root, suite)
        plan["reuse"] = status
        if not status["available"]:
            plan["blocked"] = True
            plan["blockers"] = status["blockers"]
    return plan


def run_mode(dataset: str, args: Any, *, project_root: Path,
                 log: Callable[[str], None] = print) -> int:
    """Dispatch one suite mode.  Returns a process exit code."""

    mode = args.mode
    execution = getattr(args, "execution", "reuse-required")
    smoke = bool(getattr(args, "smoke", False))
    suite = SUITES[dataset]

    reason = unsupported_execution_reason(mode, execution)
    if reason and not getattr(args, "dry_run", False):
        raise ExecutionPolicyError(reason)

    plan = build_run_plan(dataset, mode, project_root=project_root,
                          execution=execution, smoke=smoke,
                          with_learning_curves=bool(getattr(args, "with_learning_curves", False)),
                          beta_source=getattr(args, "beta_source", None))

    log(f"{dataset.title()}: mode {mode}, execution {execution}"
        + (", SMOKE (not reportable)" if smoke else ""))
    log(f"  conditions       : {', '.join(plan['conditions'])}")
    log(f"  methods          : {', '.join(plan['method_inventory'])}")
    log(f"  splits x seeds   : {plan['splits']} x {plan['seeds']} = {plan['splits'] * plan['seeds']}")
    if mode in ("protocol-beta", "panel-permutations"):
        if getattr(args, "dry_run", False):
            _emit(args, plan)
            return 1 if plan.get("blocked") else 0
        from empirical_testing.experiments.fresh import run_beta, run_panels
        output = Path(getattr(args, "output", None) or project_root / f"results/{dataset}-{mode}-rerun")
        kwargs = dict(project_root=project_root, output_dir=output, smoke=smoke,
                      workers=int(getattr(args, "workers", 1) or 1))
        if mode == "protocol-beta":
            return run_beta(dataset, **kwargs).exit_code
        run_panels(**kwargs)
        return 0
    if mode == "learning-curves":
        log(f"  learning curves  : full {plan['training_horizon']}-comparison horizon, "
            "active/passive MULTIGROUP, beta=inf, tau=0.5")
        log(f"  reconstructions  : {plan['groupwise_reconstructions']:,} groupwise; 0 pooled; 0 NLHF")
        if getattr(args, "dry_run", False):
            log("  dry run: no work performed")
            _emit(args, plan)
            return 0
        from empirical_testing.experiments.learning_curves import (
            run_learning_curves,
        )
        output = Path(getattr(args, "output", None) or
                      Path(project_root) / "results" / dataset / "learning-curves-rerun")
        result = run_learning_curves(
            dataset, project_root=Path(project_root), output_dir=output,
            imported_source=getattr(args, "curve_source", None),
            config_path=_base_config_path(project_root, dataset),
            workers=int(getattr(args, "workers", 1) or 1), smoke=smoke,
        )
        return result.exit_code
    log(f"  scheduled fits   : GLHF {plan['scheduled_fits']['glhf']:,}, "
        f"NLHF {plan['scheduled_fits']['nlhf']:,}")
    log(f"  beta GLHF fits   : {plan['scheduled_beta_glhf_fits']:,}")
    log(f"  NLHF baseline    : {plan['nlhf_baseline']}")
    if plan.get("optional_learning_curves"):
        log(f"  optional curves  : {plan['optional_learning_curves']['groupwise_reconstructions']:,} "
            "groupwise reconstructions after suite; 0 pooled; 0 NLHF")

    if plan.get("blocked"):
        log("  BLOCKED:")
        for blocker in plan["blockers"]:
            log(f"    - {blocker}")
        if getattr(args, "dry_run", False):
            _emit(args, plan)
            return 1
        raise ReuseBlockedError(
            f"{dataset.title()}: " + "; ".join(str(b) for b in plan["blockers"])
        )

    if getattr(args, "dry_run", False):
        log("  dry run: no work performed")
        _emit(args, plan)
        return 0

    from empirical_testing.experiments.run import run_suite

    config_path = _base_config_path(project_root, dataset)
    output = Path(getattr(args, "output", None) or
                  Path(project_root) / f"results/{dataset}-comparison-rerun")
    imported = (output / "beta-source" if execution == "fresh" else
                getattr(args, "beta_source", None) or resolve_beta_source(project_root, suite))
    result = run_suite(
        dataset, project_root=Path(project_root), output_dir=output, suite=suite,
        config_path=config_path, imported_source=imported,
        workers=int(getattr(args, "workers", 1) or 1), smoke=smoke,
        # Smoke reduces the number of *splits*, never the number of learner
        # seeds: the seed axis is part of the identity the imported beta
        # results are joined on, so a reduced-seed run could not be combined
        # with them at all.  A short split axis still exercises every stage.
        learner_seed_limit=None,
        split_limit=SMOKE_SPLITS if smoke else None,
        log=log,
        execution=execution,
        supplied_beta_source=getattr(args, "beta_source", None) is not None,
    )
    if result.exit_code != 0 or not getattr(args, "with_learning_curves", False):
        return result.exit_code
    from empirical_testing.experiments.learning_curves import run_learning_curves
    curve_result = run_learning_curves(
        dataset, project_root=Path(project_root), output_dir=output / "learning_curves",
        imported_source=imported, config_path=config_path,
        workers=int(getattr(args, "workers", 1) or 1), smoke=smoke,
    )
    return curve_result.exit_code


#: The frozen design each dataset's suite analysis sits on.  Recorded here rather
#: than guessed at the call site, so the two datasets cannot be crossed.
BASE_CONFIGS: Mapping[str, str] = {
    "coffee": "configs/coffee.json",
    "persona": "configs/persona.json",
}


def _base_config_path(project_root: Path, dataset: str) -> Path:
    try:
        return Path(project_root) / BASE_CONFIGS[dataset]
    except KeyError:
        raise ExecutionPolicyError(
            f"unknown dataset {dataset!r}; expected one of {sorted(BASE_CONFIGS)}"
        ) from None


def _emit(args: Any, plan: dict[str, Any]) -> None:
    output = getattr(args, "output", None)
    if output is None:
        print(json.dumps(plan, indent=2, default=str))
        return
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    (path / "plan.json").write_text(
        json.dumps(plan, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"plan written to {path / 'plan.json'}")
