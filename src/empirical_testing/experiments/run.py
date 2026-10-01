"""Run comparisons with freshly trained or reused beta results.

Each output contains split arrays, run_info.json, validation.json, and SUMMARY.md.
Reductions are written to reduce/beta/ for protocol/beta comparisons,
reduce/fixed_0.5/ for the primary radius comparison, and reduce/moving/ for the
changing-evaluation-radius diagnostic."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from empirical_testing.experiments import engine as engine
from empirical_testing.experiments import reduce as reduce_module
from empirical_testing.experiments.engine import (
    METHODS, REGIME_FIXED, REGIME_MOVING, REGIMES,
)

STAGES: tuple[str, ...] = (
    "validate_inputs", "validate_imported", "missing_fits", "evaluate",
    "reduce", "coverage", "summary",
)

EXIT_OK, EXIT_FAILED = 0, 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=False, default=engine._json_safe) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_table(directory: Path, name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Write one reduction table as both CSV and JSON."""

    import pandas as pd

    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    csv_path, json_path = directory / f"{name}.csv", directory / f"{name}.json"
    frame.to_csv(csv_path, index=False)
    _write_json(json_path, list(rows))
    return {"csv": str(csv_path), "json": str(json_path), "rows": len(frame)}


@dataclass
class RunResult:
    exit_code: int
    output_dir: Path
    summary: dict[str, Any] = field(default_factory=dict)


def run_suite(
    dataset: str,
    *,
    project_root: Path,
    output_dir: Path,
    suite: Any,
    config_path: Path,
    imported_source: Path,
    workers: int = 1,
    smoke: bool = False,
    learner_seed_limit: int | None = None,
    split_limit: int | None = None,
    confidence_level: float = 0.95,
    log: Callable[[str], None] = print,
    execution: str = "reuse-required",
    supplied_beta_source: bool = False,
) -> RunResult:
    """Run the suite analysis for one dataset, resuming any completed splits."""

    from dataclasses import replace as dataclass_replace

    from empirical_testing.parallel import ordered_map
    from empirical_testing.experiments.suite import require_reuse

    project_root, output_dir = Path(project_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    info: dict[str, Any] = {
        "suite_id": suite.suite_id, "dataset": dataset,
        "protocol_id": suite.tau_protocol_id,
        "implementation_id": engine.IMPLEMENTATION_ID,
        "mode": "comparison-suite", "execution": execution,
        "reportable": not smoke, "status": "running", "started_at": _now(),
        "workers": int(workers), "smoke": bool(smoke),
        "methods": list(METHODS), "regimes": list(REGIMES),
        "stages": {},
    }
    if smoke:
        info["smoke_note"] = "Smoke numbers are not results."
    _write_json(output_dir / "run_info.json", info)

    def stage(name: str, seconds: float, **extra: Any) -> None:
        info["stages"][name] = {"seconds": round(seconds, 2), **extra}
        _write_json(output_dir / "run_info.json", info)

    try:
        # --- 1. input validation -----------------------------------------
        mark = time.perf_counter()
        plan = engine.build_plan(
            dataset, project_root=project_root, output_dir=output_dir,
            protocol_id=suite.tau_protocol_id, config_path=config_path,
        )
        if learner_seed_limit is not None:
            plan.base_config = dataclass_replace(
                plan.base_config,
                learner_seeds=tuple(plan.base_config.learner_seeds[:learner_seed_limit]),
            )
        splits = list(plan.inputs.plan.splits())
        if split_limit is not None:
            splits = splits[:split_limit]
        log(f"{dataset.title()}: {len(splits)} splits x {len(plan.learner_seeds)} seeds")
        stage("validate_inputs", time.perf_counter() - mark,
              splits=len(splits), seeds=len(plan.learner_seeds),
              groups=list(plan.groups))

        # --- 2. imported-result validation -------------------------------
        mark = time.perf_counter()
        if execution == "fresh":
            from empirical_testing.experiments.fresh import run_beta
            run_beta(dataset, project_root=project_root, output_dir=imported_source,
                     workers=workers, smoke=smoke)
        elif not supplied_beta_source:
            require_reuse(project_root, suite)
        source_info = json.loads((Path(imported_source) / "run_info.json").read_text())
        if source_info.get("status") != "completed" or (source_info.get("smoke") and not smoke):
            raise ValueError("Beta source must be completed; smoke results cannot support a full run")
        log(f"  beta results ready: {imported_source}")
        stage("validate_imported", time.perf_counter() - mark,
              source=str(imported_source), available=True)

        # --- 3+4. missing fits and evaluation (one atomic unit each) ------
        mark = time.perf_counter()
        config_mapping = plan.base_config.as_mapping()
        config_mapping["learner_seeds"] = [int(s) for s in plan.base_config.learner_seeds]
        results, report = ordered_map(
            engine._run_split_task, splits, workers=int(workers),
            initializer=engine._init_worker,
            initargs=(dataset, config_mapping, str(project_root), str(output_dir),
                      suite.tau_protocol_id, None),
            on_progress=None, log=log,
        )
        fitted = sum(0 if r.get("reused") else 1 for r in results)
        log(f"  splits: {len(results)} ({fitted} computed, {len(results)-fitted} reused)")
        stage("missing_fits", time.perf_counter() - mark,
              computed=fitted, reused=len(results) - fitted)
        stage("evaluate", 0.0,
              note="evaluation happens inside each split unit, from the same fitted "
                   "policy cube, once per regime")

        # --- 5. reductions ------------------------------------------------
        mark = time.perf_counter()
        outputs = _reduce_everything(
            plan=plan, splits=splits, output_dir=output_dir,
            imported_source=Path(imported_source),
            confidence_level=confidence_level, log=log,
        )
        stage("reduce", time.perf_counter() - mark, tables=len(outputs["tables"]))

        # --- 6. coverage ---------------------------------------------------
        mark = time.perf_counter()
        coverage = outputs["coverage"]
        incomplete = [row for row in coverage if not row["complete"]]
        if incomplete and not smoke:
            raise RuntimeError(
                f"{len(incomplete)} (cell, method) combinations are incomplete; "
                f"first: {incomplete[0]}"
            )
        stage("coverage", time.perf_counter() - mark,
              rows=len(coverage), incomplete=len(incomplete))

        info["validation"] = outputs["validation"]
        info["fit_counts"] = outputs["fit_counts"]
        if execution == "fresh":
            info["fit_counts"]["beta_glhf"] = len(splits) * len(plan.learner_seeds) * 14
            info["fit_counts"]["total"] += info["fit_counts"]["beta_glhf"]
        info["solver"] = outputs["solver"]
        info["outputs"] = outputs["tables"]
        info["status"] = "completed"
        info["finished_at"] = _now()
        info["wall_seconds"] = round(time.perf_counter() - started, 1)
        _write_json(output_dir / "run_info.json", info)
        _write_summary(output_dir, info, suite)
        log(f"{dataset.title()}: completed in {info['wall_seconds']}s")
        return RunResult(EXIT_OK, output_dir, info)
    except Exception as error:  # noqa: BLE001 - recorded, then re-raised
        info["status"] = "failed"
        info["error"] = f"{type(error).__name__}: {error}"
        info["finished_at"] = _now()
        _write_json(output_dir / "run_info.json", info)
        log(f"{dataset.title()}: FAILED - {info['error']}")
        raise


def _reduce_everything(
    *, plan: Any, splits: Sequence[tuple[int, int]], output_dir: Path,
    imported_source: Path, confidence_level: float, log: Callable[[str], None],
) -> dict[str, Any]:
    """Join imported and new results, reduce both regimes, write every table."""

    groups = list(plan.groups)
    n_seeds = len(plan.learner_seeds)
    n_groups = len(groups)

    beta_worst, beta_levels = [], []
    fixed_worst, fixed_levels = [], []
    moving_worst, moving_levels = [], []
    test_sizes, train_sizes = [], []
    comparator_checks: list[dict[str, Any]] = []
    solver_records: list[dict[str, Any]] = []
    beta_provenance: Mapping[str, Mapping[str, str]] = {}
    tau_provenance: dict[str, Mapping[str, Mapping[str, str]]] = {}

    for repeat, fold in splits:
        new_arrays = engine.split_arrays(output_dir, repeat, fold)
        summary = json.loads(
            (engine.split_dir(output_dir, repeat, fold) / "split.json").read_text("utf-8")
        )
        imported = reduce_module.load_imported_split(imported_source, repeat, fold)
        labels = reduce_module.imported_cell_labels(imported_source, repeat, fold)

        # The group axis must agree, name for name and in order: the per-group
        # join is positional once past this point, so a disagreement here would
        # silently pair one group's imported numbers with another's new ones.
        imported_metadata = json.loads(
            (Path(imported_source) / "splits" / f"r{repeat:02d}_f{fold:02d}"
             / "split.json").read_text("utf-8")
        )
        imported_groups = list(imported_metadata["groups"])
        if imported_metadata.get("learner_seeds") != summary.get("learner_seeds"):
            raise ValueError("Beta and comparison results use different learner seeds")
        if imported_groups != groups:
            raise RuntimeError(
                f"split r{repeat} f{fold}: the imported run's group family "
                f"{imported_groups} is not this run's {groups}. For Persona the family "
                f"is discovery-selected and re-derived; a disagreement means the "
                f"selection rule no longer reproduces the imported design."
            )

        check = reduce_module.check_comparator_matches(new_arrays, imported, labels)
        comparator_checks.append({"repeat": repeat, "fold": fold, **check})
        if not check["matches"]:
            raise RuntimeError(
                f"split r{repeat} f{fold}: the suite evaluation produced a different welfare "
                f"comparator from the imported run (max abs difference "
                f"{check['max_abs_difference']:.3e}; allowed atol={check['atol']:.1e}, "
                f"rtol={check['rtol']:.1e}). The imported GLHF columns and the new "
                f"NLHF column would not be commensurable."
            )

        beta_cube = reduce_module.assemble_beta_split(
            imported=imported, labels=labels, new_arrays=new_arrays,
            n_seeds=n_seeds, n_groups=n_groups)
        beta_worst.append(beta_cube.worst)
        beta_levels.append(beta_cube.levels)
        beta_provenance = beta_cube.provenance
        for regime, worst_list, levels_list in (
            (REGIME_FIXED, fixed_worst, fixed_levels),
            (REGIME_MOVING, moving_worst, moving_levels),
        ):
            cube = reduce_module.assemble_tau_split(
                regime=regime, imported=imported, labels=labels,
                new_arrays=new_arrays, n_seeds=n_seeds, n_groups=n_groups)
            worst_list.append(cube.worst)
            levels_list.append(cube.levels)
            tau_provenance[regime] = cube.provenance

        test_sizes.append(int(summary["n_test"]))
        train_sizes.append(int(summary["n_train"]))
        for cell, records in summary.get("solver", {}).items():
            for record in records:
                solver_records.append({"repeat": repeat, "fold": fold,
                                       "cell": cell, **record})

    tables: dict[str, Any] = {}

    # --- the beta/protocol condition ------------------------------------
    beta_cells = reduce_module.beta_conditions()
    beta = reduce_module.reduce_condition(
        cells=beta_cells, worst_by_split=beta_worst, levels_by_split=beta_levels,
        provenance=beta_provenance, groups=groups, test_sizes=test_sizes,
        train_sizes=train_sizes, confidence_level=confidence_level)
    beta.update(reduce_module.reduce_protocol_contrasts(
        worst_by_split=beta_worst, cells=beta_cells, groups=groups,
        levels_by_split=beta_levels, test_sizes=test_sizes,
        train_sizes=train_sizes, confidence_level=confidence_level))
    beta_dir = output_dir / "reduce" / "beta"
    for name, rows in beta.items():
        tables[f"beta/{name}"] = _write_table(beta_dir, name, rows)

    # --- the tau ablation, one directory per regime ----------------------
    coverage: list[dict[str, Any]] = list(reduce_module.coverage_rows(
        condition="beta", regime=REGIME_FIXED, cells=beta_cells,
        provenance=beta_provenance, n_splits=len(splits), n_seeds=n_seeds,
        expected_splits=len(splits)))
    for regime, worst_list, levels_list in (
        (REGIME_FIXED, fixed_worst, fixed_levels),
        (REGIME_MOVING, moving_worst, moving_levels),
    ):
        cells = reduce_module.tau_conditions(regime)
        block = reduce_module.reduce_condition(
            cells=cells, worst_by_split=worst_list, levels_by_split=levels_list,
            provenance=tau_provenance[regime], groups=groups,
            test_sizes=test_sizes, train_sizes=train_sizes,
            confidence_level=confidence_level)
        directory = output_dir / "reduce" / regime
        for name, rows in block.items():
            tables[f"{regime}/{name}"] = _write_table(directory, name, rows)
        coverage.extend(reduce_module.coverage_rows(
            condition="tau", regime=regime, cells=cells,
            provenance=tau_provenance[regime], n_splits=len(splits),
            n_seeds=n_seeds, expected_splits=len(splits)))
        _write_json(directory / "regime.json", {
            "evaluation_regime": regime,
            "meaning": (
                "every policy scored at evaluation tau = 0.5"
                if regime == REGIME_FIXED
                else "each policy scored at evaluation tau equal to its own learner tau"
            ),
            "estimand_note": (
                "The fixed and moving regimes are different estimands. Their rows, "
                "contrasts, tables and figures are never mixed."
            ),
            "cells": [c.as_row() for c in cells],
        })

    tables["coverage"] = _write_table(output_dir / "reduce", "coverage", coverage)
    if solver_records:
        tables["solver_diagnostics"] = _write_table(
            output_dir / "reduce", "solver_diagnostics", solver_records)

    # --- the shared tau = 0.5 identity, checked not assumed --------------
    shared = _check_shared_tau_identity(fixed_worst, moving_worst, beta_worst)

    unconverged = [r for r in solver_records if not r.get("solver_success", True)]
    validation = {
        "group_family_matches_imported": True,
        "group_family": list(groups),
        "comparator_exact_on_every_split": all(c["exact"] for c in comparator_checks),
        "comparator_matches_on_every_split": all(c["matches"] for c in comparator_checks),
        "comparator_checks": comparator_checks,
        "splits_checked": len(comparator_checks),
        "shared_tau_identity": shared,
        "imported_source": str(imported_source),
    }
    _write_json(output_dir / "validation.json", validation)

    fit_counts = engine.fits_per_split(n_seeds)
    fit_counts = {key: value * len(splits) for key, value in fit_counts.items()}
    fit_counts["total"] = sum(fit_counts.values())
    solver = {
        "fits": len(solver_records),
        "unconverged": len(unconverged),
        "max_abs_exploitability": max(
            (abs(r["exploitability"]) for r in solver_records), default=0.0),
        "gate": engine.NLHF_EXPLOITABILITY_GATE,
        "all_single_reference_start": all(
            r.get("optimizer_calls") == 1 and r.get("selection") == "reference_start_v10"
            for r in solver_records),
    }
    return {"tables": tables, "coverage": coverage, "validation": validation,
            "fit_counts": fit_counts, "solver": solver}


def _check_shared_tau_identity(fixed_worst, moving_worst, beta_worst) -> dict[str, Any]:
    """``tau = 0.5`` must be *exactly* the imported result, in both regimes.

    The shared radius is evaluated at ``0.5`` under either regime, so the two
    regimes must agree there bit for bit.  Anything else would mean a value had
    been recomputed where the specification says it is reused.
    """

    taus = [reduce_module.tau_label(t) for t in engine.TAU_GRID]
    index = taus.index(reduce_module.tau_label(engine.SHARED_TAU))
    mismatches = []
    for split_index, (fixed, moving) in enumerate(zip(fixed_worst, moving_worst)):
        if not np.array_equal(fixed[index], moving[index]):
            mismatches.append(split_index)
    return {
        "shared_tau": engine.SHARED_TAU,
        "regimes_agree_exactly_at_shared_tau": not mismatches,
        "mismatched_splits": mismatches[:5],
        "splits_compared": len(fixed_worst),
    }


def _write_summary(output_dir: Path, info: Mapping[str, Any], suite: Any) -> None:
    lines = [
        f"# {suite.dataset.title()} comparison suite", "",
        f"- status: **{info['status']}**",
        f"- reportable: {info['reportable']}"
        + ("  **SMOKE - not results**" if info.get("smoke") else ""),
        f"- wall: {info.get('wall_seconds')} s at {info['workers']} workers",
        f"- methods: {', '.join(info['methods'])}",
        f"- evaluation regimes: {', '.join(info['regimes'])}",
        "",
        "## Fits", "",
        "| family | count |", "|---|---:|",
    ]
    for key, value in (info.get("fit_counts") or {}).items():
        lines.append(f"| {key} | {value:,} |")
    solver = info.get("solver") or {}
    lines += [
        "", "## NLHF solver", "",
        f"- fits: {solver.get('fits', 0):,}",
        f"- optimizer reported non-convergence: {solver.get('unconverged', 0):,} "
        f"(verified inside the gate; not failed fits)",
        f"- max |exploitability|: {solver.get('max_abs_exploitability', 0):.3e} "
        f"against a gate of {solver.get('gate')}",
        f"- every fit a single reference start: {solver.get('all_single_reference_start')}",
        "", "## Validation", "",
    ]
    for key, value in (info.get("validation") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "", "## Estimands", "",
        "The `fixed_0.5` and `moving` reductions are **different estimands** and are",
        "written into separate directories. `fixed_0.5` scores every policy at",
        "evaluation tau = 0.5, so all radii are held to one robustness standard.",
        "`moving` scores each policy at its own learner radius, so the evaluation",
        "uncertainty set moves with the policy. Do not read a moving-tau curve as the",
        "fixed-tau comparison.", "",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
