"""Small raw-input drivers for the shared beta grid and Persona panel check."""

from dataclasses import replace
import json
from pathlib import Path

from empirical_testing.experiments import engine
from empirical_testing.experiments import coffee_protocol_beta_sweep as beta
from empirical_testing.experiments.run import RunResult, _write_json
from empirical_testing.parallel import ordered_map

_STATE = {}
PANEL_SEEDS = (20260916, 20260917, 20260918, 20260919, 20260920)


def _init_worker(dataset, project_root, output_dir, config, kind, smoke):
    plan = engine.build_plan(dataset, project_root=Path(project_root),
                             output_dir=Path(output_dir), protocol_id=f"{dataset}-{kind}",
                             config_mapping=config)
    if kind == "panel" and smoke:
        plan.base_config = replace(plan.base_config, learner_seeds=plan.learner_seeds[:2])
    _STATE.update(plan=plan, kind=kind)


def _run_split(task):
    plan = _STATE["plan"]
    repeat, fold = task
    if _STATE["kind"] == "panel":
        from empirical_testing.experiments.coffee_crossfit import run_split
        return run_split(plan.base_config, plan.inputs, plan.output_dir, repeat, fold)
    train, test, _, _ = engine._views(plan, repeat, fold)
    base = plan.base_config
    summary, arrays, logs = beta.compute_split_cells(
        algorithm=base.algorithm, cv_seed=base.cv_seed, learner_seeds=base.learner_seeds,
        comparisons_per_respondent=base.comparisons_per_respondent, tau=base.tau,
        groups=base.groups, cells=beta.default_cells(), train_view=train, test_view=test,
        repeat=repeat, fold=fold)
    identity = engine.expected_identity(plan, repeat, fold)
    for key in ("data_sha256", "train_ids_sha256", "test_ids_sha256", "discovery_ids_sha256"):
        summary[key] = identity[key]
    summary.update(n_train=len(train.utilities), n_test=len(test.utilities))
    beta.publish_split(beta.split_dir(plan.output_dir, repeat, fold), summary, arrays, logs)
    return summary


def _run(dataset, project_root, output_dir, *, kind, smoke, workers, panel_seed=None):
    project_root, output_dir = Path(project_root).resolve(), Path(output_dir).resolve()
    config = json.loads((project_root / f"configs/{dataset}.json").read_text())
    for key in ("data_path", "study_path"):
        if key in config:
            config[key] = str((project_root / config[key]).resolve())
    if panel_seed is not None:
        config["panel_seed"] = panel_seed
    _init_worker(dataset, str(project_root), str(output_dir), config, kind, smoke)
    plan = _STATE["plan"]
    tasks = list(plan.inputs.plan.splits())[:2 if smoke else None]
    info = dict(dataset=dataset, kind=kind, status="running", smoke=smoke,
                reportable=False, n_splits=len(tasks), n_seeds=len(plan.learner_seeds),
                groups=list(plan.groups), config=plan.base_config.as_mapping())
    if panel_seed is not None:
        info["panel_seed"] = panel_seed
    if kind == "beta":
        info["cells"] = [cell.as_mapping() for cell in beta.default_cells()]
    _write_json(output_dir / "run_info.json", info)
    try:
        records, _ = ordered_map(
            _run_split, tasks, workers=workers, initializer=_init_worker,
            initargs=(dataset, str(project_root), str(output_dir), config, kind, smoke),
            on_progress=lambda done, total, task, seconds: print(
                f"{dataset} {kind}: {done}/{total} splits", flush=True), log=print)
        # Smoke uses the first two original folds, with their actual train/test
        # sizes. Only the reducer's list of expected units is shortened.
        repeats, folds = (1, 2) if smoke else (plan.base_config.repeats, plan.base_config.folds)
        if kind == "beta":
            inference = beta.reduce_cells(
                output_dir=output_dir, cells=beta.default_cells(), groups=plan.groups,
                learner_seeds=plan.learner_seeds, repeats=repeats, folds=folds,
                confidence_level=plan.base_config.confidence_level, split_summaries=records)
        else:
            from empirical_testing.experiments.coffee_crossfit import reduce_splits
            inference = reduce_splits(replace(plan.base_config, repeats=repeats, folds=folds),
                                      output_dir, records)
        _write_json(output_dir / "reduce/inference.json", inference)
        info.update(status="completed", reportable=not smoke)
        _write_json(output_dir / "run_info.json", info)
        return RunResult(0, output_dir, info)
    except Exception as error:
        info.update(status="failed", error=str(error))
        _write_json(output_dir / "run_info.json", info)
        raise


def run_beta(dataset, *, project_root, output_dir, workers=1, smoke=False):
    """Train all seven GLHF protocol/beta cells, with no saved-run dependency."""
    return _run(dataset, project_root, output_dir, kind="beta", workers=workers, smoke=smoke)


def run_panels(*, project_root, output_dir, workers=1, smoke=False):
    """Repeat panel construction, group selection, fitting and inference five times."""
    import pandas as pd

    output_dir = Path(output_dir)
    rows = []
    for index, seed in enumerate(PANEL_SEEDS):
        destination = output_dir / ("base" if index == 0 else f"seed-{seed}")
        result = _run("persona", project_root, destination, kind="panel", panel_seed=seed,
                      workers=workers, smoke=smoke)
        inference = json.loads((destination / "reduce/inference.json").read_text())
        rows.append(dict(panel_seed=seed, groups=", ".join(result.summary["groups"]),
                         **{key: inference[key] for key in
                            ("estimate", "corrected_se", "ci_lower", "ci_upper")}))
    pd.DataFrame(rows).to_csv(output_dir / "panel_summary.csv", index=False)
    _write_json(output_dir / "run_info.json", dict(status="completed", smoke=smoke,
                reportable=not smoke, panel_seeds=list(PANEL_SEEDS)))
