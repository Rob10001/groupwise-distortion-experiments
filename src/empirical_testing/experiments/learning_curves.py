"""Reconstruct the two groupwise, beta=infinity, tau=0.5 trajectories.

These are prefixes of the original full-horizon runs, not independently tuned
short-budget fits. Every checkpoint averages all pre-round policy maps so far.
The frozen test fold and its welfare comparator stay fixed along each curve.
No pooled/NLHF fits, beta sweeps, or radius sweeps are scheduled here.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from empirical_testing.experiments import engine as engine
from empirical_testing.experiments.reduce import corrected
from empirical_testing.experiments.run import RunResult, _write_json, _write_table
from empirical_testing.experiments.trajectory import (
    checkpoint_rounds,
    predict_prefixes,
)

IMPLEMENTATION_ID = "v10-learning-curves-2026-09-21.1"
PROTOCOLS = ("passive", "active")
FRACTIONS = (
    0.0,
    0.01,
    0.025,
    0.05,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
)
# Cross-platform reconstructed endpoints must agree well below plot precision.
ENDPOINT_ATOL = 1e-9
ENDPOINT_RTOL = 1e-8
_STATE: dict[str, Any] = {}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_files(source_dir: Path, meta: dict) -> tuple[str, ...]:
    """Only a declared submission omission permits reconstruction without logs."""
    names = ("split.json", "split.npz")
    if (source_dir / "comparisons.npz").is_file():
        return (*names, "comparisons.npz")
    if "comparisons.npz" not in meta.get("submission_omitted_files", []):
        raise FileNotFoundError(f"Missing undeclared training log: {source_dir / 'comparisons.npz'}")
    return names


def resolve_source(
    project_root: Path, dataset: str, imported_source: Path | None, *, smoke: bool = False
) -> Path:
    if imported_source is not None:
        source = Path(imported_source).resolve()
    else:
        from empirical_testing.experiments.suite import SUITES, resolve_beta_source

        source = resolve_beta_source(project_root, SUITES[dataset])
    if not (source / "run_info.json").is_file():
        raise FileNotFoundError(
            f"Completed beta/protocol artifacts required: {source}; use --curve-source"
        )
    info = json.loads((source / "run_info.json").read_text(encoding="utf-8"))
    if not (smoke and info.get("smoke") and info.get("status") == "completed") and (
        not info.get("reportable") or info.get("smoke")
    ):
        raise ValueError(
            "Learning-curve reconstruction requires reportable source artifacts"
        )
    return source


def _init_worker(
    dataset, project_root, output_dir, source, config_mapping, smoke, code_digest
):
    plan = engine.build_plan(
        dataset,
        project_root=Path(project_root),
        output_dir=Path(output_dir),
        protocol_id=IMPLEMENTATION_ID,
        config_mapping=config_mapping,
    )
    _STATE.clear()
    _STATE.update(plan=plan, source=Path(source), smoke=smoke, code_digest=code_digest)


def _run_split(task):
    from empirical_testing.algorithms.registry import fit_online_learner
    from empirical_testing.experiments.coffee_crossfit import (
        make_uniform_passive_log,
        split_learner_seed,
    )

    repeat, fold = task
    plan, source = _STATE["plan"], _STATE["source"]
    source_dir = source / "splits" / f"r{repeat:02d}_f{fold:02d}"
    meta = json.loads((source_dir / "split.json").read_text(encoding="utf-8"))
    identity = engine.expected_identity(plan, repeat, fold)
    for key in (
        "data_sha256",
        "train_ids_sha256",
        "test_ids_sha256",
        "discovery_ids_sha256",
        "groups",
        "learner_seeds",
    ):
        if meta[key] != identity[key]:
            raise ValueError(f"Frozen source mismatch in {source_dir.name}: {key}")
    if float(meta["learner_tau"]) != 0.5 or float(meta["evaluation_tau"]) != 0.5:
        raise ValueError("Only the frozen tau_L=tau_E=0.5 anchor is supported")
    seeds = list(plan.learner_seeds[:2] if _STATE["smoke"] else plan.learner_seeds)
    source_files = _source_files(source_dir, meta)
    verify_comparisons = "comparisons.npz" in source_files
    expected = {
        "implementation_id": IMPLEMENTATION_ID,
        "code_digest": _STATE["code_digest"],
        "source_sha256": {
            name: _sha(source_dir / name)
            for name in source_files
        },
        "fractions": list(FRACTIONS),
        "learner_seeds": seeds,
        "config_sha256": plan.base_config.digest(),
        "smoke": _STATE["smoke"],
    }
    destination = plan.output_dir / "splits" / source_dir.name
    if (destination / "split.json").is_file() and (destination / "split.npz").is_file():
        cached = json.loads((destination / "split.json").read_text(encoding="utf-8"))
        if cached.get("identity") != expected:
            raise ValueError(
                f"Refusing incompatible cached reconstruction: {destination}"
            )
        if cached.get("arrays_sha256") != _sha(destination / "split.npz"):
            raise ValueError(f"Cached reconstruction checksum failed: {destination}")
        return {**cached, "reused": True}

    started = time.perf_counter()
    train, test, _, _ = engine._views(plan, repeat, fold)
    n_train, n_actions = train.utilities.shape
    checkpoints = checkpoint_rounds(n_train, FRACTIONS)
    if len(checkpoints) != len(FRACTIONS):
        raise ValueError(
            "Checkpoints collide at this horizon; not a frozen production split"
        )
    with np.load(source_dir / "split.npz", allow_pickle=False) as f:
        frozen = dict(f)
    labels = meta["cell_labels"]
    method_index = meta["methods"].index("multigroup")
    optimum = frozen["welfare_optimum"][labels.index("passive@inf")]
    np.testing.assert_array_equal(
        optimum, frozen["welfare_optimum"][labels.index("active@inf")]
    )
    reference = np.full(n_actions, 1.0 / n_actions)
    group_mask = np.asarray(test.groups, dtype=float)
    group_counts = group_mask.sum(axis=0)
    if np.any(group_counts == 0):
        raise ValueError("Empty held-out group")
    welfare = np.empty((len(checkpoints), len(seeds), 2, len(plan.groups)))
    max_endpoint_error = max_policy_error = max_propensity_error = 0.0
    if verify_comparisons:
        with np.load(source_dir / "comparisons.npz", allow_pickle=False) as logs:
            saved_order = logs["respondent_order"]
            saved_draws = {
                key: logs[f"active@inf|multigroup|{key}"]
                for key in ("first", "second", "sign", "propensity")
            }
    for seed_index, seed in enumerate(seeds):
        training_seed = split_learner_seed(plan.base_config.cv_seed, repeat, fold, seed)
        log = make_uniform_passive_log(
            train,
            np.arange(n_train),
            seed=training_seed,
            comparisons_per_respondent=1,
            beta=np.inf,
        )
        order = np.asarray(log.respondent_indices, dtype=np.int64)
        np.testing.assert_array_equal(np.sort(order), np.arange(n_train))
        if verify_comparisons:
            np.testing.assert_array_equal(order, saved_order[seed_index])
        record = meta["seed_records"][seed_index]
        if (
            record["training_seed"] != training_seed
            or record["respondent_order_sha256"]
            != hashlib.sha256(order.tobytes()).hexdigest()
        ):
            raise ValueError("Frozen training seed/order mismatch")
        for protocol_index, protocol in enumerate(PROTOCOLS):
            kwargs = (
                {
                    "protocol": "active",
                    "utilities": train.utilities[order],
                    "beta": np.inf,
                }
                if protocol == "active"
                else {
                    "protocol": "logged",
                    "utilities": None,
                    "pair_indices": log.pairs,
                    "signs": log.signs,
                    "propensities": log.propensities,
                }
            )
            model = fit_online_learner(
                plan.base_config.algorithm,
                groups=train.groups[order],
                reference=reference,
                tau=0.5,
                seed=training_seed,
                **kwargs,
            )
            if protocol == "active" and verify_comparisons:
                drawn = np.asarray(model.drawn_comparisons)
                for j, key in enumerate(("first", "second", "sign")):
                    np.testing.assert_array_equal(
                        drawn[:, j], saved_draws[key][seed_index]
                    )
                difference = float(
                    np.max(np.abs(drawn[:, 3] - saved_draws["propensity"][seed_index]))
                )
                max_propensity_error = max(max_propensity_error, difference)
                np.testing.assert_allclose(
                    drawn[:, 3],
                    saved_draws["propensity"][seed_index],
                    atol=ENDPOINT_ATOL,
                    rtol=ENDPOINT_RTOL,
                )
            predictions = predict_prefixes(model, test.groups, checkpoints)
            final = model.predict(test.groups)
            max_policy_error = max(
                max_policy_error, float(np.max(np.abs(predictions[-1] - final)))
            )
            np.testing.assert_allclose(predictions[-1], final, atol=1e-12, rtol=1e-12)
            values = np.einsum("cna,na->cn", predictions, test.utilities)
            current = values @ group_mask / group_counts
            welfare[:, seed_index, protocol_index] = current
            frozen_index = labels.index(f"{protocol}@inf")
            target_w = frozen["welfare"][frozen_index, seed_index, method_index]
            target_d = frozen["group_distortion"][
                frozen_index, seed_index, method_index
            ]
            for actual, target in (
                (current[-1], target_w),
                (optimum / current[-1], target_d),
            ):
                max_endpoint_error = max(
                    max_endpoint_error, float(np.max(np.abs(actual - target)))
                )
                np.testing.assert_allclose(
                    actual, target, atol=ENDPOINT_ATOL, rtol=ENDPOINT_RTOL
                )
    if not np.isfinite(welfare).all() or np.any(welfare <= 0):
        raise ValueError("Invalid welfare denominator")
    distortion = optimum / welfare
    arrays = {
        "checkpoints": checkpoints,
        "fractions": np.array(FRACTIONS),
        "welfare": welfare,
        "welfare_optimum": optimum,
        "group_distortion": distortion,
        "worst_group_distortion": distortion.max(axis=-1),
        "test_group_counts": group_counts,
    }
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / "split.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(destination / "split.npz")
    summary = {
        "identity": expected,
        "repeat": repeat,
        "fold": fold,
        "n_train": int(n_train),
        "n_test": len(test.utilities),
        "groups": list(plan.groups),
        "protocols": list(PROTOCOLS),
        "reconstructed_fits": len(seeds) * 2,
        "max_endpoint_error": max_endpoint_error,
        "max_policy_error": max_policy_error,
        "max_propensity_error": max_propensity_error if verify_comparisons else None,
        "comparison_logs_verified": verify_comparisons,
        "endpoint_verified": True,
        "seconds": time.perf_counter() - started,
        "arrays_sha256": _sha(destination / "split.npz"),
    }
    _write_json(destination / "split.json", summary)
    return {**summary, "reused": False}


def reduce_curves(output_dir: Path, records: list[dict], dataset: str) -> dict:
    """Max groups per seed, average seeds per split, then corrected inference."""
    worst, groups, checkpoints = [], [], []
    for record in records:
        path = (
            output_dir
            / "splits"
            / f"r{record['repeat']:02d}_f{record['fold']:02d}"
            / "split.npz"
        )
        with np.load(path, allow_pickle=False) as f:
            worst.append(f["worst_group_distortion"].mean(axis=1))
            groups.append(f["group_distortion"].mean(axis=1))
            checkpoints.append(f["checkpoints"])
    worst, groups, checkpoints = (
        np.stack(worst),
        np.stack(groups),
        np.stack(checkpoints),
    )
    n_test, n_train = [r["n_test"] for r in records], [r["n_train"] for r in records]
    rows, group_rows, differences, split_rows = [], [], [], []
    for c, fraction in enumerate(FRACTIONS):
        base = {
            "dataset": dataset,
            "beta": "inf",
            "learner_tau": 0.5,
            "evaluation_tau": 0.5,
            "checkpoint_fraction": fraction,
            "mean_comparisons": float(checkpoints[:, c].mean()),
            "min_comparisons": int(checkpoints[:, c].min()),
            "max_comparisons": int(checkpoints[:, c].max()),
            "n_seeds_per_split": len(records[0]["identity"]["learner_seeds"]),
            "interval": "marginal approximate corrected 95%; not simultaneous",
        }
        for pi, protocol in enumerate(PROTOCOLS):
            rows.append(
                {
                    **base,
                    "protocol": protocol,
                    **corrected(worst[:, c, pi], n_test, n_train, 0.95),
                }
            )
            for gi, group in enumerate(records[0]["groups"]):
                group_rows.append(
                    {
                        **base,
                        "protocol": protocol,
                        "group": group,
                        **corrected(groups[:, c, pi, gi], n_test, n_train, 0.95),
                    }
                )
        differences.append(
            {
                **base,
                "contrast": "passive-minus-active",
                "positive_favors": "active",
                **corrected(worst[:, c, 0] - worst[:, c, 1], n_test, n_train, 0.95),
            }
        )
        for i, record in enumerate(records):
            split_rows.append(
                {
                    "repeat": record["repeat"],
                    "fold": record["fold"],
                    "checkpoint_fraction": fraction,
                    "comparisons": int(checkpoints[i, c]),
                    "passive": float(worst[i, c, 0]),
                    "active": float(worst[i, c, 1]),
                    "passive_minus_active": float(worst[i, c, 0] - worst[i, c, 1]),
                }
            )
    return {
        name: _write_table(output_dir / "reduce", name, table)
        for name, table in (
            ("method_levels", rows),
            ("group_levels", group_rows),
            ("protocol_contrasts", differences),
            ("split_scores", split_rows),
        )
    }


def plot_curves(output_dir: Path, dataset: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.read_csv(output_dir / "reduce/method_levels.csv")
    contrast = pd.read_csv(output_dir / "reduce/protocol_contrasts.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), layout="constrained")
    for protocol, color, style in (
        ("active", "#D9650B", "-"),
        ("passive", "#546477", "--"),
    ):
        d = frame[frame.protocol == protocol]
        axes[0].plot(
            d.mean_comparisons,
            d.estimate,
            color=color,
            ls=style,
            label=protocol.title(),
            lw=1.8,
        )
        axes[0].fill_between(
            d.mean_comparisons,
            d.ci_lower,
            d.ci_upper,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    axes[0].axhline(1, color="#777777", lw=0.8)
    axes[0].set(
        ylabel="Mean worst-group distortion", title=f"{dataset.title()}: multigroup GLHF"
    )
    axes[0].legend(frameon=False)
    axes[1].plot(contrast.mean_comparisons, contrast.estimate, color="#D9650B", lw=1.8)
    axes[1].fill_between(
        contrast.mean_comparisons,
        contrast.ci_lower,
        contrast.ci_upper,
        color="#D9650B",
        alpha=0.15,
        linewidth=0,
    )
    axes[1].axhline(0, color="#777777", lw=0.8)
    axes[1].set(
        ylabel="Passive minus active distortion", title="Paired protocol difference"
    )
    for ax in axes:
        ax.set_xlabel("Training comparisons (one per respondent)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle(
        r"$\beta=\infty$, $\tau_L=\tau_E=0.5$; prefix-averaged policies"
        + f"\n{int(frame.n_splits.iloc[0])} CV splits, {int(frame.n_seeds_per_split.iloc[0])} seeds; marginal corrected 95% intervals",
        fontsize=9,
    )
    dest = output_dir / "figures"
    dest.mkdir(exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        p = dest / f"{dataset}_learning_curves.{suffix}"
        fig.savefig(p, dpi=200)
        paths.append(str(p))
    plt.close(fig)
    (dest / "CAPTION.md").write_text(
        "Multigroup GLHF under active and passive sampling, beta=infinity and "
        "learner/evaluation KL radii 0.5. Each checkpoint averages the first t "
        "pre-round policy maps of the original full-horizon run; these are not "
        "independently refitted short-budget policies. The same held-out respondents "
        "and fixed-policy welfare benchmarks are used throughout. The horizontal "
        "coordinate is the mean comparison count across matched split fractions "
        "(individual fold sizes differ slightly). Group maxima are taken per seed, "
        "then seeds are averaged within each split. Bands are approximate corrected "
        "95% repeated-CV intervals, marginal at each checkpoint, not simultaneous. "
        "The right panel uses paired passive-minus-active differences; positive "
        "values favor active sampling. The protocol comparison includes the native "
        "score normalization of each sampler.\n",
        encoding="utf-8",
    )
    return paths


def run_learning_curves(
    dataset: str,
    *,
    project_root: Path,
    output_dir: Path,
    imported_source: Path | None = None,
    config_path: Path | None = None,
    workers: int = 1,
    smoke: bool = False,
) -> RunResult:
    from empirical_testing.parallel import ordered_map

    project_root, output_dir = Path(project_root).resolve(), Path(output_dir).resolve()
    source = resolve_source(project_root, dataset, imported_source, smoke=smoke)
    if config_path is None:
        config_path = (
            project_root
            / "configs"
            / (
                "coffee.json"
                if dataset == "coffee"
                else "persona.json"
            )
        )
    config_mapping = json.loads(Path(config_path).read_text(encoding="utf-8"))
    # Persona's adapter consumes study_path directly; resolve it independently
    # of the shell working directory. This changes paths, not the study design.
    for key in ("data_path", "study_path"):
        if config_mapping.get(key):
            path = Path(config_mapping[key])
            config_mapping[key] = str(
                (project_root / path).resolve() if not path.is_absolute() else path
            )
    plan = engine.build_plan(
        dataset,
        project_root=project_root,
        output_dir=output_dir,
        protocol_id=IMPLEMENTATION_ID,
        config_mapping=config_mapping,
    )
    if plan.base_config.comparisons_per_respondent != 1:
        raise ValueError("Reconstruction requires one comparison per respondent")
    code_root = Path(__file__).resolve().parents[1]
    code_paths = [Path(__file__), Path(__file__).with_name("trajectory.py")]
    code_paths += [
        code_root / "algorithms" / f
        for f in ("span_adaptive.py", "one_expert.py", "registry.py")
    ]
    code_paths.append(code_root / "compiled_kernels.py")
    hashes = {p.name: _sha(p) for p in code_paths}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    tasks = list(plan.inputs.plan.splits())[: 2 if smoke else None]
    if not smoke and (len(tasks) != 100 or len(plan.learner_seeds) != 50):
        raise ValueError(
            "Production curves require the frozen 100-split, 50-seed design"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    info = {
        "implementation_id": IMPLEMENTATION_ID,
        "dataset": dataset,
        "status": "running",
        "reportable": False,
        "smoke": smoke,
        "source": str(source),
        "config_path": str(config_path),
        "code_sha256": hashes,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "beta": "inf",
        "learner_tau": 0.5,
        "evaluation_tau": 0.5,
        "protocols": list(PROTOCOLS),
        "methods": ["multigroup"],
        "fractions": list(FRACTIONS),
        "n_splits": len(tasks),
        "n_seeds": 2 if smoke else len(plan.learner_seeds),
        "endpoint_atol": ENDPOINT_ATOL,
        "endpoint_rtol": ENDPOINT_RTOL,
        "estimand": "full-horizon training trajectory; first t pre-round maps averaged; fixed held-out comparator",
        "interval": "seed means within split, then approximate Nadeau-Bengio corrected marginal 95%; not simultaneous",
    }
    _write_json(output_dir / "run_info.json", info)
    try:
        records, _report = ordered_map(
            _run_split,
            tasks,
            workers=workers,
            initializer=_init_worker,
            initargs=(
                dataset,
                str(project_root),
                str(output_dir),
                str(source),
                plan.base_config.as_mapping(),
                smoke,
                digest,
            ),
            on_progress=lambda done, total, task, seconds: print(
                f"{dataset}: {done}/{total} splits completed", flush=True
            ),
            log=print,
        )
        info["outputs"] = reduce_curves(output_dir, records, dataset)
        info["figures"] = plot_curves(output_dir, dataset)
        info.update(
            status="completed",
            reportable=not smoke,
            seconds=time.perf_counter() - started,
            computed_splits=sum(not r["reused"] for r in records),
            reused_splits=sum(r["reused"] for r in records),
            reconstructed_fits=sum(
                r["reconstructed_fits"] for r in records if not r["reused"]
            ),
            max_endpoint_error=max(r["max_endpoint_error"] for r in records),
            endpoint_verified=True,
            comparison_logs_verified=all(r["comparison_logs_verified"] for r in records),
        )
        _write_json(output_dir / "run_info.json", info)
        return RunResult(0, output_dir, info)
    except Exception as exc:
        info.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _write_json(output_dir / "run_info.json", info)
        raise
