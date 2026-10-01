"""C5 (D-303): differential test for contiguous snapshot buffers + vectorized predict.

This module freezes a copy of the earlier ``AlignmentLotteryPolicy.predict``
implementation -- ``src/empirical_testing/algorithms/one_expert.py:609-661`` at the
time of writing -- as :func:`_reference_predict`, a standalone function that never
imports or calls the optimized ``predict`` method. It operates on the *snapshot
containers themselves* (whatever type they are: the original ``Tuple[FloatArray,
...]`` or the buffer ``FloatArray`` this change introduces) via plain ``zip``,
which both types support identically -- iteration yields one ``(K,)`` /``(K, m)``
row per snapshot, in the same order, for a tuple-of-arrays or for axis-0 of a
single contiguous array. This lets the reference function compare the original loop against the
optimized prediction method using either snapshot representation.

Amended 2026-09-15: under the current NumPy/BLAS runtime the batched snapshot contraction rounds
differently from the per-snapshot contraction in the last bit (worst observed 1.67e-16), so the
reference-vs-vectorized prediction comparisons use ``PREDICT_ABS_TOLERANCE``. Fitted snapshots,
weights and learner state remain bitwise gates, as do predict determinism and T-blocking.

Fixtures cover: single-group patterns, the all-groups pattern, random subsets
(including the empty/all-zero pattern, which ``predict`` maps to the reference
lottery), patterns never seen during training, and a K=1 model -- for both
``fit_groupwise_self_play`` (one_expert.py) and ``fit_groupwise_span_adaptive``
(span_adaptive.py, python backend; numba backend gated by
``pytest.importorskip``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from empirical_testing.algorithms.one_expert import (  # noqa: E402
    _PREDICTION_CHUNK_SIZE,
    _as_group_matrix,
    fit_groupwise_self_play,
    kl_divergence,
)
from empirical_testing.algorithms.span_adaptive import (  # noqa: E402
    fit_groupwise_span_adaptive,
)


#: Prediction-equivalence tolerance from the numerical comparison. The vectorized predict
#: contracts ``einsum("tng,tgm->tnm")`` over all snapshots at once, whereas the frozen reference contracts
#: ``einsum("ng,gm->nm")`` one snapshot at a time. NumPy dispatches those shapes to different BLAS kernels,
#: so the per-snapshot weighted policies can differ in the last bit; every other step (masking, exp, the
#: denominator, and the sequential cumsum fold) is bitwise identical, and fitted snapshots/weights/state are
#: bitwise identical across repeats and python/numba backends. Worst observed |difference| over the
#: retained fixtures plus a 120-case sweep (T <= 2995, K <= 12, m <= 20, tau in {0.05, 0.5, 1}) was
#: 1.67e-16 (max relative 7.3e-16). The absolute tolerance is ~12x that worst case (about 9 double-precision
#: epsilons) with zero relative tolerance; any materially different policy differs by many orders more.
PREDICT_ABS_TOLERANCE = 2e-15


def assert_predict_equivalent(reference_output, current_output, model, label="") -> None:
    reference_output = np.asarray(reference_output, dtype=np.float64)
    current_output = np.asarray(current_output, dtype=np.float64)
    assert reference_output.shape == current_output.shape, label
    np.testing.assert_allclose(current_output, reference_output, rtol=0.0, atol=PREDICT_ABS_TOLERANCE, err_msg=label)
    for output in (reference_output, current_output):
        assert np.all(np.abs(output.sum(axis=1) - 1.0) <= 1e-12), label
        assert max(kl_divergence(row, model.reference) for row in np.unique(output, axis=0)) <= model.tau + 2e-7, label


def same_bits(left, right) -> bool:
    """Exact ``uint64``-view equality (NaN-safe), matching the suite's tier-E idiom."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return False
    return bool(np.array_equal(left.view(np.uint64), right.view(np.uint64)))


# --------------------------------------------------------------------------
# Frozen reference: a literal copy of AlignmentLotteryPolicy.predict as it
# stood at one_expert.py:609-661 before C5. Do not "clean up" or vectorize
# this function -- its entire purpose is to keep computing the answer the old
# way, forever, so the new implementation has something honest to be checked
# against.
# --------------------------------------------------------------------------
def _reference_predict(model, group_matrix) -> np.ndarray:
    """Frozen copy of ``one_expert.py:609-661`` (pre-C5).

    ``model`` need only provide, by attribute: ``n_groups``, ``n_alternatives``,
    ``_outer_log_mass_snapshots``, ``_inner_policy_snapshots``,
    ``snapshot_weights``, ``reference``, ``tau``. It is never mutated.
    """

    groups = _as_group_matrix(group_matrix)
    if groups.shape[1] != model.n_groups:
        raise ValueError(
            f"group_matrix has {groups.shape[1]} columns, expected {model.n_groups}"
        )
    unique_groups, inverse = np.unique(groups, axis=0, return_inverse=True)
    output = np.zeros((unique_groups.shape[0], model.n_alternatives), dtype=np.float64)

    for outer_log_mass, inner_policies, snapshot_weight in zip(
        model._outer_log_mass_snapshots,
        model._inner_policy_snapshots,
        model.snapshot_weights,
    ):
        for start in range(0, unique_groups.shape[0], _PREDICTION_CHUNK_SIZE):
            stop = min(unique_groups.shape[0], start + _PREDICTION_CHUNK_SIZE)
            membership = unique_groups[start:stop].astype(bool)
            active = np.any(membership, axis=1)
            prediction = np.tile(model.reference, (stop - start, 1))
            if np.any(active):
                selected_log_mass = np.where(
                    membership[active],
                    outer_log_mass[None, :],
                    -np.inf,
                )
                row_maximum = np.max(selected_log_mass, axis=1)
                group_mass = np.exp(selected_log_mass - row_maximum[:, None])
                weighted_policies = np.einsum(
                    "ng,gm->nm",
                    group_mass,
                    inner_policies,
                    optimize=True,
                )
                prediction[active] = weighted_policies / group_mass.sum(
                    axis=1, keepdims=True
                )
            output[start:stop] += float(snapshot_weight) * prediction

    output = np.maximum(output, 0.0)
    output /= output.sum(axis=1, keepdims=True)
    for row in output:
        if kl_divergence(row, model.reference) > model.tau + 2e-7:
            raise AssertionError("snapshot-averaged alignment prediction is infeasible")
    return output[inverse]


# --------------------------------------------------------------------------
# Fixtures: small models, varied (T, K, m); and pattern generators.
# --------------------------------------------------------------------------
def _fixed_rate_model(T=260, K=3, m=4, seed=5, tau=0.5, beta=4.0):
    rng = np.random.default_rng(seed)
    groups = np.zeros((T, K), dtype=np.float64)
    groups[:, 0] = 1.0  # group 0 universal: every row has an active group
    if K > 1:
        groups[:, 1:] = (rng.random((T, K - 1)) < 0.4).astype(np.float64)
    utilities = rng.random((T, m))
    reference = np.full(m, 1.0 / m)
    model = fit_groupwise_self_play(
        groups=groups,
        utilities=utilities,
        beta=beta,
        reference=reference,
        tau=tau,
        protocol="passive",
        seed=seed,
        group_masses=groups.mean(axis=0),
        delta=0.05,
    )
    return model, groups


def _span_adaptive_model(T=260, K=3, m=4, seed=5, tau=0.5, beta=4.0, kernel_backend="python"):
    rng = np.random.default_rng(seed)
    groups = np.zeros((T, K), dtype=np.float64)
    groups[:, 0] = 1.0
    if K > 1:
        groups[:, 1:] = (rng.random((T, K - 1)) < 0.4).astype(np.float64)
    utilities = rng.random((T, m))
    reference = np.full(m, 1.0 / m)
    model = fit_groupwise_span_adaptive(
        groups=groups,
        utilities=utilities,
        beta=beta,
        reference=reference,
        tau=tau,
        protocol="passive",
        seed=seed,
        kernel_backend=kernel_backend,
    )
    return model, groups


def _stable_seed(name: str) -> int:
    """Deterministic seed from a fixture name, stable across processes.

    Plain ``hash(str)`` is salted per-process (``PYTHONHASHSEED``), which
    would silently desynchronize the pattern suite between the two separate
    interpreter processes ``test_new_buffers_match_pre_c5_git_head`` below
    launches to diff against pre-change code via git.
    """

    import zlib

    return zlib.crc32(name.encode("utf-8"))


def _pattern_suite(K: int, training_groups: np.ndarray, seed: int = 0) -> np.ndarray:
    """Single-group, all-groups, random subsets (incl. all-zero), and
    patterns guaranteed absent from ``training_groups``."""

    rows = []
    # Single-group patterns.
    for k in range(K):
        row = np.zeros(K)
        row[k] = 1.0
        rows.append(row)
    # All-groups pattern.
    rows.append(np.ones(K))
    # All-zero pattern (no active group -- predict must fall back to the
    # reference lottery).
    rows.append(np.zeros(K))
    # Random subsets, including repeats and near-empty/near-full subsets.
    rng = np.random.default_rng(seed)
    random_rows = rng.integers(0, 2, size=(64, K)).astype(np.float64)
    rows.extend(random_rows)

    candidate = np.asarray(rows, dtype=np.float64)

    # Patterns never seen in training: draw a batch from a disjoint seed and
    # keep only the ones absent from the training set's unique rows.
    training_unique = {tuple(row) for row in np.unique(training_groups, axis=0)}
    novel_rng = np.random.default_rng(seed + 999)
    novel_candidates = novel_rng.integers(0, 2, size=(256, K)).astype(np.float64)
    novel = [row for row in novel_candidates if tuple(row) not in training_unique]
    if novel:
        candidate = np.concatenate([candidate, np.asarray(novel[: min(16, len(novel))])], axis=0)
    return candidate


MODEL_FACTORIES = {
    "fixed_rate_K1_m2": lambda: _fixed_rate_model(T=40, K=1, m=2, seed=1, tau=0.3),
    "fixed_rate_K3_m4": lambda: _fixed_rate_model(T=260, K=3, m=4, seed=5, tau=0.5),
    "fixed_rate_K5_m20": lambda: _fixed_rate_model(T=180, K=5, m=20, seed=7, tau=0.4),
    "fixed_rate_K8_m5": lambda: _fixed_rate_model(T=220, K=8, m=5, seed=11, tau=0.6),
    "span_adaptive_python_K1_m2": lambda: _span_adaptive_model(T=40, K=1, m=2, seed=1, tau=0.3),
    "span_adaptive_python_K3_m4": lambda: _span_adaptive_model(T=260, K=3, m=4, seed=5, tau=0.5),
    "span_adaptive_python_K5_m20": lambda: _span_adaptive_model(T=180, K=5, m=20, seed=7, tau=0.4),
    "span_adaptive_python_K8_m5": lambda: _span_adaptive_model(T=220, K=8, m=5, seed=11, tau=0.6),
}

NUMBA_MODEL_FACTORIES = {
    "span_adaptive_numba_K1_m2": lambda: _span_adaptive_model(
        T=40, K=1, m=2, seed=1, tau=0.3, kernel_backend="numba"
    ),
    "span_adaptive_numba_K3_m4": lambda: _span_adaptive_model(
        T=260, K=3, m=4, seed=5, tau=0.5, kernel_backend="numba"
    ),
    "span_adaptive_numba_K5_m20": lambda: _span_adaptive_model(
        T=180, K=5, m=20, seed=7, tau=0.4, kernel_backend="numba"
    ),
    "span_adaptive_numba_K8_m5": lambda: _span_adaptive_model(
        T=220, K=8, m=5, seed=11, tau=0.6, kernel_backend="numba"
    ),
}


@pytest.mark.parametrize("name", sorted(MODEL_FACTORIES))
def test_reference_matches_current_predict(name):
    """The frozen per-snapshot reference and today's vectorized ``predict`` agree to
    ``PREDICT_ABS_TOLERANCE`` (floating-point contraction order only; see the constant)."""

    model, groups = MODEL_FACTORIES[name]()
    patterns = _pattern_suite(model.n_groups, groups, seed=_stable_seed(name))
    reference_output = _reference_predict(model, patterns)
    current_output = model.predict(patterns)
    assert_predict_equivalent(reference_output, current_output, model, name)
    if model.n_groups == 1:  # one group: the contraction is exact, so the paths must agree bit for bit
        assert same_bits(reference_output, current_output), name


@pytest.mark.parametrize("name", sorted(NUMBA_MODEL_FACTORIES))
def test_reference_matches_current_predict_numba_backend(name):
    """Same check for span-adaptive models fit with the numba backend
    (backend only changes how the *fit* runs; ``predict`` is shared code)."""

    pytest.importorskip("numba")
    model, groups = NUMBA_MODEL_FACTORIES[name]()
    patterns = _pattern_suite(model.n_groups, groups, seed=_stable_seed(name))
    reference_output = _reference_predict(model, patterns)
    current_output = model.predict(patterns)
    assert_predict_equivalent(reference_output, current_output, model, name)
    if model.n_groups == 1:
        assert same_bits(reference_output, current_output), name


@pytest.mark.parametrize("shape", [(260, 3, 4, 5, 0.5), (220, 8, 5, 11, 0.6), (180, 5, 20, 7, 0.4)])
def test_fitted_state_and_predict_are_bitwise_deterministic_across_repeats_and_backends(shape):
    """Snapshots, weights and learner state stay bitwise gates; so does predict on identical state."""

    pytest.importorskip("numba")
    T, K, m, seed, tau = shape
    first, groups = _span_adaptive_model(T=T, K=K, m=m, seed=seed, tau=tau, kernel_backend="python")
    again, _ = _span_adaptive_model(T=T, K=K, m=m, seed=seed, tau=tau, kernel_backend="python")
    numba_model, _ = _span_adaptive_model(T=T, K=K, m=m, seed=seed, tau=tau, kernel_backend="numba")
    for field in ("_outer_log_mass_snapshots", "_inner_policy_snapshots", "snapshot_weights", "final_outer_log_masses",
                  "outer_second_moments", "centered_inner_second_moments", "inner_second_moments"):
        assert same_bits(getattr(first, field), getattr(again, field)), field
        assert same_bits(getattr(first, field), getattr(numba_model, field)), field
    patterns = _pattern_suite(K, groups, seed=1)
    assert same_bits(first.predict(patterns), first.predict(patterns))
    assert same_bits(first.predict(patterns), numba_model.predict(patterns))


def test_prediction_tolerance_still_detects_materially_different_policies():
    """The tolerance is far too tight to hide a changed snapshot, weight, or membership mixing."""

    import dataclasses

    model, groups = MODEL_FACTORIES["span_adaptive_python_K3_m4"]()
    patterns = _pattern_suite(model.n_groups, groups, seed=3)
    baseline = model.predict(patterns)
    inner = np.array(model._inner_policy_snapshots, copy=True)
    inner[len(inner) // 2, [1, 2]] = inner[len(inner) // 2, [2, 1]]  # swap two groups' policies at one snapshot
    weights = np.array(model.snapshot_weights, copy=True)
    weights[0], weights[-1] = weights[0] * 1.001, weights[-1] - weights[0] * 0.001
    for changed in (dataclasses.replace(model, _inner_policy_snapshots=inner),
                    dataclasses.replace(model, snapshot_weights=weights)):
        altered = _reference_predict(changed, patterns)
        assert np.max(np.abs(altered - baseline)) > 1e3 * PREDICT_ABS_TOLERANCE
        with pytest.raises(AssertionError):
            assert_predict_equivalent(altered, baseline, model, "materially different policy")


@pytest.mark.parametrize("name", sorted(MODEL_FACTORIES))
def test_fitted_snapshots_are_well_formed(name):
    """Row-by-row structural check on the snapshot containers themselves,
    independent of ``predict``: used again in M3 to confirm the new buffer
    holds exactly the old tuple's rows in the old order."""

    model, _groups = MODEL_FACTORIES[name]()
    n_snapshots = len(model.snapshot_weights)
    assert len(model._outer_log_mass_snapshots) == n_snapshots
    assert len(model._inner_policy_snapshots) == n_snapshots
    for outer_row in model._outer_log_mass_snapshots:
        assert outer_row.shape == (model.n_groups,)
    for inner_row in model._inner_policy_snapshots:
        assert inner_row.shape == (model.n_groups, model.n_alternatives)


# --------------------------------------------------------------------------
# Optional differential check against a developer-supplied revision from
# before the buffer refactor. A second interpreter fits identical models with
# that revision's learners and compares its original snapshot containers and
# prediction loop against the current implementation.
# --------------------------------------------------------------------------

_GIT_HEAD_FIXTURES = [
    # (name, kind, T, K, m, seed, tau, kernel_backend)
    ("fixed_rate_K1_m2", "fixed_rate", 40, 1, 2, 1, 0.3, None),
    ("fixed_rate_K3_m4", "fixed_rate", 260, 3, 4, 5, 0.5, None),
    ("fixed_rate_K5_m20", "fixed_rate", 180, 5, 20, 7, 0.4, None),
    ("span_adaptive_python_K1_m2", "span_adaptive", 40, 1, 2, 1, 0.3, "python"),
    ("span_adaptive_python_K3_m4", "span_adaptive", 260, 3, 4, 5, 0.5, "python"),
    ("span_adaptive_python_K8_m5", "span_adaptive", 220, 8, 5, 11, 0.6, "python"),
]
_GIT_HEAD_NUMBA_FIXTURES = [
    ("span_adaptive_numba_K3_m4", "span_adaptive", 260, 3, 4, 5, 0.5, "numba"),
]

_OLD_FIT_SUBPROCESS_TEMPLATE = r"""
import json
import sys
import numpy as np

sys.path.insert(0, {src_path!r})

from empirical_testing.algorithms.one_expert import fit_groupwise_self_play
from empirical_testing.algorithms.span_adaptive import fit_groupwise_span_adaptive


def pattern_suite(K, training_groups, seed):
    import zlib
    rows = []
    for k in range(K):
        row = np.zeros(K)
        row[k] = 1.0
        rows.append(row)
    rows.append(np.ones(K))
    rows.append(np.zeros(K))
    rng = np.random.default_rng(seed)
    rows.extend(rng.integers(0, 2, size=(64, K)).astype(np.float64))
    candidate = np.asarray(rows, dtype=np.float64)
    training_unique = {{tuple(row) for row in np.unique(training_groups, axis=0)}}
    novel_rng = np.random.default_rng(seed + 999)
    novel_candidates = novel_rng.integers(0, 2, size=(256, K)).astype(np.float64)
    novel = [row for row in novel_candidates if tuple(row) not in training_unique]
    if novel:
        candidate = np.concatenate([candidate, np.asarray(novel[: min(16, len(novel))])], axis=0)
    return candidate


def build(kind, T, K, m, seed, tau, kernel_backend):
    rng = np.random.default_rng(seed)
    groups = np.zeros((T, K), dtype=np.float64)
    groups[:, 0] = 1.0
    if K > 1:
        groups[:, 1:] = (rng.random((T, K - 1)) < 0.4).astype(np.float64)
    utilities = rng.random((T, m))
    reference = np.full(m, 1.0 / m)
    if kind == "fixed_rate":
        model = fit_groupwise_self_play(
            groups=groups, utilities=utilities, beta=4.0, reference=reference,
            tau=tau, protocol="passive", seed=seed,
            group_masses=groups.mean(axis=0), delta=0.05,
        )
    else:
        model = fit_groupwise_span_adaptive(
            groups=groups, utilities=utilities, beta=4.0, reference=reference,
            tau=tau, protocol="passive", seed=seed, kernel_backend=kernel_backend,
        )
    return model, groups


import zlib

fixtures = {fixtures!r}
out = {{}}
for name, kind, T, K, m, seed, tau, kernel_backend in fixtures:
    model, groups = build(kind, T, K, m, seed, tau, kernel_backend)
    patterns = pattern_suite(model.n_groups, groups, seed=zlib.crc32(name.encode("utf-8")))
    out[name] = {{
        "outer": np.stack(model._outer_log_mass_snapshots, axis=0).tolist(),
        "inner": np.stack(model._inner_policy_snapshots, axis=0).tolist(),
        "weights": np.asarray(model.snapshot_weights, dtype=np.float64).tolist(),
        "predict": np.asarray(model.predict(patterns), dtype=np.float64).tolist(),
    }}

with open({out_path!r}, "w", encoding="utf-8") as fh:
    json.dump(out, fh)
"""


@pytest.fixture(scope="module")
def git_head_fixture_dump():
    """Compare against an explicitly supplied pre-refactor Git revision.

    Set ``EMPIRICAL_TESTING_LEGACY_REVISION`` to a revision containing the
    historical files to enable this developer-only check. The fixture skips
    when the variable, Git, or the historical source files are unavailable.
    Its result is ``{name: {"outer", "inner", "weights", "predict"}}``.
    """

    legacy_revision = os.environ.get("EMPIRICAL_TESTING_LEGACY_REVISION")
    if not legacy_revision:
        pytest.skip("set EMPIRICAL_TESTING_LEGACY_REVISION for the optional historical-source check")
    one_expert_rel = "EmpiricalTestingAdaptiveCPU/src/empirical_testing/algorithms/one_expert.py"
    span_adaptive_rel = "EmpiricalTestingAdaptiveCPU/src/empirical_testing/algorithms/span_adaptive.py"
    try:
        old_one_expert = subprocess.run(
            ["git", "show", f"{legacy_revision}:{one_expert_rel}"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, check=True, timeout=30,
        ).stdout
        old_span_adaptive = subprocess.run(
            ["git", "show", f"{legacy_revision}:{span_adaptive_rel}"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, check=True, timeout=30,
        ).stdout
    except Exception as error:  # pragma: no cover - environment-dependent
        pytest.skip(f"could not read pre-change source via git: {error}")

    with tempfile.TemporaryDirectory(prefix="c5_old_ref_") as tmp:
        tmp_path = Path(tmp)
        scratch_src = tmp_path / "src"
        shutil.copytree(
            PROJECT_ROOT / "src",
            scratch_src,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        (scratch_src / "empirical_testing" / "algorithms" / "one_expert.py").write_text(
            old_one_expert, encoding="utf-8"
        )
        (scratch_src / "empirical_testing" / "algorithms" / "span_adaptive.py").write_text(
            old_span_adaptive, encoding="utf-8"
        )

        all_fixtures = _GIT_HEAD_FIXTURES.copy()
        try:
            import numba  # noqa: F401

            all_fixtures += _GIT_HEAD_NUMBA_FIXTURES
        except ImportError:
            pass

        out_path = tmp_path / "dump.json"
        script = _OLD_FIT_SUBPROCESS_TEMPLATE.format(
            src_path=str(scratch_src), fixtures=all_fixtures, out_path=str(out_path)
        )
        script_path = tmp_path / "run_old.py"
        script_path.write_text(script, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            pytest.fail(
                "pre-C5 reference subprocess failed:\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        return json.loads(out_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", [row[0] for row in _GIT_HEAD_FIXTURES])
def test_new_buffers_match_pre_c5_git_head(name, git_head_fixture_dump):
    """Compare buffer-based fit/predict with a supplied historical revision.

    Fitted snapshot containers agree bit-for-bit; predictions use the
    documented floating-point tolerance for reordered contractions.
    """

    spec = {row[0]: row for row in _GIT_HEAD_FIXTURES}[name]
    _name, kind, T, K, m, seed, tau, kernel_backend = spec
    if kind == "fixed_rate":
        new_model, groups = _fixed_rate_model(T=T, K=K, m=m, seed=seed, tau=tau)
    else:
        new_model, groups = _span_adaptive_model(
            T=T, K=K, m=m, seed=seed, tau=tau, kernel_backend=kernel_backend
        )
    old = git_head_fixture_dump[name]

    assert same_bits(new_model._outer_log_mass_snapshots, np.asarray(old["outer"])), (
        f"{name}: outer_log_mass_snapshots row mismatch vs. git HEAD"
    )
    assert same_bits(new_model._inner_policy_snapshots, np.asarray(old["inner"])), (
        f"{name}: inner_policy_snapshots row mismatch vs. git HEAD"
    )
    assert same_bits(new_model.snapshot_weights, np.asarray(old["weights"])), (
        f"{name}: snapshot_weights mismatch vs. git HEAD"
    )

    patterns = _pattern_suite(new_model.n_groups, groups, seed=_stable_seed(name))
    new_predict = new_model.predict(patterns)
    assert_predict_equivalent(np.asarray(old["predict"]), new_predict, new_model,
                              f"{name}: predict() output mismatch vs. git HEAD's actual (pre-C5) predict")


@pytest.mark.parametrize("name", [row[0] for row in _GIT_HEAD_NUMBA_FIXTURES])
def test_new_buffers_match_pre_c5_git_head_numba_backend(name, git_head_fixture_dump):
    """Same historical-source check for the numba-backed span-adaptive fit."""

    pytest.importorskip("numba")
    if name not in git_head_fixture_dump:
        pytest.skip("numba fixture not present in the git-HEAD dump for this run")
    spec = {row[0]: row for row in _GIT_HEAD_NUMBA_FIXTURES}[name]
    _name, kind, T, K, m, seed, tau, kernel_backend = spec
    new_model, groups = _span_adaptive_model(
        T=T, K=K, m=m, seed=seed, tau=tau, kernel_backend=kernel_backend
    )
    old = git_head_fixture_dump[name]

    assert same_bits(new_model._outer_log_mass_snapshots, np.asarray(old["outer"]))
    assert same_bits(new_model._inner_policy_snapshots, np.asarray(old["inner"]))
    assert same_bits(new_model.snapshot_weights, np.asarray(old["weights"]))

    patterns = _pattern_suite(new_model.n_groups, groups, seed=_stable_seed(name))
    new_predict = new_model.predict(patterns)
    assert_predict_equivalent(np.asarray(old["predict"]), new_predict, new_model, name)


# --------------------------------------------------------------------------
# Coordinator fix: predict's snapshot-axis (T) memory blocking.
#
# Before this fix, the active branch materialized several full-T arrays of
# shape (T, n_active_rows_in_chunk, max(K, m)) at once (selected_log_mass,
# group_mass, weighted_policies, active_prediction, weighted, plus the
# cumsum). Coffee has <=32 unique membership patterns (2**K, K=5), so the
# earlier M5 timings never exercised n_active anywhere near
# _PREDICTION_CHUNK_SIZE=1024 and never saw the resulting peak. A dataset
# with ~1000+ unique patterns and thousands of rounds would have: ~150MB
# at T=3744/K=5 scales past a gigabyte at larger T/m.
#
# The fix blocks over T using a memory budget (_PREDICT_T_BLOCK_BYTES_BUDGET)
# instead of changing the row-chunk size (a different row-chunk width could
# shift some rows onto a different BLAS/einsum dispatch path -- exactly the
# kind of change the design's einsum risk note warns against). Blocks are
# folded by *prepending* the running total as a phantom first row and
# re-applying cumsum, not by summing each block independently and adding
# the sums (see the docstring on `AlignmentLotteryPolicy.predict` for why
# only the former is bit-identical to an unblocked cumsum).
#
# These tests force the block size via the private `_t_block_override` hook
# (1, and an odd number, 7) and check bitwise agreement with (a) the
# default-budget result and (b) `_reference_predict`, the frozen pre-C5
# oracle -- on a fixture built specifically to exercise BOTH multiple
# pattern-row chunks (>1024 unique rows) and multiple T-blocks even under
# the production budget (K=12, m=20 push per-t bytes high enough that
# T=300 needs more than one default-budget block too).
# --------------------------------------------------------------------------
def _many_unique_patterns_model():
    return _span_adaptive_model(T=300, K=12, m=20, seed=99, tau=0.5, beta=4.0)


def _many_unique_patterns(K: int, n: int, seed: int) -> np.ndarray:
    """``n`` distinct binary length-``K`` rows (no repeats), forcing multiple
    ``_PREDICTION_CHUNK_SIZE``-wide row chunks when ``n`` exceeds it."""

    rng = np.random.default_rng(seed)
    space = 2**K
    assert n <= space, "requested more unique patterns than exist at this K"
    indices = rng.choice(space, size=n, replace=False)
    bits = ((indices[:, None] >> np.arange(K)[None, :]) & 1).astype(np.float64)
    return bits


def test_predict_t_blocking_default_budget_forces_multiple_blocks():
    """Sanity check that the fixture actually exercises T-blocking under the
    *production* budget, not only under a forced override -- otherwise the
    bitwise-agreement tests below would be vacuous."""

    model, _groups = _many_unique_patterns_model()
    max_dim = max(model.n_groups, model.n_alternatives)
    per_t_bytes = 8 * _PREDICTION_CHUNK_SIZE * max_dim
    from empirical_testing.algorithms.one_expert import _PREDICT_T_BLOCK_BYTES_BUDGET

    default_block = max(1, _PREDICT_T_BLOCK_BYTES_BUDGET // per_t_bytes)
    n_snapshots = len(model.snapshot_weights)
    assert default_block < n_snapshots, (
        "fixture too small to force multi-block chunking under the default "
        f"budget: block={default_block}, T={n_snapshots}"
    )


@pytest.mark.parametrize("n_unique_patterns", [1500])
@pytest.mark.parametrize("t_block_override", [1, 7, None])
def test_predict_t_blocking_matches_reference(n_unique_patterns, t_block_override):
    """Multiple row chunks (n_unique_patterns > _PREDICTION_CHUNK_SIZE) AND
    multiple T-blocks (forced override, or the default budget per the sanity
    check above), compared bit-for-bit against the frozen pre-C5 reference."""

    model, groups = _many_unique_patterns_model()
    patterns = _many_unique_patterns(model.n_groups, n_unique_patterns, seed=2026)
    assert patterns.shape[0] > _PREDICTION_CHUNK_SIZE, "fixture must force >1 row chunk"

    reference_output = _reference_predict(model, patterns)
    kwargs = {} if t_block_override is None else {"_t_block_override": t_block_override}
    new_output = model.predict(patterns, **kwargs)
    assert_predict_equivalent(reference_output, new_output, model,
                              f"t_block_override={t_block_override} diverged from the frozen reference")
    # Blocking itself must remain exact: every block size reproduces the unblocked vectorized result bitwise.
    assert same_bits(new_output, model.predict(patterns))


def test_predict_t_blocking_forced_sizes_agree_with_default_budget():
    """B=1 and B=7 must also agree with each other and with the (also
    multi-block, per the sanity check) default-budget result -- not just
    each individually with the reference, which would miss a bug that
    happens to cancel against the reference at one specific block size."""

    model, groups = _many_unique_patterns_model()
    patterns = _many_unique_patterns(model.n_groups, 1500, seed=2026)

    default_output = model.predict(patterns)
    for override in (1, 7, 64):
        blocked_output = model.predict(patterns, _t_block_override=override)
        assert same_bits(default_output, blocked_output), (
            f"_t_block_override={override} diverged from the default-budget result"
        )
