"""Acceptance tests for the suite import of the completed protocol x beta runs.

``empirical_testing.experiments.provenance`` relocates two *finished* runs
into their suite namespaces.  The runs themselves are irreplaceable -- they are
the only copy of results suite adopts rather than retrains -- so the properties
pinned here are the ones that protect them:

* the relocation preserves every source byte, and says so with a per-file
  sha256 map plus ``hashes_unchanged_by_move``;
* a second call is a no-op that reports ``already_complete``, and a call after
  an interrupted move finishes the job instead of moving anything twice;
* a destination that already holds content is a *refusal*, never an overwrite
  or a merge, and both trees survive the refusal untouched;
* every verification failure blocks the move entirely, and one ``verify_source``
  call reports all of them rather than stopping at the first;
* the manifest preserves the source run's historical ids verbatim, so the
  import can never be mistaken for fresh suite training, and leaves
  ``v10_complete`` false because importing beta results is only one of suite's
  prerequisites;
* ``compare_with_frozen_arrays`` compares *saved* arrays and never recomputes,
  and compares the anchor cell's slice -- not the raw tensors, whose rank and
  names differ between a sweep run and the frozen run it was seeded from.

Everything here is synthetic and lives in ``tmp_path``.  No test reads, moves
or writes anything under ``results/``: those are the unmigrated production
directories, and a test that touched them would be the exact accident this
module exists to prevent.
"""

from __future__ import annotations

import json
import shutil
import stat
from pathlib import Path

import numpy as np
import pytest

from empirical_testing.experiments.coffee_protocol_beta_sweep import (
    ANCHOR_ARRAY_MAP,
    DIAGNOSTIC_KEYS,
    SWEEP_METHODS,
)
from empirical_testing.experiments.provenance import (
    EXPECTED_CELLS,
    EXPECTED_SPLITS,
    IMPORT_MANIFEST,
    SOURCE_RUNS,
    SOURCE_SUBDIR,
    MigrationError,
    SourceRun,
    compare_with_frozen_arrays,
    hash_tree,
    migrate,
    migration_state,
    verify_source,
)

#: The run this suite migrates.  Which of the two real entries is used does not
#: matter -- they differ only in their ids -- but taking a real one keeps the
#: fixture honest about the shape ``migrate`` will meet in production.
SOURCE: SourceRun = SOURCE_RUNS["coffee"]

VALID_METHODS = ("pooled", "multigroup", "uniform")


def _split_names(count: int) -> list[str]:
    """``rNN_fMM`` names, the layout the crossfit runs save splits under."""

    return [f"r{i // 10:02d}_f{i % 10:02d}" for i in range(count)]


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_run(
    run_dir: Path,
    *,
    source: SourceRun = SOURCE,
    status: str = "completed",
    protocol_id: str | None = None,
    n_splits: int = EXPECTED_SPLITS,
    methods: tuple[str, ...] = VALID_METHODS,
    cells=EXPECTED_CELLS,
    drop_split_json: str | None = None,
    drop_split_npz: str | None = None,
    anchor: dict | None = None,
    write_anchor: bool = True,
    score: float = 1.0,
) -> Path:
    """A miniature but *valid* completed beta run, ready to be migrated.

    Every keyword exists so a single test can introduce exactly one defect;
    with no keywords the result passes ``verify_source`` cleanly.  The arrays
    are one element each: ``verify_source`` only reads them for existence, and
    ``compare_with_frozen_arrays`` only needs them to be comparable.
    """

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "run_info.json", {
        "status": status,
        "protocol_id": source.protocol_id if protocol_id is None else protocol_id,
        "implementation_id": "coffee-crossfit-v4-protocol-beta-impl-1",
        "frozen_run": "results/coffee-crossfit-10x10-v4-20260916-114746",
        "config_sha256": "0" * 64,
        "cells": [{"protocol": p, "beta": b} for p, b in cells],
    })

    for name in _split_names(n_splits):
        split_dir = run_dir / "splits" / name
        split_dir.mkdir(parents=True, exist_ok=True)
        if name != drop_split_json:
            _write_json(split_dir / "split.json", {
                "split": name,
                "methods": list(methods),
                "cells": [f"{p}@{b}" for p, b in cells],
            })
        if name != drop_split_npz:
            np.savez(split_dir / "split.npz",
                     welfare_ratio=np.array([score]),
                     distortion=np.array([2.0 * score]))

    _write_json(run_dir / "design" / "design.json", {
        "protocol_id": source.protocol_id,
        "config_sha256": "0" * 64,
    })

    if write_anchor:
        _write_json(run_dir / "reduce" / "anchor_check.json", anchor if anchor is not None else {
            "checked": True,
            "agrees": True,
            "anchor_cell": "passive@inf",
            "arrays_compared": 200,
            "array_mismatches": [],
            "frozen_run": "results/coffee-crossfit-10x10-v4-20260916-114746",
        })
    return run_dir


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A fake project root holding one valid, unmigrated source run."""

    build_run(tmp_path / SOURCE.original_path)
    return tmp_path


def _original(project: Path) -> Path:
    return project / SOURCE.original_path


def _destination(project: Path) -> Path:
    return project / SOURCE.suite_directory / SOURCE_SUBDIR


def _manifest_path(project: Path) -> Path:
    return project / SOURCE.suite_directory / IMPORT_MANIFEST


# ------------------------------------------------------------------- baseline

def test_fixture_run_verifies_cleanly(project: Path):
    """The fixture must be a *valid* run, or every failure test below is vacuous."""

    report = verify_source(_original(project), SOURCE)
    assert report.ok, f"the baseline fixture should verify, but reported: {report.failures}"
    assert report.checks["splits"] == EXPECTED_SPLITS
    assert report.checks["methods_canonical"] == sorted(VALID_METHODS)


# ------------------------------------------------------------ hash preservation

def test_migration_preserves_source_hashes_exactly(project: Path):
    """"Migration preserves source hashes exactly."

    Hashed before, hashed after, compared -- and the manifest has to carry the
    evidence, because the manifest is what a later reader has instead of the
    pre-move tree.
    """

    before = hash_tree(_original(project))
    assert before, "the fixture produced no files to hash"

    manifest = migrate(project, SOURCE)

    after = hash_tree(_destination(project))
    assert after == before, "the relocation changed the source bytes"
    assert manifest["hashes_unchanged_by_move"] is True
    # The map is the audit trail: a count alone could not detect a swap.
    assert manifest["file_sha256"] == before
    assert manifest["file_count"] == len(before)


def test_migrated_tree_is_write_protected(project: Path):
    """``source/`` is declared immutable, so the imported files lose their write bit."""

    migrate(project, SOURCE)
    info = _destination(project) / "run_info.json"
    assert not info.stat().st_mode & stat.S_IWUSR, "imported files should not be writable in place"


# ---------------------------------------------------------------- idempotency

def test_migrate_twice_leaves_one_copy_and_reports_already_complete(project: Path):
    """"Migration is idempotent": the second call must not re-move anything."""

    first = migrate(project, SOURCE)
    manifest_bytes = _manifest_path(project).read_bytes()
    after_first = hash_tree(_destination(project))

    second = migrate(project, SOURCE)

    assert second.get("already_complete") is True, "a completed migration must say so"
    assert second["file_sha256"] == first["file_sha256"]
    # A re-move would have produced source/source or rewritten the manifest.
    assert not (_destination(project) / SOURCE_SUBDIR).exists()
    assert not _original(project).exists()
    assert hash_tree(_destination(project)) == after_first
    assert _manifest_path(project).read_bytes() == manifest_bytes,\
        "the recorded manifest must be returned, not regenerated"


def test_verified_partial_migration_is_completed_not_redone(project: Path):
    """"Idempotent continuation": destination bytes present, manifest absent.

    This is the interrupted-move case -- ``shutil.move`` finished but the
    process died before the manifest was written.  The second call must finish
    the job by writing the manifest, not move anything a second time.
    """

    before = hash_tree(_original(project))
    destination = _destination(project)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(_original(project)), str(destination))
    assert not _manifest_path(project).exists()
    assert migration_state(project, SOURCE) == "partial"

    manifest = migrate(project, SOURCE)

    assert manifest["resumed_from_partial_migration"] is True
    assert manifest["hashes_unchanged_by_move"] is True
    assert manifest["file_sha256"] == before, "the resumed manifest must describe the bytes on disk"
    assert _manifest_path(project).is_file()
    assert migration_state(project, SOURCE) == "complete"
    # Nothing moved twice: no nested copy, and the tree is exactly what arrived.
    assert not (destination / SOURCE_SUBDIR).exists()
    assert hash_tree(destination) == before


def test_partial_migration_that_does_not_verify_is_refused(project: Path):
    """A half-copied destination must not be blessed with a manifest."""

    destination = _destination(project)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(_original(project)), str(destination))
    shutil.rmtree(destination / "splits" / "r00_f00")

    with pytest.raises(MigrationError, match="does not verify"):
        migrate(project, SOURCE)
    assert not _manifest_path(project).exists()


# ------------------------------------------------------------------- collision

def test_destination_collision_is_refused_and_both_trees_survive(project: Path):
    """"A destination collision is REFUSED, never overwritten or merged."

    Both sides present means somebody already put something there; deleting or
    merging either one could destroy the only copy of a 500MB run.
    """

    destination = _destination(project)
    build_run(destination, score=9.0)  # a *different* run already occupying the slot
    original_hashes = hash_tree(_original(project))
    destination_hashes = hash_tree(destination)
    assert original_hashes != destination_hashes, "the fixture must make the two trees distinguishable"

    assert migration_state(project, SOURCE) == "blocked"
    with pytest.raises(MigrationError, match="refusing to overwrite or merge"):
        migrate(project, SOURCE)

    assert hash_tree(_original(project)) == original_hashes, "the original was touched by a refusal"
    assert hash_tree(destination) == destination_hashes, "the destination was touched by a refusal"
    # Spot-check real content, not only digests: the occupant's arrays are its own.
    with np.load(destination / "splits" / "r00_f00" / "split.npz") as arrays:
        assert arrays["welfare_ratio"] == pytest.approx(9.0)
    assert not _manifest_path(project).exists()


def test_collision_is_refused_even_when_a_manifest_is_present(project: Path):
    """A manifest does not license a merge while the original still exists."""

    build_run(_destination(project), score=9.0)
    _write_json(_manifest_path(project), {"migration_version": "v10", "dataset": "coffee"})

    assert migration_state(project, SOURCE) == "blocked"
    with pytest.raises(MigrationError):
        migrate(project, SOURCE)
    assert _original(project).is_dir()


# ---------------------------------------------------- verification blocks move

def _mutate_status(run: Path):
    info = json.loads((run / "run_info.json").read_text())
    info["status"] = "running"
    _write_json(run / "run_info.json", info)


def _mutate_protocol_id(run: Path):
    info = json.loads((run / "run_info.json").read_text())
    info["protocol_id"] = "some-other-sweep"
    _write_json(run / "run_info.json", info)


def _mutate_split_count(run: Path):
    shutil.rmtree(run / "splits" / "r09_f09")


def _mutate_missing_split_json(run: Path):
    (run / "splits" / "r03_f07" / "split.json").unlink()


def _mutate_missing_split_npz(run: Path):
    (run / "splits" / "r05_f02" / "split.npz").unlink()


def _mutate_cells(run: Path):
    info = json.loads((run / "run_info.json").read_text())
    info["cells"] = info["cells"][:5] + [{"protocol": "active", "beta": "0.5"}]
    _write_json(run / "run_info.json", info)


def _methods_mutator(methods):
    def mutate(run: Path):
        for split_dir in sorted((run / "splits").iterdir()):
            unit = json.loads((split_dir / "split.json").read_text())
            unit["methods"] = list(methods)
            _write_json(split_dir / "split.json", unit)
    return mutate


def _mutate_missing_anchor(run: Path):
    (run / "reduce" / "anchor_check.json").unlink()


def _anchor_mutator(**fields):
    def mutate(run: Path):
        anchor = json.loads((run / "reduce" / "anchor_check.json").read_text())
        anchor.update(fields)
        _write_json(run / "reduce" / "anchor_check.json", anchor)
    return mutate


@pytest.mark.parametrize("mutate, expected", [
    (_mutate_status, "not 'completed'"),
    (_mutate_protocol_id, "protocol_id is"),
    (_mutate_split_count, "99 saved splits"),
    (_mutate_missing_split_json, "missing split.json or split.npz"),
    (_mutate_missing_split_npz, "missing split.json or split.npz"),
    (_mutate_cells, "not the expected seven-cell grid"),
    # ``pooled_nlhf`` is a *permitted* method, but no beta run contains it: an
    # NLHF fit in the source would mean suite imported something it must fit itself.
    (_methods_mutator(("pooled", "multigroup", "uniform", "pooled_nlhf")), "saved methods"),
    # ``intersection_nlhf`` is retired; it must be reported, not raised through.
    (_methods_mutator(("pooled", "multigroup", "intersection_nlhf")), "intersection_nlhf"),
    (_mutate_missing_anchor, "anchor_check.json is missing"),
    (_anchor_mutator(agrees=False, array_mismatches=["r00_f00:welfare_ratio"]),
     "anchor gate did not agree"),
    (_anchor_mutator(checked=False), "anchor gate was not run"),
], ids=["status", "protocol_id", "split_count", "no_split_json", "no_split_npz",
        "cells", "extra_nlhf_method", "retired_method", "no_anchor_check",
        "anchor_disagrees", "anchor_unchecked"])
def test_verification_failure_blocks_the_move_entirely(project: Path, mutate, expected):
    """"Verification failures block the move entirely."

    Each case introduces exactly one defect into an otherwise valid run and
    asserts three things: the failure is *reported* (not raised as some other
    exception type), nothing reached the destination, and the original tree is
    byte-identical afterwards.
    """

    original = _original(project)
    mutate(original)
    before = hash_tree(original)

    report = verify_source(original, SOURCE)
    assert not report.ok, "the defect should have been detected"
    assert any(expected in failure for failure in report.failures),\
        f"expected a failure mentioning {expected!r}, got {report.failures}"

    with pytest.raises(MigrationError) as raised:
        migrate(project, SOURCE)
    assert "nothing was moved" in str(raised.value)

    assert not _destination(project).exists(), "a failed verification still created a destination"
    assert not _manifest_path(project).exists()
    assert hash_tree(original) == before, "the original was modified by a refused migration"


def test_missing_run_info_is_reported_not_raised(tmp_path: Path):
    """An empty directory is a failure report, not a ``KeyError`` from deep inside."""

    run = tmp_path / SOURCE.original_path
    run.mkdir(parents=True)
    report = verify_source(run, SOURCE)
    assert not report.ok
    assert any("run_info.json is missing" in f for f in report.failures)


def test_verify_source_collects_every_failure(project: Path):
    """"``verify_source`` collects ALL failures rather than stopping at the first."

    A migration operator should learn everything wrong with a run in one pass,
    not fix one problem at a time across eleven re-runs.
    """

    original = _original(project)
    _mutate_status(original)
    _mutate_protocol_id(original)
    _mutate_split_count(original)
    _mutate_cells(original)
    _methods_mutator(("pooled",))(original)
    _anchor_mutator(checked=False, agrees=False)(original)

    report = verify_source(original, SOURCE)

    joined = " | ".join(report.failures)
    for expected in ["not 'completed'", "protocol_id is", "99 saved splits",
                     "seven-cell grid", "saved methods", "anchor gate was not run",
                     "anchor gate did not agree"]:
        assert expected in joined, f"{expected!r} missing from collected failures: {report.failures}"
    assert len(report.failures) >= 7
    assert report.as_json()["ok"] is False


# -------------------------------------------------------------------- manifest

def test_manifest_records_provenance_and_preserves_historical_ids(project: Path):
    """"The manifest records original path, new path, source/historical ids...

    ...and historical ids are preserved verbatim, NOT rewritten to look like
    fresh suite training."  The two id namespaces have to coexist: the suite
    protocol id says where the results now live, the historical ids say what
    actually produced them.
    """

    info = json.loads((_original(project) / "run_info.json").read_text())

    manifest = migrate(project, SOURCE)

    assert manifest["original_path"] == SOURCE.original_path
    assert manifest["new_path"] == f"{SOURCE.suite_directory}/{SOURCE_SUBDIR}"
    assert manifest["source_protocol_id"] == SOURCE.protocol_id
    assert manifest["v10_protocol_id"] == SOURCE.suite_protocol_id

    historical = manifest["historical_ids_preserved"]
    assert historical["protocol_id"] == info["protocol_id"],\
        "the source protocol id was rewritten; the import would look like a suite fit"
    assert historical["implementation_id"] == info["implementation_id"]
    assert historical["frozen_run"] == info["frozen_run"]
    assert historical["config_sha256"] == info["config_sha256"]
    assert historical["protocol_id"] != SOURCE.suite_protocol_id
    assert manifest["immutable"] is True
    assert manifest["validation"]["ok"] is True

    # The manifest on disk is the same document, not a summary of it.
    assert json.loads(_manifest_path(project).read_text()) == manifest


def test_manifest_does_not_claim_suite_is_complete(project: Path):
    """"``v10_complete`` is false: importing beta results is only one prerequisite."""

    manifest = migrate(project, SOURCE)
    assert manifest["v10_complete"] is False
    assert "NLHF" in manifest["v10_completion_note"]
    # The note must not be the only place the distinction lives.
    assert manifest["migration_version"] == "v10"


def test_dry_run_moves_nothing_and_writes_no_manifest(project: Path):
    """"``dry_run=True`` moves nothing and writes no manifest file."""

    before = hash_tree(_original(project))

    manifest = migrate(project, SOURCE, dry_run=True)

    assert manifest["dry_run"] is True
    assert manifest["hashes_unchanged_by_move"] is True
    assert hash_tree(_original(project)) == before, "a dry run modified the source"
    assert not _destination(project).exists()
    assert not _manifest_path(project).exists()
    assert not (project / SOURCE.suite_directory).exists(),\
        "a dry run should not even create the suite directory"
    # And it must still be pending afterwards, so a real run can follow.
    assert migration_state(project, SOURCE) == "pending"


# ------------------------------------------------- direct array comparison
#
# The comparison is a *slice*, not a whole-array match: a sweep run saves a
# ``(cell, seed, method, group)`` tensor over all seven cells and all three
# methods, while the frozen source run saved ``(seed, arm, group)`` for its two
# trained arms only.  The fixtures below therefore build the two shapes
# separately, and the frozen side is written by indexing the sweep side
# explicitly here -- not by calling ``anchor_arrays`` -- so the tests pin the
# reduction rather than restate it.

#: The seven sweep cells, labelled the way ``split.json`` records them.
SWEEP_CELL_LABELS = [f"{protocol}@{beta}" for protocol, beta in EXPECTED_CELLS]
ANCHOR_INDEX = SWEEP_CELL_LABELS.index("passive@inf")

N_CELLS, N_SEEDS, N_METHODS, N_ARMS, N_GROUPS = len(SWEEP_CELL_LABELS), 3, 3, 2, 2


def _ramp(shape: tuple[int, ...], offset: int, *, integer: bool = False) -> np.ndarray:
    """Distinct values in every position, so any misindexing shows up."""

    values = np.arange(int(np.prod(shape))) + offset
    return values.reshape(shape).astype(np.int64 if integer else float)


def _sweep_arrays(offset: int) -> dict[str, np.ndarray]:
    """One split's arrays at *sweep* rank: cell-indexed, three methods wide."""

    cell_seed_method = (N_CELLS, N_SEEDS, N_METHODS)
    cell_seed_method_group = (*cell_seed_method, N_GROUPS)
    arrays = {
        "worst_group_distortion": _ramp(cell_seed_method, offset + 1_000),
        "group_distortion": _ramp(cell_seed_method_group, offset + 2_000),
        "welfare": _ramp(cell_seed_method_group, offset + 3_000),
        "welfare_optimum": _ramp((N_CELLS, N_GROUPS), offset + 4_000),
        "worst_group_index": _ramp(cell_seed_method, offset + 5, integer=True),
        "distortion_status": _ramp(cell_seed_method_group, offset + 6, integer=True),
        "seed_difference": _ramp((N_CELLS, N_SEEDS), offset + 7_000),
        # The two count arrays are per-split, not per-cell: they carry no cell
        # axis on either side and must be compared as saved.
        "train_group_counts": _ramp((N_GROUPS,), offset + 8, integer=True),
        "test_group_counts": _ramp((N_GROUPS,), offset + 9, integer=True),
    }
    # Diagnostics the sweep also saves.  They are outside ANCHOR_ARRAY_MAP and
    # must simply be ignored by the comparison.
    arrays.update({
        f"label_{key}": _ramp((N_CELLS, N_SEEDS, N_ARMS), offset + index)
        for index, key in enumerate(DIAGNOSTIC_KEYS)
    })
    return arrays


def _frozen_arrays(sweep: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The values the anchor slice must reduce to, under the *frozen* names.

    Written out index by index: the anchor cell is selected, and the untrained
    uniform method is dropped by keeping only the two fitted arms.
    """

    cell = ANCHOR_INDEX
    return {
        "worst_group_distortion": sweep["worst_group_distortion"][cell][:, :N_ARMS],
        # The frozen run called this one ``distortion``.
        "distortion": sweep["group_distortion"][cell][:, :N_ARMS, :],
        "welfare": sweep["welfare"][cell][:, :N_ARMS, :],
        "welfare_optimum": sweep["welfare_optimum"][cell],
        "worst_group_index": sweep["worst_group_index"][cell][:, :N_ARMS],
        "distortion_status": sweep["distortion_status"][cell][:, :N_ARMS, :],
        "seed_difference": sweep["seed_difference"][cell],
        "train_group_counts": sweep["train_group_counts"],
        "test_group_counts": sweep["test_group_counts"],
    }


def build_sweep_run(
    run_dir: Path,
    *,
    n_splits: int = 4,
    cell_labels: list[str] | None = None,
) -> Path:
    """A sweep-shaped run: seven cells, three methods, saved per split."""

    for offset, name in enumerate(_split_names(n_splits)):
        split_dir = run_dir / "splits" / name
        split_dir.mkdir(parents=True, exist_ok=True)
        _write_json(split_dir / "split.json", {
            "split": name,
            "methods": list(SWEEP_METHODS),
            "cell_labels": list(SWEEP_CELL_LABELS if cell_labels is None else cell_labels),
        })
        np.savez(split_dir / "split.npz", **_sweep_arrays(offset))
    return run_dir


def build_frozen_run(run_dir: Path, *, n_splits: int = 4) -> Path:
    """The frozen source run: two trained arms, no cell axis, frozen names."""

    for offset, name in enumerate(_split_names(n_splits)):
        split_dir = run_dir / "splits" / name
        split_dir.mkdir(parents=True, exist_ok=True)
        np.savez(split_dir / "split.npz", **_frozen_arrays(_sweep_arrays(offset)))
    return run_dir


def _edit_npz(path: Path, edit) -> None:
    """Rewrite one saved bundle with a single deliberate change."""

    with np.load(path, allow_pickle=False) as bundle:
        arrays = {name: bundle[name] for name in bundle.files}
    edit(arrays)
    np.savez(path, **arrays)


def test_compare_with_frozen_arrays_agrees_on_identical_runs(tmp_path: Path):
    """The anchor slice of a sweep run matches the frozen run it was seeded from.

    The two trees are written independently and differ in rank, name and axis
    order; agreement can only come from performing the anchor reduction.  The
    result has to say it reached that conclusion by reading files.
    """

    run = build_sweep_run(tmp_path / "run", n_splits=4)
    frozen = build_frozen_run(tmp_path / "frozen", n_splits=4)

    result = compare_with_frozen_arrays(run, frozen)

    assert result["agrees"] is True, result
    assert result["splits_compared"] == 4
    assert result["arrays_compared"] == 4 * len(ANCHOR_ARRAY_MAP),\
        "every mapped array in every split should be compared"
    assert result["mismatches"] == [] and result["missing"] == []
    assert result["anchor_cell"] == "passive@inf"


def test_compare_with_frozen_arrays_records_that_nothing_was_recomputed(tmp_path: Path):
    """The result carries its own provenance: files read, uniform axis excluded."""

    run = build_sweep_run(tmp_path / "run", n_splits=1)
    frozen = build_frozen_run(tmp_path / "frozen", n_splits=1)

    result = compare_with_frozen_arrays(run, frozen)

    assert "nothing recomputed" in result["method"]
    assert "no route rerun" in result["method"]
    excluded = " ".join(result["excluded_from_equality"])
    assert "uniform" in excluded, "the untrained benchmark axis must be declared excluded"


def test_compare_with_frozen_arrays_reports_the_specific_mismatch(tmp_path: Path):
    """A single differing anchor-cell value must be named, not summarised away."""

    run = build_sweep_run(tmp_path / "run", n_splits=4)
    frozen = build_frozen_run(tmp_path / "frozen", n_splits=4)
    # One seed, one trained arm, one group, inside the anchor cell.
    _edit_npz(run / "splits" / "r00_f02" / "split.npz",
              lambda arrays: arrays["group_distortion"].__setitem__(
                  (ANCHOR_INDEX, 0, 0, 0), 99.0))

    result = compare_with_frozen_arrays(run, frozen)

    assert result["agrees"] is False
    assert result["mismatches"] == ["r00_f02:group_distortion"], result["mismatches"]
    assert result["missing"] == []


def test_compare_with_frozen_arrays_ignores_non_anchor_cells_and_the_uniform_arm(
    tmp_path: Path,
):
    """Everything outside the anchor slice is irrelevant to the check.

    This is the property the whole-array comparison got wrong: it compared a
    seven-cell three-method tensor against a one-cell two-arm one and reported
    a mismatch on nearly every array.  Changing the six other cells, and the
    untrained uniform column of the anchor cell itself, must change nothing --
    the frozen run never computed either.
    """

    run = build_sweep_run(tmp_path / "run", n_splits=3)
    frozen = build_frozen_run(tmp_path / "frozen", n_splits=3)

    def scramble(arrays: dict[str, np.ndarray]) -> None:
        for name in ("worst_group_distortion", "group_distortion", "welfare",
                     "distortion_status", "worst_group_index"):
            # Every cell but the anchor, at every method.
            for cell in range(N_CELLS):
                if cell != ANCHOR_INDEX:
                    arrays[name][cell] = -7
            # And the uniform method axis of the anchor cell itself.
            arrays[name][ANCHOR_INDEX, :, N_ARMS:] = -7
        for name in ("welfare_optimum", "seed_difference"):
            for cell in range(N_CELLS):
                if cell != ANCHOR_INDEX:
                    arrays[name][cell] = -7

    for name in _split_names(3):
        _edit_npz(run / "splits" / name / "split.npz", scramble)

    result = compare_with_frozen_arrays(run, frozen)

    assert result["agrees"] is True, result
    assert result["mismatches"] == [], "a non-anchor cell or the uniform arm is not a mismatch"
    assert result["splits_compared"] == 3


def test_compare_with_frozen_arrays_reports_missing_counterparts(tmp_path: Path):
    """A frozen split that does not exist is reported, never skipped.

    Silently skipping would let a run "agree" with a frozen source it barely
    overlaps -- the failure mode this check exists to catch.
    """

    run = build_sweep_run(tmp_path / "run", n_splits=4)
    frozen = build_frozen_run(tmp_path / "frozen", n_splits=4)
    shutil.rmtree(frozen / "splits" / "r00_f01")

    result = compare_with_frozen_arrays(run, frozen)

    assert result["agrees"] is False, "a missing counterpart must not count as agreement"
    assert result["missing"] == ["r00_f01: no frozen counterpart"]
    assert result["splits_compared"] == 3, "the remaining splits are still compared"


def test_compare_with_frozen_arrays_reports_a_named_array_that_is_absent(tmp_path: Path):
    """A mapped array the frozen run never saved lands in ``missing``.

    The name reported is the *frozen* one, since that is the side at fault.
    """

    run = build_sweep_run(tmp_path / "run", n_splits=2)
    frozen = build_frozen_run(tmp_path / "frozen", n_splits=2)
    for name in _split_names(2):
        _edit_npz(frozen / "splits" / name / "split.npz",
                  lambda arrays: arrays.pop("welfare_optimum"))

    result = compare_with_frozen_arrays(run, frozen)

    assert result["arrays_compared"] == 2 * (len(ANCHOR_ARRAY_MAP) - 1),\
        "only the arrays both runs saved are comparable"
    assert result["missing"] == ["r00_f00:welfare_optimum (frozen)",
                                 "r00_f01:welfare_optimum (frozen)"]
    assert result["agrees"] is False


def test_compare_with_frozen_arrays_reports_a_mapped_array_absent_from_the_sweep(
    tmp_path: Path,
):
    """A sweep split that cannot even be sliced is reported, not raised."""

    run = build_sweep_run(tmp_path / "run", n_splits=1)
    frozen = build_frozen_run(tmp_path / "frozen", n_splits=1)
    _edit_npz(run / "splits" / "r00_f00" / "split.npz",
              lambda arrays: arrays.pop("seed_difference"))

    result = compare_with_frozen_arrays(run, frozen)

    assert result["agrees"] is False
    assert len(result["missing"]) == 1
    assert "seed_difference" in result["missing"][0]
    assert result["splits_compared"] == 0, "an unsliceable split was not compared"


def test_compare_with_frozen_arrays_reports_a_split_without_the_anchor_cell(tmp_path: Path):
    """A split whose ``cell_labels`` lack the anchor cell is reported, not guessed at.

    Falling back to cell zero would compare ``passive@0.5`` against the frozen
    deterministic run and call the difference a mismatch -- or, worse, agree.
    """

    labels = [label for label in SWEEP_CELL_LABELS if label != "passive@inf"]
    run = build_sweep_run(tmp_path / "run", n_splits=2, cell_labels=labels)
    frozen = build_frozen_run(tmp_path / "frozen", n_splits=2)

    result = compare_with_frozen_arrays(run, frozen)

    assert result["agrees"] is False
    assert result["missing"] == ["r00_f00: no 'passive@inf' cell",
                                 "r00_f01: no 'passive@inf' cell"]
    assert result["splits_compared"] == 0 and result["arrays_compared"] == 0


def test_compare_with_frozen_arrays_accepts_another_anchor_cell(tmp_path: Path):
    """The anchor cell is a parameter, and selecting a different one selects it."""

    run = build_sweep_run(tmp_path / "run", n_splits=2)
    frozen = tmp_path / "frozen"
    other = SWEEP_CELL_LABELS.index("active@4")
    for offset, name in enumerate(_split_names(2)):
        sweep = _sweep_arrays(offset)
        split_dir = frozen / "splits" / name
        split_dir.mkdir(parents=True, exist_ok=True)
        np.savez(split_dir / "split.npz", **{
            "worst_group_distortion": sweep["worst_group_distortion"][other][:, :N_ARMS],
            "distortion": sweep["group_distortion"][other][:, :N_ARMS, :],
            "welfare": sweep["welfare"][other][:, :N_ARMS, :],
            "welfare_optimum": sweep["welfare_optimum"][other],
            "worst_group_index": sweep["worst_group_index"][other][:, :N_ARMS],
            "distortion_status": sweep["distortion_status"][other][:, :N_ARMS, :],
            "seed_difference": sweep["seed_difference"][other],
            "train_group_counts": sweep["train_group_counts"],
            "test_group_counts": sweep["test_group_counts"],
        })

    assert compare_with_frozen_arrays(run, frozen, anchor_cell="active@4")["agrees"] is True
    # ...and the default anchor, whose values differ, disagrees against it.
    assert compare_with_frozen_arrays(run, frozen)["agrees"] is False


def test_compare_with_frozen_arrays_reports_an_absent_frozen_run(tmp_path: Path):
    """A frozen directory that does not exist is a reported disagreement."""

    run = build_sweep_run(tmp_path / "run", n_splits=2)

    result = compare_with_frozen_arrays(run, tmp_path / "nowhere")

    assert result["agrees"] is False
    assert any("does not exist" in item for item in result["missing"])


def test_compare_with_frozen_arrays_reports_a_run_without_splits(tmp_path: Path):
    """A run with no ``splits/`` directory reports rather than raising."""

    run = tmp_path / "run"
    run.mkdir()
    frozen = build_frozen_run(tmp_path / "frozen", n_splits=1)

    result = compare_with_frozen_arrays(run, frozen)

    assert result["agrees"] is False
    assert any("no splits/ directory" in item for item in result["missing"])
    assert result["splits_compared"] == 0


# --------------------------------------------------------------------- states

def test_migration_state_pending(project: Path):
    assert migration_state(project, SOURCE) == "pending"


def test_migration_state_complete(project: Path):
    migrate(project, SOURCE)
    assert migration_state(project, SOURCE) == "complete"


def test_migration_state_partial(project: Path):
    """Bytes at the destination but no manifest: the move ran, the record did not."""

    _destination(project).parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(_original(project)), str(_destination(project)))
    assert migration_state(project, SOURCE) == "partial"


def test_migration_state_blocked_when_both_sides_exist(project: Path):
    build_run(_destination(project), score=9.0)
    assert migration_state(project, SOURCE) == "blocked"


def test_migration_state_blocked_when_neither_side_exists(tmp_path: Path):
    """Nothing to migrate and nothing migrated is a blockage, not a silent success."""

    assert migration_state(tmp_path, SOURCE) == "blocked"
    with pytest.raises(MigrationError, match="does not exist"):
        migrate(tmp_path, SOURCE)


def test_both_declared_source_runs_are_distinct_namespaces():
    """The two entries must not share a path, or one would migrate over the other."""

    paths = {run.original_path for run in SOURCE_RUNS.values()}
    destinations = {run.suite_directory for run in SOURCE_RUNS.values()}
    assert len(paths) == len(SOURCE_RUNS) == len(destinations)
    for run in SOURCE_RUNS.values():
        assert run.protocol_id != run.suite_protocol_id


def test_persona_run_migrates_with_its_own_ids(tmp_path: Path):
    """The module is not hard-wired to the coffee run: the persona entry works too."""

    persona = SOURCE_RUNS["persona"]
    build_run(tmp_path / persona.original_path, source=persona, n_splits=4)
    # A short run still has to fail the split count, so pin that first...
    with pytest.raises(MigrationError, match="4 saved splits"):
        migrate(tmp_path, persona)
    # ...then migrate a full one and check the ids that travelled with it.
    shutil.rmtree(tmp_path / persona.original_path)
    build_run(tmp_path / persona.original_path, source=persona)

    manifest = migrate(tmp_path, persona)

    assert manifest["dataset"] == "persona"
    assert manifest["source_protocol_id"] == persona.protocol_id
    assert manifest["v10_protocol_id"] == persona.suite_protocol_id
    assert manifest["historical_ids_preserved"]["protocol_id"] == persona.protocol_id
