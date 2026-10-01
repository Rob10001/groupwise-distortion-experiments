"""Validate and import saved beta runs for comparison-suite reuse.

Import checks saved arrays and manifests without fitting policies. Existing
destinations are verified before continuation; conflicting contents are refused.
Suite completion is recorded by the comparison runner."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from empirical_testing.methods import (
    RetiredMethodError,
    UnknownMethodError,
    canonical,
)

#: Filename of the manifest this module writes into the suite beta directory.
IMPORT_MANIFEST = "import_manifest.json"

#: The immutable subdirectory the source run is relocated into.
SOURCE_SUBDIR = "source"

#: Version stamped into the manifest.
MIGRATION_VERSION = "v10"

#: The methods a completed beta run is expected to carry.  NLHF is deliberately
#: absent: neither run contains it, and the suite adds those fits separately rather
#: than pretending they were imported.
EXPECTED_SOURCE_METHODS = ("pooled", "multigroup", "uniform")

#: The seven ``(protocol, beta)`` cells the beta grid declares.
EXPECTED_CELLS = (
    ("passive", "0.5"), ("passive", "1"), ("passive", "2"),
    ("passive", "4"), ("passive", "inf"),
    ("active", "4"), ("active", "inf"),
)

EXPECTED_SPLITS = 100


class MigrationError(RuntimeError):
    """A migration precondition failed.  Nothing has been moved."""


@dataclass(frozen=True)
class SourceRun:
    """One completed beta run and where the suite keeps it."""

    dataset: str
    #: Historical run location retained for provenance import tooling.
    original_path: str
    #: The suite beta directory this run becomes the ``source/`` of.
    suite_directory: str
    #: The sweep protocol id the run must declare.
    protocol_id: str
    #: The suite protocol id that now owns this directory.
    suite_protocol_id: str


SOURCE_RUNS: Mapping[str, SourceRun] = {
    "coffee": SourceRun(
        dataset="coffee",
        original_path="results/coffee-crossfit-v4-protocol-beta-run1",
        suite_directory="results/coffee/beta-source",
        protocol_id="coffee-crossfit-v4-protocol-beta-sweep",
        suite_protocol_id="coffee-protocol-beta-v10",
    ),
    "persona": SourceRun(
        dataset="persona",
        original_path="results/persona-fixed-v8-protocol-beta-run1",
        suite_directory="results/persona/beta-source",
        protocol_id="persona-fixed-v8-protocol-beta-sweep",
        suite_protocol_id="persona-protocol-beta-v10",
    ),
}


# --------------------------------------------------------------------- hashing

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


#: Operating-system metadata that is not part of a run.  macOS writes
#: ``.DS_Store`` into any directory a Finder window touches, which would
#: otherwise show up as a difference against the import manifest and look like
#: the immutable source had been modified.  Excluded from hashing so the
#: comparison stays about the run's own artifacts.
NON_ARTIFACT_NAMES = frozenset({".DS_Store", "Thumbs.db", ".localized"})


def hash_tree(root: Path) -> dict[str, str]:
    """``{path relative to root: sha256}`` for every run artifact beneath ``root``."""

    return {
        str(path.relative_to(root)).replace(os.sep, "/"): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in NON_ARTIFACT_NAMES
    }


# ---------------------------------------------------------------- verification

@dataclass
class VerificationReport:
    """What the source run was checked for, and what it said."""

    dataset: str
    path: str
    checks: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def as_json(self) -> dict[str, Any]:
        return {"dataset": self.dataset, "path": self.path, "ok": self.ok,
                "checks": self.checks, "failures": list(self.failures)}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_source(run_dir: Path, source: SourceRun) -> VerificationReport:
    """Check a completed beta run without recomputing any of it.

    Reads the run's own manifests and saved splits.  Every failure is collected
    rather than raised, so one call reports everything that is wrong.
    """

    report = VerificationReport(dataset=source.dataset, path=str(run_dir))

    def fail(message: str) -> None:
        report.failures.append(message)

    if not run_dir.is_dir():
        fail(f"{run_dir} does not exist")
        return report

    info_path = run_dir / "run_info.json"
    if not info_path.is_file():
        fail("run_info.json is missing; this is not a completed run directory")
        return report
    info = _read_json(info_path)

    report.checks["status"] = info.get("status")
    if info.get("status") != "completed":
        fail(f"run status is {info.get('status')!r}, not 'completed'")

    report.checks["protocol_id"] = info.get("protocol_id")
    if info.get("protocol_id") != source.protocol_id:
        fail(f"protocol_id is {info.get('protocol_id')!r}, expected {source.protocol_id!r}")

    report.checks["implementation_id"] = info.get("implementation_id")
    report.checks["frozen_run"] = info.get("frozen_run")

    declared = tuple(sorted((str(c["protocol"]), str(c["beta"])) for c in info.get("cells", [])))
    report.checks["cells"] = [f"{p}@{b}" for p, b in declared]
    if declared != tuple(sorted(EXPECTED_CELLS)):
        fail(f"declared cells {report.checks['cells']} are not the expected seven-cell grid")

    # --- split coverage and method identity, read from the saved units -------
    splits_dir = run_dir / "splits"
    split_dirs = sorted(p for p in splits_dir.iterdir() if p.is_dir()) if splits_dir.is_dir() else []
    report.checks["splits"] = len(split_dirs)
    if len(split_dirs) != EXPECTED_SPLITS:
        fail(f"{len(split_dirs)} saved splits, expected {EXPECTED_SPLITS}")

    methods_seen: set[str] = set()
    cells_seen: set[str] = set()
    incomplete: list[str] = []
    for split_dir in split_dirs:
        unit = split_dir / "split.json"
        if not unit.is_file() or not (split_dir / "split.npz").is_file():
            incomplete.append(split_dir.name)
            continue
        payload = _read_json(unit)
        methods_seen.update(str(m) for m in payload.get("methods", ()))
        for cell in payload.get("cells", ()):
            # Saved either as a label or as a ``{protocol, beta, cell}`` record.
            cells_seen.add(str(cell["cell"]) if isinstance(cell, dict) else str(cell))
    if incomplete:
        fail(f"{len(incomplete)} split(s) are missing split.json or split.npz: {incomplete[:5]}")

    report.checks["methods"] = sorted(methods_seen)
    resolved: set[str] = set()
    unresolvable: list[str] = []
    for method in sorted(methods_seen):
        try:
            resolved.add(canonical(method))
        except (RetiredMethodError, UnknownMethodError) as error:
            # A retired or unknown name is a *finding about the source run*, not
            # a crash in the migration: verify_source reports every problem in
            # one pass, so this is collected like any other failure.
            unresolvable.append(method)
            fail(f"saved method {method!r} does not name a permitted family: {error}")
    report.checks["methods_canonical"] = sorted(resolved)
    report.checks["methods_unresolvable"] = unresolvable
    if resolved != set(EXPECTED_SOURCE_METHODS):
        fail(f"saved methods {sorted(resolved)} are not {sorted(EXPECTED_SOURCE_METHODS)}")
    report.checks["cells_in_splits"] = sorted(cells_seen)

    # --- design identity -----------------------------------------------------
    design_path = run_dir / "design" / "design.json"
    if design_path.is_file():
        design = _read_json(design_path)
        report.checks["design_protocol_id"] = design.get("protocol_id")
        report.checks["design_config_sha256"] = design.get("config_sha256") or info.get("config_sha256")
    else:
        report.checks["design_protocol_id"] = None
    report.checks["config_sha256"] = info.get("config_sha256")

    # --- the source run's own anchor report, read not rerun ------------------
    anchor_path = run_dir / "reduce" / "anchor_check.json"
    if not anchor_path.is_file():
        fail("reduce/anchor_check.json is missing; the anchor gate was never recorded")
    else:
        anchor = _read_json(anchor_path)
        report.checks["anchor"] = {
            "checked": anchor.get("checked"),
            "agrees": anchor.get("agrees"),
            "anchor_cell": anchor.get("anchor_cell"),
            "arrays_compared": anchor.get("arrays_compared"),
            "array_mismatches": anchor.get("array_mismatches"),
            "frozen_run": anchor.get("frozen_run"),
        }
        if not anchor.get("checked"):
            fail("the anchor gate was not run for this source")
        if not anchor.get("agrees"):
            fail(f"the anchor gate did not agree: {anchor.get('array_mismatches')}")

    return report


def compare_with_frozen_arrays(run_dir: Path, frozen_dir: Path, *,
                               anchor_cell: str = "passive@inf") -> dict[str, Any]:
    """Compare the anchor cell's saved arrays with the frozen source arrays.

    This is the anchor check done by **reading files**, never by rerunning a
    route: no policy is fitted and no split is recomputed.

    The comparison is a slice, not a whole-array match.  A sweep run stores a
    ``(cell, seed, method, group)`` tensor over all seven cells and three
    methods, while the frozen run stores ``(seed, arm, group)`` for its two
    trained arms only.  Comparing the raw arrays would compare a seven-cell
    tensor against a one-cell one and report a mismatch on every split, so the
    anchor cell is selected and reduced onto the frozen run's own axes first --
    using the sweep's own :func:`anchor_arrays`, so this cannot drift from the
    reduction the production gate used.  The untrained uniform axis is excluded
    because the frozen run never computed it.
    """

    import numpy as np

    from empirical_testing.experiments.coffee_protocol_beta_sweep import (
        ANCHOR_ARRAY_MAP, anchor_arrays,
    )

    result: dict[str, Any] = {
        "run": str(run_dir), "frozen_run": str(frozen_dir), "anchor_cell": anchor_cell,
        "method": ("direct comparison of the anchor cell's saved arrays with the frozen "
                   "run's saved arrays; nothing recomputed, no route rerun"),
        "excluded_from_equality": [
            "the untrained uniform benchmark axis (the frozen run never fitted it)",
            "timings, seed records and route-specific metadata",
        ],
        "splits_compared": 0, "arrays_compared": 0, "mismatches": [], "missing": [],
    }
    if not frozen_dir.is_dir():
        result["missing"].append(f"frozen run {frozen_dir} does not exist")
        result["agrees"] = False
        return result
    splits_root = run_dir / "splits"
    if not splits_root.is_dir():
        result["missing"].append(f"{run_dir} has no splits/ directory")
        result["agrees"] = False
        return result

    for split_dir in sorted(p for p in splits_root.iterdir() if p.is_dir()):
        frozen_split = frozen_dir / "splits" / split_dir.name
        if not (frozen_split / "split.npz").is_file():
            result["missing"].append(f"{split_dir.name}: no frozen counterpart")
            continue
        labels = _read_json(split_dir / "split.json").get("cell_labels", [])
        if anchor_cell not in labels:
            result["missing"].append(f"{split_dir.name}: no {anchor_cell!r} cell")
            continue
        index = list(labels).index(anchor_cell)
        with np.load(split_dir / "split.npz", allow_pickle=False) as here,\
                np.load(frozen_split / "split.npz", allow_pickle=False) as there:
            try:
                ours = anchor_arrays({name: here[name] for name in here.files}, index)
            except KeyError as error:
                result["missing"].append(f"{split_dir.name}: {error}")
                continue
            for name, frozen_name in ANCHOR_ARRAY_MAP.items():
                if frozen_name not in there.files:
                    result["missing"].append(f"{split_dir.name}:{frozen_name} (frozen)")
                    continue
                result["arrays_compared"] += 1
                if not np.array_equal(ours[name], there[frozen_name]):
                    result["mismatches"].append(f"{split_dir.name}:{name}")
        result["splits_compared"] += 1

    result["agrees"] = not result["mismatches"] and not result["missing"]
    return result


# ------------------------------------------------------------------ relocation

def migration_state(project_root: Path, source: SourceRun) -> str:
    """``"pending"``, ``"complete"``, ``"partial"`` or ``"blocked"``."""

    original = project_root / source.original_path
    destination = project_root / source.suite_directory / SOURCE_SUBDIR
    manifest = project_root / source.suite_directory / IMPORT_MANIFEST

    if manifest.is_file() and destination.is_dir() and not original.exists():
        return "complete"
    if destination.exists() and original.exists():
        # Both sides present: a relocation that did not finish, or a genuine
        # collision.  Never resolved by deleting either one.
        return "blocked"
    if destination.exists() and not manifest.is_file():
        return "partial"
    if original.is_dir():
        return "pending"
    return "blocked"


def migrate(project_root: Path, source: SourceRun, *, dry_run: bool = False) -> dict[str, Any]:
    """Relocate one completed beta run into its suite ``source/`` directory.

    Returns the manifest.  Raises :class:`MigrationError` before moving
    anything if the source does not verify or the destination is occupied.
    """

    project_root = Path(project_root)
    original = project_root / source.original_path
    suite_dir = project_root / source.suite_directory
    destination = suite_dir / SOURCE_SUBDIR
    manifest_path = suite_dir / IMPORT_MANIFEST

    state = migration_state(project_root, source)
    if state == "complete":
        # Idempotent continuation: already done, and verified when it was done.
        return _read_json(manifest_path) | {"already_complete": True}
    if state == "blocked":
        if destination.exists() and original.exists():
            raise MigrationError(
                f"both {original} and {destination} exist; refusing to overwrite or merge. "
                f"Inspect them and remove neither automatically."
            )
        raise MigrationError(f"{original} does not exist and no migration is recorded")
    if state == "partial":
        # The bytes arrived but the manifest did not.  Finish by re-hashing what
        # is there and writing the manifest; nothing is moved twice.
        report = verify_source(destination, source)
        if not report.ok:
            raise MigrationError(
                f"a partially migrated {destination} does not verify: {report.failures}"
            )
        after = hash_tree(destination)
        manifest = _build_manifest(source, report, after, after, resumed=True)
        if not dry_run:
            _write_json(manifest_path, manifest)
        return manifest

    # --- pending: verify, hash, move, re-hash, manifest ----------------------
    report = verify_source(original, source)
    if not report.ok:
        raise MigrationError(
            f"{original} did not verify, so nothing was moved:\n  - "
            + "\n  - ".join(report.failures)
        )

    before = hash_tree(original)
    if dry_run:
        return _build_manifest(source, report, before, before, dry_run=True)

    suite_dir.mkdir(parents=True, exist_ok=True)
    if destination.exists():  # pragma: no cover - guarded by migration_state
        raise MigrationError(f"{destination} appeared while verifying; refusing to overwrite")
    shutil.move(str(original), str(destination))

    after = hash_tree(destination)
    if after != before:
        changed = sorted(set(before) ^ set(after)) or [
            name for name in before if before[name] != after.get(name)
        ]
        raise MigrationError(
            f"the relocation changed {len(changed)} file(s) under {destination}: {changed[:5]}"
        )

    manifest = _build_manifest(source, report, before, after)
    _write_json(manifest_path, manifest)
    _mark_immutable(destination)
    return manifest


def _build_manifest(source: SourceRun, report: VerificationReport,
                    before: Mapping[str, str], after: Mapping[str, str],
                    *, dry_run: bool = False, resumed: bool = False) -> dict[str, Any]:
    return {
        "migration_version": MIGRATION_VERSION,
        "dataset": source.dataset,
        "original_path": source.original_path,
        "new_path": f"{source.suite_directory}/{SOURCE_SUBDIR}",
        "source_protocol_id": source.protocol_id,
        "v10_protocol_id": source.suite_protocol_id,
        "historical_ids_preserved": {
            "protocol_id": report.checks.get("protocol_id"),
            "implementation_id": report.checks.get("implementation_id"),
            "frozen_run": report.checks.get("frozen_run"),
            "config_sha256": report.checks.get("config_sha256"),
        },
        "note": (
            "The source run's own protocol and implementation ids are recorded "
            "unchanged as historical ids. suite adopts these results; it did not "
            "retrain them, and no beta GLHF fit was performed to produce this "
            "import."
        ),
        "immutable": True,
        "validation": report.as_json(),
        "file_count": len(after),
        "file_sha256": dict(sorted(after.items())),
        "hashes_unchanged_by_move": dict(before) == dict(after),
        "dry_run": dry_run,
        "resumed_from_partial_migration": resumed,
        "v10_complete": False,
        "v10_completion_note": (
            "Importing the beta results is one prerequisite of the suite condition, "
            "not the whole of it; completion is marked only after the missing NLHF "
            "fits, evaluation, reduction and coverage checks pass."
        ),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _mark_immutable(destination: Path) -> None:
    """Drop the owner write bit on the imported tree, best effort.

    A convenience guard against an accidental in-place edit, not a security
    boundary: the manifest's hashes are what actually detect tampering.
    """

    try:
        for path in destination.rglob("*"):
            if path.is_file():
                mode = path.stat().st_mode
                path.chmod(mode & ~0o222)
    except OSError:  # pragma: no cover - platform dependent
        pass
