"""Build and verify the anonymous, self-contained empirical submission archive.

Only explicitly selected source, frozen inputs, and reference artifacts are
included. Research originals are never edited. Uses the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT
ARCHIVE_ROOT = "EmpiricalTesting"
EXCLUDED_PARTS = {"__pycache__", ".venv", ".git", ".pytest_cache", ".ruff_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".nbc", ".nbi", ".log", ".tmp"}
TEXT_SUFFIXES = {".py", ".json", ".md", ".csv", ".txt", ".toml"}
ALLOWED_SUFFIXES = TEXT_SUFFIXES | {".pdf", ".png", ".npz"}
LOCAL_PREFIX = re.compile(
    r"(?:[A-Za-z]:[\\/]+Users[\\/]+|/(?:Users|home)/)[^\r\n\"'<>]*?"
    r"(?=(?:EmpiricalTesting|experiment|figures|scripts)[\\/])"
)
LOCAL_PATH = re.compile(r"(?:[A-Za-z]:[\\/]+Users[\\/]+|/(?:Users|home)/)[^\r\n\"'<>]+")
SECRET_PATTERNS = {
    "provider credential": re.compile(r"\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{30,}|AIza[A-Za-z0-9_-]{35}|xox[baprs]-[A-Za-z0-9-]{20,})\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "email address": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}
REVISION_KEYS = {"git_revision", "git_commit", "git_head", "source_revision"}


def check_file_type(path: Path) -> None:
    if path.name in {".gitignore", ".gitattributes"}:
        return
    if path.name.lower().startswith(".env") or path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"Unreviewed submission file type: {path.name}")


def selected_files(*, include_trajectories: bool = False) -> list[Path]:
    """An allowlist prevents future local logs, credentials, or runs leaking in."""
    files: set[Path] = set()

    def add(relative: str, *, suffix: str | None = None) -> None:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(f"Required submission input is missing: {relative}")
        candidates = path.rglob("*") if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_symlink():
                raise ValueError(f"Refusing symbolic link: {candidate.relative_to(ROOT)}")
            if not candidate.is_file():
                continue
            rel = candidate.relative_to(ROOT)
            if not include_trajectories and rel.parts[0] == "results":
                if candidate.name == "comparisons.npz":
                    continue
                if "learning-curves" in rel.parts and "splits" in rel.parts:
                    continue
            if EXCLUDED_PARTS.intersection(rel.parts) or candidate.suffix in EXCLUDED_SUFFIXES:
                continue
            if candidate.name in {".DS_Store", "Thumbs.db", ".localized"}:
                continue
            if suffix is None or candidate.suffix == suffix:
                check_file_type(candidate)
                files.add(candidate)

    for name in ("README.md", "requirements.txt", "pyproject.toml", ".gitattributes",
                 ".gitignore", "run_coffee.py", "run_persona.py", "reproduce.py", "build_submission.py"):
        add(name)
    for directory in ("src", "tests", "scripts"):
        add(directory, suffix=".py")
    add("configs/coffee.json")
    add("configs/persona.json")
    add("data")
    for dataset in ("coffee", "persona"):
        for directory in ("comparison", "beta-source", "learning-curves", "figures"):
            add(f"results/{dataset}/{directory}")
    add("results/persona/panel-permutations")
    for directory in ("paper", "learning-curves"):
        add(f"figures/{directory}")
    return sorted(files)


def portable_text(value: str) -> str:
    value = LOCAL_PREFIX.sub("", value)
    # Paths outside the project are provenance metadata, never runtime inputs.
    return LOCAL_PATH.sub("[local path removed]", value)


def anonymous_json(value):
    if isinstance(value, dict):
        return {portable_text(key): ("[revision omitted for anonymous submission]"
                if key in REVISION_KEYS else anonymous_json(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [anonymous_json(item) for item in value]
    return portable_text(value) if isinstance(value, str) else value


def inspect_text(name: str, data: bytes) -> list[str]:
    if Path(name).suffix not in TEXT_SUFFIXES:
        return []
    value = data.decode("utf-8-sig")
    findings = []
    if LOCAL_PATH.search(value):
        findings.append("local user path")
    for category, pattern in SECRET_PATTERNS.items():
        for match in pattern.finditer(value):
            if category == "email address" and match.group().endswith(("@example.com", "@example.org")):
                continue
            findings.append(category)
            break
    return findings


def prepare_metadata(files: list[Path]) -> tuple[dict[str, bytes], dict[str, str]]:
    replacements: dict[str, bytes] = {}
    changes: dict[str, str] = {}
    selected = set(files)
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix not in TEXT_SUFFIXES:
            continue
        original = path.read_bytes()
        output = original
        # Preserve raw collection bytes and frozen design hashes exactly.
        if path.suffix == ".json" and ("/results/" in f"/{rel}" or rel.startswith("figures/")):
            value = json.loads(original)
            sanitized = anonymous_json(value)
            # Explicitly distinguish a deliberately slimmed source from a damaged
            # run. Reconstruction still verifies seed/order hashes and endpoints.
            if path.name == "split.json" and "beta-source" in path.parts:
                if path.with_name("comparisons.npz") not in selected:
                    sanitized["submission_omitted_files"] = ["comparisons.npz"]
            if sanitized != value:
                output = (json.dumps(sanitized, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        elif path.suffix == ".md":
            output = portable_text(original.decode("utf-8-sig")).encode("utf-8")
        if output != original:
            replacements[rel] = output
            changes[rel] = "Anonymized metadata or recorded omitted training logs; numerical values unchanged."
        findings = inspect_text(rel, output)
        if findings:
            raise ValueError(f"Submission audit failed in {rel}: {', '.join(findings)}")

    # Imported metadata can change during anonymization. Record checksums of the
    # distributed files, retaining the frozen scientific IDs and all array bytes.
    for path in files:
        if path.name != "import_manifest.json":
            continue
        rel = path.relative_to(ROOT).as_posix()
        source = path.parent / "source"
        payload = json.loads(replacements.get(rel, path.read_bytes()))
        hashes = {}
        for artifact in files:
            if not artifact.is_relative_to(source):
                continue
            item = artifact.relative_to(ROOT).as_posix()
            data = replacements.get(item)
            hashes[artifact.relative_to(source).as_posix()] = (
                hashlib.sha256(data).hexdigest() if data is not None else file_sha256(artifact)
            )
        payload["file_sha256"] = hashes
        payload["submission_metadata"] = {
            "anonymized": True,
            "note": "Local paths and source revision metadata were anonymized; run logs omitted. "
                    "This file_sha256 map describes the distributed source files. "
                    "Raw data and numerical arrays retain their original bytes. "
                    "Historical migration statements refer to the original migration.",
        }
        replacements[rel] = (json.dumps(payload, indent=2) + "\n").encode()
        changes[rel] = "Updated checksum map for the anonymous distributed source."
    return replacements, changes


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(output: Path, *, include_trajectories: bool = False,
          max_size_mb: float | None = 100.0) -> None:
    files = selected_files(include_trajectories=include_trajectories)
    print(f"Auditing {len(files)} selected files before packaging.", flush=True)
    replacements, changes = prepare_metadata(files)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    if temporary.exists():
        raise FileExistsError(f"Unfinished archive exists; inspect it before retrying: {temporary.name}")
    checksums = {}
    print(f"Packaging {len(files)} files; anonymizing {len(changes)} metadata files.", flush=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as archive:
        for index, path in enumerate(files):
            rel = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{rel}", date_time=(2026, 1, 1, 0, 0, 0))
            # NPZ, PDF, and PNG payloads already use compression.
            info.compress_type = (zipfile.ZIP_STORED if path.suffix in {".npz", ".pdf", ".png"}
                                  else zipfile.ZIP_DEFLATED)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            digest = hashlib.sha256()
            with archive.open(info, "w", force_zip64=True) as target:
                if rel in replacements:
                    data = replacements[rel]
                    target.write(data)
                    digest.update(data)
                else:
                    with path.open("rb") as source:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            target.write(chunk)
                            digest.update(chunk)
            checksums[rel] = digest.hexdigest()
            if index % 250 == 0:
                print(f"  {index + 1}/{len(files)} files", flush=True)
        manifest = {
            "format": 1,
            "profile": "full" if include_trajectories else "submission",
            "scope": "Current comparison suites, learning curves, panel sensitivity, frozen inputs, source and regression tests.",
            "excluded": ["virtual environments and caches", "API collection logs and credentials",
                         "historical runs", "duplicate code/data snapshots", "Git history",
                         "historical run logs", "superseded collections and workflows"],
            "anonymized_files": changes,
            "preservation": "Raw data, configurations, and included NPZ arrays are copied byte for byte. "
                            "SHA-256 checksums below identify the distributed files.",
            "sha256": checksums,
        }
        if not include_trajectories:
            manifest["excluded"] += ["per-comparison training logs (comparisons.npz)",
                                     "per-split reconstructed learning-curve arrays and metadata"]
        archive.writestr(f"{ARCHIVE_ROOT}/SUBMISSION_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    if max_size_mb is not None and temporary.stat().st_size > max_size_mb * 1e6:
        raise ValueError(f"Archive exceeds the {max_size_mb:g} MB packaging budget; "
                         f"previous archive preserved, inspect {temporary.name}")
    verify(temporary)
    temporary.replace(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8")
    print(f"Archive: {output.name} ({output.stat().st_size / 1e6:.2f} MB)", flush=True)


def verify(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read(f"{ARCHIVE_ROOT}/SUBMISSION_MANIFEST.json"))
        actual = set(archive.namelist())
        expected = {f"{ARCHIVE_ROOT}/{name}" for name in manifest["sha256"]}
        expected.add(f"{ARCHIVE_ROOT}/SUBMISSION_MANIFEST.json")
        if actual != expected or len(actual) != len(archive.namelist()):
            raise ValueError("Unexpected, missing, or duplicate ZIP entries")
        for name, expected_hash in manifest["sha256"].items():
            if ".." in Path(name).parts or Path(name).is_absolute():
                raise ValueError("Unsafe archive path")
            check_file_type(Path(name))
            digest = hashlib.sha256()
            with archive.open(f"{ARCHIVE_ROOT}/{name}") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                raise ValueError(f"Checksum mismatch: {name}")
            if Path(name).suffix in TEXT_SUFFIXES:
                findings = inspect_text(name, archive.read(f"{ARCHIVE_ROOT}/{name}"))
                if findings:
                    raise ValueError(f"Submission audit failed in {name}: {', '.join(findings)}")
    print(f"Verified {len(manifest['sha256'])} payload checksums and text privacy scan.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT.parent / "submission" / "EmpiricalTesting-submission.zip")
    parser.add_argument("--verify", type=Path, help="Verify an existing ZIP instead of building")
    parser.add_argument("--include-trajectories", action="store_true",
                        help="Include bulky training logs and reconstructed split arrays")
    parser.add_argument("--max-size-mb", type=float,
                        help="Decimal MB budget (default: 100 for submission; unlimited for full)")
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
    else:
        budget = args.max_size_mb if args.max_size_mb is not None else (
            None if args.include_trajectories else 100.0)
        if budget is not None and budget <= 0:
            parser.error("--max-size-mb must be positive")
        build(args.output, include_trajectories=args.include_trajectories, max_size_mb=budget)


if __name__ == "__main__":
    main()
