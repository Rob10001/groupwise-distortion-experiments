"""Privacy and file-boundary regressions for the submission builder."""

import importlib.util
import hashlib
import json
from pathlib import Path
import zipfile

import pytest


_spec = importlib.util.spec_from_file_location(
    "submission_builder", Path(__file__).resolve().parents[1] / "build_submission.py"
)
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


@pytest.mark.parametrize("path", [
    "/".join(["", "Users", "example", "research", "EmpiricalTesting", "results", "run", "summary.json"]),
    "\\".join(["C:", "Users", "example", "research", "EmpiricalTesting", "results", "run", "summary.json"]),
])
def test_anonymization_preserves_project_relative_location(path):
    result = builder.anonymous_json({"source": path, "estimate": 1.25})
    assert result["source"].replace("\\", "/") == "EmpiricalTesting/results/run/summary.json"
    assert result["estimate"] == 1.25
    assert not builder.LOCAL_PATH.search(result["source"])


@pytest.mark.parametrize("name", [".env", ".env.json", "credentials.yaml", "private.pem", "unknown"])
def test_unreviewed_or_secret_file_types_are_refused(name):
    with pytest.raises(ValueError, match="Unreviewed"):
        builder.check_file_type(Path(name))


@pytest.mark.parametrize("prefix,length", [("sk-proj-", 40), ("hf_", 35), ("AIza", 35), ("xoxb-", 30)])
def test_provider_credentials_are_detected_without_echoing_them(prefix, length):
    data = ('{"credential": "' + prefix + "a" * length + '"}').encode()
    assert "provider credential" in builder.inspect_text("artifact.json", data)


def test_nested_revision_metadata_is_anonymized_without_changing_scientific_hashes():
    result = builder.anonymous_json({"run": {"git_revision": "revision", "data_sha256": "data-hash"}})
    assert result["run"]["git_revision"] == "[revision omitted for anonymous submission]"
    assert result["run"]["data_sha256"] == "data-hash"


@pytest.fixture
def package_root(tmp_path, monkeypatch):
    root = tmp_path / "package"
    for name in ("README.md", "CITATION.cff", "LICENSE", "LICENSE-DATA.md", "RELEASE.md",
                 "requirements.txt", "pyproject.toml",
                 ".gitattributes", ".gitignore", "run_coffee.py", "run_persona.py",
                 "build_submission.py", "reproduce.py", "configs/coffee.json", "configs/persona.json",
                 "data/persona/ratings.csv", "data/coffee/raw/coffee_survey.csv",
                 "examples/minimal_glhf.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if path.suffix == ".json" else "fixture\n")
    for name in ("src", "tests", "scripts", "figures/paper", "figures/learning-curves",
                 "results/persona/panel-permutations"):
        (root / name).mkdir(parents=True)
    for dataset in ("coffee", "persona"):
        for part in ("comparison", "beta-source/source", "learning-curves", "figures"):
            directory = root / "results" / dataset / part
            directory.mkdir(parents=True)
            if part == "figures":
                continue
            split = directory / "splits/r00_f00"
            split.mkdir(parents=True)
            (split / "split.json").write_text('{"estimate": 1.25}')
            (split / "split.npz").write_bytes(b"frozen final arrays")
            if part != "learning-curves":
                (split / "comparisons.npz").write_bytes(b"bulky training log")
            (directory / "reduce").mkdir()
            (directory / "reduce/method_levels.csv").write_text("estimate\n1.25\n")
        (root / "results" / dataset / "beta-source/import_manifest.json").write_text("{}")
    monkeypatch.setattr(builder, "ROOT", root)
    return root


def test_submission_retains_inputs_endpoints_and_reductions_without_trajectories(package_root):
    selected = {p.relative_to(package_root).as_posix() for p in builder.selected_files()}
    for dataset in ("coffee", "persona"):
        assert f"results/{dataset}/beta-source/source/splits/r00_f00/split.npz" in selected
        assert f"results/{dataset}/comparison/splits/r00_f00/split.npz" in selected
        assert f"results/{dataset}/learning-curves/reduce/method_levels.csv" in selected
    assert "data/persona/ratings.csv" in selected
    assert "data/coffee/raw/coffee_survey.csv" in selected
    assert {"CITATION.cff", "LICENSE", "LICENSE-DATA.md", "RELEASE.md",
            "examples/minimal_glhf.py"} <= selected
    assert not any(p.endswith("comparisons.npz") or "learning-curves/splits/" in p for p in selected)
    full = builder.selected_files(include_trajectories=True)
    assert any(p.name == "comparisons.npz" for p in full)
    assert any("learning-curves" in p.parts and p.name == "split.npz" for p in full)


def test_built_archive_records_omissions_preserves_data_and_updates_hashes(package_root, tmp_path):
    output = tmp_path / "submission.zip"
    builder.build(output)
    with zipfile.ZipFile(output) as archive:
        prefix = "EmpiricalTesting/"
        manifest = json.loads(archive.read(prefix + "SUBMISSION_MANIFEST.json"))
        assert manifest["profile"] == "submission"
        for name in ("CITATION.cff", "LICENSE", "LICENSE-DATA.md", "RELEASE.md",
                     "examples/minimal_glhf.py"):
            assert archive.read(prefix + name) == (package_root / name).read_bytes()
        for dataset in ("coffee", "persona"):
            beta = prefix + f"results/{dataset}/beta-source/"
            meta = json.loads(archive.read(beta + "source/splits/r00_f00/split.json"))
            assert meta["submission_omitted_files"] == ["comparisons.npz"]
            assert meta["estimate"] == 1.25
            imported = json.loads(archive.read(beta + "import_manifest.json"))
            for name, digest in imported["file_sha256"].items():
                assert hashlib.sha256(archive.read(beta + "source/" + name)).hexdigest() == digest
            assert not any(p.endswith("comparisons.npz") for p in imported["file_sha256"])
        for name in ("data/persona/ratings.csv", "data/coffee/raw/coffee_survey.csv"):
            assert archive.read(prefix + name) == (package_root / name).read_bytes()
    assert output.with_suffix(".zip.sha256").read_text().split()[0] == builder.file_sha256(output)
    assert (package_root / "results/coffee/beta-source/source/splits/r00_f00/comparisons.npz").is_file()


def test_size_budget_does_not_replace_previous_archive(package_root, tmp_path):
    output = tmp_path / "submission.zip"
    output.write_bytes(b"previous archive")
    with pytest.raises(ValueError, match="exceeds"):
        builder.build(output, max_size_mb=0.000001)
    assert output.read_bytes() == b"previous archive"
