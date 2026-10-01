"""Regression checks for the narrowly scoped suite trajectory driver."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from empirical_testing.experiments import learning_curves as curves
from empirical_testing.experiments.reduce import corrected


def test_missing_logs_require_explicit_submission_omission(tmp_path):
    with pytest.raises(FileNotFoundError, match="undeclared"):
        curves._source_files(tmp_path, {})
    meta = {"submission_omitted_files": ["comparisons.npz"]}
    assert curves._source_files(tmp_path, meta) == ("split.json", "split.npz")
    (tmp_path / "comparisons.npz").write_bytes(b"present")
    assert curves._source_files(tmp_path, meta) == ("split.json", "split.npz", "comparisons.npz")


def test_reduce_maxes_before_seed_mean_and_pairs_split_differences(tmp_path, monkeypatch):
    monkeypatch.setattr(curves, "FRACTIONS", (0.0, 1.0))
    records = []
    expected = []
    for split, offset in enumerate((0.0, 0.2, -0.1, 0.3)):
        # Axes: checkpoint, seed, protocol, group. The seeds' worst groups
        # differ, so max(mean over seeds) gives the wrong split score.
        group = np.array(
            [
                [[[1.0, 3.0], [2.0, 2.0]], [[4.0, 1.0], [1.0, 3.0]]],
                [[[2.0 + offset, 5.0], [3.0 + offset, 2.0]],
                 [[6.0, 2.0 + offset], [2.0, 4.0 + offset]]],
            ]
        )
        worst = group.max(axis=-1)
        directory = tmp_path / "splits" / f"r00_f{split:02d}"
        directory.mkdir(parents=True)
        np.savez_compressed(
            directory / "split.npz",
            group_distortion=group,
            worst_group_distortion=worst,
            checkpoints=np.array([0, 100]),
        )
        records.append({
            "repeat": 0, "fold": split, "n_test": 20, "n_train": 80,
            "groups": ["a", "b"], "identity": {"learner_seeds": [11, 12]},
        })
        expected.append(worst[1].mean(axis=0))
    curves.reduce_curves(tmp_path, records, "coffee")
    levels = pd.read_csv(tmp_path / "reduce" / "method_levels.csv")
    split_scores = pd.read_csv(tmp_path / "reduce" / "split_scores.csv")
    contrasts = pd.read_csv(tmp_path / "reduce" / "protocol_contrasts.csv")
    final = split_scores[split_scores.checkpoint_fraction == 1.0]
    expected = np.asarray(expected)
    np.testing.assert_allclose(final.passive, expected[:, 0])
    np.testing.assert_allclose(final.active, expected[:, 1])
    assert final.iloc[0].passive == 5.5  # Mean of maxima 5 and 6.
    assert final.iloc[0].passive != 4.0  # Max of seed means would be 4.
    np.testing.assert_allclose(final.passive_minus_active, expected[:, 0] - expected[:, 1])
    paired = corrected(expected[:, 0] - expected[:, 1], [20] * 4, [80] * 4, .95)
    observed = contrasts[contrasts.checkpoint_fraction == 1.0].iloc[0]
    for key in ("estimate", "corrected_se", "ci_lower", "ci_upper"):
        assert observed[key] == pytest.approx(paired[key])
    assert levels[levels.checkpoint_fraction == 1.0].shape[0] == 2


def test_resolve_source_rejects_missing_and_smoke(tmp_path):
    with pytest.raises(FileNotFoundError):
        curves.resolve_source(tmp_path, "coffee", tmp_path / "missing")
    source = tmp_path / "source"
    source.mkdir()
    for info in ({"reportable": False, "smoke": False},
                 {"reportable": True, "smoke": True}):
        (source / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
        with pytest.raises(ValueError, match="reportable source"):
            curves.resolve_source(tmp_path, "coffee", source)
    (source / "run_info.json").write_text(
        json.dumps({"reportable": True, "smoke": False}), encoding="utf-8"
    )
    assert curves.resolve_source(tmp_path, "coffee", source) == source


def test_fresh_smoke_source_can_only_feed_smoke_curves(tmp_path):
    (tmp_path / "run_info.json").write_text(json.dumps(
        {"status": "completed", "reportable": False, "smoke": True}))
    assert curves.resolve_source(tmp_path, "coffee", tmp_path, smoke=True) == tmp_path
    with pytest.raises(ValueError, match="reportable"):
        curves.resolve_source(tmp_path, "coffee", tmp_path)


def test_endpoint_failure_blocks_reduction_and_completion(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "run_info.json").write_text(
        json.dumps({"reportable": True, "smoke": False}), encoding="utf-8"
    )
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    base_config = SimpleNamespace(
        comparisons_per_respondent=1,
        as_mapping=dict,
    )
    plan = SimpleNamespace(
        base_config=base_config,
        learner_seeds=[1, 2],
        inputs=SimpleNamespace(plan=SimpleNamespace(splits=lambda: [(0, 0)])),
    )
    monkeypatch.setattr(curves.engine, "build_plan", lambda *a, **kw: plan)

    def failed_endpoint(*args, **kwargs):
        raise AssertionError("reconstructed endpoint differs from frozen result")

    def forbidden(*args, **kwargs):
        raise AssertionError("reduction or plotting ran after endpoint failure")

    monkeypatch.setattr("empirical_testing.parallel.ordered_map", failed_endpoint)
    monkeypatch.setattr(curves, "reduce_curves", forbidden)
    monkeypatch.setattr(curves, "plot_curves", forbidden)
    output = tmp_path / "output"
    with pytest.raises(AssertionError, match="reconstructed endpoint"):
        curves.run_learning_curves(
            "coffee", project_root=tmp_path, output_dir=output,
            imported_source=source, config_path=config, smoke=True,
        )
    info = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
    assert info["status"] == "failed"
    assert "endpoint_verified" not in info
    assert not (output / "reduce").exists()
