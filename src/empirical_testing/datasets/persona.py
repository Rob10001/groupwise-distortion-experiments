"""Adapter for the synthetic persona-evaluation panel.

The source is one ``results.csv`` written by
``experiment/persona-eval/persona_eval.py``: one model-generated rating per
row, each row naming a persona **cell** (a bitstring over the study's binary
features), one candidate answer, and one demographic-line ordering.  The
collection analysed must be a complete factorial grid::

    C cells x A answers x D draws = C * A * D ratings,

where ``C = 2 ** len(study["features"])`` and ``A = len(study["alternatives"])``
are read from the frozen study file and ``D`` is read from the data.  Nothing
about the grid is hard-coded here: the two collections analysed so far are
``16 x 5 x 125 = 10,000`` (``sky-blue-persona-api-v5``) and ``16 x 6 x 104 =
9,984`` (``sky-blue-persona-api-v8``), and both go through this same code.

What this adapter builds, and what it does not
----------------------------------------------
A dense preference dataset needs a complete utility vector per row, but the
ratings were elicited **one answer at a time, in separate API calls**.  No
``A``-answer vector was ever jointly elicited, so none exists in the data.  This
adapter *constructs* pseudo-respondents: within each cell it independently
permutes that cell's ``D`` draws for each answer and reads the permuted columns
off side by side, giving ``D`` ``A``-answer vectors per cell and ``C * D`` in
total.

That alignment is an artifact of this construction.  It assumes independence
across the answer-specific rating draws within a cell, it is reproducible only
against the recorded ``panel_seed``, and a pseudo-respondent is **not** ``A``
ratings from one simulated individual.  A persona bitstring identifies a
demographic cell, not a person, and nothing here is an observation or a
prediction of human demographic preference.

Scores are mapped ``U = (score - 1) / 9`` from the 1--10 rating scale, so the
map is linear onto ``[0, 1]`` and no rating is clipped, imputed or repaired.

Refusals
--------
The adapter fails rather than analysing a partial grid.  A non-``ok`` row, a
missing or out-of-range score, an unknown answer id, a cell that is not a
complete bitstring, or an unbalanced ``(cell, answer)`` count all raise.  There
is no dropping, no back-filling and no resampling to a common count.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from empirical_testing.core.types import DensePreferenceDataset

#: The rating scale the collector's structured output enforces.
SCORE_MIN = 1
SCORE_MAX = 10
#: ``all_personas`` in the study's ``analysis_groups`` is the everyone group;
#: the dense contract requires that column to be named ``universal``.
UNIVERSAL_STUDY_ID = "all_personas"
UNIVERSAL = "universal"
REQUIRED_COLUMNS = ("persona_bits", "alternative_id", "source_row", "status", "score")
#: Written by the collector when it varies or fixes the demographic-line order.
#: Optional, because the earliest collections predate the column; when it is
#: present the adapter records whether the order was held constant, which is a
#: property of the *collection* and not something the analysis can vary.
ORDER_COLUMN = "demographic_order_bits"
DEFAULT_PANEL_SEED = 20260916


def demographic_order_condition(frame: pd.DataFrame) -> dict[str, Any]:
    """What the collection did with demographic-line position.

    Reported, never acted on.  A fixed order means line position is constant
    across every request, so it cannot be a source of within-collection
    variation and its effect is not identifiable from this collection alone.
    """

    if ORDER_COLUMN not in frame.columns:
        return {"recorded": False, "fixed": None, "orderings": None,
                "note": f"the collection's results.csv has no {ORDER_COLUMN} column"}
    values = sorted(set(frame[ORDER_COLUMN].astype("string").dropna().tolist()))
    fixed = len(values) == 1
    return {
        "recorded": True,
        "fixed": fixed,
        "orderings": values if fixed or len(values) <= 8 else len(values),
        "n_orderings": len(values),
        "note": (
            f"demographic-line position was held constant at {values[0]!r} in every request, so it is "
            "not a source of within-collection variation and its causal effect is not identifiable here"
            if fixed else
            f"demographic-line position varied over {len(values)} orderings across requests"
        ),
    }


def utility_from_score(scores: np.ndarray) -> np.ndarray:
    """``U = (score - 1) / 9``: the declared linear map onto ``[0, 1]``."""

    return (np.asarray(scores, dtype=np.float64) - float(SCORE_MIN)) / float(SCORE_MAX - SCORE_MIN)


def permutation_seed(panel_seed: int, cell_index: int, answer_index: int) -> int:
    """The 32-bit seed for one ``(cell, answer)`` permutation.

    Derived only from the declared panel seed and the two indices, so the whole
    panel is reproducible from ``panel_seed`` alone and is identical however the
    rows happen to be ordered in memory or how the run is parallelized.
    """

    state = np.random.SeedSequence(
        [int(panel_seed), int(cell_index), int(answer_index)]
    ).generate_state(1, dtype=np.uint32)
    return int(state[0])


def candidate_groups(study: Mapping[str, Any]) -> tuple[str, ...]:
    """The candidate group universe named by the study, universal first.

    This is read from ``analysis_groups`` in the study file the collection was
    run against, which was written before any score existed.  The adapter never
    invents a group and never adds an intersection.
    """

    groups = list(study["analysis_groups"])
    names = [UNIVERSAL if item["id"] == UNIVERSAL_STUDY_ID else str(item["id"]) for item in groups]
    if not names or names[0] != UNIVERSAL:
        raise ValueError(f"the study's first analysis group must be {UNIVERSAL_STUDY_ID!r}")
    if len(set(names)) != len(names):
        raise ValueError("the study's analysis groups have duplicate ids")
    return tuple(names)


def _membership(study: Mapping[str, Any], cell_bits: Sequence[str], requested: Sequence[str]) -> np.ndarray:
    """Binary membership of each cell in each requested group.

    Membership is a deterministic function of the persona bitstring: bit *i*
    selects ``features[i].levels[bit]``, and a feature-level group holds exactly
    the cells whose bit equals that group's declared level.  Nothing is
    imputed, and the universal column is one for every cell.
    """

    feature_ids = [str(item["id"]) for item in study["features"]]
    by_name: dict[str, dict[str, Any]] = {}
    for item in study["analysis_groups"]:
        name = UNIVERSAL if item["id"] == UNIVERSAL_STUDY_ID else str(item["id"])
        by_name[name] = item
    unknown = sorted(set(requested) - set(by_name))
    if unknown:
        raise ValueError(f"unknown persona groups {unknown}; available={tuple(by_name)}")
    columns = []
    for name in requested:
        item = by_name[name]
        if name == UNIVERSAL:
            columns.append(np.ones(len(cell_bits), dtype=np.int8))
            continue
        feature_id = str(item["feature_id"])
        level = int(item["level"])
        position = feature_ids.index(feature_id)
        columns.append(
            np.asarray([1 if int(bits[position]) == level else 0 for bits in cell_bits], dtype=np.int8)
        )
    return np.stack(columns, axis=1)


class PersonaAdapter:
    """Build the synthetic pseudo-respondent panel from one collection CSV."""

    name = "persona"

    def load(
        self, path: Path, group_names: Sequence[str], options: Mapping[str, Any] | None = None
    ) -> DensePreferenceDataset:
        options = dict(options or {})
        unknown = sorted(set(options) - {"study_path", "panel_seed"})
        if unknown:
            raise ValueError(f"persona accepts only study_path and panel_seed; got {unknown}")
        study_path = options.get("study_path")
        if study_path is None:
            raise ValueError("persona requires dataset_options.study_path")
        panel_seed = int(options.get("panel_seed", DEFAULT_PANEL_SEED))
        study = json.loads(Path(study_path).read_text(encoding="utf-8"))

        # Both bit columns are read as strings: they are bitstrings, and letting
        # pandas infer would silently turn "00011011" into the integer 11011.
        frame = pd.read_csv(
            path, dtype={"persona_bits": "string", ORDER_COLUMN: "string"}, low_memory=False
        )
        missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"persona results are missing required columns: {missing}")
        bad_status = frame.loc[frame["status"].astype("string") != "ok"]
        if len(bad_status):
            raise ValueError(
                f"{len(bad_status)} of {len(frame)} rows are not status 'ok'; the panel requires a "
                "complete grid and this adapter never drops, imputes or resamples a row"
            )
        scores = pd.to_numeric(frame["score"], errors="coerce")
        if scores.isna().any():
            raise ValueError("every accepted row must carry a numeric score")
        if not scores.between(SCORE_MIN, SCORE_MAX).all() or not (scores == scores.round()).all():
            raise ValueError(f"scores must be integers in [{SCORE_MIN}, {SCORE_MAX}]")

        answers = tuple(str(item["id"]) for item in study["alternatives"])
        if len(set(answers)) != len(answers) or len(answers) < 2:
            raise ValueError("the study must declare at least two distinct alternatives")
        seen_answers = set(frame["alternative_id"].astype("string"))
        if seen_answers != set(answers):
            raise ValueError(
                f"the collection's answers {sorted(seen_answers)} do not match the study's {list(answers)}"
            )
        n_features = len(study["features"])
        cell_bits = tuple(format(value, f"0{n_features}b") for value in range(2 ** n_features))
        seen_cells = set(frame["persona_bits"])
        if seen_cells != set(cell_bits):
            raise ValueError(
                f"the collection covers {len(seen_cells)} persona cells; the complete factorial grid "
                f"over {n_features} binary features has {len(cell_bits)}"
            )

        answer_index = {name: index for index, name in enumerate(answers)}
        cell_index = {bits: index for index, bits in enumerate(cell_bits)}
        frame = frame.assign(
            _score=scores.to_numpy(dtype=np.int64),
            _cell=frame["persona_bits"].map(cell_index).to_numpy(dtype=np.int64),
            _answer=frame["alternative_id"].astype("string").map(answer_index).to_numpy(dtype=np.int64),
            _row=pd.to_numeric(frame["source_row"], errors="raise").to_numpy(dtype=np.int64),
        )
        counts = frame.groupby(["_cell", "_answer"], sort=True).size()
        if len(counts) != len(cell_bits) * len(answers) or counts.nunique() != 1:
            raise ValueError(
                "the panel requires the same number of draws for every (cell, answer) pair; "
                f"observed counts range over {sorted(set(counts.tolist()))[:5]}"
            )
        draws = int(counts.iloc[0])

        utilities = np.empty((len(cell_bits) * draws, len(answers)), dtype=np.float64)
        respondent_ids = np.empty(len(cell_bits) * draws, dtype=object)
        row_cell = np.empty(len(cell_bits) * draws, dtype=np.int64)
        for index, bits in enumerate(cell_bits):
            block = slice(index * draws, (index + 1) * draws)
            for answer in range(len(answers)):
                # Source-row order fixes the pre-permutation draw order, so the
                # permutation -- not the CSV's incidental ordering -- is the
                # only thing that decides which draws line up.
                cell_rows = frame.loc[(frame["_cell"] == index) & (frame["_answer"] == answer)]
                ordered = cell_rows.sort_values("_row")["_score"].to_numpy(dtype=np.int64)
                seed = permutation_seed(panel_seed, index, answer)
                order = np.random.default_rng(seed).permutation(draws)
                utilities[block, answer] = utility_from_score(ordered[order])
            respondent_ids[block] = [f"{bits}-{position:04d}" for position in range(draws)]
            row_cell[block] = index

        order = demographic_order_condition(frame)
        groups = _membership(study, [cell_bits[value] for value in row_cell], tuple(group_names))
        requested = tuple(group_names)
        if not requested or requested[0] != UNIVERSAL:
            raise ValueError("persona groups must list universal first")
        return DensePreferenceDataset(
            name=self.name,
            utilities=utilities,
            groups=groups,
            group_names=requested,
            action_names=answers,
            respondent_ids=respondent_ids.astype(str),
            source=str(path),
            notes=(
                f"Synthetic panel: {len(cell_bits)} cells x {len(answers)} answers x {draws} draws "
                f"= {len(cell_bits) * len(answers) * draws} model-generated ratings.",
                "Ratings are model outputs under persona prompts, not human responses; a persona "
                "bitstring identifies a demographic cell, not a person.",
                f"The {len(answers)} answers were each rated in a separate API call. "
                f"{len(answers)}-answer vectors are constructed by independently permuting each "
                "answer's draws within a cell, so the alignment is an artifact of "
                f"panel_seed={panel_seed} and assumes independence across answer-specific draws.",
                f"A pseudo-respondent is not {len(answers)} ratings from one simulated individual.",
                f"Utilities are U = (score - 1) / {SCORE_MAX - SCORE_MIN}; no score is clipped or imputed.",
                "Group membership is a deterministic function of the persona bitstring.",
                order["note"] if order["recorded"] else
                "the collection does not record demographic-line position",
            ),
            auxiliary={
                "cell_index": row_cell,
                "cell_bits": np.asarray([cell_bits[value] for value in row_cell], dtype=str),
            },
        ).validate()


def permutation_table(panel_seed: int, cells: Sequence[str], answers: Sequence[str], draws: int) -> list[dict]:
    """Every ``(cell, answer)`` permutation seed, in the order the panel uses them.

    The permutations are not stored with the panel: they are a pure function of
    ``panel_seed`` and the two indices, so recording the seeds is enough to
    rebuild the exact alignment.
    """

    return [
        {
            "cell": str(cell), "cell_index": cell_index,
            "answer": str(answer), "answer_index": answer_index,
            "seed": permutation_seed(panel_seed, cell_index, answer_index),
            "draws": int(draws),
        }
        for cell_index, cell in enumerate(cells)
        for answer_index, answer in enumerate(answers)
    ]


def panel_record(path: Path, study_path: Path, panel_seed: int, dataset: DensePreferenceDataset) -> dict:
    """The auditable record of how the panel was constructed.

    The demographic-order condition is re-read from the collection CSV rather
    than carried on the dataset: it is one fact about the whole collection, not
    a per-respondent column, and the manifest is the place it belongs.
    """

    order = demographic_order_condition(
        pd.read_csv(
            path, usecols=lambda name: name == ORDER_COLUMN,
            dtype={ORDER_COLUMN: "string"}, low_memory=False,
        )
    )
    cells = np.unique(np.asarray(dataset.auxiliary["cell_bits"]).astype(str))
    draws = int(dataset.utilities.shape[0] // max(cells.size, 1))
    return {
        "permutations": permutation_table(panel_seed, sorted(cells.tolist()), dataset.action_names, draws),
        "results_csv": str(path),
        "study_json": str(study_path),
        "panel_seed": int(panel_seed),
        "cells": int(cells.size),
        "answers": list(dataset.action_names),
        "n_answers": len(dataset.action_names),
        "demographic_order": order,
        "draws_per_cell_answer": draws,
        "pseudo_respondents": int(dataset.utilities.shape[0]),
        "utility_map": f"(score - 1) / {SCORE_MAX - SCORE_MIN}",
        "alignment": (
            "within each cell, each answer's draws are independently permuted with "
            "default_rng(SeedSequence([panel_seed, cell_index, answer_index])) and read side by side"
        ),
        "alignment_caveat": (
            f"the {len(dataset.action_names)} answers were each rated in a separate API call; the "
            "aligned vectors assume independence across answer-specific draws and are not "
            f"{len(dataset.action_names)} ratings from one individual"
        ),
    }
