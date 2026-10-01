"""Adapter for the Great American Coffee Taste Test."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from empirical_testing.core.types import DensePreferenceDataset


class CoffeeAdapter:
    name = "coffee"

    _score_columns = tuple(
        f"coffee_{letter}_personal_preference" for letter in "abcd"
    )

    def load(
        self, path: Path, group_names: Sequence[str], options: Mapping[str, Any] | None = None
    ) -> DensePreferenceDataset:
        if options:
            raise ValueError(f"coffee accepts no dataset_options; got {sorted(options)}")
        frame = pd.read_csv(path, low_memory=False)
        missing = sorted(
            set(self._score_columns + ("submission_id",)) - set(frame.columns)
        )
        if missing:
            raise ValueError(f"coffee source is missing required columns: {missing}")

        scores = frame.loc[:, self._score_columns].apply(pd.to_numeric, errors="coerce")
        complete = scores.notna().all(axis=1) & frame["submission_id"].notna()
        frame = frame.loc[complete].reset_index(drop=True)
        ratings = scores.loc[complete].to_numpy(dtype=np.float64)
        utilities = (ratings - 1.0) / 4.0

        age = frame.get("age", pd.Series(index=frame.index, dtype="object")).astype("string")
        gender = frame.get("gender", pd.Series(index=frame.index, dtype="object")).astype("string")
        total_spend = frame.get(
            "total_spend", pd.Series(index=frame.index, dtype="object")
        ).astype("string")
        number_children = frame.get(
            "number_children", pd.Series(index=frame.index, dtype="object")
        ).astype("string")
        ethnicity_race = frame.get(
            "ethnicity_race", pd.Series(index=frame.index, dtype="object")
        ).astype("string")
        wfh = frame.get("wfh", pd.Series(index=frame.index, dtype="object")).astype("string")
        most_willing = frame.get(
            "most_willing", pd.Series(index=frame.index, dtype="object")
        ).astype("string")
        # ``expertise`` is a 1--10 self-rating that pandas may present as float
        # ("8.0"), int or text depending on the column's missing values.  Coerce
        # once, here: anything that is not a number is missing, and a missing
        # rating is a nonmember rather than an imputed one.
        expertise = pd.to_numeric(
            frame.get("expertise", pd.Series(index=frame.index, dtype="object")),
            errors="coerce",
        )

        candidates = {
            "universal": np.ones(len(frame), dtype=np.int8),
            "age:45+": age.isin(
                ("45-54 years old", "55-64 years old", ">65 years old")
            ).to_numpy(dtype=np.int8),
            "gender:female": gender.eq("Female").fillna(False).to_numpy(dtype=np.int8),
            "spend:<20": total_spend.eq("<$20").fillna(False).to_numpy(dtype=np.int8),
            "children:reported-1+": number_children.isin(
                ("1", "2", "3", "More than 3")
            ).to_numpy(dtype=np.int8),
            "race:asian-pacific-islander": ethnicity_race.eq(
                "Asian/Pacific Islander"
            ).fillna(False).to_numpy(dtype=np.int8),
            # The two definitions below exist only for the repeated-CV route
            # (``coffee-repeated-cv-v1``); the frozen primary family above is
            # unchanged. ``work:from-home`` is the survey's explicit
            # primarily-at-home category and deliberately excludes the hybrid
            # answer "I do a mix of both".
            "race:hispanic-latino": ethnicity_race.eq(
                "Hispanic/Latino"
            ).fillna(False).to_numpy(dtype=np.int8),
            "work:from-home": wfh.eq(
                "I primarily work from home"
            ).fillna(False).to_numpy(dtype=np.int8),
            # The three definitions below exist only for the 10x10 sensitivity
            # route (``coffee-repeated-cv-v2-10x10``).  Every family above is
            # unchanged.  ``wfh:in-person`` is the survey's explicit
            # primarily-in-person category: both "I primarily work from home"
            # and the hybrid "I do a mix of both" are excluded.
            "most-willing:$10-$15": most_willing.eq(
                "$10-$15"
            ).fillna(False).to_numpy(dtype=np.int8),
            # 8, 9 or 10 on the self-rated expertise scale. ``expertise`` was
            # coerced to a number above, so "8", 8 and "8.0" agree and a
            # nonnumeric or missing rating is a nonmember.
            "expertise:8+": expertise.isin(
                (8, 9, 10)
            ).fillna(False).to_numpy(dtype=np.int8),
            "wfh:in-person": wfh.eq(
                "I primarily work in person"
            ).fillna(False).to_numpy(dtype=np.int8),
            # The two definitions below exist only for the second 10x10
            # sensitivity route (``coffee-repeated-cv-v3-10x10``).  Every
            # family above is unchanged.  ``expertise:1-3`` is the bottom of
            # the same self-rating ``expertise:8+`` reads from the top of: the
            # column was coerced to a number above, so "2", 2 and "2.0" agree
            # and a nonnumeric or missing rating is a nonmember rather than a
            # low one.
            "expertise:1-3": expertise.isin(
                (1, 2, 3)
            ).fillna(False).to_numpy(dtype=np.int8),
            # The survey's single "$40-$60" monthly-spend answer; the adjacent
            # "$20-$40" and "$60-$80" categories are nonmembers.
            "spend:$40-$60": total_spend.eq(
                "$40-$60"
            ).fillna(False).to_numpy(dtype=np.int8),
            # The definition below exists only for the fourth 10x10 route
            # (``coffee-repeated-cv-v4-10x10``).  Every family above is
            # unchanged.  ``expertise:1-4`` is one step wider at the bottom of
            # the same self-rating than ``expertise:1-3``: the column was
            # coerced to a number above, so "4", 4 and "4.0" agree, and a
            # nonnumeric or missing rating is a nonmember rather than a low one.
            "expertise:1-4": expertise.isin(
                (1, 2, 3, 4)
            ).fillna(False).to_numpy(dtype=np.int8),
        }

        requested = tuple(group_names)
        if not requested or requested[0] != "universal":
            raise ValueError("Coffee groups must list universal first")
        unknown = sorted(set(requested) - set(candidates))
        if unknown:
            raise ValueError(
                f"unknown Coffee groups {unknown}; available={tuple(candidates)}"
            )

        notes = [
            "Complete personal-preference profiles only.",
            "Scores are mapped linearly from 1--5 to [0,1].",
            "The sample is self-selected and is not representative of US adults.",
            "Missing or unlisted demographic values imply subgroup nonmembership.",
            "children:reported-1+ means reporting one or more children; missing child counts are not imputed as zero.",
        ]
        if "work:from-home" in requested:
            notes.append(
                "work:from-home is the survey's explicit 'I primarily work from home' answer; "
                "'I do a mix of both' is not included."
            )
        if "wfh:in-person" in requested:
            notes.append(
                "wfh:in-person is the survey's explicit 'I primarily work in person' answer; "
                "'I primarily work from home' and 'I do a mix of both' are not included."
            )
        if "expertise:8+" in requested:
            notes.append(
                "expertise:8+ means a self-rated expertise of 8, 9 or 10; missing or nonnumeric "
                "ratings imply nonmembership and are not imputed."
            )
        if "expertise:1-4" in requested:
            notes.append(
                "expertise:1-4 means a self-rated expertise of 1, 2, 3 or 4; missing or nonnumeric "
                "ratings imply nonmembership and are not imputed as low."
            )
        if "expertise:1-3" in requested:
            notes.append(
                "expertise:1-3 means a self-rated expertise of 1, 2 or 3; missing or nonnumeric "
                "ratings imply nonmembership and are not imputed as low."
            )
        if "spend:$40-$60" in requested:
            notes.append(
                "spend:$40-$60 is the survey's single '$40-$60' total_spend answer; adjacent "
                "categories such as '$20-$40' and '$60-$80' are nonmembers."
            )
        if "most-willing:$10-$15" in requested:
            notes.append(
                "most-willing:$10-$15 is the survey's single '$10-$15' willingness-to-pay answer; "
                "adjacent categories such as '$8-$10' and '$15-$20' are nonmembers."
            )
        if "race:hispanic-latino" in requested:
            notes.append(
                "race:hispanic-latino is the survey's single 'Hispanic/Latino' ethnicity_race answer; "
                "respondents selecting another or no category are nonmembers."
            )

        prefer_ad = frame.get(
            "prefer_ad", pd.Series(index=frame.index, dtype="object")
        ).astype("string").to_numpy()
        dataset = DensePreferenceDataset(
            name="Great-American-Coffee-Taste-Test",
            utilities=utilities,
            groups=np.column_stack([candidates[name] for name in requested]),
            group_names=requested,
            action_names=("Coffee A", "Coffee B", "Coffee C", "Coffee D"),
            respondent_ids=frame["submission_id"].astype(str).to_numpy(),
            source=(
                "https://github.com/rfordatascience/tidytuesday/"
                "tree/main/data/2024/2024-05-14"
            ),
            notes=tuple(notes),
            # ``ratings``: the raw 1-5 scores, for the utility-sensitivity readings.
            # ``stated_choice``: the survey's own A-vs-D pick, for the generic
            # stated-choice agreement check in the runner.
            auxiliary={"ratings": ratings, "stated_choice": prefer_ad},
        )
        return dataset.validate()

