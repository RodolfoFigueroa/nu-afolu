from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nu_afolu.transition_feasibility import (
    CAPACITY_WATCH,
    FEASIBLE,
    INFEASIBLE,
    build_source_availability_table,
    build_transition_feasibility_table,
)

SOURCE_CLASSES = ("croplands", "forests_primary")


@pytest.fixture
def area_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "zone": "zone-a",
                "year": 2020,
                "croplands": 100.0,
                "forests_primary": 50.0,
                "settlements": 10.0,
            },
        ],
    ).set_index(["zone", "year"])


def _transitions(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "zone": "zone-a",
                "scenario": "SSP1",
                "year": row["year"],
                "from_class": row["from_class"],
                "calibration": row["calibration"],
                "area_m2": row["area_m2"],
            }
            for row in rows
        ],
    )


def test_source_availability_uses_requested_year(area_df: pd.DataFrame) -> None:
    result = build_source_availability_table(
        area_df,
        source_year=2020,
        source_classes=SOURCE_CLASSES,
    )

    assert set(result["from_class"]) == set(SOURCE_CLASSES)
    assert result["available_source_area_2020_m2"].sum() == 150.0


def test_feasible_when_cumulative_transitions_remain_below_stock(
    area_df: pd.DataFrame,
) -> None:
    availability = build_source_availability_table(
        area_df,
        source_year=2020,
        source_classes=SOURCE_CLASSES,
    )
    transitions = _transitions(
        [
            {
                "year": 2030,
                "from_class": "croplands",
                "calibration": "raw",
                "area_m2": 20.0,
            },
            {
                "year": 2040,
                "from_class": "croplands",
                "calibration": "raw",
                "area_m2": 30.0,
            },
            {
                "year": 2030,
                "from_class": "forests_primary",
                "calibration": "raw",
                "area_m2": 10.0,
            },
        ],
    )

    result = build_transition_feasibility_table(transitions, availability)

    row = result.iloc[0]
    assert row["transition_feasibility"] == FEASIBLE
    assert row["max_capacity_ratio"] == 0.5
    assert row["total_overrun_area_m2"] == 0.0


def test_capacity_watch_when_cumulative_transition_reaches_watch_ratio(
    area_df: pd.DataFrame,
) -> None:
    availability = build_source_availability_table(
        area_df,
        source_year=2020,
        source_classes=SOURCE_CLASSES,
    )
    transitions = _transitions(
        [
            {
                "year": 2030,
                "from_class": "croplands",
                "calibration": "calibrated",
                "area_m2": 50.0,
            },
            {
                "year": 2040,
                "from_class": "croplands",
                "calibration": "calibrated",
                "area_m2": 30.0,
            },
        ],
    )

    result = build_transition_feasibility_table(transitions, availability)

    row = result.iloc[0]
    assert row["transition_feasibility"] == CAPACITY_WATCH
    assert row["max_capacity_ratio"] == 0.8
    assert row["overrun_source_classes"] == 0


def test_infeasible_when_cumulative_transition_exceeds_source_stock(
    area_df: pd.DataFrame,
) -> None:
    availability = build_source_availability_table(
        area_df,
        source_year=2020,
        source_classes=SOURCE_CLASSES,
    )
    transitions = _transitions(
        [
            {
                "year": 2030,
                "from_class": "croplands",
                "calibration": "calibrated",
                "area_m2": 70.0,
            },
            {
                "year": 2040,
                "from_class": "croplands",
                "calibration": "calibrated",
                "area_m2": 60.0,
            },
        ],
    )

    result = build_transition_feasibility_table(transitions, availability)

    row = result.iloc[0]
    assert row["transition_feasibility"] == INFEASIBLE
    assert row["max_capacity_ratio"] == 1.3
    assert row["total_overrun_area_m2"] == 30.0
    assert row["overrun_source_classes"] == 1
    assert row["first_overrun_year"] == 2040
    assert row["limiting_from_class"] == "croplands"


def test_zero_available_source_with_positive_transition_is_infeasible(
    area_df: pd.DataFrame,
) -> None:
    zero_area = area_df.copy()
    zero_area.loc[("zone-a", 2020), "croplands"] = 0.0
    availability = build_source_availability_table(
        zero_area,
        source_year=2020,
        source_classes=SOURCE_CLASSES,
    )
    transitions = _transitions(
        [
            {
                "year": 2030,
                "from_class": "croplands",
                "calibration": "calibrated",
                "area_m2": 1.0,
            },
        ],
    )

    result = build_transition_feasibility_table(transitions, availability)

    row = result.iloc[0]
    assert row["transition_feasibility"] == INFEASIBLE
    assert np.isinf(row["max_capacity_ratio"])
    assert row["total_overrun_area_m2"] == 1.0
