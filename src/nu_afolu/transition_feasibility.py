from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

FEASIBLE = "feasible"
CAPACITY_WATCH = "capacity_watch"
INFEASIBLE = "infeasible"
TRANSITION_FEASIBILITY_LABELS = frozenset(
    {FEASIBLE, CAPACITY_WATCH, INFEASIBLE},
)
DEFAULT_CAPACITY_WATCH_RATIO = 0.80
AREA_TOLERANCE_M2 = 1e-3

SOURCE_AVAILABILITY_COLUMNS = (
    "zone",
    "from_class",
    "available_source_area_2020_m2",
)
TRANSITION_FEASIBILITY_COLUMNS = (
    "zone",
    "scenario",
    "calibration",
    "transition_feasibility",
    "max_capacity_ratio",
    "total_overrun_area_m2",
    "overrun_source_classes",
    "first_overrun_year",
    "limiting_from_class",
)


def build_source_availability_table(
    area_df: pd.DataFrame,
    *,
    source_year: int,
    source_classes: Sequence[str],
) -> pd.DataFrame:
    """Return 2020 source-class stock available for future settlement conversion."""
    area_2020 = _select_year(area_df, source_year)
    missing_classes = sorted(set(source_classes).difference(area_2020.columns))
    if missing_classes:
        message = f"Missing source classes in area table: {missing_classes}"
        raise ValueError(message)

    return (
        area_2020.reset_index()
        .melt(
            id_vars="zone",
            value_vars=list(source_classes),
            var_name="from_class",
            value_name="available_source_area_2020_m2",
        )
        .astype({"zone": "string", "from_class": "string"})
        .assign(
            available_source_area_2020_m2=lambda frame: pd.to_numeric(
                frame["available_source_area_2020_m2"],
                errors="coerce",
            ).fillna(0.0),
        )
    )


def build_transition_feasibility_table(
    df_chen_transitions: pd.DataFrame,
    df_source_availability: pd.DataFrame,
    *,
    watch_ratio: float = DEFAULT_CAPACITY_WATCH_RATIO,
) -> pd.DataFrame:
    """Screen cumulative transition demand against 2020 source-class stock."""
    _validate_watch_ratio(watch_ratio)
    if df_chen_transitions.empty:
        return pd.DataFrame(columns=TRANSITION_FEASIBILITY_COLUMNS)

    required_transition_columns = {
        "zone",
        "scenario",
        "year",
        "from_class",
        "calibration",
        "area_m2",
    }
    required_availability_columns = set(SOURCE_AVAILABILITY_COLUMNS)
    _raise_for_missing_columns(
        df_chen_transitions,
        required_transition_columns,
        "transition table",
    )
    _raise_for_missing_columns(
        df_source_availability,
        required_availability_columns,
        "source availability table",
    )

    period = (
        df_chen_transitions.groupby(
            ["zone", "scenario", "year", "from_class", "calibration"],
            as_index=False,
        )["area_m2"]
        .sum()
        .sort_values(["zone", "scenario", "from_class", "calibration", "year"])
        .merge(df_source_availability, on=["zone", "from_class"], how="left")
    )
    period["available_source_area_2020_m2"] = period[
        "available_source_area_2020_m2"
    ].fillna(0.0)
    period["cumulative_transition_area_m2"] = period.groupby(
        ["zone", "scenario", "from_class", "calibration"],
    )["area_m2"].cumsum()
    period["capacity_ratio"] = _capacity_ratio(
        period["cumulative_transition_area_m2"],
        period["available_source_area_2020_m2"],
    )
    period["overrun_area_m2"] = (
        period["cumulative_transition_area_m2"]
        - period["available_source_area_2020_m2"]
    ).clip(lower=0.0)
    period["is_overrun"] = period["overrun_area_m2"].gt(AREA_TOLERANCE_M2)

    summaries: list[dict[str, object]] = []
    for key, group in period.groupby(["zone", "scenario", "calibration"], sort=False):
        zone, scenario, calibration = key
        max_ratio = float(group["capacity_ratio"].max())
        total_overrun = float(group["overrun_area_m2"].sum())
        overrun_classes = int(
            group.loc[group["is_overrun"], "from_class"].nunique(),
        )
        overrun_rows = group[group["is_overrun"]]
        if overrun_rows.empty:
            first_overrun_year = pd.NA
        else:
            first_overrun_year = int(overrun_rows["year"].min())

        limiting_row = group.sort_values(
            ["capacity_ratio", "overrun_area_m2"],
            ascending=[False, False],
        ).iloc[0]
        summaries.append(
            {
                "zone": zone,
                "scenario": scenario,
                "calibration": calibration,
                "transition_feasibility": _classify_feasibility(
                    max_ratio,
                    total_overrun,
                    watch_ratio,
                ),
                "max_capacity_ratio": max_ratio,
                "total_overrun_area_m2": total_overrun,
                "overrun_source_classes": overrun_classes,
                "first_overrun_year": first_overrun_year,
                "limiting_from_class": limiting_row["from_class"],
            },
        )

    return pd.DataFrame(summaries, columns=TRANSITION_FEASIBILITY_COLUMNS)


def _select_year(area_df: pd.DataFrame, source_year: int) -> pd.DataFrame:
    if isinstance(area_df.index, pd.MultiIndex) and "year" in area_df.index.names:
        area_2020 = area_df.xs(source_year, level="year")
    elif "year" in area_df.columns:
        area_2020 = area_df[area_df["year"].eq(source_year)].set_index("zone")
    else:
        message = "area_df must have a year index level or year column."
        raise ValueError(message)

    if "zone" not in area_2020.index.names:
        area_2020.index.name = "zone"
    return area_2020


def _capacity_ratio(
    cumulative_transition: pd.Series,
    available_source_area: pd.Series,
) -> pd.Series:
    zero_available = available_source_area.le(AREA_TOLERANCE_M2)
    positive_transition = cumulative_transition.gt(AREA_TOLERANCE_M2)
    out = cumulative_transition.div(available_source_area.replace(0, np.nan))
    out = out.mask(zero_available & ~positive_transition, 0.0)
    return out.mask(zero_available & positive_transition, np.inf)


def _classify_feasibility(
    max_capacity_ratio: float,
    total_overrun_area_m2: float,
    watch_ratio: float,
) -> str:
    if total_overrun_area_m2 > AREA_TOLERANCE_M2:
        return INFEASIBLE
    if max_capacity_ratio >= watch_ratio:
        return CAPACITY_WATCH
    return FEASIBLE


def _validate_watch_ratio(watch_ratio: float) -> None:
    if not 0 < watch_ratio <= 1:
        message = "watch_ratio must be greater than 0 and less than or equal to 1."
        raise ValueError(message)


def _raise_for_missing_columns(
    df: pd.DataFrame,
    columns: set[str],
    table_name: str,
) -> None:
    missing = sorted(columns.difference(df.columns))
    if missing:
        message = f"Missing required columns in {table_name}: {missing}"
        raise ValueError(message)
