from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

GROWTH_CONSISTENT = "consistent"
LOW_GROWTH = "low_growth"
HIGH_GROWTH = "high_growth"
EXTREME_GROWTH = "extreme_growth"
INSUFFICIENT_HISTORY = "insufficient_history"
GROWTH_PLAUSIBILITY_LABELS = frozenset(
    {
        GROWTH_CONSISTENT,
        LOW_GROWTH,
        HIGH_GROWTH,
        EXTREME_GROWTH,
        INSUFFICIENT_HISTORY,
    },
)

STABLE_HISTORY = "stable_history"
HISTORICAL_GROWTH_UNSTABLE = "historical_growth_unstable"
HISTORICAL_SETTLEMENT_DECLINE = "historical_settlement_decline"
HISTORICAL_GROWTH_CONTEXT_LABELS = frozenset(
    {
        STABLE_HISTORY,
        HISTORICAL_GROWTH_UNSTABLE,
        HISTORICAL_SETTLEMENT_DECLINE,
        INSUFFICIENT_HISTORY,
    },
)

WITHIN_HISTORICAL_ENVELOPE = "within_historical_envelope"
ABOVE_HISTORICAL_ENVELOPE = "above_historical_envelope"
BELOW_HISTORICAL_ENVELOPE = "below_historical_envelope"
INSUFFICIENT_HISTORICAL_ENVELOPE = "insufficient_historical_envelope"
HISTORICAL_ENVELOPE_LABELS = frozenset(
    {
        WITHIN_HISTORICAL_ENVELOPE,
        ABOVE_HISTORICAL_ENVELOPE,
        BELOW_HISTORICAL_ENVELOPE,
        INSUFFICIENT_HISTORICAL_ENVELOPE,
    },
)

SUPPORTS_MANUAL_REVIEW = "historical_growth_supports_review"
CHEN_LOW_VS_RECENT = "chen_low_vs_recent"
CHEN_HIGH_VS_RECENT = "chen_high_vs_recent"
CHEN_EXTREME_VS_RECENT = "chen_extreme_vs_recent"
CHEN_ABOVE_HISTORICAL_ENVELOPE = "chen_above_historical_envelope"
INSUFFICIENT_RECENT_HISTORY = "insufficient_recent_history"
HISTORICAL_DECLINE_REVIEW = "historical_settlement_decline"
HISTORICAL_UNSTABLE_REVIEW = "historical_growth_unstable"
HISTORICAL_GROWTH_REVIEW_NOTES = frozenset(
    {
        SUPPORTS_MANUAL_REVIEW,
        CHEN_LOW_VS_RECENT,
        CHEN_HIGH_VS_RECENT,
        CHEN_EXTREME_VS_RECENT,
        CHEN_ABOVE_HISTORICAL_ENVELOPE,
        INSUFFICIENT_RECENT_HISTORY,
        HISTORICAL_DECLINE_REVIEW,
        HISTORICAL_UNSTABLE_REVIEW,
    },
)

DEFAULT_LOWER_RATIO = 0.25
DEFAULT_UPPER_RATIO = 4.0
DEFAULT_EXTREME_RATIO = 8.0
DEFAULT_VOLATILITY_RATIO = 4.0
AREA_TOLERANCE_M2 = 1e-3

HISTORICAL_GROWTH_DIAGNOSTIC_COLUMNS = (
    "zone",
    "scenario",
    "recent_growth_area_m2",
    "recent_growth_pct",
    "median_historical_decadal_growth_area_m2",
    "max_historical_decadal_growth_area_m2",
    "min_historical_decadal_growth_area_m2",
    "historical_growth_range_ratio",
    "negative_historical_growth_periods",
    "historical_growth_context",
    "max_chen_new_area_m2",
    "max_chen_to_recent_growth_ratio",
    "max_chen_to_historical_median_ratio",
    "max_chen_to_historical_max_ratio",
    "worst_growth_plausibility",
    "historical_envelope_alignment",
    "historical_growth_review_note",
)


def classify_growth_plausibility(
    ratio: float,
    historical_growth_area_m2: float,
    *,
    min_historical_growth_area_m2: float,
    lower_ratio: float = DEFAULT_LOWER_RATIO,
    upper_ratio: float = DEFAULT_UPPER_RATIO,
    extreme_ratio: float = DEFAULT_EXTREME_RATIO,
) -> str:
    if historical_growth_area_m2 < min_historical_growth_area_m2:
        return INSUFFICIENT_HISTORY
    if not _finite(ratio):
        return INSUFFICIENT_HISTORY
    if ratio < lower_ratio:
        return LOW_GROWTH
    if ratio <= upper_ratio:
        return GROWTH_CONSISTENT
    if ratio <= extreme_ratio:
        return HIGH_GROWTH
    return EXTREME_GROWTH


def classify_historical_growth_context(
    *,
    recent_growth_area_m2: float,
    historical_growth_range_ratio: float,
    negative_historical_growth_periods: int,
    min_historical_growth_area_m2: float,
    volatility_ratio: float = DEFAULT_VOLATILITY_RATIO,
) -> str:
    if negative_historical_growth_periods > 0:
        return HISTORICAL_SETTLEMENT_DECLINE
    if recent_growth_area_m2 < min_historical_growth_area_m2:
        return INSUFFICIENT_HISTORY
    if (
        _finite(historical_growth_range_ratio)
        and historical_growth_range_ratio > volatility_ratio
    ):
        return HISTORICAL_GROWTH_UNSTABLE
    return STABLE_HISTORY


def build_historical_settlement_growth(
    area_df: pd.DataFrame,
    *,
    periods: Sequence[tuple[int, int]],
    settlement_column: str = "settlements",
) -> pd.DataFrame:
    settlement_area = _settlement_area_by_zone_year(area_df, settlement_column)
    missing_years = sorted(
        {
            year
            for period in periods
            for year in period
            if year not in settlement_area.columns
        },
    )
    if missing_years:
        message = f"Missing settlement area years: {missing_years}"
        raise ValueError(message)

    rows: list[pd.DataFrame] = []
    for start_year, end_year in periods:
        years_per_decade = (end_year - start_year) / 10
        start_area = settlement_area[start_year]
        end_area = settlement_area[end_year]
        delta_area = end_area - start_area
        rows.append(
            pd.DataFrame(
                {
                    "zone": settlement_area.index,
                    "period_start_year": start_year,
                    "year": end_year,
                    "period_length_years": end_year - start_year,
                    "start_settlement_area_m2": start_area.to_numpy(),
                    "end_settlement_area_m2": end_area.to_numpy(),
                    "historical_growth_area_m2": delta_area.to_numpy(),
                    "historical_decadal_growth_area_m2": (
                        delta_area.div(years_per_decade).to_numpy()
                    ),
                    "historical_growth_pct": delta_area.div(
                        start_area.replace(0, np.nan),
                    ).to_numpy(),
                },
            ),
        )
    return pd.concat(rows, ignore_index=True)


def build_historical_growth_context(
    df_historical_growth: pd.DataFrame,
    *,
    recent_growth_period: tuple[int, int],
    min_historical_growth_area_m2: float,
    volatility_ratio: float = DEFAULT_VOLATILITY_RATIO,
) -> pd.DataFrame:
    required = {
        "zone",
        "period_start_year",
        "year",
        "historical_growth_area_m2",
        "historical_decadal_growth_area_m2",
        "historical_growth_pct",
    }
    _raise_for_missing_columns(df_historical_growth, required, "historical growth")
    recent_start, recent_end = recent_growth_period
    recent = df_historical_growth[
        df_historical_growth["period_start_year"].eq(recent_start)
        & df_historical_growth["year"].eq(recent_end)
    ][["zone", "historical_growth_area_m2", "historical_growth_pct"]].rename(
        columns={
            "historical_growth_area_m2": "recent_growth_area_m2",
            "historical_growth_pct": "recent_growth_pct",
        },
    )
    if recent["zone"].duplicated().any():
        message = "Recent historical growth period must have one row per zone."
        raise ValueError(message)

    context = (
        df_historical_growth.groupby("zone", as_index=False)
        .agg(
            median_historical_decadal_growth_area_m2=(
                "historical_decadal_growth_area_m2",
                "median",
            ),
            max_historical_decadal_growth_area_m2=(
                "historical_decadal_growth_area_m2",
                "max",
            ),
            min_historical_decadal_growth_area_m2=(
                "historical_decadal_growth_area_m2",
                "min",
            ),
            negative_historical_growth_periods=(
                "historical_growth_area_m2",
                lambda series: int(series.lt(-AREA_TOLERANCE_M2).sum()),
            ),
        )
        .merge(recent, on="zone", how="left")
    )
    positive_growth = df_historical_growth[
        df_historical_growth["historical_decadal_growth_area_m2"].gt(
            AREA_TOLERANCE_M2,
        )
    ]
    positive_range = (
        positive_growth.groupby("zone")["historical_decadal_growth_area_m2"]
        .agg(["min", "max"])
        .assign(
            historical_growth_range_ratio=lambda frame: frame["max"].div(
                frame["min"].replace(0, np.nan),
            ),
        )[["historical_growth_range_ratio"]]
        .reset_index()
    )
    context = context.merge(positive_range, on="zone", how="left")
    context["historical_growth_context"] = [
        classify_historical_growth_context(
            recent_growth_area_m2=recent_growth,
            historical_growth_range_ratio=range_ratio,
            negative_historical_growth_periods=int(negative_periods),
            min_historical_growth_area_m2=min_historical_growth_area_m2,
            volatility_ratio=volatility_ratio,
        )
        for recent_growth, range_ratio, negative_periods in zip(
            context["recent_growth_area_m2"],
            context["historical_growth_range_ratio"],
            context["negative_historical_growth_periods"],
            strict=True,
        )
    ]
    return context


def build_chen_growth_plausibility(
    df_expansion: pd.DataFrame,
    df_historical_context: pd.DataFrame,
    *,
    min_historical_growth_area_m2: float,
) -> pd.DataFrame:
    required_expansion = {
        "zone",
        "scenario",
        "period_start_year",
        "year",
        "chen_new_area_m2",
    }
    required_context = {
        "zone",
        "recent_growth_area_m2",
        "recent_growth_pct",
        "median_historical_decadal_growth_area_m2",
        "max_historical_decadal_growth_area_m2",
        "min_historical_decadal_growth_area_m2",
        "historical_growth_range_ratio",
        "negative_historical_growth_periods",
        "historical_growth_context",
    }
    _raise_for_missing_columns(df_expansion, required_expansion, "Chen expansion")
    _raise_for_missing_columns(
        df_historical_context,
        required_context,
        "historical growth context",
    )

    out = df_expansion.merge(df_historical_context, on="zone", how="left")
    out["chen_to_recent_growth_ratio"] = _positive_ratio(
        out["chen_new_area_m2"],
        out["recent_growth_area_m2"],
    )
    out["chen_to_historical_median_ratio"] = _positive_ratio(
        out["chen_new_area_m2"],
        out["median_historical_decadal_growth_area_m2"],
    )
    out["chen_to_historical_max_ratio"] = _positive_ratio(
        out["chen_new_area_m2"],
        out["max_historical_decadal_growth_area_m2"],
    )
    out["growth_plausibility"] = [
        classify_growth_plausibility(
            ratio,
            growth,
            min_historical_growth_area_m2=min_historical_growth_area_m2,
        )
        for ratio, growth in zip(
            out["chen_to_recent_growth_ratio"],
            out["recent_growth_area_m2"],
            strict=True,
        )
    ]
    out["historical_envelope_alignment"] = [
        _classify_historical_envelope_alignment(
            chen_to_median_ratio=median_ratio,
            chen_to_max_ratio=max_ratio,
            historical_context=context,
        )
        for median_ratio, max_ratio, context in zip(
            out["chen_to_historical_median_ratio"],
            out["chen_to_historical_max_ratio"],
            out["historical_growth_context"],
            strict=True,
        )
    ]
    return out


def summarize_growth_plausibility(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "zone",
        "scenario",
        "chen_new_area_m2",
        "recent_growth_area_m2",
        "recent_growth_pct",
        "median_historical_decadal_growth_area_m2",
        "max_historical_decadal_growth_area_m2",
        "min_historical_decadal_growth_area_m2",
        "historical_growth_range_ratio",
        "negative_historical_growth_periods",
        "historical_growth_context",
        "chen_to_recent_growth_ratio",
        "chen_to_historical_median_ratio",
        "chen_to_historical_max_ratio",
        "growth_plausibility",
        "historical_envelope_alignment",
    }
    _raise_for_missing_columns(df, required, "Chen growth plausibility")
    label_rank = {
        GROWTH_CONSISTENT: 0,
        LOW_GROWTH: 1,
        HIGH_GROWTH: 2,
        EXTREME_GROWTH: 3,
        INSUFFICIENT_HISTORY: 4,
    }
    rank_label = {rank: label for label, rank in label_rank.items()}
    envelope_rank = {
        WITHIN_HISTORICAL_ENVELOPE: 0,
        BELOW_HISTORICAL_ENVELOPE: 1,
        INSUFFICIENT_HISTORICAL_ENVELOPE: 2,
        ABOVE_HISTORICAL_ENVELOPE: 3,
    }
    rank_envelope = {rank: label for label, rank in envelope_rank.items()}
    summary = (
        df.assign(
            _growth_rank=lambda frame: frame["growth_plausibility"].map(label_rank),
            _envelope_rank=lambda frame: frame["historical_envelope_alignment"].map(
                envelope_rank,
            ),
        )
        .groupby(["zone", "scenario"], as_index=False)
        .agg(
            recent_growth_area_m2=("recent_growth_area_m2", "first"),
            recent_growth_pct=("recent_growth_pct", "first"),
            median_historical_decadal_growth_area_m2=(
                "median_historical_decadal_growth_area_m2",
                "first",
            ),
            max_historical_decadal_growth_area_m2=(
                "max_historical_decadal_growth_area_m2",
                "first",
            ),
            min_historical_decadal_growth_area_m2=(
                "min_historical_decadal_growth_area_m2",
                "first",
            ),
            historical_growth_range_ratio=("historical_growth_range_ratio", "first"),
            negative_historical_growth_periods=(
                "negative_historical_growth_periods",
                "first",
            ),
            historical_growth_context=("historical_growth_context", "first"),
            max_chen_new_area_m2=("chen_new_area_m2", "max"),
            max_chen_to_recent_growth_ratio=("chen_to_recent_growth_ratio", "max"),
            max_chen_to_historical_median_ratio=(
                "chen_to_historical_median_ratio",
                "max",
            ),
            max_chen_to_historical_max_ratio=("chen_to_historical_max_ratio", "max"),
            worst_growth_rank=("_growth_rank", "max"),
            worst_envelope_rank=("_envelope_rank", "max"),
        )
    )
    summary["worst_growth_plausibility"] = summary["worst_growth_rank"].map(rank_label)
    summary["historical_envelope_alignment"] = summary["worst_envelope_rank"].map(
        rank_envelope,
    )
    summary["historical_growth_review_note"] = [
        _review_note(plausibility, alignment, context)
        for plausibility, alignment, context in zip(
            summary["worst_growth_plausibility"],
            summary["historical_envelope_alignment"],
            summary["historical_growth_context"],
            strict=True,
        )
    ]
    return summary.drop(columns=["worst_growth_rank", "worst_envelope_rank"])


def build_historical_growth_diagnostics(df_summary: pd.DataFrame) -> pd.DataFrame:
    _raise_for_missing_columns(
        df_summary,
        set(HISTORICAL_GROWTH_DIAGNOSTIC_COLUMNS),
        "growth plausibility summary",
    )
    return df_summary[list(HISTORICAL_GROWTH_DIAGNOSTIC_COLUMNS)].copy()


def _settlement_area_by_zone_year(
    area_df: pd.DataFrame,
    settlement_column: str,
) -> pd.DataFrame:
    if isinstance(area_df.index, pd.MultiIndex) and {"zone", "year"}.issubset(
        set(area_df.index.names),
    ):
        table = area_df[[settlement_column]].reset_index().pivot_table(
            index="zone",
            columns="year",
            values=settlement_column,
            aggfunc="first",
        )
    elif {"zone", "year", settlement_column}.issubset(area_df.columns):
        table = area_df.pivot_table(
            index="zone",
            columns="year",
            values=settlement_column,
            aggfunc="first",
        )
    else:
        message = (
            "area_df must have zone/year index levels or zone, year, and settlement "
            "columns."
        )
        raise ValueError(message)
    table.columns = table.columns.astype(int)
    return table


def _classify_historical_envelope_alignment(
    *,
    chen_to_median_ratio: float,
    chen_to_max_ratio: float,
    historical_context: str,
    lower_ratio: float = DEFAULT_LOWER_RATIO,
    upper_ratio: float = DEFAULT_UPPER_RATIO,
) -> str:
    if historical_context in {INSUFFICIENT_HISTORY, HISTORICAL_SETTLEMENT_DECLINE}:
        return INSUFFICIENT_HISTORICAL_ENVELOPE
    if not _finite(chen_to_median_ratio) or not _finite(chen_to_max_ratio):
        return INSUFFICIENT_HISTORICAL_ENVELOPE
    if chen_to_max_ratio > upper_ratio:
        return ABOVE_HISTORICAL_ENVELOPE
    if chen_to_median_ratio < lower_ratio:
        return BELOW_HISTORICAL_ENVELOPE
    return WITHIN_HISTORICAL_ENVELOPE


def _review_note(plausibility: str, alignment: str, context: str) -> str:
    if plausibility == EXTREME_GROWTH:
        note = CHEN_EXTREME_VS_RECENT
    elif alignment == ABOVE_HISTORICAL_ENVELOPE:
        note = CHEN_ABOVE_HISTORICAL_ENVELOPE
    elif context == HISTORICAL_SETTLEMENT_DECLINE:
        note = HISTORICAL_DECLINE_REVIEW
    elif context == HISTORICAL_GROWTH_UNSTABLE:
        note = HISTORICAL_UNSTABLE_REVIEW
    elif plausibility == HIGH_GROWTH:
        note = CHEN_HIGH_VS_RECENT
    elif plausibility == LOW_GROWTH:
        note = CHEN_LOW_VS_RECENT
    elif plausibility == INSUFFICIENT_HISTORY:
        note = INSUFFICIENT_RECENT_HISTORY
    else:
        note = SUPPORTS_MANUAL_REVIEW
    return note


def _positive_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce")
    numerator = pd.to_numeric(numerator, errors="coerce")
    return numerator.div(denominator.where(denominator.gt(AREA_TOLERANCE_M2)))


def _finite(value: float) -> bool:
    return isinstance(value, int | float | np.number) and math.isfinite(value)


def _raise_for_missing_columns(
    df: pd.DataFrame,
    columns: set[str],
    table_name: str,
) -> None:
    missing = sorted(columns.difference(df.columns))
    if missing:
        message = f"Missing required columns in {table_name}: {missing}"
        raise ValueError(message)
