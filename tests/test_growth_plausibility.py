from __future__ import annotations

import numpy as np
import pandas as pd

from nu_afolu.growth_plausibility import (
    ABOVE_HISTORICAL_ENVELOPE,
    EXTREME_GROWTH,
    GROWTH_CONSISTENT,
    HIGH_GROWTH,
    HISTORICAL_GROWTH_UNSTABLE,
    HISTORICAL_SETTLEMENT_DECLINE,
    INSUFFICIENT_HISTORY,
    LOW_GROWTH,
    STABLE_HISTORY,
    build_chen_growth_plausibility,
    build_historical_growth_context,
    build_historical_growth_diagnostics,
    build_historical_settlement_growth,
    classify_growth_plausibility,
    summarize_growth_plausibility,
)

PERIODS = ((2000, 2010), (2010, 2020), (2000, 2020))
RECENT_PERIOD = (2010, 2020)
MIN_GROWTH = 1_000_000.0


def _area_df(
    *,
    y2000: float = 10_000_000.0,
    y2010: float = 12_000_000.0,
    y2020: float = 14_000_000.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"zone": "zone-a", "year": 2000, "settlements": y2000},
            {"zone": "zone-a", "year": 2010, "settlements": y2010},
            {"zone": "zone-a", "year": 2020, "settlements": y2020},
        ],
    ).set_index(["zone", "year"])


def _context(area_df: pd.DataFrame) -> pd.DataFrame:
    historical = build_historical_settlement_growth(area_df, periods=PERIODS)
    return build_historical_growth_context(
        historical,
        recent_growth_period=RECENT_PERIOD,
        min_historical_growth_area_m2=MIN_GROWTH,
    )


def test_growth_plausibility_labels() -> None:
    assert (
        classify_growth_plausibility(
            1.0,
            MIN_GROWTH,
            min_historical_growth_area_m2=MIN_GROWTH,
        )
        == GROWTH_CONSISTENT
    )
    assert (
        classify_growth_plausibility(
            0.1,
            MIN_GROWTH,
            min_historical_growth_area_m2=MIN_GROWTH,
        )
        == LOW_GROWTH
    )
    assert (
        classify_growth_plausibility(
            5.0,
            MIN_GROWTH,
            min_historical_growth_area_m2=MIN_GROWTH,
        )
        == HIGH_GROWTH
    )
    assert (
        classify_growth_plausibility(
            9.0,
            MIN_GROWTH,
            min_historical_growth_area_m2=MIN_GROWTH,
        )
        == EXTREME_GROWTH
    )
    assert (
        classify_growth_plausibility(
            np.nan,
            1.0,
            min_historical_growth_area_m2=MIN_GROWTH,
        )
        == INSUFFICIENT_HISTORY
    )


def test_historical_growth_context_stable_when_periods_are_consistent() -> None:
    context = _context(_area_df())

    row = context.iloc[0]
    assert row["historical_growth_context"] == STABLE_HISTORY
    assert row["recent_growth_area_m2"] == 2_000_000.0
    assert row["median_historical_decadal_growth_area_m2"] == 2_000_000.0


def test_historical_growth_context_flags_volatile_history() -> None:
    context = _context(_area_df(y2000=10_000_000, y2010=11_000_000, y2020=20_000_000))

    row = context.iloc[0]
    assert row["historical_growth_context"] == HISTORICAL_GROWTH_UNSTABLE
    assert row["historical_growth_range_ratio"] == 9.0


def test_historical_growth_context_flags_negative_growth() -> None:
    context = _context(_area_df(y2000=10_000_000, y2010=14_000_000, y2020=12_000_000))

    row = context.iloc[0]
    assert row["historical_growth_context"] == HISTORICAL_SETTLEMENT_DECLINE
    assert row["negative_historical_growth_periods"] == 1


def test_negative_recent_growth_does_not_create_ratio_support() -> None:
    context = _context(_area_df(y2000=10_000_000, y2010=14_000_000, y2020=12_000_000))
    expansion = pd.DataFrame(
        [
            {
                "zone": "zone-a",
                "scenario": "SSP1",
                "period_start_year": 2020,
                "year": 2030,
                "chen_new_area_m2": 4_000_000.0,
            },
        ],
    )

    result = build_chen_growth_plausibility(
        expansion,
        context,
        min_historical_growth_area_m2=MIN_GROWTH,
    )

    row = result.iloc[0]
    assert row["growth_plausibility"] == INSUFFICIENT_HISTORY
    assert np.isnan(row["chen_to_recent_growth_ratio"])


def test_diagnostics_capture_extreme_growth_and_historical_envelope() -> None:
    context = _context(_area_df())
    expansion = pd.DataFrame(
        [
            {
                "zone": "zone-a",
                "scenario": "SSP1",
                "period_start_year": 2020,
                "year": 2030,
                "chen_new_area_m2": 20_000_000.0,
            },
        ],
    )
    plausibility = build_chen_growth_plausibility(
        expansion,
        context,
        min_historical_growth_area_m2=MIN_GROWTH,
    )
    summary = build_historical_growth_diagnostics(
        summarize_growth_plausibility(plausibility),
    )

    row = summary.iloc[0]
    assert row["worst_growth_plausibility"] == EXTREME_GROWTH
    assert row["historical_envelope_alignment"] == ABOVE_HISTORICAL_ENVELOPE
    assert row["max_chen_to_recent_growth_ratio"] == 10.0
