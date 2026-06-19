from __future__ import annotations

import pandas as pd

from nu_afolu.artifact_validation import validate_external_validation_artifacts
from nu_afolu.chen import SSP_NAMES
from nu_afolu.external_validation import (
    COMPARATOR_CONFLICT,
    COMPARATOR_INSUFFICIENT,
    COMPARATOR_SUPPORTED,
    EXTERNAL_BASELINE_CONFLICT,
    EXTERNAL_CONFLICT,
    EXTERNAL_REVIEW,
    EXTERNAL_SUPPORT,
    EXTERNAL_SUPPORTS_BOTH,
    GROWTH_CONSISTENT,
    HIGH_CHEN_GROWTH,
    INSUFFICIENT_EXTERNAL_GROWTH,
    LOW_CHEN_GROWTH,
    QUESTIONS_CHEN_BASELINE,
    QUESTIONS_GLC_BASELINE,
    classify_baseline_comparator_support,
    classify_external_advisory,
    classify_growth_alignment,
    combine_baseline_support,
)
from nu_afolu.transition_feasibility import FEASIBLE

ZONES = ("zone-a", "zone-b")
EXTERNAL_AREA = 100.0
COMPARATOR_AREA = 90.0
TRUE_POSITIVE_AREA = 80.0
FALSE_POSITIVE_AREA = 10.0
FALSE_NEGATIVE_AREA = 20.0
IOU = TRUE_POSITIVE_AREA / (
    EXTERNAL_AREA + COMPARATOR_AREA - TRUE_POSITIVE_AREA
)
AREA_BIAS = 0.9
APE = 0.1
GHSL_GROWTH = 100.0
RAW_CHEN_GROWTH = 80.0
CALIBRATED_CHEN_GROWTH = 100.0


def _baseline_agreement() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "zone": zone,
                "scenario": scenario,
                "comparator": comparator,
                "external_dataset": "ghsl_built_surface",
                "external_year": 2020,
                "external_area_m2": EXTERNAL_AREA,
                "comparator_area_m2": COMPARATOR_AREA,
                "tp_area_m2": TRUE_POSITIVE_AREA,
                "fp_area_m2": FALSE_POSITIVE_AREA,
                "fn_area_m2": FALSE_NEGATIVE_AREA,
                "precision": TRUE_POSITIVE_AREA / COMPARATOR_AREA,
                "recall": TRUE_POSITIVE_AREA / EXTERNAL_AREA,
                "iou": IOU,
                "area_bias": AREA_BIAS,
                "ape": APE,
                "comparator_support": COMPARATOR_SUPPORTED,
            }
            for zone in ZONES
            for scenario in SSP_NAMES
            for comparator in ("glc_settlements_2020", "chen_urban_2020")
        ],
    )


def _growth_alignment() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for zone in ZONES:
        for scenario in SSP_NAMES:
            for calibration, chen_growth in (
                ("raw", RAW_CHEN_GROWTH),
                ("calibrated", CALIBRATED_CHEN_GROWTH),
            ):
                rows.append(
                    {
                        "zone": zone,
                        "scenario": scenario,
                        "calibration": calibration,
                        "external_dataset": "ghsl_built_surface",
                        "period_start_year": 2020,
                        "year": 2030,
                        "ghsl_growth_area_m2": GHSL_GROWTH,
                        "chen_growth_area_m2": chen_growth,
                        "chen_to_external_growth_ratio": chen_growth / GHSL_GROWTH,
                        "growth_alignment": GROWTH_CONSISTENT,
                    },
                )
    return pd.DataFrame(rows)


def _assessment() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "zone": zone,
                "scenario": scenario,
                "land_estimate_readiness": "ready_for_manual_review",
                "manual_review_priority": "low",
                "overall_assessment": "ready_for_manual_review",
                "transition_feasibility": FEASIBLE,
                "reliability": "high",
            }
            for zone in ZONES
            for scenario in SSP_NAMES
        ],
    )


def _review_flags() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "zone": zone,
                "scenario": scenario,
                "glc_baseline_support": COMPARATOR_SUPPORTED,
                "chen_baseline_support": COMPARATOR_SUPPORTED,
                "external_baseline_validation": EXTERNAL_SUPPORTS_BOTH,
                "calibrated_growth_alignment": GROWTH_CONSISTENT,
                "external_advisory": EXTERNAL_SUPPORT,
                "land_estimate_readiness": "ready_for_manual_review",
                "manual_review_priority": "low",
                "overall_assessment": "ready_for_manual_review",
                "transition_feasibility": FEASIBLE,
                "reliability": "high",
                "iou": IOU,
                "ape": APE,
                "ghsl_growth_area_m2": GHSL_GROWTH,
                "calibrated_chen_growth_area_m2": CALIBRATED_CHEN_GROWTH,
                "calibrated_chen_to_external_growth_ratio": (
                    CALIBRATED_CHEN_GROWTH / GHSL_GROWTH
                ),
            }
            for zone in ZONES
            for scenario in SSP_NAMES
        ],
    )


def _summary(flags: pd.DataFrame) -> pd.DataFrame:
    return (
        flags.groupby(
            [
                "scenario",
                "external_advisory",
                "external_baseline_validation",
                "calibrated_growth_alignment",
            ],
            as_index=False,
        )
        .agg(
            rows=("zone", "count"),
            median_ghsl_growth_area_m2=("ghsl_growth_area_m2", "median"),
            median_calibrated_chen_to_external_growth_ratio=(
                "calibrated_chen_to_external_growth_ratio",
                "median",
            ),
        )
        .assign(share=1.0)[
            [
                "scenario",
                "external_advisory",
                "external_baseline_validation",
                "calibrated_growth_alignment",
                "rows",
                "share",
                "median_ghsl_growth_area_m2",
                "median_calibrated_chen_to_external_growth_ratio",
            ]
        ]
    )


def test_baseline_comparator_support_labels() -> None:
    assert (
        classify_baseline_comparator_support(
            external_area_m2=1_000_000,
            comparator_area_m2=1_000_000,
            iou=0.25,
            ape=0.8,
        )
        == COMPARATOR_SUPPORTED
    )
    assert (
        classify_baseline_comparator_support(
            external_area_m2=1_000_000,
            comparator_area_m2=1_000_000,
            iou=0.1,
            ape=0.35,
        )
        == COMPARATOR_SUPPORTED
    )
    assert (
        classify_baseline_comparator_support(
            external_area_m2=1_000_000,
            comparator_area_m2=1_000_000,
            iou=0.1,
            ape=0.8,
        )
        == COMPARATOR_CONFLICT
    )
    assert (
        classify_baseline_comparator_support(
            external_area_m2=1.0,
            comparator_area_m2=1_000_000,
            iou=1.0,
            ape=0.0,
        )
        == COMPARATOR_INSUFFICIENT
    )


def test_combined_baseline_and_advisory_labels() -> None:
    assert (
        combine_baseline_support(COMPARATOR_SUPPORTED, COMPARATOR_SUPPORTED)
        == EXTERNAL_SUPPORTS_BOTH
    )
    assert (
        combine_baseline_support(COMPARATOR_CONFLICT, COMPARATOR_SUPPORTED)
        == QUESTIONS_GLC_BASELINE
    )
    assert (
        combine_baseline_support(COMPARATOR_SUPPORTED, COMPARATOR_CONFLICT)
        == QUESTIONS_CHEN_BASELINE
    )
    assert (
        combine_baseline_support(COMPARATOR_CONFLICT, COMPARATOR_CONFLICT)
        == EXTERNAL_BASELINE_CONFLICT
    )
    assert (
        classify_external_advisory(EXTERNAL_SUPPORTS_BOTH, GROWTH_CONSISTENT)
        == EXTERNAL_SUPPORT
    )
    assert (
        classify_external_advisory(QUESTIONS_GLC_BASELINE, GROWTH_CONSISTENT)
        == EXTERNAL_REVIEW
    )
    assert (
        classify_external_advisory(EXTERNAL_SUPPORTS_BOTH, HIGH_CHEN_GROWTH)
        == EXTERNAL_CONFLICT
    )


def test_growth_alignment_labels() -> None:
    assert (
        classify_growth_alignment(
            external_growth_area_m2=1_000_000,
            chen_growth_area_m2=1_000_000,
        )
        == GROWTH_CONSISTENT
    )
    assert (
        classify_growth_alignment(
            external_growth_area_m2=1_000_000,
            chen_growth_area_m2=100_000,
        )
        == LOW_CHEN_GROWTH
    )
    assert (
        classify_growth_alignment(
            external_growth_area_m2=1_000_000,
            chen_growth_area_m2=5_000_000,
        )
        == HIGH_CHEN_GROWTH
    )
    assert (
        classify_growth_alignment(
            external_growth_area_m2=10.0,
            chen_growth_area_m2=5_000_000,
        )
        == INSUFFICIENT_EXTERNAL_GROWTH
    )


def test_external_validation_accepts_valid_artifacts() -> None:
    flags = _review_flags()
    report = validate_external_validation_artifacts(
        _baseline_agreement(),
        _growth_alignment(),
        flags,
        _summary(flags),
        _assessment(),
        zone_names=ZONES,
    )

    assert report.empty


def test_external_validation_reports_invalid_label() -> None:
    flags = _review_flags()
    invalid = flags.copy()
    invalid.loc[0, "external_advisory"] = "approved_for_model_input"

    report = validate_external_validation_artifacts(
        _baseline_agreement(),
        _growth_alignment(),
        invalid,
        _summary(flags),
        _assessment(),
        zone_names=ZONES,
    )

    assert "external_advisory_allowed_values" in set(report["check"])


def test_external_validation_reports_metric_range_error() -> None:
    flags = _review_flags()
    invalid = _baseline_agreement()
    invalid.loc[0, "iou"] = 2.0

    report = validate_external_validation_artifacts(
        invalid,
        _growth_alignment(),
        flags,
        _summary(flags),
        _assessment(),
        zone_names=ZONES,
    )

    assert "iou_range" in set(report["check"])


def test_external_validation_reports_pairing_mismatch() -> None:
    flags = _review_flags()
    invalid = flags.copy()
    invalid.loc[0, "chen_baseline_support"] = COMPARATOR_CONFLICT

    report = validate_external_validation_artifacts(
        _baseline_agreement(),
        _growth_alignment(),
        invalid,
        _summary(flags),
        _assessment(),
        zone_names=ZONES,
    )

    assert "chen_baseline_support_pairing" in set(report["check"])
