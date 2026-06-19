from __future__ import annotations

import math

import pandas as pd

from nu_afolu.metrics import add_area_calibration_fields, agreement_metrics_from_areas

OBSERVED_AREA = 100.0
CHEN_AREA = 80.0
TRUE_POSITIVE_AREA = 40.0
FALSE_POSITIVE_AREA = -0.001
FALSE_NEGATIVE_AREA = -0.001
LOW_CORRECTION_CHEN_AREA = 1_000.0
HIGH_CORRECTION_CHEN_AREA = 10.0
LOWER_BOUND = 0.25
UPPER_BOUND = 4.0
EXPECTED_PRECISION = 0.5
EXPECTED_RECALL = 0.4
EXPECTED_AREA_BIAS = 0.8
EXPECTED_APE = 0.2
EXPECTED_CORRECTION_FACTOR = 1.25


def test_agreement_metrics_from_areas_calculates_overlap_metrics() -> None:
    metrics = agreement_metrics_from_areas(
        observed_area_m2=OBSERVED_AREA,
        chen_area_m2=CHEN_AREA,
        tp_area_m2=TRUE_POSITIVE_AREA,
    )

    expected_iou = TRUE_POSITIVE_AREA / (
        OBSERVED_AREA + CHEN_AREA - TRUE_POSITIVE_AREA
    )
    assert metrics["precision"] == EXPECTED_PRECISION
    assert metrics["recall"] == EXPECTED_RECALL
    assert metrics["iou"] == expected_iou
    assert metrics["area_bias"] == EXPECTED_AREA_BIAS
    assert metrics["ape"] == EXPECTED_APE
    assert metrics["correction_factor_raw"] == EXPECTED_CORRECTION_FACTOR
    assert metrics["correction_factor"] == EXPECTED_CORRECTION_FACTOR
    assert metrics["calibration_valid"] is True


def test_agreement_metrics_from_areas_handles_zero_observed_area() -> None:
    metrics = agreement_metrics_from_areas(
        observed_area_m2=0,
        chen_area_m2=CHEN_AREA,
        tp_area_m2=0,
    )

    assert math.isnan(metrics["recall"])
    assert math.isnan(metrics["area_bias"])
    assert math.isnan(metrics["ape"])
    assert metrics["calibration_valid"] is False
    assert metrics["correction_factor"] == 1.0


def test_agreement_metrics_from_areas_handles_zero_chen_area() -> None:
    metrics = agreement_metrics_from_areas(
        observed_area_m2=OBSERVED_AREA,
        chen_area_m2=0,
        tp_area_m2=0,
    )

    assert math.isnan(metrics["precision"])
    assert math.isnan(metrics["correction_factor_raw"])
    assert metrics["calibration_valid"] is False
    assert metrics["correction_factor"] == 1.0


def test_agreement_metrics_from_areas_clips_correction_factor() -> None:
    low = agreement_metrics_from_areas(
        observed_area_m2=OBSERVED_AREA,
        chen_area_m2=LOW_CORRECTION_CHEN_AREA,
        tp_area_m2=TRUE_POSITIVE_AREA,
    )
    high = agreement_metrics_from_areas(
        observed_area_m2=OBSERVED_AREA,
        chen_area_m2=HIGH_CORRECTION_CHEN_AREA,
        tp_area_m2=HIGH_CORRECTION_CHEN_AREA,
    )

    assert low["correction_factor"] == LOWER_BOUND
    assert high["correction_factor"] == UPPER_BOUND


def test_agreement_metrics_from_areas_clamps_negative_overlap_components() -> None:
    metrics = agreement_metrics_from_areas(
        observed_area_m2=OBSERVED_AREA,
        chen_area_m2=CHEN_AREA,
        tp_area_m2=TRUE_POSITIVE_AREA,
        fp_area_m2=FALSE_POSITIVE_AREA,
        fn_area_m2=FALSE_NEGATIVE_AREA,
    )

    assert metrics["fp_area_m2"] == 0.0
    assert metrics["fn_area_m2"] == 0.0


def test_add_area_calibration_fields_returns_copy_and_marks_invalid_rows() -> None:
    raw = pd.DataFrame(
        {
            "observed_area_m2": [OBSERVED_AREA, 0.0],
            "chen_area_m2": [CHEN_AREA, CHEN_AREA],
        },
    )

    out = add_area_calibration_fields(raw)

    assert "area_error_m2" not in raw
    assert out.loc[0, "area_error_m2"] == CHEN_AREA - OBSERVED_AREA
    assert bool(out.loc[0, "calibration_valid"]) is True
    assert math.isnan(out.loc[1, "ape"])
    assert bool(out.loc[1, "calibration_valid"]) is False
    assert out.loc[1, "correction_factor"] == 1.0
