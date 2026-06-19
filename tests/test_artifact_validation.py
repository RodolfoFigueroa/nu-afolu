from __future__ import annotations

import pandas as pd
import pytest

from nu_afolu.artifact_validation import (
    FUTURE_YEARS,
    SOURCE_CLASSES,
    raise_for_validation_errors,
    validate_calibration_artifacts,
    validate_calibration_table,
    validate_exploration_artifacts,
    validate_transition_closure_artifacts,
)
from nu_afolu.chen import SSP_NAMES
from nu_afolu.transition_feasibility import CAPACITY_WATCH, FEASIBLE, INFEASIBLE

ZONES = ("zone-a", "zone-b")
THRESHOLDS = (0.10, 0.25)
OBSERVED_AREA = 100.0
CHEN_AREA = 80.0
AREA_ERROR = -20.0
AREA_BIAS = 0.8
APE = 0.2
TRUE_POSITIVE_AREA = 40.0
FALSE_POSITIVE_AREA = 40.0
FALSE_NEGATIVE_AREA = 60.0
PRECISION = 0.5
RECALL = 0.4
IOU = TRUE_POSITIVE_AREA / (OBSERVED_AREA + CHEN_AREA - TRUE_POSITIVE_AREA)
CORRECTION_FACTOR = 1.25
RAW_SOURCE_AREA = 1.0
EXISTING_SETTLEMENT_AREA = 2.0


@pytest.fixture
def calibration() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "zone": zone,
                "scenario": scenario,
                "observed_area_m2": OBSERVED_AREA,
                "chen_area_m2": CHEN_AREA,
                "area_error_m2": AREA_ERROR,
                "area_bias": AREA_BIAS,
                "ape": APE,
                "tp_area_m2": TRUE_POSITIVE_AREA,
                "fp_area_m2": FALSE_POSITIVE_AREA,
                "fn_area_m2": FALSE_NEGATIVE_AREA,
                "precision": PRECISION,
                "recall": RECALL,
                "iou": IOU,
                "correction_factor_raw": CORRECTION_FACTOR,
                "correction_factor": CORRECTION_FACTOR,
                "calibration_valid": True,
                "reliability": "high",
            }
            for zone in ZONES
            for scenario in SSP_NAMES
        ],
    )


@pytest.fixture
def scale_sensitivity() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "zone": zone,
                "scenario": scenario,
                "threshold": threshold,
                "observed_area_m2": OBSERVED_AREA,
                "chen_area_m2": CHEN_AREA,
                "tp_area_m2": TRUE_POSITIVE_AREA,
                "fp_area_m2": FALSE_POSITIVE_AREA,
                "fn_area_m2": FALSE_NEGATIVE_AREA,
                "precision": PRECISION,
                "recall": RECALL,
                "iou": IOU,
                "area_bias": AREA_BIAS,
            }
            for zone in ZONES
            for scenario in SSP_NAMES
            for threshold in THRESHOLDS
        ],
    )


@pytest.fixture
def expansion() -> pd.DataFrame:
    nonsettlement_area = RAW_SOURCE_AREA * len(SOURCE_CLASSES)
    return pd.DataFrame(
        [
            {
                "zone": zone,
                "scenario": scenario,
                "period_start_year": year - 10,
                "year": year,
                "chen_new_area_m2": nonsettlement_area + EXISTING_SETTLEMENT_AREA,
                "nonsettlement_source_area_m2": nonsettlement_area,
                "existing_settlement_area_m2": EXISTING_SETTLEMENT_AREA,
                "correction_factor": CORRECTION_FACTOR,
                "reliability": "high",
            }
            for zone in ZONES
            for scenario in SSP_NAMES
            for year in FUTURE_YEARS
        ],
    )


@pytest.fixture
def transitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "zone": zone,
                "scenario": scenario,
                "period_start_year": year - 10,
                "year": year,
                "from_class": source_class,
                "to_class": "settlements",
                "correction_factor": CORRECTION_FACTOR,
                "reliability": "high",
                "calibration": calibration,
                "area_m2": (
                    RAW_SOURCE_AREA
                    if calibration == "raw"
                    else RAW_SOURCE_AREA * CORRECTION_FACTOR
                ),
                "scaled_up_area_only": calibration == "calibrated",
            }
            for zone in ZONES
            for scenario in SSP_NAMES
            for year in FUTURE_YEARS
            for source_class in SOURCE_CLASSES
            for calibration in ("raw", "calibrated")
        ],
    )


@pytest.fixture
def transition_feasibility() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "zone": zone,
                "scenario": scenario,
                "calibration": calibration,
                "transition_feasibility": FEASIBLE,
                "max_capacity_ratio": 0.5,
                "total_overrun_area_m2": 0.0,
                "overrun_source_classes": 0,
                "first_overrun_year": pd.NA,
                "limiting_from_class": "croplands",
            }
            for zone in ZONES
            for scenario in SSP_NAMES
            for calibration in ("raw", "calibrated")
        ],
    )


@pytest.fixture
def assessment(calibration: pd.DataFrame) -> pd.DataFrame:
    out = calibration.copy()
    out["observed_settlement_area_2020_m2"] = OBSERVED_AREA
    out["observed_total_area_2020_m2"] = 1_000.0
    out["observed_settlement_fraction_2020"] = 0.1
    out["recent_growth_area_m2"] = 20.0
    out["max_chen_new_area_m2"] = 12.0
    out["max_chen_to_recent_growth_ratio"] = 0.6
    out["worst_growth_plausibility"] = "consistent"
    out["max_sensitive_share"] = 0.01
    out["max_watch_share"] = 0.02
    out["max_sensitive_area_m2"] = 0.5
    out["worst_sensitive_flag"] = "low"
    out["area_adequacy"] = "good"
    out["spatial_adequacy"] = "good"
    out["calibration_adequacy"] = "good"
    out["growth_risk"] = "low"
    out["sensitive_class_risk"] = "low"
    out["transition_feasibility"] = FEASIBLE
    out["max_transition_capacity_ratio"] = 0.5
    out["total_transition_overrun_area_m2"] = 0.0
    out["overrun_source_classes"] = 0
    out["limiting_transition_source_class"] = "croplands"
    out["land_estimate_readiness"] = "ready_for_manual_review"
    out["manual_review_priority"] = "low"
    out["overall_assessment"] = "ready_for_manual_review"
    return out


@pytest.fixture
def review_candidates(assessment: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "zone",
        "scenario",
        "land_estimate_readiness",
        "manual_review_priority",
        "calibration_adequacy",
        "growth_risk",
        "sensitive_class_risk",
        "transition_feasibility",
        "max_transition_capacity_ratio",
        "total_transition_overrun_area_m2",
        "overrun_source_classes",
        "limiting_transition_source_class",
        "reliability",
        "area_adequacy",
        "spatial_adequacy",
        "iou",
        "area_bias",
        "correction_factor",
        "worst_growth_plausibility",
        "max_chen_to_recent_growth_ratio",
        "worst_sensitive_flag",
        "max_sensitive_share",
        "max_sensitive_area_m2",
    ]
    out = assessment.loc[[0], columns].copy()
    out.insert(2, "review_reason", "representative_ready_for_manual_review")
    return out


@pytest.fixture
def method_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "zone": zone,
                "scenario": scenario,
                "method": method,
                "observed_threshold": pd.NA if method == "fractional_current" else 0.5,
                "buffer_m": pd.NA,
                "observed_area_m2": OBSERVED_AREA,
                "chen_area_m2": CHEN_AREA,
                "area_error_m2": AREA_ERROR,
                "area_bias": AREA_BIAS,
                "ape": APE,
                "correction_factor_raw": CORRECTION_FACTOR,
                "correction_factor": CORRECTION_FACTOR,
                "calibration_valid": True,
                "precision": PRECISION,
                "recall": RECALL,
                "iou": IOU,
                "buffered_precision": pd.NA,
                "buffered_recall": pd.NA,
                "buffered_f1": pd.NA,
                "spatial_metric_name": "iou",
                "spatial_score": IOU,
                "valid_comparison": True,
            }
            for zone in ZONES
            for scenario in SSP_NAMES
            for method in ("fractional_current", "threshold_50")
        ],
    )


@pytest.fixture
def method_summary(method_comparison: pd.DataFrame) -> pd.DataFrame:
    return (
        method_comparison.groupby("method", as_index=False)
        .agg(
            rows=("method", "size"),
            valid_comparisons=("valid_comparison", "sum"),
            median_spatial_score=("spatial_score", "median"),
            mean_spatial_score=("spatial_score", "mean"),
            median_ape_pct=("ape", lambda series: series.median() * 100),
            median_area_bias=("area_bias", "median"),
            median_correction_factor=("correction_factor", "median"),
        )
        .assign(valid_share=1.0)
    )


@pytest.fixture
def method_recommendations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "zone": zone,
                "scenario": scenario,
                "method": "fractional_current",
                "spatial_metric_name": "iou",
                "spatial_score": IOU,
                "ape": APE,
                "area_bias": AREA_BIAS,
                "correction_factor": CORRECTION_FACTOR,
                "observed_threshold": pd.NA,
                "buffer_m": pd.NA,
                "valid_comparison": True,
            }
            for zone in ZONES
            for scenario in SSP_NAMES
        ],
    )


@pytest.fixture
def disagreement_typology() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "zone": zone,
                "scenario": scenario,
                "diagnostic_type": "stable_current_candidate",
                "area_error_class": "area_close",
                "spatial_agreement_class": "moderate_current_overlap",
                "current_iou": IOU,
                "current_ape": APE,
                "current_area_bias": AREA_BIAS,
                "current_correction_factor": CORRECTION_FACTOR,
                "strict_iou": IOU,
                "strict_ape": APE,
                "buffered_f1_widest": 0.8,
                "buffered_gain_over_current": 0.8 - IOU,
                "strict_iou_gain_over_current": 0.0,
                "strict_ape_delta": 0.0,
                "current_valid": True,
                "review_score": 1.0,
            }
            for zone in ZONES
            for scenario in SSP_NAMES
        ],
    )


@pytest.fixture
def disagreement_summary(disagreement_typology: pd.DataFrame) -> pd.DataFrame:
    rows = len(disagreement_typology)
    return pd.DataFrame(
        [
            {
                "diagnostic_type": "stable_current_candidate",
                "rows": rows,
                "median_current_iou": IOU,
                "median_current_ape": APE,
                "median_buffered_gain": 0.8 - IOU,
                "median_strict_iou_gain": 0.0,
                "max_review_score": 1.0,
                "share": 1.0,
            },
        ],
    )


def test_validation_accepts_valid_artifacts(
    calibration: pd.DataFrame,
    scale_sensitivity: pd.DataFrame,
    expansion: pd.DataFrame,
    transitions: pd.DataFrame,
    transition_feasibility: pd.DataFrame,
    assessment: pd.DataFrame,
    review_candidates: pd.DataFrame,
    method_comparison: pd.DataFrame,
    method_summary: pd.DataFrame,
    method_recommendations: pd.DataFrame,
    disagreement_typology: pd.DataFrame,
    disagreement_summary: pd.DataFrame,
) -> None:
    calibration_report = validate_calibration_artifacts(
        calibration,
        scale_sensitivity,
        zone_names=ZONES,
        thresholds=THRESHOLDS,
    )
    closure_report = validate_transition_closure_artifacts(
        calibration,
        expansion,
        transitions,
        transition_feasibility,
        assessment,
        review_candidates,
        zone_names=ZONES,
    )
    exploration_report = validate_exploration_artifacts(
        method_comparison,
        method_summary,
        method_recommendations,
        disagreement_typology,
        disagreement_summary,
        zone_names=ZONES,
    )

    assert calibration_report.empty
    assert closure_report.empty
    assert exploration_report.empty
    assert validate_calibration_table(calibration, zone_names=ZONES).empty


def test_raise_for_validation_errors_raises(calibration: pd.DataFrame) -> None:
    invalid = calibration.drop(columns=["zone"])
    report = validate_calibration_artifacts(
        invalid,
        pd.DataFrame(),
        zone_names=ZONES,
        thresholds=THRESHOLDS,
    )

    with pytest.raises(ValueError, match="artifact validation error"):
        raise_for_validation_errors(report)


def test_calibration_validation_reports_duplicate_keys(
    calibration: pd.DataFrame,
    scale_sensitivity: pd.DataFrame,
) -> None:
    invalid = pd.concat([calibration, calibration.iloc[[0]]], ignore_index=True)

    report = validate_calibration_artifacts(
        invalid,
        scale_sensitivity,
        zone_names=ZONES,
        thresholds=THRESHOLDS,
    )

    assert "unique_key" in set(report["check"])


def test_transition_validation_reports_raw_total_mismatch(
    calibration: pd.DataFrame,
    expansion: pd.DataFrame,
    transitions: pd.DataFrame,
    transition_feasibility: pd.DataFrame,
    assessment: pd.DataFrame,
    review_candidates: pd.DataFrame,
) -> None:
    invalid = transitions.copy()
    raw_mask = invalid["calibration"].eq("raw")
    invalid.loc[raw_mask.idxmax(), "area_m2"] += 1.0

    report = validate_transition_closure_artifacts(
        calibration,
        expansion,
        invalid,
        transition_feasibility,
        assessment,
        review_candidates,
        zone_names=ZONES,
    )

    assert "raw_transition_total_consistency" in set(report["check"])


def test_transition_validation_reports_scaled_up_flag_mismatch(
    calibration: pd.DataFrame,
    expansion: pd.DataFrame,
    transitions: pd.DataFrame,
    transition_feasibility: pd.DataFrame,
    assessment: pd.DataFrame,
    review_candidates: pd.DataFrame,
) -> None:
    invalid = transitions.copy()
    calibrated_mask = invalid["calibration"].eq("calibrated")
    invalid.loc[calibrated_mask.idxmax(), "scaled_up_area_only"] = False

    report = validate_transition_closure_artifacts(
        calibration,
        expansion,
        invalid,
        transition_feasibility,
        assessment,
        review_candidates,
        zone_names=ZONES,
    )

    assert "calibrated_scaled_up_flag" in set(report["check"])


def test_transition_validation_reports_raw_infeasibility(
    calibration: pd.DataFrame,
    expansion: pd.DataFrame,
    transitions: pd.DataFrame,
    transition_feasibility: pd.DataFrame,
    assessment: pd.DataFrame,
    review_candidates: pd.DataFrame,
) -> None:
    invalid = transition_feasibility.copy()
    raw_mask = invalid["calibration"].eq("raw")
    invalid.loc[raw_mask.idxmax(), "transition_feasibility"] = INFEASIBLE
    invalid.loc[raw_mask.idxmax(), "total_overrun_area_m2"] = 1.0
    invalid.loc[raw_mask.idxmax(), "overrun_source_classes"] = 1
    invalid.loc[raw_mask.idxmax(), "first_overrun_year"] = FUTURE_YEARS[0]

    report = validate_transition_closure_artifacts(
        calibration,
        expansion,
        transitions,
        invalid,
        assessment,
        review_candidates,
        zone_names=ZONES,
    )

    assert "raw_transition_feasibility" in set(report["check"])


def test_transition_validation_accepts_calibrated_infeasibility(
    calibration: pd.DataFrame,
    expansion: pd.DataFrame,
    transitions: pd.DataFrame,
    transition_feasibility: pd.DataFrame,
    assessment: pd.DataFrame,
    review_candidates: pd.DataFrame,
) -> None:
    infeasible = transition_feasibility.copy()
    calibrated_mask = infeasible["calibration"].eq("calibrated")
    row_idx = calibrated_mask.idxmax()
    zone = infeasible.loc[row_idx, "zone"]
    scenario = infeasible.loc[row_idx, "scenario"]
    infeasible.loc[row_idx, "transition_feasibility"] = INFEASIBLE
    infeasible.loc[row_idx, "max_capacity_ratio"] = 1.2
    infeasible.loc[row_idx, "total_overrun_area_m2"] = 1.0
    infeasible.loc[row_idx, "overrun_source_classes"] = 1
    infeasible.loc[row_idx, "first_overrun_year"] = FUTURE_YEARS[0]

    gated_assessment = assessment.copy()
    assessment_mask = gated_assessment["zone"].eq(zone) & gated_assessment[
        "scenario"
    ].eq(scenario)
    gated_assessment.loc[assessment_mask, "transition_feasibility"] = INFEASIBLE
    gated_assessment.loc[assessment_mask, "max_transition_capacity_ratio"] = 1.2
    gated_assessment.loc[assessment_mask, "total_transition_overrun_area_m2"] = 1.0
    gated_assessment.loc[assessment_mask, "overrun_source_classes"] = 1
    gated_assessment.loc[assessment_mask, "land_estimate_readiness"] = "not_ready"
    gated_assessment.loc[assessment_mask, "manual_review_priority"] = "high"
    gated_assessment.loc[assessment_mask, "overall_assessment"] = "not_ready"

    gated_review_candidates = review_candidates.copy()
    candidate_mask = gated_review_candidates["zone"].eq(zone) & gated_review_candidates[
        "scenario"
    ].eq(scenario)
    gated_review_candidates.loc[candidate_mask, "transition_feasibility"] = INFEASIBLE
    gated_review_candidates.loc[
        candidate_mask,
        "max_transition_capacity_ratio",
    ] = 1.2
    gated_review_candidates.loc[
        candidate_mask,
        "total_transition_overrun_area_m2",
    ] = 1.0
    gated_review_candidates.loc[candidate_mask, "overrun_source_classes"] = 1
    gated_review_candidates.loc[candidate_mask, "land_estimate_readiness"] = "not_ready"
    gated_review_candidates.loc[candidate_mask, "manual_review_priority"] = "high"

    report = validate_transition_closure_artifacts(
        calibration,
        expansion,
        transitions,
        infeasible,
        gated_assessment,
        gated_review_candidates,
        zone_names=ZONES,
    )

    assert report.empty


def test_transition_validation_reports_capacity_watch_ready_label(
    calibration: pd.DataFrame,
    expansion: pd.DataFrame,
    transitions: pd.DataFrame,
    transition_feasibility: pd.DataFrame,
    assessment: pd.DataFrame,
    review_candidates: pd.DataFrame,
) -> None:
    watched = transition_feasibility.copy()
    calibrated_mask = watched["calibration"].eq("calibrated")
    row_idx = calibrated_mask.idxmax()
    zone = watched.loc[row_idx, "zone"]
    scenario = watched.loc[row_idx, "scenario"]
    watched.loc[row_idx, "transition_feasibility"] = CAPACITY_WATCH
    watched.loc[row_idx, "max_capacity_ratio"] = 0.8

    watched_assessment = assessment.copy()
    assessment_mask = watched_assessment["zone"].eq(zone) & watched_assessment[
        "scenario"
    ].eq(scenario)
    watched_assessment.loc[assessment_mask, "transition_feasibility"] = CAPACITY_WATCH
    watched_assessment.loc[assessment_mask, "max_transition_capacity_ratio"] = 0.8

    watched_candidates = review_candidates.copy()
    candidate_mask = watched_candidates["zone"].eq(zone) & watched_candidates[
        "scenario"
    ].eq(scenario)
    watched_candidates.loc[candidate_mask, "transition_feasibility"] = CAPACITY_WATCH
    watched_candidates.loc[candidate_mask, "max_transition_capacity_ratio"] = 0.8

    report = validate_transition_closure_artifacts(
        calibration,
        expansion,
        transitions,
        watched,
        watched_assessment,
        watched_candidates,
        zone_names=ZONES,
    )

    assert "capacity_watch_readiness_gate" in set(report["check"])


def test_exploration_validation_reports_invalid_diagnostic_type(
    method_comparison: pd.DataFrame,
    method_summary: pd.DataFrame,
    method_recommendations: pd.DataFrame,
    disagreement_typology: pd.DataFrame,
    disagreement_summary: pd.DataFrame,
) -> None:
    invalid = disagreement_typology.copy()
    invalid.loc[0, "diagnostic_type"] = "approved_for_model_input"

    report = validate_exploration_artifacts(
        method_comparison,
        method_summary,
        method_recommendations,
        invalid,
        disagreement_summary,
        zone_names=ZONES,
    )

    assert "diagnostic_type_allowed_values" in set(report["check"])
