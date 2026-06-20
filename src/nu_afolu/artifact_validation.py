from __future__ import annotations

from typing import TYPE_CHECKING

from nu_afolu._artifact_validation_core import (
    AREA_TOLERANCE_M2,
    ERROR,
    RATIO_TOLERANCE,
    REPORT_COLUMNS,
    ValidationRow,
    _report,
)
from nu_afolu._artifact_validation_schemas import (
    ADEQUACY_LABELS,
    AREA_ERROR_CLASSES,
    ASSESSMENT_COLUMNS,
    ASSESSMENT_FEASIBILITY_COLUMNS,
    CALIBRATION_COLUMNS,
    CALIBRATION_TYPES,
    DIAGNOSTIC_TYPES,
    DISAGREEMENT_SUMMARY_COLUMNS,
    DISAGREEMENT_TYPOLOGY_COLUMNS,
    EXPANSION_COLUMNS,
    EXTERNAL_BASELINE_COLUMNS,
    EXTERNAL_BASELINE_COMPARATORS,
    EXTERNAL_DATASETS,
    EXTERNAL_GROWTH_COLUMNS,
    EXTERNAL_REVIEW_FLAG_COLUMNS,
    EXTERNAL_SUMMARY_COLUMNS,
    FUTURE_YEARS,
    GROWTH_RISK_LABELS,
    METHOD_COMPARISON_COLUMNS,
    METHOD_RECOMMENDATION_COLUMNS,
    METHOD_SUMMARY_COLUMNS,
    READINESS_LABELS,
    RELIABILITY_LABELS,
    REVIEW_CANDIDATE_COLUMNS,
    REVIEW_PRIORITY_LABELS,
    SCALE_SENSITIVITY_COLUMNS,
    SENSITIVE_FLAG_LABELS,
    SENSITIVE_RISK_LABELS,
    SOURCE_CLASSES,
    SPATIAL_AGREEMENT_CLASSES,
    TRANSITION_COLUMNS,
)
from nu_afolu._calibration_validation import (
    _validate_calibration_table,
    _validate_scale_sensitivity_table,
)
from nu_afolu._exploration_validation import (
    _validate_disagreement_summary,
    _validate_disagreement_typology,
    _validate_method_comparison,
    _validate_method_recommendations,
    _validate_method_summary,
)
from nu_afolu._external_artifact_validation import (
    _validate_external_baseline_agreement,
    _validate_external_growth_alignment,
    _validate_external_review_flags,
    _validate_external_validation_summary,
)
from nu_afolu._transition_artifact_validation import (
    _validate_assessment_feasibility_consistency,
    _validate_assessment_growth_consistency,
    _validate_assessment_table,
    _validate_expansion_table,
    _validate_historical_growth_diagnostics,
    _validate_review_candidates,
    _validate_transition_feasibility_table,
    _validate_transition_table,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

__all__ = [
    "ADEQUACY_LABELS",
    "AREA_ERROR_CLASSES",
    "AREA_TOLERANCE_M2",
    "ASSESSMENT_COLUMNS",
    "ASSESSMENT_FEASIBILITY_COLUMNS",
    "CALIBRATION_COLUMNS",
    "CALIBRATION_TYPES",
    "DIAGNOSTIC_TYPES",
    "DISAGREEMENT_SUMMARY_COLUMNS",
    "DISAGREEMENT_TYPOLOGY_COLUMNS",
    "ERROR",
    "EXPANSION_COLUMNS",
    "EXTERNAL_BASELINE_COLUMNS",
    "EXTERNAL_BASELINE_COMPARATORS",
    "EXTERNAL_DATASETS",
    "EXTERNAL_GROWTH_COLUMNS",
    "EXTERNAL_REVIEW_FLAG_COLUMNS",
    "EXTERNAL_SUMMARY_COLUMNS",
    "FUTURE_YEARS",
    "GROWTH_RISK_LABELS",
    "METHOD_COMPARISON_COLUMNS",
    "METHOD_RECOMMENDATION_COLUMNS",
    "METHOD_SUMMARY_COLUMNS",
    "RATIO_TOLERANCE",
    "READINESS_LABELS",
    "RELIABILITY_LABELS",
    "REPORT_COLUMNS",
    "REVIEW_CANDIDATE_COLUMNS",
    "REVIEW_PRIORITY_LABELS",
    "SCALE_SENSITIVITY_COLUMNS",
    "SENSITIVE_FLAG_LABELS",
    "SENSITIVE_RISK_LABELS",
    "SOURCE_CLASSES",
    "SPATIAL_AGREEMENT_CLASSES",
    "TRANSITION_COLUMNS",
    "ValidationRow",
    "raise_for_validation_errors",
    "validate_calibration_artifacts",
    "validate_calibration_table",
    "validate_exploration_artifacts",
    "validate_external_validation_artifacts",
    "validate_transition_closure_artifacts",
]


def validate_calibration_artifacts(
    df_calibration: pd.DataFrame,
    df_scale_sensitivity: pd.DataFrame,
    *,
    zone_names: Iterable[str],
    thresholds: Iterable[float],
) -> pd.DataFrame:
    zones = tuple(zone_names)
    issues: list[ValidationRow] = []
    _validate_calibration_table(df_calibration, zones, issues)
    _validate_scale_sensitivity_table(df_scale_sensitivity, zones, thresholds, issues)
    return _report(issues)


def validate_calibration_table(
    df_calibration: pd.DataFrame,
    *,
    zone_names: Iterable[str],
) -> pd.DataFrame:
    issues: list[ValidationRow] = []
    _validate_calibration_table(df_calibration, tuple(zone_names), issues)
    return _report(issues)


def validate_transition_closure_artifacts(
    df_calibration: pd.DataFrame,
    df_chen_expansion: pd.DataFrame,
    df_chen_transitions: pd.DataFrame,
    df_transition_feasibility: pd.DataFrame,
    df_historical_growth_diagnostics: pd.DataFrame,
    df_land_estimation_assessment: pd.DataFrame,
    df_review_candidates: pd.DataFrame,
    *,
    zone_names: Iterable[str],
) -> pd.DataFrame:
    zones = tuple(zone_names)
    issues: list[ValidationRow] = []
    _validate_calibration_table(df_calibration, zones, issues)
    _validate_expansion_table(df_chen_expansion, zones, issues)
    _validate_transition_table(df_chen_expansion, df_chen_transitions, zones, issues)
    _validate_transition_feasibility_table(df_transition_feasibility, zones, issues)
    _validate_historical_growth_diagnostics(
        df_historical_growth_diagnostics,
        zones,
        issues,
    )
    _validate_assessment_table(df_land_estimation_assessment, zones, issues)
    _validate_assessment_feasibility_consistency(
        df_land_estimation_assessment,
        df_transition_feasibility,
        issues,
    )
    _validate_assessment_growth_consistency(
        df_land_estimation_assessment,
        df_historical_growth_diagnostics,
        issues,
    )
    _validate_review_candidates(
        df_review_candidates,
        df_land_estimation_assessment,
        zones,
        issues,
    )
    return _report(issues)


def validate_exploration_artifacts(
    df_method_comparison: pd.DataFrame,
    df_method_summary: pd.DataFrame,
    df_method_recommendation_candidates: pd.DataFrame,
    df_disagreement_typology: pd.DataFrame,
    df_disagreement_summary: pd.DataFrame,
    *,
    zone_names: Iterable[str],
) -> pd.DataFrame:
    zones = tuple(zone_names)
    issues: list[ValidationRow] = []
    _validate_method_comparison(df_method_comparison, zones, issues)
    _validate_method_summary(df_method_summary, df_method_comparison, issues)
    _validate_method_recommendations(
        df_method_recommendation_candidates,
        df_method_comparison,
        zones,
        issues,
    )
    _validate_disagreement_typology(df_disagreement_typology, zones, issues)
    _validate_disagreement_summary(
        df_disagreement_summary,
        df_disagreement_typology,
        issues,
    )
    return _report(issues)


def validate_external_validation_artifacts(
    df_external_baseline_agreement: pd.DataFrame,
    df_external_growth_alignment: pd.DataFrame,
    df_external_review_flags: pd.DataFrame,
    df_external_validation_summary: pd.DataFrame,
    df_land_estimation_assessment: pd.DataFrame,
    *,
    zone_names: Iterable[str],
) -> pd.DataFrame:
    zones = tuple(zone_names)
    issues: list[ValidationRow] = []
    _validate_external_baseline_agreement(
        df_external_baseline_agreement,
        zones,
        issues,
    )
    _validate_external_growth_alignment(
        df_external_growth_alignment,
        zones,
        issues,
    )
    _validate_external_review_flags(
        df_external_review_flags,
        df_external_baseline_agreement,
        df_external_growth_alignment,
        df_land_estimation_assessment,
        zones,
        issues,
    )
    _validate_external_validation_summary(
        df_external_validation_summary,
        df_external_review_flags,
        issues,
    )
    return _report(issues)


def raise_for_validation_errors(report: pd.DataFrame) -> None:
    if report.empty:
        return
    errors = report[report["severity"].eq(ERROR)]
    if errors.empty:
        return
    first = errors.iloc[0]
    message = (
        f"{len(errors)} artifact validation error(s); first error in "
        f"{first['artifact']}::{first['check']}: {first['message']}"
    )
    raise ValueError(message)
