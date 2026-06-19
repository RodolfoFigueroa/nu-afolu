from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from nu_afolu.chen import CHEN_YEARS, SSP_NAMES
from nu_afolu.constants import LABEL_LIST
from nu_afolu.external_validation import (
    BASELINE_COMPARATOR_SUPPORT_LABELS,
    EXTERNAL_ADVISORY_LABELS,
    EXTERNAL_BASELINE_LABELS,
    EXTERNAL_GROWTH_LABELS,
    classify_external_advisory,
    combine_baseline_support,
)
from nu_afolu.metrics import DEFAULT_CORRECTION_FACTOR_BOUNDS
from nu_afolu.transition_feasibility import (
    CAPACITY_WATCH,
    FEASIBLE,
    INFEASIBLE,
    TRANSITION_FEASIBILITY_COLUMNS,
    TRANSITION_FEASIBILITY_LABELS,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

REPORT_COLUMNS = ("artifact", "check", "severity", "message", "rows")
ERROR = "error"
AREA_TOLERANCE_M2 = 1e-3
RATIO_TOLERANCE = 1e-9

CALIBRATION_COLUMNS = (
    "zone",
    "scenario",
    "observed_area_m2",
    "chen_area_m2",
    "area_error_m2",
    "area_bias",
    "ape",
    "tp_area_m2",
    "fp_area_m2",
    "fn_area_m2",
    "precision",
    "recall",
    "iou",
    "correction_factor_raw",
    "correction_factor",
    "calibration_valid",
    "reliability",
)
SCALE_SENSITIVITY_COLUMNS = (
    "zone",
    "scenario",
    "threshold",
    "observed_area_m2",
    "chen_area_m2",
    "tp_area_m2",
    "fp_area_m2",
    "fn_area_m2",
    "precision",
    "recall",
    "iou",
    "area_bias",
)
EXPANSION_COLUMNS = (
    "zone",
    "scenario",
    "period_start_year",
    "year",
    "chen_new_area_m2",
    "nonsettlement_source_area_m2",
    "existing_settlement_area_m2",
    "correction_factor",
    "reliability",
)
TRANSITION_COLUMNS = (
    "zone",
    "scenario",
    "period_start_year",
    "year",
    "from_class",
    "to_class",
    "correction_factor",
    "reliability",
    "calibration",
    "area_m2",
    "scaled_up_area_only",
)
ASSESSMENT_FEASIBILITY_COLUMNS = (
    "transition_feasibility",
    "max_transition_capacity_ratio",
    "total_transition_overrun_area_m2",
    "overrun_source_classes",
    "limiting_transition_source_class",
)
ASSESSMENT_COLUMNS = (
    "zone",
    "scenario",
    "observed_area_m2",
    "chen_area_m2",
    "area_error_m2",
    "area_bias",
    "ape",
    "tp_area_m2",
    "fp_area_m2",
    "fn_area_m2",
    "precision",
    "recall",
    "iou",
    "correction_factor_raw",
    "correction_factor",
    "calibration_valid",
    "reliability",
    "observed_settlement_area_2020_m2",
    "observed_total_area_2020_m2",
    "observed_settlement_fraction_2020",
    "recent_growth_area_m2",
    "max_chen_new_area_m2",
    "max_chen_to_recent_growth_ratio",
    "worst_growth_plausibility",
    "max_sensitive_share",
    "max_watch_share",
    "max_sensitive_area_m2",
    "worst_sensitive_flag",
    "area_adequacy",
    "spatial_adequacy",
    "calibration_adequacy",
    "growth_risk",
    "sensitive_class_risk",
    *ASSESSMENT_FEASIBILITY_COLUMNS,
    "land_estimate_readiness",
    "manual_review_priority",
    "overall_assessment",
)
REVIEW_CANDIDATE_COLUMNS = (
    "zone",
    "scenario",
    "review_reason",
    "land_estimate_readiness",
    "manual_review_priority",
    "calibration_adequacy",
    "growth_risk",
    "sensitive_class_risk",
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
    *ASSESSMENT_FEASIBILITY_COLUMNS,
)
METHOD_COMPARISON_COLUMNS = (
    "zone",
    "scenario",
    "method",
    "observed_threshold",
    "buffer_m",
    "observed_area_m2",
    "chen_area_m2",
    "area_error_m2",
    "area_bias",
    "ape",
    "correction_factor_raw",
    "correction_factor",
    "calibration_valid",
    "precision",
    "recall",
    "iou",
    "buffered_precision",
    "buffered_recall",
    "buffered_f1",
    "spatial_metric_name",
    "spatial_score",
    "valid_comparison",
)
METHOD_SUMMARY_COLUMNS = (
    "method",
    "rows",
    "valid_comparisons",
    "median_spatial_score",
    "mean_spatial_score",
    "median_ape_pct",
    "median_area_bias",
    "median_correction_factor",
    "valid_share",
)
METHOD_RECOMMENDATION_COLUMNS = (
    "zone",
    "scenario",
    "method",
    "spatial_metric_name",
    "spatial_score",
    "ape",
    "area_bias",
    "correction_factor",
    "observed_threshold",
    "buffer_m",
    "valid_comparison",
)
DISAGREEMENT_TYPOLOGY_COLUMNS = (
    "zone",
    "scenario",
    "diagnostic_type",
    "area_error_class",
    "spatial_agreement_class",
    "current_iou",
    "current_ape",
    "current_area_bias",
    "current_correction_factor",
    "strict_iou",
    "strict_ape",
    "buffered_f1_widest",
    "buffered_gain_over_current",
    "strict_iou_gain_over_current",
    "strict_ape_delta",
    "current_valid",
    "review_score",
)
DISAGREEMENT_SUMMARY_COLUMNS = (
    "diagnostic_type",
    "rows",
    "median_current_iou",
    "median_current_ape",
    "median_buffered_gain",
    "median_strict_iou_gain",
    "max_review_score",
    "share",
)
EXTERNAL_BASELINE_COMPARATORS = frozenset(
    {"glc_settlements_2020", "chen_urban_2020"},
)
EXTERNAL_DATASETS = frozenset({"ghsl_built_surface"})
EXTERNAL_BASELINE_COLUMNS = (
    "zone",
    "scenario",
    "comparator",
    "external_dataset",
    "external_year",
    "external_area_m2",
    "comparator_area_m2",
    "tp_area_m2",
    "fp_area_m2",
    "fn_area_m2",
    "precision",
    "recall",
    "iou",
    "area_bias",
    "ape",
    "comparator_support",
)
EXTERNAL_GROWTH_COLUMNS = (
    "zone",
    "scenario",
    "calibration",
    "external_dataset",
    "period_start_year",
    "year",
    "ghsl_growth_area_m2",
    "chen_growth_area_m2",
    "chen_to_external_growth_ratio",
    "growth_alignment",
)
EXTERNAL_REVIEW_FLAG_COLUMNS = (
    "zone",
    "scenario",
    "glc_baseline_support",
    "chen_baseline_support",
    "external_baseline_validation",
    "calibrated_growth_alignment",
    "external_advisory",
    "land_estimate_readiness",
    "manual_review_priority",
    "overall_assessment",
    "transition_feasibility",
    "reliability",
    "iou",
    "ape",
    "ghsl_growth_area_m2",
    "calibrated_chen_growth_area_m2",
    "calibrated_chen_to_external_growth_ratio",
)
EXTERNAL_SUMMARY_COLUMNS = (
    "scenario",
    "external_advisory",
    "external_baseline_validation",
    "calibrated_growth_alignment",
    "rows",
    "share",
    "median_ghsl_growth_area_m2",
    "median_calibrated_chen_to_external_growth_ratio",
)

RELIABILITY_LABELS = frozenset({"high", "medium", "low"})
CALIBRATION_TYPES = frozenset({"raw", "calibrated"})
READINESS_LABELS = frozenset(
    {"ready_for_manual_review", "needs_targeted_review", "not_ready"},
)
REVIEW_PRIORITY_LABELS = frozenset({"low", "medium", "high"})
ADEQUACY_LABELS = frozenset({"good", "moderate", "poor"})
GROWTH_RISK_LABELS = frozenset({"low", "watch", "high", "review"})
SENSITIVE_RISK_LABELS = frozenset({"low", "watch", "high"})
GROWTH_PLAUSIBILITY_LABELS = frozenset(
    {
        "consistent",
        "low_growth",
        "high_growth",
        "extreme_growth",
        "insufficient_history",
    },
)
SENSITIVE_FLAG_LABELS = frozenset({"low", "watch", "high"})
DIAGNOSTIC_TYPES = frozenset(
    {
        "invalid_current_calibration",
        "tolerance_masks_area_mismatch",
        "weak_even_with_tolerance",
        "strict_threshold_improves_overlap",
        "stable_current_candidate",
        "needs_targeted_method_review",
    },
)
AREA_ERROR_CLASSES = frozenset(
    {"area_close", "moderate_area_mismatch", "large_area_mismatch"},
)
SPATIAL_AGREEMENT_CLASSES = frozenset(
    {"strong_current_overlap", "moderate_current_overlap", "weak_current_overlap"},
)
SOURCE_CLASSES = tuple(label for label in LABEL_LIST if label != "settlements")
FUTURE_YEARS = tuple(year for year in CHEN_YEARS if year > 2020)

ValidationRow = dict[str, object]


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
    _validate_assessment_table(df_land_estimation_assessment, zones, issues)
    _validate_assessment_feasibility_consistency(
        df_land_estimation_assessment,
        df_transition_feasibility,
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


def _validate_calibration_table(
    df: pd.DataFrame,
    zones: Sequence[str],
    issues: list[ValidationRow],
) -> None:
    artifact = "calibration"
    _check_required_columns(df, artifact, CALIBRATION_COLUMNS, issues)
    if not _has_columns(df, CALIBRATION_COLUMNS):
        return

    _check_exact_rows(df, artifact, len(zones) * len(SSP_NAMES), issues)
    _check_key_set(df, artifact, ["zone", "scenario"], [zones, SSP_NAMES], issues)
    _check_unique_key(df, artifact, ["zone", "scenario"], issues)
    _check_allowed_values(df, artifact, "reliability", RELIABILITY_LABELS, issues)
    _check_boolean_values(df, artifact, "calibration_valid", issues)
    _check_nonnegative(
        df,
        artifact,
        [
            "observed_area_m2",
            "chen_area_m2",
            "tp_area_m2",
            "fp_area_m2",
            "fn_area_m2",
            "correction_factor",
        ],
        issues,
    )
    _check_between(
        df,
        artifact,
        ["precision", "recall", "iou"],
        0,
        1,
        issues,
        allow_null=True,
    )
    _check_close(
        df,
        artifact,
        "area_error_m2",
        df["chen_area_m2"] - df["observed_area_m2"],
        "area_error_consistency",
        issues,
    )
    observed = df["observed_area_m2"].replace(0, np.nan)
    chen = df["chen_area_m2"].replace(0, np.nan)
    _check_close(
        df,
        artifact,
        "area_bias",
        df["chen_area_m2"].div(observed),
        "area_bias_consistency",
        issues,
        allow_null=True,
    )
    _check_close(
        df,
        artifact,
        "ape",
        df["area_error_m2"].abs().div(observed),
        "ape_consistency",
        issues,
        allow_null=True,
    )
    _check_close(
        df,
        artifact,
        "correction_factor_raw",
        df["observed_area_m2"].div(chen),
        "correction_factor_raw_consistency",
        issues,
        allow_null=True,
    )
    valid = df["calibration_valid"].astype(bool)
    lower, upper = DEFAULT_CORRECTION_FACTOR_BOUNDS
    valid_factors = df.loc[valid, "correction_factor"]
    invalid_factors = df.loc[~valid, "correction_factor"]
    out_of_bounds = valid_factors.lt(lower - RATIO_TOLERANCE) | valid_factors.gt(
        upper + RATIO_TOLERANCE,
    )
    _add_if(
        issues,
        bool(out_of_bounds.sum()),
        artifact,
        "correction_factor_bounds",
        "Valid calibration rows must have clipped correction factors within bounds.",
        int(out_of_bounds.sum()),
    )
    _add_if(
        issues,
        bool((invalid_factors.sub(1.0).abs().gt(RATIO_TOLERANCE)).sum()),
        artifact,
        "invalid_correction_factor_fallback",
        "Invalid calibration rows must use correction_factor == 1.0.",
        int((invalid_factors.sub(1.0).abs().gt(RATIO_TOLERANCE)).sum()),
    )


def _validate_scale_sensitivity_table(
    df: pd.DataFrame,
    zones: Sequence[str],
    thresholds: Iterable[float],
    issues: list[ValidationRow],
) -> None:
    artifact = "scale_sensitivity"
    threshold_values = tuple(float(value) for value in thresholds)
    _check_required_columns(df, artifact, SCALE_SENSITIVITY_COLUMNS, issues)
    if not _has_columns(df, SCALE_SENSITIVITY_COLUMNS):
        return

    expected_rows = len(zones) * len(SSP_NAMES) * len(threshold_values)
    _check_exact_rows(df, artifact, expected_rows, issues)
    _check_key_set(
        df,
        artifact,
        ["zone", "scenario", "threshold"],
        [zones, SSP_NAMES, threshold_values],
        issues,
    )
    _check_unique_key(df, artifact, ["zone", "scenario", "threshold"], issues)
    _check_nonnegative(
        df,
        artifact,
        ["observed_area_m2", "chen_area_m2", "tp_area_m2", "fp_area_m2", "fn_area_m2"],
        issues,
    )
    _check_between(
        df,
        artifact,
        ["precision", "recall", "iou"],
        0,
        1,
        issues,
        allow_null=True,
    )
    _check_nonnegative(df, artifact, ["area_bias"], issues, allow_null=True)


def _validate_expansion_table(
    df: pd.DataFrame,
    zones: Sequence[str],
    issues: list[ValidationRow],
) -> None:
    artifact = "chen_expansion"
    _check_required_columns(df, artifact, EXPANSION_COLUMNS, issues)
    if not _has_columns(df, EXPANSION_COLUMNS):
        return

    _check_exact_rows(
        df,
        artifact,
        len(zones) * len(SSP_NAMES) * len(FUTURE_YEARS),
        issues,
    )
    _check_key_set(
        df,
        artifact,
        ["zone", "scenario", "year"],
        [zones, SSP_NAMES, FUTURE_YEARS],
        issues,
    )
    _check_unique_key(df, artifact, ["zone", "scenario", "year"], issues)
    _check_allowed_values(df, artifact, "reliability", RELIABILITY_LABELS, issues)
    _check_nonnegative(
        df,
        artifact,
        [
            "chen_new_area_m2",
            "nonsettlement_source_area_m2",
            "existing_settlement_area_m2",
            "correction_factor",
        ],
        issues,
    )
    _check_close(
        df,
        artifact,
        "period_start_year",
        df["year"] - 10,
        "period_start_year_consistency",
        issues,
    )
    _check_close(
        df,
        artifact,
        "chen_new_area_m2",
        df["nonsettlement_source_area_m2"] + df["existing_settlement_area_m2"],
        "chen_new_area_consistency",
        issues,
    )


def _validate_transition_table(
    df_expansion: pd.DataFrame,
    df: pd.DataFrame,
    zones: Sequence[str],
    issues: list[ValidationRow],
) -> None:
    artifact = "chen_transitions"
    _check_required_columns(df, artifact, TRANSITION_COLUMNS, issues)
    if not _has_columns(df, TRANSITION_COLUMNS):
        return

    expected_rows = (
        len(zones) * len(SSP_NAMES) * len(FUTURE_YEARS) * len(SOURCE_CLASSES) * 2
    )
    _check_exact_rows(df, artifact, expected_rows, issues)
    _check_key_set(
        df,
        artifact,
        ["zone", "scenario", "year", "from_class", "calibration"],
        [zones, SSP_NAMES, FUTURE_YEARS, SOURCE_CLASSES, CALIBRATION_TYPES],
        issues,
    )
    _check_unique_key(
        df,
        artifact,
        ["zone", "scenario", "period_start_year", "year", "from_class", "calibration"],
        issues,
    )
    _check_allowed_values(df, artifact, "reliability", RELIABILITY_LABELS, issues)
    _check_allowed_values(df, artifact, "from_class", SOURCE_CLASSES, issues)
    _check_allowed_values(df, artifact, "to_class", {"settlements"}, issues)
    _check_allowed_values(df, artifact, "calibration", CALIBRATION_TYPES, issues)
    _check_boolean_values(df, artifact, "scaled_up_area_only", issues)
    _check_nonnegative(df, artifact, ["correction_factor", "area_m2"], issues)
    _check_close(
        df,
        artifact,
        "period_start_year",
        df["year"] - 10,
        "period_start_year_consistency",
        issues,
    )

    raw = df[df["calibration"].eq("raw")]
    calibrated = df[df["calibration"].eq("calibrated")]
    _add_if(
        issues,
        bool(raw["scaled_up_area_only"].astype(bool).sum()),
        artifact,
        "raw_scaled_up_flag",
        "Raw transition rows must not be marked scaled_up_area_only.",
        int(raw["scaled_up_area_only"].astype(bool).sum()),
    )
    scaled_expected = calibrated["correction_factor"].gt(1.0 + RATIO_TOLERANCE)
    scaled_actual = calibrated["scaled_up_area_only"].astype(bool)
    _add_if(
        issues,
        bool((scaled_actual != scaled_expected).sum()),
        artifact,
        "calibrated_scaled_up_flag",
        "Calibrated transition scaled_up_area_only must match correction_factor > 1.",
        int((scaled_actual != scaled_expected).sum()),
    )

    key = ["zone", "scenario", "period_start_year", "year", "from_class"]
    paired = raw[[*key, "area_m2", "correction_factor"]].merge(
        calibrated[[*key, "area_m2"]],
        on=key,
        suffixes=("_raw", "_calibrated"),
        how="outer",
    )
    _add_if(
        issues,
        bool(paired.isna().any(axis=1).sum()),
        artifact,
        "raw_calibrated_pairing",
        "Every raw transition row must have one calibrated counterpart.",
        int(paired.isna().any(axis=1).sum()),
    )
    paired_complete = paired.dropna()
    if not paired_complete.empty:
        expected_calibrated = (
            paired_complete["area_m2_raw"] * paired_complete["correction_factor"]
        )
        mismatched = (
            paired_complete["area_m2_calibrated"]
            .sub(expected_calibrated)
            .abs()
            .gt(AREA_TOLERANCE_M2)
        )
        _add_if(
            issues,
            bool(mismatched.sum()),
            artifact,
            "calibrated_area_consistency",
            "Calibrated transition areas must equal raw area times correction factor.",
            int(mismatched.sum()),
        )

    if _has_columns(df_expansion, EXPANSION_COLUMNS):
        raw_totals = (
            raw.groupby(["zone", "scenario", "year"], as_index=False)["area_m2"]
            .sum()
            .rename(columns={"area_m2": "raw_transition_area_m2"})
        )
        expansion_totals = df_expansion[
            ["zone", "scenario", "year", "nonsettlement_source_area_m2"]
        ]
        compared = raw_totals.merge(
            expansion_totals,
            on=["zone", "scenario", "year"],
            how="outer",
        )
        missing = compared.isna().any(axis=1)
        _add_if(
            issues,
            bool(missing.sum()),
            artifact,
            "raw_transition_expansion_pairing",
            "Raw transition totals must pair with expansion source totals.",
            int(missing.sum()),
        )
        compared_complete = compared.dropna()
        mismatched = (
            compared_complete["raw_transition_area_m2"]
            .sub(compared_complete["nonsettlement_source_area_m2"])
            .abs()
            .gt(AREA_TOLERANCE_M2)
        )
        _add_if(
            issues,
            bool(mismatched.sum()),
            artifact,
            "raw_transition_total_consistency",
            "Raw transition totals must equal nonsettlement source area.",
            int(mismatched.sum()),
        )


def _validate_transition_feasibility_table(
    df: pd.DataFrame,
    zones: Sequence[str],
    issues: list[ValidationRow],
) -> None:
    artifact = "transition_feasibility"
    _check_required_columns(df, artifact, TRANSITION_FEASIBILITY_COLUMNS, issues)
    if not _has_columns(df, TRANSITION_FEASIBILITY_COLUMNS):
        return

    _check_exact_rows(
        df,
        artifact,
        len(zones) * len(SSP_NAMES) * len(CALIBRATION_TYPES),
        issues,
    )
    _check_key_set(
        df,
        artifact,
        ["zone", "scenario", "calibration"],
        [zones, SSP_NAMES, CALIBRATION_TYPES],
        issues,
    )
    _check_unique_key(df, artifact, ["zone", "scenario", "calibration"], issues)
    _check_allowed_values(df, artifact, "calibration", CALIBRATION_TYPES, issues)
    _check_allowed_values(
        df,
        artifact,
        "transition_feasibility",
        TRANSITION_FEASIBILITY_LABELS,
        issues,
    )
    _check_allowed_values(df, artifact, "limiting_from_class", SOURCE_CLASSES, issues)
    _check_allowed_values(
        df,
        artifact,
        "first_overrun_year",
        FUTURE_YEARS,
        issues,
        allow_null=True,
    )
    _check_nonnegative(
        df,
        artifact,
        [
            "max_capacity_ratio",
            "total_overrun_area_m2",
            "overrun_source_classes",
        ],
        issues,
    )

    raw = df[df["calibration"].eq("raw")]
    raw_not_feasible = raw["transition_feasibility"].ne(FEASIBLE)
    _add_if(
        issues,
        bool(raw_not_feasible.sum()),
        artifact,
        "raw_transition_feasibility",
        "Raw transition feasibility rows must remain feasible.",
        int(raw_not_feasible.sum()),
    )

    has_overrun = df["total_overrun_area_m2"].gt(AREA_TOLERANCE_M2)
    labeled_infeasible = df["transition_feasibility"].eq(INFEASIBLE)
    _add_if(
        issues,
        bool((has_overrun != labeled_infeasible).sum()),
        artifact,
        "infeasible_label_consistency",
        "Rows with transition overrun must be labeled infeasible, and vice versa.",
        int((has_overrun != labeled_infeasible).sum()),
    )
    overrun_classes = df["overrun_source_classes"].gt(0)
    _add_if(
        issues,
        bool((has_overrun != overrun_classes).sum()),
        artifact,
        "overrun_source_class_consistency",
        "Rows with transition overrun must report at least one overrun source class.",
        int((has_overrun != overrun_classes).sum()),
    )
    first_year_null = df["first_overrun_year"].isna()
    first_year_mismatch = (has_overrun & first_year_null) | (
        ~has_overrun & ~first_year_null
    )
    _add_if(
        issues,
        bool(first_year_mismatch.sum()),
        artifact,
        "first_overrun_year_consistency",
        "first_overrun_year must be present only for infeasible rows.",
        int(first_year_mismatch.sum()),
    )


def _validate_assessment_table(
    df: pd.DataFrame,
    zones: Sequence[str],
    issues: list[ValidationRow],
) -> None:
    artifact = "land_estimation_assessment"
    _check_required_columns(df, artifact, ASSESSMENT_COLUMNS, issues)
    if not _has_columns(df, ASSESSMENT_COLUMNS):
        return

    _check_exact_rows(df, artifact, len(zones) * len(SSP_NAMES), issues)
    _check_key_set(df, artifact, ["zone", "scenario"], [zones, SSP_NAMES], issues)
    _check_unique_key(df, artifact, ["zone", "scenario"], issues)
    _check_allowed_values(df, artifact, "reliability", RELIABILITY_LABELS, issues)
    _check_allowed_values(df, artifact, "area_adequacy", ADEQUACY_LABELS, issues)
    _check_allowed_values(df, artifact, "spatial_adequacy", ADEQUACY_LABELS, issues)
    _check_allowed_values(df, artifact, "calibration_adequacy", ADEQUACY_LABELS, issues)
    _check_allowed_values(df, artifact, "growth_risk", GROWTH_RISK_LABELS, issues)
    _check_allowed_values(
        df,
        artifact,
        "sensitive_class_risk",
        SENSITIVE_RISK_LABELS,
        issues,
    )
    _check_allowed_values(
        df,
        artifact,
        "transition_feasibility",
        TRANSITION_FEASIBILITY_LABELS,
        issues,
    )
    _check_allowed_values(
        df,
        artifact,
        "land_estimate_readiness",
        READINESS_LABELS,
        issues,
    )
    _check_allowed_values(
        df,
        artifact,
        "manual_review_priority",
        REVIEW_PRIORITY_LABELS,
        issues,
    )
    _check_allowed_values(df, artifact, "overall_assessment", READINESS_LABELS, issues)
    _check_allowed_values(
        df,
        artifact,
        "worst_growth_plausibility",
        GROWTH_PLAUSIBILITY_LABELS,
        issues,
        allow_null=True,
    )
    _check_allowed_values(
        df,
        artifact,
        "worst_sensitive_flag",
        SENSITIVE_FLAG_LABELS,
        issues,
        allow_null=True,
    )
    _check_nonnegative(
        df,
        artifact,
        [
            "observed_settlement_area_2020_m2",
            "observed_total_area_2020_m2",
            "recent_growth_area_m2",
            "max_chen_new_area_m2",
            "max_sensitive_area_m2",
        ],
        issues,
        allow_null=True,
    )
    _check_between(
        df,
        artifact,
        ["observed_settlement_fraction_2020", "max_sensitive_share", "max_watch_share"],
        0,
        1,
        issues,
        allow_null=True,
    )
    _check_allowed_values(
        df,
        artifact,
        "limiting_transition_source_class",
        SOURCE_CLASSES,
        issues,
        allow_null=True,
    )
    _check_nonnegative(
        df,
        artifact,
        [
            "max_transition_capacity_ratio",
            "total_transition_overrun_area_m2",
            "overrun_source_classes",
        ],
        issues,
    )
    infeasible_not_blocked = df["transition_feasibility"].eq(INFEASIBLE) & ~df[
        "land_estimate_readiness"
    ].eq("not_ready")
    _add_if(
        issues,
        bool(infeasible_not_blocked.sum()),
        artifact,
        "infeasible_readiness_gate",
        "Infeasible transition rows must be marked not_ready.",
        int(infeasible_not_blocked.sum()),
    )
    watch_ready = df["transition_feasibility"].eq(CAPACITY_WATCH) & df[
        "land_estimate_readiness"
    ].eq("ready_for_manual_review")
    _add_if(
        issues,
        bool(watch_ready.sum()),
        artifact,
        "capacity_watch_readiness_gate",
        "Capacity-watch transition rows must receive targeted review.",
        int(watch_ready.sum()),
    )


def _validate_assessment_feasibility_consistency(
    df_assessment: pd.DataFrame,
    df_transition_feasibility: pd.DataFrame,
    issues: list[ValidationRow],
) -> None:
    artifact = "land_estimation_assessment"
    required_assessment_columns = {
        "zone",
        "scenario",
        *ASSESSMENT_FEASIBILITY_COLUMNS,
    }
    if not _has_columns(df_assessment, required_assessment_columns) or not _has_columns(
        df_transition_feasibility,
        TRANSITION_FEASIBILITY_COLUMNS,
    ):
        return

    calibrated = (
        df_transition_feasibility[
            df_transition_feasibility["calibration"].eq("calibrated")
        ][
            [
                "zone",
                "scenario",
                "transition_feasibility",
                "max_capacity_ratio",
                "total_overrun_area_m2",
                "overrun_source_classes",
                "limiting_from_class",
            ]
        ]
        .rename(
            columns={
                "max_capacity_ratio": "expected_max_transition_capacity_ratio",
                "total_overrun_area_m2": "expected_total_transition_overrun_area_m2",
                "overrun_source_classes": "expected_overrun_source_classes",
                "limiting_from_class": "expected_limiting_transition_source_class",
                "transition_feasibility": "expected_transition_feasibility",
            },
        )
    )
    compared = df_assessment.merge(calibrated, on=["zone", "scenario"], how="outer")
    expected_columns = [
        "expected_transition_feasibility",
        "expected_max_transition_capacity_ratio",
        "expected_total_transition_overrun_area_m2",
        "expected_overrun_source_classes",
        "expected_limiting_transition_source_class",
    ]
    missing = compared[expected_columns].isna().any(axis=1)
    _add_if(
        issues,
        bool(missing.sum()),
        artifact,
        "transition_feasibility_pairing",
        "Assessment rows must pair with calibrated transition feasibility rows.",
        int(missing.sum()),
    )
    complete = compared[~missing]
    if complete.empty:
        return

    label_mismatch = complete["transition_feasibility"].ne(
        complete["expected_transition_feasibility"],
    )
    _add_if(
        issues,
        bool(label_mismatch.sum()),
        artifact,
        "transition_feasibility_consistency",
        "Assessment transition_feasibility must match calibrated feasibility.",
        int(label_mismatch.sum()),
    )
    class_mismatch = complete["limiting_transition_source_class"].ne(
        complete["expected_limiting_transition_source_class"],
    )
    _add_if(
        issues,
        bool(class_mismatch.sum()),
        artifact,
        "limiting_source_consistency",
        "Assessment limiting source class must match calibrated feasibility.",
        int(class_mismatch.sum()),
    )
    numeric_pairs = [
        ("max_transition_capacity_ratio", "expected_max_transition_capacity_ratio"),
        (
            "total_transition_overrun_area_m2",
            "expected_total_transition_overrun_area_m2",
        ),
        ("overrun_source_classes", "expected_overrun_source_classes"),
    ]
    for observed_column, expected_column in numeric_pairs:
        mismatch = (
            pd.to_numeric(complete[observed_column], errors="coerce")
            .sub(pd.to_numeric(complete[expected_column], errors="coerce"))
            .abs()
            .gt(AREA_TOLERANCE_M2)
        )
        _add_if(
            issues,
            bool(mismatch.sum()),
            artifact,
            f"{observed_column}_consistency",
            f"{observed_column} must match calibrated transition feasibility.",
            int(mismatch.sum()),
        )


def _validate_review_candidates(
    df: pd.DataFrame,
    df_assessment: pd.DataFrame,
    zones: Sequence[str],
    issues: list[ValidationRow],
) -> None:
    artifact = "review_candidates"
    _check_required_columns(df, artifact, REVIEW_CANDIDATE_COLUMNS, issues)
    if not _has_columns(df, REVIEW_CANDIDATE_COLUMNS):
        return

    _check_allowed_values(df, artifact, "zone", zones, issues)
    _check_allowed_values(df, artifact, "scenario", SSP_NAMES, issues)
    _check_unique_key(df, artifact, ["zone", "scenario", "review_reason"], issues)
    _check_allowed_values(df, artifact, "reliability", RELIABILITY_LABELS, issues)
    _check_allowed_values(df, artifact, "area_adequacy", ADEQUACY_LABELS, issues)
    _check_allowed_values(df, artifact, "spatial_adequacy", ADEQUACY_LABELS, issues)
    _check_allowed_values(df, artifact, "calibration_adequacy", ADEQUACY_LABELS, issues)
    _check_allowed_values(df, artifact, "growth_risk", GROWTH_RISK_LABELS, issues)
    _check_allowed_values(
        df,
        artifact,
        "sensitive_class_risk",
        SENSITIVE_RISK_LABELS,
        issues,
    )
    _check_allowed_values(
        df,
        artifact,
        "transition_feasibility",
        TRANSITION_FEASIBILITY_LABELS,
        issues,
    )
    _check_allowed_values(
        df,
        artifact,
        "land_estimate_readiness",
        READINESS_LABELS,
        issues,
    )
    _check_allowed_values(
        df,
        artifact,
        "manual_review_priority",
        REVIEW_PRIORITY_LABELS,
        issues,
    )
    _check_allowed_values(
        df,
        artifact,
        "worst_growth_plausibility",
        GROWTH_PLAUSIBILITY_LABELS,
        issues,
        allow_null=True,
    )
    _check_allowed_values(
        df,
        artifact,
        "worst_sensitive_flag",
        SENSITIVE_FLAG_LABELS,
        issues,
        allow_null=True,
    )
    _check_allowed_values(
        df,
        artifact,
        "limiting_transition_source_class",
        SOURCE_CLASSES,
        issues,
        allow_null=True,
    )
    _check_between(
        df,
        artifact,
        ["iou", "max_sensitive_share"],
        0,
        1,
        issues,
        allow_null=True,
    )
    _check_nonnegative(
        df,
        artifact,
        [
            "correction_factor",
            "max_chen_to_recent_growth_ratio",
            "max_sensitive_area_m2",
            "max_transition_capacity_ratio",
            "total_transition_overrun_area_m2",
            "overrun_source_classes",
        ],
        issues,
        allow_null=True,
    )
    if _has_columns(df_assessment, ("zone", "scenario")):
        assessment_keys = set(
            df_assessment[["zone", "scenario"]].itertuples(index=False, name=None),
        )
        candidate_keys = set(
            df[["zone", "scenario"]].itertuples(index=False, name=None),
        )
        missing = candidate_keys.difference(assessment_keys)
        _add_if(
            issues,
            bool(missing),
            artifact,
            "assessment_key_subset",
            "Review candidates must refer to land-estimation assessment rows.",
            len(missing),
        )
    assessment_columns = ("zone", "scenario", *ASSESSMENT_FEASIBILITY_COLUMNS)
    if _has_columns(df_assessment, assessment_columns):
        expected = df_assessment[
            ["zone", "scenario", *ASSESSMENT_FEASIBILITY_COLUMNS]
        ].rename(
            columns={
                column: f"expected_{column}"
                for column in ASSESSMENT_FEASIBILITY_COLUMNS
            },
        )
        compared = df.merge(expected, on=["zone", "scenario"], how="left")
        missing_expected = compared[
            [f"expected_{column}" for column in ASSESSMENT_FEASIBILITY_COLUMNS]
        ].isna().any(axis=1)
        _add_if(
            issues,
            bool(missing_expected.sum()),
            artifact,
            "assessment_feasibility_pairing",
            "Review candidate feasibility context must pair with assessment rows.",
            int(missing_expected.sum()),
        )
        complete = compared[~missing_expected]
        if complete.empty:
            return

        for column in (
            "transition_feasibility",
            "limiting_transition_source_class",
        ):
            mismatch = complete[column].ne(complete[f"expected_{column}"])
            _add_if(
                issues,
                bool(mismatch.sum()),
                artifact,
                f"{column}_consistency",
                f"Review candidate {column} must match assessment.",
                int(mismatch.sum()),
            )
        for column in (
            "max_transition_capacity_ratio",
            "total_transition_overrun_area_m2",
            "overrun_source_classes",
        ):
            mismatch = (
                pd.to_numeric(complete[column], errors="coerce")
                .sub(pd.to_numeric(complete[f"expected_{column}"], errors="coerce"))
                .abs()
                .gt(AREA_TOLERANCE_M2)
            )
            _add_if(
                issues,
                bool(mismatch.sum()),
                artifact,
                f"{column}_consistency",
                f"Review candidate {column} must match assessment.",
                int(mismatch.sum()),
            )


def _validate_method_comparison(
    df: pd.DataFrame,
    zones: Sequence[str],
    issues: list[ValidationRow],
) -> None:
    artifact = "method_comparison"
    _check_required_columns(df, artifact, METHOD_COMPARISON_COLUMNS, issues)
    if not _has_columns(df, METHOD_COMPARISON_COLUMNS):
        return

    _check_allowed_values(df, artifact, "zone", zones, issues)
    _check_allowed_values(df, artifact, "scenario", SSP_NAMES, issues)
    _check_unique_key(df, artifact, ["zone", "scenario", "method"], issues)
    _check_boolean_values(df, artifact, "calibration_valid", issues)
    _check_boolean_values(df, artifact, "valid_comparison", issues)
    _check_nonnegative(
        df,
        artifact,
        ["observed_area_m2", "chen_area_m2", "correction_factor"],
        issues,
    )
    _check_between(
        df,
        artifact,
        [
            "precision",
            "recall",
            "iou",
            "buffered_precision",
            "buffered_recall",
            "buffered_f1",
            "spatial_score",
        ],
        0,
        1,
        issues,
        allow_null=True,
    )
    counts = df.groupby(["zone", "scenario"])["method"].nunique()
    expected_count = counts.iloc[0] if not counts.empty else 0
    _add_if(
        issues,
        bool(counts.ne(expected_count).sum()),
        artifact,
        "method_count_consistency",
        "Each zone-scenario pair must have the same number of methods.",
        int(counts.ne(expected_count).sum()),
    )
    _check_exact_rows(
        df,
        artifact,
        len(zones) * len(SSP_NAMES) * int(expected_count),
        issues,
    )


def _validate_method_summary(
    df_summary: pd.DataFrame,
    df_comparison: pd.DataFrame,
    issues: list[ValidationRow],
) -> None:
    artifact = "method_summary"
    _check_required_columns(df_summary, artifact, METHOD_SUMMARY_COLUMNS, issues)
    if not _has_columns(df_summary, METHOD_SUMMARY_COLUMNS):
        return

    methods = set(df_comparison["method"]) if "method" in df_comparison else set()
    _check_allowed_values(df_summary, artifact, "method", methods, issues)
    _check_unique_key(df_summary, artifact, ["method"], issues)
    _check_between(df_summary, artifact, ["valid_share"], 0, 1, issues)
    _check_nonnegative(
        df_summary,
        artifact,
        ["rows", "valid_comparisons", "median_correction_factor"],
        issues,
        allow_null=True,
    )
    if _has_columns(df_comparison, ("method", "valid_comparison")):
        counts = (
            df_comparison.groupby("method", as_index=False)
            .agg(
                expected_rows=("method", "size"),
                expected_valid=("valid_comparison", "sum"),
            )
            .merge(df_summary, on="method", how="outer")
        )
        missing = counts.isna().any(axis=1)
        _add_if(
            issues,
            bool(missing.sum()),
            artifact,
            "method_summary_pairing",
            "Method summary rows must match method comparison methods.",
            int(missing.sum()),
        )
        complete = counts.dropna()
        mismatched = complete["rows"].ne(complete["expected_rows"]) | complete[
            "valid_comparisons"
        ].ne(complete["expected_valid"])
        _add_if(
            issues,
            bool(mismatched.sum()),
            artifact,
            "method_summary_counts",
            "Method summary counts must match method comparison rows.",
            int(mismatched.sum()),
        )


def _validate_method_recommendations(
    df: pd.DataFrame,
    df_comparison: pd.DataFrame,
    zones: Sequence[str],
    issues: list[ValidationRow],
) -> None:
    artifact = "method_recommendation_candidates"
    _check_required_columns(df, artifact, METHOD_RECOMMENDATION_COLUMNS, issues)
    if not _has_columns(df, METHOD_RECOMMENDATION_COLUMNS):
        return

    _check_exact_rows(df, artifact, len(zones) * len(SSP_NAMES), issues)
    _check_key_set(df, artifact, ["zone", "scenario"], [zones, SSP_NAMES], issues)
    _check_unique_key(df, artifact, ["zone", "scenario"], issues)
    methods = set(df_comparison["method"]) if "method" in df_comparison else set()
    _check_allowed_values(df, artifact, "method", methods, issues)
    _check_boolean_values(df, artifact, "valid_comparison", issues)
    _check_between(df, artifact, ["spatial_score"], 0, 1, issues, allow_null=True)
    _check_nonnegative(
        df,
        artifact,
        ["ape", "correction_factor"],
        issues,
        allow_null=True,
    )


def _validate_disagreement_typology(
    df: pd.DataFrame,
    zones: Sequence[str],
    issues: list[ValidationRow],
) -> None:
    artifact = "disagreement_typology"
    _check_required_columns(df, artifact, DISAGREEMENT_TYPOLOGY_COLUMNS, issues)
    if not _has_columns(df, DISAGREEMENT_TYPOLOGY_COLUMNS):
        return

    _check_exact_rows(df, artifact, len(zones) * len(SSP_NAMES), issues)
    _check_key_set(df, artifact, ["zone", "scenario"], [zones, SSP_NAMES], issues)
    _check_unique_key(df, artifact, ["zone", "scenario"], issues)
    _check_allowed_values(df, artifact, "diagnostic_type", DIAGNOSTIC_TYPES, issues)
    _check_allowed_values(df, artifact, "area_error_class", AREA_ERROR_CLASSES, issues)
    _check_allowed_values(
        df,
        artifact,
        "spatial_agreement_class",
        SPATIAL_AGREEMENT_CLASSES,
        issues,
    )
    _check_boolean_values(df, artifact, "current_valid", issues)
    _check_between(
        df,
        artifact,
        ["current_iou", "strict_iou", "buffered_f1_widest"],
        0,
        1,
        issues,
        allow_null=True,
    )
    _check_nonnegative(
        df,
        artifact,
        ["current_ape", "current_correction_factor", "review_score"],
        issues,
        allow_null=True,
    )


def _validate_disagreement_summary(
    df_summary: pd.DataFrame,
    df_typology: pd.DataFrame,
    issues: list[ValidationRow],
) -> None:
    artifact = "disagreement_summary"
    _check_required_columns(df_summary, artifact, DISAGREEMENT_SUMMARY_COLUMNS, issues)
    if not _has_columns(df_summary, DISAGREEMENT_SUMMARY_COLUMNS):
        return

    _check_allowed_values(
        df_summary,
        artifact,
        "diagnostic_type",
        DIAGNOSTIC_TYPES,
        issues,
    )
    _check_unique_key(df_summary, artifact, ["diagnostic_type"], issues)
    _check_between(df_summary, artifact, ["share"], 0, 1, issues)
    _check_nonnegative(df_summary, artifact, ["rows", "max_review_score"], issues)
    if _has_columns(df_typology, ("diagnostic_type",)):
        expected = (
            df_typology["diagnostic_type"]
            .value_counts()
            .rename_axis("diagnostic_type")
            .reset_index(name="expected_rows")
        )
        compared = expected.merge(df_summary, on="diagnostic_type", how="outer")
        missing = compared.isna().any(axis=1)
        _add_if(
            issues,
            bool(missing.sum()),
            artifact,
            "disagreement_summary_pairing",
            "Disagreement summary rows must match typology diagnostic types.",
            int(missing.sum()),
        )
        complete = compared.dropna()
        mismatched = complete["rows"].ne(complete["expected_rows"])
        _add_if(
            issues,
            bool(mismatched.sum()),
            artifact,
            "disagreement_summary_counts",
            "Disagreement summary counts must match typology rows.",
            int(mismatched.sum()),
        )


def _validate_external_baseline_agreement(
    df: pd.DataFrame,
    zones: Sequence[str],
    issues: list[ValidationRow],
) -> None:
    artifact = "external_baseline_agreement"
    _check_required_columns(df, artifact, EXTERNAL_BASELINE_COLUMNS, issues)
    if not _has_columns(df, EXTERNAL_BASELINE_COLUMNS):
        return

    _check_exact_rows(
        df,
        artifact,
        len(zones) * len(SSP_NAMES) * len(EXTERNAL_BASELINE_COMPARATORS),
        issues,
    )
    _check_key_set(
        df,
        artifact,
        ["zone", "scenario", "comparator"],
        [zones, SSP_NAMES, EXTERNAL_BASELINE_COMPARATORS],
        issues,
    )
    _check_unique_key(df, artifact, ["zone", "scenario", "comparator"], issues)
    _check_allowed_values(df, artifact, "external_dataset", EXTERNAL_DATASETS, issues)
    _check_allowed_values(df, artifact, "external_year", {2020}, issues)
    _check_allowed_values(
        df,
        artifact,
        "comparator_support",
        BASELINE_COMPARATOR_SUPPORT_LABELS,
        issues,
    )
    _check_nonnegative(
        df,
        artifact,
        [
            "external_area_m2",
            "comparator_area_m2",
            "tp_area_m2",
            "fp_area_m2",
            "fn_area_m2",
        ],
        issues,
    )
    _check_between(
        df,
        artifact,
        ["precision", "recall", "iou"],
        0,
        1,
        issues,
        allow_null=True,
    )
    _check_nonnegative(df, artifact, ["area_bias", "ape"], issues, allow_null=True)


def _validate_external_growth_alignment(
    df: pd.DataFrame,
    zones: Sequence[str],
    issues: list[ValidationRow],
) -> None:
    artifact = "external_growth_alignment"
    _check_required_columns(df, artifact, EXTERNAL_GROWTH_COLUMNS, issues)
    if not _has_columns(df, EXTERNAL_GROWTH_COLUMNS):
        return

    _check_exact_rows(
        df,
        artifact,
        len(zones) * len(SSP_NAMES) * len(CALIBRATION_TYPES),
        issues,
    )
    _check_key_set(
        df,
        artifact,
        ["zone", "scenario", "calibration"],
        [zones, SSP_NAMES, CALIBRATION_TYPES],
        issues,
    )
    _check_unique_key(df, artifact, ["zone", "scenario", "calibration"], issues)
    _check_allowed_values(df, artifact, "external_dataset", EXTERNAL_DATASETS, issues)
    _check_allowed_values(df, artifact, "calibration", CALIBRATION_TYPES, issues)
    _check_allowed_values(df, artifact, "period_start_year", {2020}, issues)
    _check_allowed_values(df, artifact, "year", {2030}, issues)
    _check_allowed_values(
        df,
        artifact,
        "growth_alignment",
        EXTERNAL_GROWTH_LABELS,
        issues,
    )
    _check_nonnegative(
        df,
        artifact,
        ["ghsl_growth_area_m2", "chen_growth_area_m2"],
        issues,
    )
    _check_nonnegative(
        df,
        artifact,
        ["chen_to_external_growth_ratio"],
        issues,
        allow_null=True,
    )


def _validate_external_review_flags(
    df: pd.DataFrame,
    df_baseline: pd.DataFrame,
    df_growth: pd.DataFrame,
    df_assessment: pd.DataFrame,
    zones: Sequence[str],
    issues: list[ValidationRow],
) -> None:
    artifact = "external_review_flags"
    _check_required_columns(df, artifact, EXTERNAL_REVIEW_FLAG_COLUMNS, issues)
    if not _has_columns(df, EXTERNAL_REVIEW_FLAG_COLUMNS):
        return

    _check_exact_rows(df, artifact, len(zones) * len(SSP_NAMES), issues)
    _check_key_set(df, artifact, ["zone", "scenario"], [zones, SSP_NAMES], issues)
    _check_unique_key(df, artifact, ["zone", "scenario"], issues)
    _check_allowed_values(
        df,
        artifact,
        "glc_baseline_support",
        BASELINE_COMPARATOR_SUPPORT_LABELS,
        issues,
    )
    _check_allowed_values(
        df,
        artifact,
        "chen_baseline_support",
        BASELINE_COMPARATOR_SUPPORT_LABELS,
        issues,
    )
    _check_allowed_values(
        df,
        artifact,
        "external_baseline_validation",
        EXTERNAL_BASELINE_LABELS,
        issues,
    )
    _check_allowed_values(
        df,
        artifact,
        "calibrated_growth_alignment",
        EXTERNAL_GROWTH_LABELS,
        issues,
    )
    _check_allowed_values(
        df,
        artifact,
        "external_advisory",
        EXTERNAL_ADVISORY_LABELS,
        issues,
    )
    _check_allowed_values(
        df,
        artifact,
        "land_estimate_readiness",
        READINESS_LABELS,
        issues,
    )
    _check_allowed_values(
        df,
        artifact,
        "manual_review_priority",
        REVIEW_PRIORITY_LABELS,
        issues,
    )
    _check_allowed_values(
        df,
        artifact,
        "overall_assessment",
        READINESS_LABELS,
        issues,
    )
    _check_allowed_values(
        df,
        artifact,
        "transition_feasibility",
        TRANSITION_FEASIBILITY_LABELS,
        issues,
    )
    _check_allowed_values(df, artifact, "reliability", RELIABILITY_LABELS, issues)
    _check_between(df, artifact, ["iou"], 0, 1, issues, allow_null=True)
    _check_nonnegative(
        df,
        artifact,
        [
            "ape",
            "ghsl_growth_area_m2",
            "calibrated_chen_growth_area_m2",
            "calibrated_chen_to_external_growth_ratio",
        ],
        issues,
        allow_null=True,
    )
    _validate_external_review_flag_pairings(
        df,
        df_baseline,
        df_growth,
        df_assessment,
        issues,
    )


def _validate_external_review_flag_pairings(
    df: pd.DataFrame,
    df_baseline: pd.DataFrame,
    df_growth: pd.DataFrame,
    df_assessment: pd.DataFrame,
    issues: list[ValidationRow],
) -> None:
    _validate_external_baseline_pairing(df, df_baseline, issues)
    _validate_external_growth_pairing(df, df_growth, issues)
    _validate_external_assessment_pairing(df, df_assessment, issues)


def _validate_external_baseline_pairing(
    df: pd.DataFrame,
    df_baseline: pd.DataFrame,
    issues: list[ValidationRow],
) -> None:
    artifact = "external_review_flags"
    key = ["zone", "scenario"]
    baseline_columns = (*key, "comparator", "comparator_support")
    if not _has_columns(df_baseline, baseline_columns):
        return

    expected_baseline = (
        df_baseline.pivot_table(
            index=key,
            columns="comparator",
            values="comparator_support",
            aggfunc="first",
        )
        .reset_index()
        .rename(
            columns={
                "glc_settlements_2020": "expected_glc_baseline_support",
                "chen_urban_2020": "expected_chen_baseline_support",
            },
        )
    )
    required = {
        "expected_glc_baseline_support",
        "expected_chen_baseline_support",
    }
    if not required.issubset(expected_baseline.columns):
        return

    compared = df.merge(expected_baseline, on=key, how="left")
    missing = compared[list(required)].isna().any(axis=1)
    _add_if(
        issues,
        bool(missing.sum()),
        artifact,
        "baseline_pairing",
        "Review flags must pair with both baseline comparator rows.",
        int(missing.sum()),
    )
    complete = compared[~missing].copy()
    if complete.empty:
        return

    complete["expected_external_baseline_validation"] = [
        combine_baseline_support(glc, chen)
        for glc, chen in zip(
            complete["expected_glc_baseline_support"],
            complete["expected_chen_baseline_support"],
            strict=True,
        )
    ]
    for observed_column, expected_column in (
        ("glc_baseline_support", "expected_glc_baseline_support"),
        ("chen_baseline_support", "expected_chen_baseline_support"),
        ("external_baseline_validation", "expected_external_baseline_validation"),
    ):
        mismatch = complete[observed_column].ne(complete[expected_column])
        _add_if(
            issues,
            bool(mismatch.sum()),
            artifact,
            f"{observed_column}_pairing",
            f"{observed_column} must match baseline agreement rows.",
            int(mismatch.sum()),
        )


def _validate_external_growth_pairing(
    df: pd.DataFrame,
    df_growth: pd.DataFrame,
    issues: list[ValidationRow],
) -> None:
    artifact = "external_review_flags"
    key = ["zone", "scenario"]
    growth_columns = (
        *key,
        "calibration",
        "ghsl_growth_area_m2",
        "chen_growth_area_m2",
        "chen_to_external_growth_ratio",
        "growth_alignment",
    )
    if not _has_columns(df_growth, growth_columns):
        return

    expected_growth = df_growth[df_growth["calibration"].eq("calibrated")][
        [
            *key,
            "ghsl_growth_area_m2",
            "chen_growth_area_m2",
            "chen_to_external_growth_ratio",
            "growth_alignment",
        ]
    ].rename(
        columns={
            "ghsl_growth_area_m2": "expected_ghsl_growth_area_m2",
            "chen_growth_area_m2": "expected_calibrated_chen_growth_area_m2",
            "chen_to_external_growth_ratio": (
                "expected_calibrated_chen_to_external_growth_ratio"
            ),
            "growth_alignment": "expected_calibrated_growth_alignment",
        },
    )
    compared = df.merge(expected_growth, on=key, how="left")
    expected_columns = [
        "expected_ghsl_growth_area_m2",
        "expected_calibrated_chen_growth_area_m2",
        "expected_calibrated_growth_alignment",
    ]
    missing = compared[expected_columns].isna().any(axis=1)
    _add_if(
        issues,
        bool(missing.sum()),
        artifact,
        "growth_pairing",
        "Review flags must pair with calibrated external growth rows.",
        int(missing.sum()),
    )
    complete = compared[~missing].copy()
    if complete.empty:
        return

    complete["expected_external_advisory"] = [
        classify_external_advisory(baseline, growth)
        for baseline, growth in zip(
            complete["external_baseline_validation"],
            complete["expected_calibrated_growth_alignment"],
            strict=True,
        )
    ]
    label_pairs = (
        ("calibrated_growth_alignment", "expected_calibrated_growth_alignment"),
        ("external_advisory", "expected_external_advisory"),
    )
    for observed_column, expected_column in label_pairs:
        mismatch = complete[observed_column].ne(complete[expected_column])
        _add_if(
            issues,
            bool(mismatch.sum()),
            artifact,
            f"{observed_column}_pairing",
            f"{observed_column} must match calibrated growth rows.",
            int(mismatch.sum()),
        )
    numeric_pairs = (
        ("ghsl_growth_area_m2", "expected_ghsl_growth_area_m2"),
        (
            "calibrated_chen_growth_area_m2",
            "expected_calibrated_chen_growth_area_m2",
        ),
        (
            "calibrated_chen_to_external_growth_ratio",
            "expected_calibrated_chen_to_external_growth_ratio",
        ),
    )
    for observed_column, expected_column in numeric_pairs:
        mismatch = (
            pd.to_numeric(complete[observed_column], errors="coerce")
            .sub(pd.to_numeric(complete[expected_column], errors="coerce"))
            .abs()
            .gt(AREA_TOLERANCE_M2)
        )
        _add_if(
            issues,
            bool(mismatch.sum()),
            artifact,
            f"{observed_column}_pairing",
            f"{observed_column} must match calibrated growth rows.",
            int(mismatch.sum()),
        )


def _validate_external_assessment_pairing(
    df: pd.DataFrame,
    df_assessment: pd.DataFrame,
    issues: list[ValidationRow],
) -> None:
    artifact = "external_review_flags"
    key = ["zone", "scenario"]
    assessment_columns = (
        *key,
        "land_estimate_readiness",
        "manual_review_priority",
        "overall_assessment",
        "transition_feasibility",
        "reliability",
    )
    if not _has_columns(df_assessment, assessment_columns):
        return

    expected_assessment = df_assessment[list(assessment_columns)].rename(
        columns={
            column: f"expected_{column}"
            for column in assessment_columns
            if column not in key
        },
    )
    compared = df.merge(expected_assessment, on=key, how="left")
    expected_columns = [
        f"expected_{column}" for column in assessment_columns if column not in key
    ]
    missing = compared[expected_columns].isna().any(axis=1)
    _add_if(
        issues,
        bool(missing.sum()),
        artifact,
        "assessment_pairing",
        "Review flags must pair with land-estimation assessment rows.",
        int(missing.sum()),
    )
    complete = compared[~missing]
    for column in (
        "land_estimate_readiness",
        "manual_review_priority",
        "overall_assessment",
        "transition_feasibility",
        "reliability",
    ):
        mismatch = complete[column].ne(complete[f"expected_{column}"])
        _add_if(
            issues,
            bool(mismatch.sum()),
            artifact,
            f"{column}_pairing",
            f"{column} must match land-estimation assessment rows.",
            int(mismatch.sum()),
        )


def _validate_external_validation_summary(
    df_summary: pd.DataFrame,
    df_flags: pd.DataFrame,
    issues: list[ValidationRow],
) -> None:
    artifact = "external_validation_summary"
    _check_required_columns(df_summary, artifact, EXTERNAL_SUMMARY_COLUMNS, issues)
    if not _has_columns(df_summary, EXTERNAL_SUMMARY_COLUMNS):
        return

    _check_allowed_values(df_summary, artifact, "scenario", SSP_NAMES, issues)
    _check_allowed_values(
        df_summary,
        artifact,
        "external_advisory",
        EXTERNAL_ADVISORY_LABELS,
        issues,
    )
    _check_allowed_values(
        df_summary,
        artifact,
        "external_baseline_validation",
        EXTERNAL_BASELINE_LABELS,
        issues,
    )
    _check_allowed_values(
        df_summary,
        artifact,
        "calibrated_growth_alignment",
        EXTERNAL_GROWTH_LABELS,
        issues,
    )
    key = [
        "scenario",
        "external_advisory",
        "external_baseline_validation",
        "calibrated_growth_alignment",
    ]
    _check_unique_key(df_summary, artifact, key, issues)
    _check_between(df_summary, artifact, ["share"], 0, 1, issues)
    _check_nonnegative(
        df_summary,
        artifact,
        [
            "rows",
            "median_ghsl_growth_area_m2",
            "median_calibrated_chen_to_external_growth_ratio",
        ],
        issues,
        allow_null=True,
    )
    if _has_columns(df_flags, key):
        expected = (
            df_flags.groupby(key, as_index=False)
            .size()
            .rename(columns={"size": "expected_rows"})
        )
        compared = expected.merge(df_summary, on=key, how="outer")
        missing = compared.isna().any(axis=1)
        _add_if(
            issues,
            bool(missing.sum()),
            artifact,
            "summary_pairing",
            "External validation summary rows must match review-flag groups.",
            int(missing.sum()),
        )
        complete = compared.dropna()
        mismatched = complete["rows"].ne(complete["expected_rows"])
        _add_if(
            issues,
            bool(mismatched.sum()),
            artifact,
            "summary_counts",
            "External validation summary counts must match review flags.",
            int(mismatched.sum()),
        )


def _report(issues: Sequence[ValidationRow]) -> pd.DataFrame:
    return pd.DataFrame(issues, columns=REPORT_COLUMNS)


def _has_columns(df: pd.DataFrame, columns: Iterable[str]) -> bool:
    return set(columns).issubset(df.columns)


def _add_issue(
    issues: list[ValidationRow],
    artifact: str,
    check: str,
    message: str,
    rows: int,
) -> None:
    issues.append(
        {
            "artifact": artifact,
            "check": check,
            "severity": ERROR,
            "message": message,
            "rows": rows,
        },
    )


def _add_if(
    issues: list[ValidationRow],
    condition: object,
    artifact: str,
    check: str,
    message: str,
    rows: int,
) -> None:
    if condition:
        _add_issue(issues, artifact, check, message, rows)


def _check_required_columns(
    df: pd.DataFrame,
    artifact: str,
    columns: Iterable[str],
    issues: list[ValidationRow],
) -> None:
    missing = sorted(set(columns).difference(df.columns))
    _add_if(
        issues,
        bool(missing),
        artifact,
        "required_columns",
        f"Missing required columns: {missing}",
        len(missing),
    )


def _check_exact_rows(
    df: pd.DataFrame,
    artifact: str,
    expected_rows: int,
    issues: list[ValidationRow],
) -> None:
    actual_rows = len(df)
    _add_if(
        issues,
        actual_rows != expected_rows,
        artifact,
        "row_count",
        f"Expected {expected_rows} rows, found {actual_rows}.",
        abs(actual_rows - expected_rows),
    )


def _check_unique_key(
    df: pd.DataFrame,
    artifact: str,
    columns: Sequence[str],
    issues: list[ValidationRow],
) -> None:
    duplicated = int(df.duplicated(list(columns), keep=False).sum())
    _add_if(
        issues,
        duplicated > 0,
        artifact,
        "unique_key",
        f"Duplicate rows found for key {list(columns)}.",
        duplicated,
    )


def _check_key_set(
    df: pd.DataFrame,
    artifact: str,
    columns: Sequence[str],
    values: Sequence[Iterable[object]],
    issues: list[ValidationRow],
) -> None:
    expected_index = pd.MultiIndex.from_product(values, names=list(columns))
    actual_index = pd.MultiIndex.from_frame(df[list(columns)].drop_duplicates())
    missing = expected_index.difference(actual_index)
    extra = actual_index.difference(expected_index)
    _add_if(
        issues,
        len(missing) > 0,
        artifact,
        "missing_keys",
        f"Missing expected keys for {list(columns)}.",
        len(missing),
    )
    _add_if(
        issues,
        len(extra) > 0,
        artifact,
        "unexpected_keys",
        f"Found unexpected keys for {list(columns)}.",
        len(extra),
    )


def _check_allowed_values(
    df: pd.DataFrame,
    artifact: str,
    column: str,
    allowed_values: Iterable[object],
    issues: list[ValidationRow],
    *,
    allow_null: bool = False,
) -> None:
    series = df[column]
    invalid_nulls = 0 if allow_null else int(series.isna().sum())
    allowed = set(allowed_values)
    invalid_values = series.dropna()[~series.dropna().isin(allowed)]
    rows = invalid_nulls + len(invalid_values)
    examples = sorted(str(value) for value in set(invalid_values.head(5)))
    _add_if(
        issues,
        rows > 0,
        artifact,
        f"{column}_allowed_values",
        f"Unexpected values in {column}: {examples}",
        rows,
    )


def _check_boolean_values(
    df: pd.DataFrame,
    artifact: str,
    column: str,
    issues: list[ValidationRow],
) -> None:
    valid = df[column].isin([True, False])
    rows = int((~valid).sum())
    _add_if(
        issues,
        rows > 0,
        artifact,
        f"{column}_boolean",
        f"{column} must contain only boolean values.",
        rows,
    )


def _check_nonnegative(
    df: pd.DataFrame,
    artifact: str,
    columns: Iterable[str],
    issues: list[ValidationRow],
    *,
    allow_null: bool = False,
) -> None:
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = values.lt(-AREA_TOLERANCE_M2)
        if not allow_null:
            invalid = invalid | values.isna()
        rows = int(invalid.sum())
        _add_if(
            issues,
            rows > 0,
            artifact,
            f"{column}_nonnegative",
            f"{column} must be nonnegative.",
            rows,
        )


def _check_between(
    df: pd.DataFrame,
    artifact: str,
    columns: Iterable[str],
    lower: float,
    upper: float,
    issues: list[ValidationRow],
    *,
    allow_null: bool = False,
) -> None:
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = values.lt(lower - RATIO_TOLERANCE) | values.gt(
            upper + RATIO_TOLERANCE,
        )
        if not allow_null:
            invalid = invalid | values.isna()
        rows = int(invalid.sum())
        _add_if(
            issues,
            rows > 0,
            artifact,
            f"{column}_range",
            f"{column} must be between {lower} and {upper}.",
            rows,
        )


def _check_close(
    df: pd.DataFrame,
    artifact: str,
    column: str,
    expected: pd.Series,
    check: str,
    issues: list[ValidationRow],
    *,
    allow_null: bool = False,
) -> None:
    observed = pd.to_numeric(df[column], errors="coerce")
    expected = pd.to_numeric(expected, errors="coerce")
    invalid = observed.sub(expected).abs().gt(AREA_TOLERANCE_M2)
    if allow_null:
        invalid = invalid & ~(observed.isna() & expected.isna())
    else:
        invalid = invalid | observed.isna() | expected.isna()
    rows = int(invalid.sum())
    _add_if(
        issues,
        rows > 0,
        artifact,
        check,
        f"{column} does not match the expected calculation.",
        rows,
    )
