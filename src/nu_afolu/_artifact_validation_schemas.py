from __future__ import annotations

from nu_afolu.chen import CHEN_YEARS
from nu_afolu.constants import LABEL_LIST

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
