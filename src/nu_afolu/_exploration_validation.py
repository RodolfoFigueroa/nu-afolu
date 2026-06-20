from __future__ import annotations

from typing import TYPE_CHECKING

from nu_afolu._artifact_validation_core import (
    _add_if,
    _check_allowed_values,
    _check_between,
    _check_boolean_values,
    _check_exact_rows,
    _check_key_set,
    _check_nonnegative,
    _check_required_columns,
    _check_unique_key,
    _has_columns,
)
from nu_afolu._artifact_validation_schemas import (
    AREA_ERROR_CLASSES,
    DIAGNOSTIC_TYPES,
    DISAGREEMENT_SUMMARY_COLUMNS,
    DISAGREEMENT_TYPOLOGY_COLUMNS,
    METHOD_COMPARISON_COLUMNS,
    METHOD_RECOMMENDATION_COLUMNS,
    METHOD_SUMMARY_COLUMNS,
    SPATIAL_AGREEMENT_CLASSES,
)
from nu_afolu.chen import SSP_NAMES

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from nu_afolu._artifact_validation_core import ValidationRow


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
