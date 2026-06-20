from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from nu_afolu._artifact_validation_core import (
    AREA_TOLERANCE_M2,
    _add_if,
    _check_allowed_values,
    _check_between,
    _check_exact_rows,
    _check_key_set,
    _check_nonnegative,
    _check_required_columns,
    _check_unique_key,
    _has_columns,
)
from nu_afolu._artifact_validation_schemas import (
    CALIBRATION_TYPES,
    EXTERNAL_BASELINE_COLUMNS,
    EXTERNAL_BASELINE_COMPARATORS,
    EXTERNAL_DATASETS,
    EXTERNAL_GROWTH_COLUMNS,
    EXTERNAL_REVIEW_FLAG_COLUMNS,
    EXTERNAL_SUMMARY_COLUMNS,
    READINESS_LABELS,
    RELIABILITY_LABELS,
    REVIEW_PRIORITY_LABELS,
)
from nu_afolu.chen import SSP_NAMES
from nu_afolu.external_validation import (
    BASELINE_COMPARATOR_SUPPORT_LABELS,
    EXTERNAL_ADVISORY_LABELS,
    EXTERNAL_BASELINE_LABELS,
    EXTERNAL_GROWTH_LABELS,
    classify_external_advisory,
    combine_baseline_support,
)
from nu_afolu.transition_feasibility import TRANSITION_FEASIBILITY_LABELS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nu_afolu._artifact_validation_core import ValidationRow


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
