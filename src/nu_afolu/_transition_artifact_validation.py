from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from nu_afolu._artifact_validation_core import (
    AREA_TOLERANCE_M2,
    RATIO_TOLERANCE,
    _add_if,
    _check_allowed_values,
    _check_between,
    _check_boolean_values,
    _check_close,
    _check_exact_rows,
    _check_key_set,
    _check_nonnegative,
    _check_required_columns,
    _check_unique_key,
    _has_columns,
    _numeric_mismatch,
)
from nu_afolu._artifact_validation_schemas import (
    ADEQUACY_LABELS,
    ASSESSMENT_COLUMNS,
    ASSESSMENT_FEASIBILITY_COLUMNS,
    CALIBRATION_TYPES,
    EXPANSION_COLUMNS,
    FUTURE_YEARS,
    GROWTH_RISK_LABELS,
    READINESS_LABELS,
    RELIABILITY_LABELS,
    REVIEW_CANDIDATE_COLUMNS,
    REVIEW_PRIORITY_LABELS,
    SENSITIVE_FLAG_LABELS,
    SENSITIVE_RISK_LABELS,
    SOURCE_CLASSES,
    TRANSITION_COLUMNS,
)
from nu_afolu.chen import SSP_NAMES
from nu_afolu.growth_plausibility import (
    GROWTH_PLAUSIBILITY_LABELS,
    HISTORICAL_ENVELOPE_LABELS,
    HISTORICAL_GROWTH_CONTEXT_LABELS,
    HISTORICAL_GROWTH_DIAGNOSTIC_COLUMNS,
    HISTORICAL_GROWTH_REVIEW_NOTES,
)
from nu_afolu.transition_feasibility import (
    CAPACITY_WATCH,
    FEASIBLE,
    INFEASIBLE,
    TRANSITION_FEASIBILITY_COLUMNS,
    TRANSITION_FEASIBILITY_LABELS,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nu_afolu._artifact_validation_core import ValidationRow


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
            raw.groupby(["zone", "scenario", "year"], as_index=False)
            .agg(raw_transition_area_m2=("area_m2", "sum"))
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


def _validate_historical_growth_diagnostics(
    df: pd.DataFrame,
    zones: Sequence[str],
    issues: list[ValidationRow],
) -> None:
    artifact = "historical_growth_diagnostics"
    _check_required_columns(df, artifact, HISTORICAL_GROWTH_DIAGNOSTIC_COLUMNS, issues)
    if not _has_columns(df, HISTORICAL_GROWTH_DIAGNOSTIC_COLUMNS):
        return

    _check_exact_rows(df, artifact, len(zones) * len(SSP_NAMES), issues)
    _check_key_set(df, artifact, ["zone", "scenario"], [zones, SSP_NAMES], issues)
    _check_unique_key(df, artifact, ["zone", "scenario"], issues)
    _check_allowed_values(
        df,
        artifact,
        "worst_growth_plausibility",
        GROWTH_PLAUSIBILITY_LABELS,
        issues,
    )
    _check_allowed_values(
        df,
        artifact,
        "historical_growth_context",
        HISTORICAL_GROWTH_CONTEXT_LABELS,
        issues,
    )
    _check_allowed_values(
        df,
        artifact,
        "historical_envelope_alignment",
        HISTORICAL_ENVELOPE_LABELS,
        issues,
    )
    _check_allowed_values(
        df,
        artifact,
        "historical_growth_review_note",
        HISTORICAL_GROWTH_REVIEW_NOTES,
        issues,
    )
    _check_nonnegative(
        df,
        artifact,
        ["max_chen_new_area_m2", "negative_historical_growth_periods"],
        issues,
    )
    _check_nonnegative(
        df,
        artifact,
        [
            "historical_growth_range_ratio",
            "max_chen_to_recent_growth_ratio",
            "max_chen_to_historical_median_ratio",
            "max_chen_to_historical_max_ratio",
        ],
        issues,
        allow_null=True,
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


def _validate_assessment_growth_consistency(
    df_assessment: pd.DataFrame,
    df_growth: pd.DataFrame,
    issues: list[ValidationRow],
) -> None:
    artifact = "land_estimation_assessment"
    key = ["zone", "scenario"]
    growth_columns = (
        *key,
        "recent_growth_area_m2",
        "max_chen_new_area_m2",
        "max_chen_to_recent_growth_ratio",
        "worst_growth_plausibility",
    )
    assessment_columns = tuple(growth_columns)
    if not _has_columns(df_assessment, assessment_columns) or not _has_columns(
        df_growth,
        growth_columns,
    ):
        return

    expected = df_growth[list(growth_columns)].rename(
        columns={
            column: f"expected_{column}"
            for column in growth_columns
            if column not in key
        },
    )
    compared = df_assessment[list(assessment_columns)].merge(
        expected,
        on=key,
        how="outer",
    )
    expected_columns = [
        f"expected_{column}" for column in growth_columns if column not in key
    ]
    missing = compared[expected_columns].isna().any(axis=1)
    _add_if(
        issues,
        bool(missing.sum()),
        artifact,
        "historical_growth_pairing",
        "Assessment rows must pair with historical growth diagnostics.",
        int(missing.sum()),
    )
    complete = compared[~missing]
    if complete.empty:
        return

    label_mismatch = complete["worst_growth_plausibility"].ne(
        complete["expected_worst_growth_plausibility"],
    )
    _add_if(
        issues,
        bool(label_mismatch.sum()),
        artifact,
        "worst_growth_plausibility_consistency",
        "Assessment worst_growth_plausibility must match growth diagnostics.",
        int(label_mismatch.sum()),
    )
    numeric_pairs = (
        ("recent_growth_area_m2", "expected_recent_growth_area_m2"),
        ("max_chen_new_area_m2", "expected_max_chen_new_area_m2"),
        (
            "max_chen_to_recent_growth_ratio",
            "expected_max_chen_to_recent_growth_ratio",
        ),
    )
    for observed_column, expected_column in numeric_pairs:
        mismatch = _numeric_mismatch(
            complete[observed_column],
            complete[expected_column],
        )
        _add_if(
            issues,
            bool(mismatch.sum()),
            artifact,
            f"{observed_column}_consistency",
            f"{observed_column} must match historical growth diagnostics.",
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
