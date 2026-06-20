from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nu_afolu._artifact_validation_core import (
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
)
from nu_afolu._artifact_validation_schemas import (
    CALIBRATION_COLUMNS,
    RELIABILITY_LABELS,
    SCALE_SENSITIVITY_COLUMNS,
)
from nu_afolu.chen import SSP_NAMES
from nu_afolu.metrics import DEFAULT_CORRECTION_FACTOR_BOUNDS

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import pandas as pd

    from nu_afolu._artifact_validation_core import ValidationRow


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
