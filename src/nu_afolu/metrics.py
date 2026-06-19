from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from nu_afolu.utils import safe_ratio

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_CORRECTION_FACTOR_BOUNDS = (0.25, 4.0)


def _area(value: float | None) -> float:
    if value is None:
        return 0.0
    return float(value)


def _nonnegative_area(value: float | None) -> float:
    return max(_area(value), 0.0)


def _clip(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    return min(max(value, lower), upper)


def agreement_metrics_from_areas(
    *,
    observed_area_m2: float,
    chen_area_m2: float,
    tp_area_m2: float,
    fp_area_m2: float | None = None,
    fn_area_m2: float | None = None,
    correction_factor_bounds: tuple[float, float] = DEFAULT_CORRECTION_FACTOR_BOUNDS,
) -> dict[str, float | bool]:
    observed_area_m2 = _area(observed_area_m2)
    chen_area_m2 = _area(chen_area_m2)
    tp_area_m2 = _nonnegative_area(tp_area_m2)
    fp_area_m2 = (
        _nonnegative_area(fp_area_m2)
        if fp_area_m2 is not None
        else _nonnegative_area(chen_area_m2 - tp_area_m2)
    )
    fn_area_m2 = (
        _nonnegative_area(fn_area_m2)
        if fn_area_m2 is not None
        else _nonnegative_area(observed_area_m2 - tp_area_m2)
    )

    precision = safe_ratio(tp_area_m2, chen_area_m2)
    recall = safe_ratio(tp_area_m2, observed_area_m2)
    union_area_m2 = chen_area_m2 + observed_area_m2 - tp_area_m2
    iou = safe_ratio(tp_area_m2, union_area_m2)
    area_error_m2 = chen_area_m2 - observed_area_m2
    area_bias = safe_ratio(chen_area_m2, observed_area_m2)
    ape = safe_ratio(abs(area_error_m2), observed_area_m2)
    correction_factor_raw = safe_ratio(observed_area_m2, chen_area_m2)
    calibration_valid = bool(
        observed_area_m2 > 0
        and chen_area_m2 > 0
        and math.isfinite(correction_factor_raw)
    )
    correction_factor = (
        _clip(correction_factor_raw, correction_factor_bounds)
        if calibration_valid
        else 1.0
    )

    return {
        "observed_area_m2": observed_area_m2,
        "chen_area_m2": chen_area_m2,
        "area_error_m2": area_error_m2,
        "area_bias": area_bias,
        "ape": ape,
        "tp_area_m2": tp_area_m2,
        "fp_area_m2": fp_area_m2,
        "fn_area_m2": fn_area_m2,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "correction_factor_raw": correction_factor_raw,
        "correction_factor": correction_factor,
        "calibration_valid": calibration_valid,
    }


def add_area_calibration_fields(
    df: pd.DataFrame,
    *,
    correction_factor_bounds: tuple[float, float] = DEFAULT_CORRECTION_FACTOR_BOUNDS,
) -> pd.DataFrame:
    out = df.copy()
    out["area_error_m2"] = out["chen_area_m2"] - out["observed_area_m2"]
    out["area_bias"] = out["chen_area_m2"].div(
        out["observed_area_m2"].replace(0, np.nan),
    )
    out["ape"] = out["area_error_m2"].abs().div(
        out["observed_area_m2"].replace(0, np.nan),
    )
    out["correction_factor_raw"] = out["observed_area_m2"].div(
        out["chen_area_m2"].replace(0, np.nan),
    )
    out["calibration_valid"] = (
        out["observed_area_m2"].gt(0)
        & out["chen_area_m2"].gt(0)
        & np.isfinite(out["correction_factor_raw"])
    )
    out["correction_factor"] = np.where(
        out["calibration_valid"],
        np.clip(out["correction_factor_raw"], *correction_factor_bounds),
        1.0,
    )
    return out
