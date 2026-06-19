from __future__ import annotations

import math

MIN_EXTERNAL_AREA_M2 = 1_000_000.0
BASELINE_APE_SUPPORT_THRESHOLD = 0.35
BASELINE_IOU_SUPPORT_THRESHOLD = 0.25
GROWTH_RATIO_LOWER = 0.25
GROWTH_RATIO_UPPER = 4.0

COMPARATOR_SUPPORTED = "supported"
COMPARATOR_CONFLICT = "conflict"
COMPARATOR_INSUFFICIENT = "insufficient_external_signal"
BASELINE_COMPARATOR_SUPPORT_LABELS = frozenset(
    {COMPARATOR_SUPPORTED, COMPARATOR_CONFLICT, COMPARATOR_INSUFFICIENT},
)

EXTERNAL_SUPPORTS_BOTH = "external_supports_both"
QUESTIONS_GLC_BASELINE = "questions_glc_baseline"
QUESTIONS_CHEN_BASELINE = "questions_chen_baseline"
EXTERNAL_BASELINE_CONFLICT = "external_baseline_conflict"
INSUFFICIENT_EXTERNAL_SIGNAL = "insufficient_external_signal"
EXTERNAL_BASELINE_LABELS = frozenset(
    {
        EXTERNAL_SUPPORTS_BOTH,
        QUESTIONS_GLC_BASELINE,
        QUESTIONS_CHEN_BASELINE,
        EXTERNAL_BASELINE_CONFLICT,
        INSUFFICIENT_EXTERNAL_SIGNAL,
    },
)

GROWTH_CONSISTENT = "consistent"
LOW_CHEN_GROWTH = "low_chen_growth"
HIGH_CHEN_GROWTH = "high_chen_growth"
INSUFFICIENT_EXTERNAL_GROWTH = "insufficient_external_growth"
EXTERNAL_GROWTH_LABELS = frozenset(
    {
        GROWTH_CONSISTENT,
        LOW_CHEN_GROWTH,
        HIGH_CHEN_GROWTH,
        INSUFFICIENT_EXTERNAL_GROWTH,
    },
)

EXTERNAL_SUPPORT = "external_support"
EXTERNAL_REVIEW = "external_review"
EXTERNAL_CONFLICT = "external_conflict"
EXTERNAL_ADVISORY_LABELS = frozenset(
    {EXTERNAL_SUPPORT, EXTERNAL_REVIEW, EXTERNAL_CONFLICT},
)


def classify_baseline_comparator_support(
    *,
    external_area_m2: float,
    comparator_area_m2: float,
    iou: float,
    ape: float,
    min_external_area_m2: float = MIN_EXTERNAL_AREA_M2,
    max_ape: float = BASELINE_APE_SUPPORT_THRESHOLD,
    min_iou: float = BASELINE_IOU_SUPPORT_THRESHOLD,
) -> str:
    if external_area_m2 < min_external_area_m2:
        return COMPARATOR_INSUFFICIENT
    if comparator_area_m2 < min_external_area_m2:
        return COMPARATOR_INSUFFICIENT

    if _finite(iou) and iou >= min_iou:
        return COMPARATOR_SUPPORTED
    if _finite(ape) and ape <= max_ape:
        return COMPARATOR_SUPPORTED
    return COMPARATOR_CONFLICT


def combine_baseline_support(
    glc_baseline_support: str,
    chen_baseline_support: str,
) -> str:
    if COMPARATOR_INSUFFICIENT in {glc_baseline_support, chen_baseline_support}:
        return INSUFFICIENT_EXTERNAL_SIGNAL
    if (
        glc_baseline_support == COMPARATOR_SUPPORTED
        and chen_baseline_support == COMPARATOR_SUPPORTED
    ):
        return EXTERNAL_SUPPORTS_BOTH
    if (
        glc_baseline_support == COMPARATOR_CONFLICT
        and chen_baseline_support == COMPARATOR_SUPPORTED
    ):
        return QUESTIONS_GLC_BASELINE
    if (
        glc_baseline_support == COMPARATOR_SUPPORTED
        and chen_baseline_support == COMPARATOR_CONFLICT
    ):
        return QUESTIONS_CHEN_BASELINE
    return EXTERNAL_BASELINE_CONFLICT


def classify_growth_alignment(
    *,
    external_growth_area_m2: float,
    chen_growth_area_m2: float,
    min_external_growth_area_m2: float = MIN_EXTERNAL_AREA_M2,
    lower_ratio: float = GROWTH_RATIO_LOWER,
    upper_ratio: float = GROWTH_RATIO_UPPER,
) -> str:
    if external_growth_area_m2 < min_external_growth_area_m2:
        return INSUFFICIENT_EXTERNAL_GROWTH

    ratio = _safe_ratio(chen_growth_area_m2, external_growth_area_m2)
    if not _finite(ratio):
        return INSUFFICIENT_EXTERNAL_GROWTH
    if ratio < lower_ratio:
        return LOW_CHEN_GROWTH
    if ratio > upper_ratio:
        return HIGH_CHEN_GROWTH
    return GROWTH_CONSISTENT


def classify_external_advisory(
    baseline_validation: str,
    growth_alignment: str,
) -> str:
    if (
        baseline_validation == EXTERNAL_SUPPORTS_BOTH
        and growth_alignment == GROWTH_CONSISTENT
    ):
        return EXTERNAL_SUPPORT
    if baseline_validation == EXTERNAL_BASELINE_CONFLICT or growth_alignment in {
        LOW_CHEN_GROWTH,
        HIGH_CHEN_GROWTH,
    }:
        return EXTERNAL_CONFLICT
    return EXTERNAL_REVIEW


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return math.nan
    return float(numerator / denominator)


def _finite(value: float) -> bool:
    return isinstance(value, int | float) and math.isfinite(value)
