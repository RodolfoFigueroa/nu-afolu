from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

REPORT_COLUMNS = ("artifact", "check", "severity", "message", "rows")
ERROR = "error"
AREA_TOLERANCE_M2 = 1e-3
RATIO_TOLERANCE = 1e-9

ValidationRow = dict[str, object]


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


def _numeric_mismatch(
    observed: pd.Series,
    expected: pd.Series,
    *,
    tolerance: float = AREA_TOLERANCE_M2,
) -> pd.Series:
    observed_numeric = pd.to_numeric(observed, errors="coerce")
    expected_numeric = pd.to_numeric(expected, errors="coerce")
    both_null = observed_numeric.isna() & expected_numeric.isna()
    both_positive_inf = np.isposinf(observed_numeric) & np.isposinf(expected_numeric)
    both_negative_inf = np.isneginf(observed_numeric) & np.isneginf(expected_numeric)
    close = observed_numeric.sub(expected_numeric).abs().le(tolerance)
    return cast(
        "pd.Series",
        ~(both_null | both_positive_inf | both_negative_inf | close),
    )
