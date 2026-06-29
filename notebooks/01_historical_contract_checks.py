import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def import_dependencies():
    import os  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import marimo as mo  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415
    from dagster_components.partitions import zone_partitions  # noqa: PLC0415

    from nu_afolu.constants import LABEL_LIST  # noqa: PLC0415

    return LABEL_LIST, Path, mo, np, os, pd, xr, zone_partitions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 01 Historical Contract Checks

    This notebook validates the historical `area_table` and `transition_table`
    artifacts before they are used as Chen SSP feasibility inputs. The goal is to
    separate trustworthy historical table artifacts from zones that need upstream
    repair or explicit exclusion.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal, Inputs, And Assumptions

    This stage checks the historical table contracts described in `docs/upstream.md`
    and `implementation_plan/analysis_contracts.md`.

    The notebook starts from zones with complete artifact sets, then focuses on:

    - `area_table/{ZONE}.parquet`
    - `transition_table/{ZONE}.nc`

    Missing `area_table` label columns are reported as notes rather than hard
    failures because the upstream reducer may omit class groups that do not appear
    in a zone. For reconciliation, missing labels and missing area cells are
    normalized to zero. Unexpected labels, bad dimensions, negative values, infinite
    values, and failed mass-balance checks are hard failures.
    """)
    return


@app.cell
def configure_contract_checks(Path, os, pd):
    PROJECT_ROOT = Path.cwd()
    OUT_PATH_KEY = "OUT_PATH"
    RECONCILIATION_ABS_TOLERANCE_M2 = 1.0

    def _read_dotenv_value(path: Path, key: str) -> str | None:
        if not path.exists():
            return None

        for _line in path.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _name, _value = _line.split("=", 1)
            if _name.strip() == key:
                return _value.strip().strip('"').strip("'")

        return None

    _out_path_raw = os.environ.get(OUT_PATH_KEY)
    OUT_PATH_SOURCE = "environment variable"
    if not _out_path_raw:
        _out_path_raw = _read_dotenv_value(PROJECT_ROOT / ".env", OUT_PATH_KEY)
        OUT_PATH_SOURCE = ".env file" if _out_path_raw else "missing"

    OUT_PATH = Path(_out_path_raw).expanduser() if _out_path_raw else None
    OUT_PATH_EXISTS = bool(OUT_PATH and OUT_PATH.exists())

    configuration_summary = pd.DataFrame(
        [
            {
                "setting": OUT_PATH_KEY,
                "source": OUT_PATH_SOURCE,
                "value": str(OUT_PATH) if OUT_PATH else "not configured",
                "exists": OUT_PATH_EXISTS,
            },
            {
                "setting": "RECONCILIATION_ABS_TOLERANCE_M2",
                "source": "notebook default",
                "value": RECONCILIATION_ABS_TOLERANCE_M2,
                "exists": True,
            },
        ]
    )

    configuration_summary
    return OUT_PATH, RECONCILIATION_ABS_TOLERANCE_M2


@app.cell(hide_code=True)
def _(RECONCILIATION_ABS_TOLERANCE_M2, mo):
    mo.md(f"""
    ## Reconciliation Tolerance

    Transition and area tables are Earth Engine reduction outputs, so exact binary
    equality is too strict for floating-point area sums. This notebook uses an
    absolute per-class tolerance of `{RECONCILIATION_ABS_TOLERANCE_M2:g} m²` when
    comparing transition row or column sums against `area_table` values.

    The tolerance is deliberately small relative to zone-scale areas. It is meant
    to absorb floating-point noise, not material mass-balance differences.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loaded Zones

    The previous notebook identified zones with complete upstream artifact sets.
    This notebook recomputes that inventory so it can be rerun independently, then
    loads only the historical table artifacts for those zones.
    """)
    return


@app.cell
def discover_candidate_zones(OUT_PATH, Path, pd, zone_partitions):
    REQUIRED_ARTIFACT_SPECS = {
        "bbox_ee": {"relative_dir": Path("bbox") / "ee", "extension": ".json"},
        "area_raster": {"relative_dir": Path("area_raster"), "extension": ".json"},
        "transition_raster": {
            "relative_dir": Path("transition_raster"),
            "extension": ".json",
        },
        "area_table": {"relative_dir": Path("area_table"), "extension": ".parquet"},
        "transition_table": {
            "relative_dir": Path("transition_table"),
            "extension": ".nc",
        },
    }

    partition_zone_names = tuple(zone_partitions.get_partition_keys())

    artifact_zone_sets: dict[str, set[str]] = {}
    for _artifact_name, _spec in REQUIRED_ARTIFACT_SPECS.items():
        _artifact_dir = OUT_PATH / _spec["relative_dir"] if OUT_PATH else None
        if _artifact_dir and _artifact_dir.exists():
            artifact_zone_sets[_artifact_name] = {
                _path.stem for _path in _artifact_dir.glob(f"*{_spec['extension']}")
            }
        else:
            artifact_zone_sets[_artifact_name] = set()

    _availability_rows = []
    for _zone_name in partition_zone_names:
        _row = {"zone": _zone_name}
        for _artifact_name in REQUIRED_ARTIFACT_SPECS:
            _row[_artifact_name] = _zone_name in artifact_zone_sets[_artifact_name]
        _row["complete_artifact_set"] = all(
            _row[_artifact_name] for _artifact_name in REQUIRED_ARTIFACT_SPECS
        )
        _availability_rows.append(_row)

    historical_artifact_availability = pd.DataFrame(_availability_rows)
    complete_zone_names = tuple(
        historical_artifact_availability.loc[
            historical_artifact_availability["complete_artifact_set"],
            "zone",
        ]
    )

    candidate_zone_summary = pd.DataFrame(
        [
            {"metric": "canonical partition zones", "value": len(partition_zone_names)},
            {
                "metric": "zones with complete upstream artifact sets",
                "value": len(complete_zone_names),
            },
            {
                "metric": "area_table files discovered",
                "value": len(artifact_zone_sets["area_table"]),
            },
            {
                "metric": "transition_table files discovered",
                "value": len(artifact_zone_sets["transition_table"]),
            },
        ]
    )

    candidate_zone_summary
    return complete_zone_names, historical_artifact_availability


@app.cell
def show_historical_artifact_availability(historical_artifact_availability):
    historical_artifact_availability
    return


@app.cell
def load_historical_tables(OUT_PATH, complete_zone_names, pd, xr):
    loaded_area_tables = {}
    loaded_transition_tables = {}
    _load_error_rows = []

    for _zone_name in complete_zone_names:
        _area_path = OUT_PATH / "area_table" / f"{_zone_name}.parquet"
        _transition_path = OUT_PATH / "transition_table" / f"{_zone_name}.nc"

        try:
            loaded_area_tables[_zone_name] = pd.read_parquet(_area_path)
        except Exception as _exc:  # noqa: BLE001
            _load_error_rows.append(
                {
                    "zone": _zone_name,
                    "artifact": "area_table",
                    "path": str(_area_path),
                    "error": str(_exc),
                }
            )

        try:
            loaded_transition_tables[_zone_name] = xr.load_dataarray(_transition_path)
        except Exception as _exc:  # noqa: BLE001
            _load_error_rows.append(
                {
                    "zone": _zone_name,
                    "artifact": "transition_table",
                    "path": str(_transition_path),
                    "error": str(_exc),
                }
            )

    loaded_zone_names = tuple(
        _zone_name
        for _zone_name in complete_zone_names
        if _zone_name in loaded_area_tables and _zone_name in loaded_transition_tables
    )

    load_errors = pd.DataFrame(
        _load_error_rows, columns=["zone", "artifact", "path", "error"]
    )
    load_summary = pd.DataFrame(
        [
            {"metric": "candidate complete zones", "value": len(complete_zone_names)},
            {
                "metric": "zones with both tables loaded",
                "value": len(loaded_zone_names),
            },
            {"metric": "table load errors", "value": len(load_errors)},
        ]
    )

    load_summary
    return (
        load_errors,
        loaded_area_tables,
        loaded_transition_tables,
        loaded_zone_names,
    )


@app.cell
def show_load_errors(load_errors):
    load_errors
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Schema And Value Checks

    The next checks validate table structure before doing reconciliation. They keep
    schema findings visible, because later notebooks should be able to filter zones
    by hard failures while still seeing expected sparsity and historical edge cases.
    """)
    return


@app.cell
def define_schema_helpers(LABEL_LIST, pd):
    def format_label_list(
        labels: list[str] | tuple[str, ...], *, limit: int = 6
    ) -> str:
        if not labels:
            return ""
        if len(labels) <= limit:
            return ", ".join(labels)
        return ", ".join(labels[:limit]) + f", ... ({len(labels)} total)"

    def normalize_area_table(area_table: pd.DataFrame) -> pd.DataFrame:
        _normalized = area_table.copy()
        _normalized.index = _normalized.index.astype(int)
        _normalized.index.name = "year"
        _normalized = _normalized.reindex(columns=list(LABEL_LIST))
        _normalized = _normalized.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return _normalized.sort_index()

    "basic schema helpers defined"
    return format_label_list, normalize_area_table


@app.cell
def define_area_label_helper(LABEL_LIST, pd):
    def summarize_area_labels(area_table: pd.DataFrame) -> dict[str, object]:
        _expected_labels = list(LABEL_LIST)
        _columns = list(area_table.columns)
        _unexpected_labels = sorted(
            str(_column) for _column in _columns if _column not in _expected_labels
        )
        _missing_labels = [
            _label for _label in _expected_labels if _label not in _columns
        ]
        _present_label_columns = [
            _label for _label in _expected_labels if _label in _columns
        ]
        return {
            "unexpected_labels": _unexpected_labels,
            "missing_labels": _missing_labels,
            "present_label_columns": _present_label_columns,
        }

    "area label helper defined"
    return (summarize_area_labels,)


@app.cell
def define_area_year_helper(pd):
    def summarize_area_years(area_table: pd.DataFrame) -> dict[str, object]:
        _out = {
            "year_index_valid": True,
            "year_min": None,
            "year_max": None,
            "duplicate_year_count": None,
            "has_2020_area": False,
            "hard_messages": [],
            "note_messages": [],
        }
        try:
            _year_index = pd.Index(area_table.index.astype(int), name="year")
        except (TypeError, ValueError, OverflowError):
            _out["year_index_valid"] = False
            _out["hard_messages"].append(
                "area_table index cannot be interpreted as integer years"
            )
            return _out

        _out["duplicate_year_count"] = int(_year_index.duplicated().sum())
        _out["year_min"] = int(_year_index.min()) if len(_year_index) else None
        _out["year_max"] = int(_year_index.max()) if len(_year_index) else None
        _out["has_2020_area"] = 2020 in set(_year_index)
        if _out["duplicate_year_count"]:
            _out["hard_messages"].append(
                f"duplicate area years: {_out['duplicate_year_count']}"
            )
        if not _out["has_2020_area"]:
            _out["hard_messages"].append(
                "missing 2020 area row required for Chen baseline checks"
            )
        if area_table.index.name != "year":
            _out["note_messages"].append(
                f"index name is {area_table.index.name!r}, expected 'year'"
            )
        return _out

    "area year helper defined"
    return (summarize_area_years,)


@app.cell
def define_area_value_helper(np, pd):
    def summarize_area_values(
        area_table: pd.DataFrame,
        present_label_columns: list[str],
    ) -> dict[str, int]:
        if not present_label_columns:
            return {
                "missing_value_count": 0,
                "non_numeric_count": 0,
                "infinite_count": 0,
                "negative_count": 0,
            }

        _raw_values = area_table.loc[:, present_label_columns]
        _numeric_values = _raw_values.apply(pd.to_numeric, errors="coerce")
        _value_array = _numeric_values.to_numpy(dtype=float, na_value=np.nan)
        return {
            "missing_value_count": int(
                (_numeric_values.isna() & _raw_values.isna()).sum().sum()
            ),
            "non_numeric_count": int(
                (_numeric_values.isna() & _raw_values.notna()).sum().sum()
            ),
            "infinite_count": int(np.isinf(_value_array).sum()),
            "negative_count": int((_numeric_values < 0).sum().sum()),
        }

    "area value helper defined"
    return (summarize_area_values,)


@app.cell
def define_area_validation_helper(
    format_label_list,
    pd,
    summarize_area_labels,
    summarize_area_values,
    summarize_area_years,
):
    def validate_area_table(
        zone_name: str, area_table: pd.DataFrame
    ) -> dict[str, object]:
        _label_summary = summarize_area_labels(area_table)
        _year_summary = summarize_area_years(area_table)
        _value_summary = summarize_area_values(
            area_table, _label_summary["present_label_columns"]
        )
        _hard_messages = list(_year_summary["hard_messages"])
        _note_messages = list(_year_summary["note_messages"])

        if _label_summary["unexpected_labels"]:
            _hard_messages.append(
                f"unexpected labels: {format_label_list(_label_summary['unexpected_labels'])}"
            )
        if _label_summary["missing_labels"]:
            _note_messages.append(
                f"missing labels normalized to zero: {format_label_list(_label_summary['missing_labels'])}"
            )
        if not _label_summary["present_label_columns"]:
            _hard_messages.append("no expected AFOLU label columns are present")
        if _value_summary["non_numeric_count"]:
            _hard_messages.append(
                f"non-numeric area cells: {_value_summary['non_numeric_count']}"
            )
        if _value_summary["infinite_count"]:
            _hard_messages.append(
                f"infinite area cells: {_value_summary['infinite_count']}"
            )
        if _value_summary["negative_count"]:
            _hard_messages.append(
                f"negative area cells: {_value_summary['negative_count']}"
            )
        if _value_summary["missing_value_count"]:
            _note_messages.append(
                f"missing area cells normalized to zero: {_value_summary['missing_value_count']}"
            )

        _status = "fail" if _hard_messages else "warn" if _note_messages else "pass"
        return {
            "zone": zone_name,
            "area_contract_status": _status,
            "area_hard_failure_count": len(_hard_messages),
            "area_note_count": len(_note_messages),
            "area_issue_summary": "; ".join(_hard_messages + _note_messages),
            "index_name": area_table.index.name,
            "year_index_valid": _year_summary["year_index_valid"],
            "year_min": _year_summary["year_min"],
            "year_max": _year_summary["year_max"],
            "duplicate_year_count": _year_summary["duplicate_year_count"],
            "has_2020_area": _year_summary["has_2020_area"],
            "present_label_count": len(_label_summary["present_label_columns"]),
            "missing_label_count": len(_label_summary["missing_labels"]),
            "missing_labels": format_label_list(_label_summary["missing_labels"]),
            "unexpected_label_count": len(_label_summary["unexpected_labels"]),
            "unexpected_labels": format_label_list(_label_summary["unexpected_labels"]),
            **_value_summary,
        }

    "area validation helper defined"
    return (validate_area_table,)


@app.cell
def define_transition_validation_helper(LABEL_LIST, np, pd, xr):
    def validate_transition_table(
        zone_name: str, transition_table: xr.DataArray
    ) -> dict[str, object]:
        _expected_labels = list(LABEL_LIST)
        _hard_messages: list[str] = []
        _note_messages: list[str] = []
        _dims = tuple(transition_table.dims)
        _dims_match = _dims == ("year", "start", "end")
        if not _dims_match:
            _hard_messages.append(
                f"dimensions are {_dims}, expected ('year', 'start', 'end')"
            )

        _start_labels = (
            [str(_label) for _label in transition_table.coords["start"].to_numpy()]
            if "start" in transition_table.coords
            else []
        )
        _end_labels = (
            [str(_label) for _label in transition_table.coords["end"].to_numpy()]
            if "end" in transition_table.coords
            else []
        )
        _start_labels_match = _start_labels == _expected_labels
        _end_labels_match = _end_labels == _expected_labels
        if not _start_labels_match:
            _hard_messages.append("start coordinates do not match LABEL_LIST")
        if not _end_labels_match:
            _hard_messages.append("end coordinates do not match LABEL_LIST")

        try:
            _year_values = pd.Index(
                transition_table.coords["year"].to_numpy().astype(int), name="year"
            )
            _year_coord_valid = True
            _duplicate_year_count = int(_year_values.duplicated().sum())
            _year_min = int(_year_values.min()) if len(_year_values) else None
            _year_max = int(_year_values.max()) if len(_year_values) else None
            _has_2021_transition = 2021 in set(_year_values)
        except (KeyError, TypeError, ValueError, OverflowError):
            _year_coord_valid = False
            _duplicate_year_count = None
            _year_min = None
            _year_max = None
            _has_2021_transition = False
            _hard_messages.append(
                "transition year coordinate cannot be interpreted as integer years"
            )

        if _duplicate_year_count:
            _hard_messages.append(
                f"duplicate transition years: {_duplicate_year_count}"
            )
        _value_array = np.asarray(transition_table.to_numpy(), dtype=float)
        _non_finite_count = int((~np.isfinite(_value_array)).sum())
        _negative_count = int((_value_array < 0).sum())
        if _non_finite_count:
            _hard_messages.append(f"non-finite transition cells: {_non_finite_count}")
        if _negative_count:
            _hard_messages.append(f"negative transition cells: {_negative_count}")
        if _has_2021_transition:
            _note_messages.append("includes expected 2021 -> 2022 transition edge")

        _status = "fail" if _hard_messages else "warn" if _note_messages else "pass"
        return {
            "zone": zone_name,
            "transition_contract_status": _status,
            "transition_hard_failure_count": len(_hard_messages),
            "transition_note_count": len(_note_messages),
            "transition_issue_summary": "; ".join(_hard_messages + _note_messages),
            "dims": str(_dims),
            "dims_match": _dims_match,
            "year_coord_valid": _year_coord_valid,
            "year_min": _year_min,
            "year_max": _year_max,
            "duplicate_year_count": _duplicate_year_count,
            "start_labels_match": _start_labels_match,
            "end_labels_match": _end_labels_match,
            "non_finite_count": _non_finite_count,
            "negative_count": _negative_count,
            "has_2021_transition": _has_2021_transition,
        }

    "transition validation helper defined"
    return (validate_transition_table,)


@app.cell
def check_area_tables(
    loaded_area_tables,
    loaded_zone_names,
    pd,
    validate_area_table,
):
    area_contract_checks = pd.DataFrame(
        [
            validate_area_table(_zone_name, loaded_area_tables[_zone_name])
            for _zone_name in loaded_zone_names
        ]
    )

    area_contract_checks
    return (area_contract_checks,)


@app.cell
def check_transition_tables(
    loaded_transition_tables,
    loaded_zone_names,
    pd,
    validate_transition_table,
):
    transition_contract_checks = pd.DataFrame(
        [
            validate_transition_table(_zone_name, loaded_transition_tables[_zone_name])
            for _zone_name in loaded_zone_names
        ]
    )

    transition_contract_checks
    return (transition_contract_checks,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Area And Transition Reconciliation

    For each transition start year `Y`, the transition table should reconcile in two
    directions:

    - summing over `end` should match the `area_table` vector for year `Y`;
    - summing over `start` should match the `area_table` vector for year `Y + 1`
      when that next area year exists.

    The known historical edge is reported separately: current artifacts include a
    `2021 -> 2022` transition band, while `area_table` may stop at `2021`.
    """)
    return


@app.cell
def define_reconciliation_helper(LABEL_LIST, normalize_area_table, np, pd, xr):
    def build_reconciliation_rows(
        zone_name: str,
        area_table: pd.DataFrame,
        transition_table: xr.DataArray,
        tolerance_m2: float,
    ) -> list[dict[str, object]]:
        _labels = list(LABEL_LIST)
        _area = normalize_area_table(area_table)
        _area_years = {int(_year) for _year in _area.index}
        _max_area_year = max(_area_years) if _area_years else None
        _transition_years = [
            int(_year) for _year in transition_table.coords["year"].to_numpy()
        ]
        _max_transition_year = max(_transition_years) if _transition_years else None

        _rows: list[dict[str, object]] = []
        for _year in _transition_years:
            if _year in _area_years:
                _start_from_transition = (
                    transition_table.sel(year=_year)
                    .sum(dim="end")
                    .to_series()
                    .reindex(_labels)
                    .fillna(0.0)
                )
                _start_from_area = _area.loc[_year, _labels].astype(float)
                _diff = (_start_from_transition - _start_from_area).abs()
                _max_abs_diff = float(_diff.max())
                _rows.append(
                    {
                        "zone": zone_name,
                        "transition_year": _year,
                        "area_year": _year,
                        "check": "start_year_area",
                        "comparison_available": True,
                        "reconciliation_status": "pass"
                        if _max_abs_diff <= tolerance_m2
                        else "fail",
                        "max_abs_diff_m2": _max_abs_diff,
                        "sum_abs_diff_m2": float(_diff.sum()),
                        "worst_label": str(_diff.idxmax()),
                        "reference_area_m2": float(_start_from_area.sum()),
                    }
                )
            else:
                _rows.append(
                    {
                        "zone": zone_name,
                        "transition_year": _year,
                        "area_year": _year,
                        "check": "start_year_area",
                        "comparison_available": False,
                        "reconciliation_status": "missing_area_year",
                        "max_abs_diff_m2": np.nan,
                        "sum_abs_diff_m2": np.nan,
                        "worst_label": "",
                        "reference_area_m2": np.nan,
                    }
                )

            _next_year = _year + 1
            if _next_year in _area_years:
                _end_from_transition = (
                    transition_table.sel(year=_year)
                    .sum(dim="start")
                    .to_series()
                    .reindex(_labels)
                    .fillna(0.0)
                )
                _end_from_area = _area.loc[_next_year, _labels].astype(float)
                _diff = (_end_from_transition - _end_from_area).abs()
                _max_abs_diff = float(_diff.max())
                _rows.append(
                    {
                        "zone": zone_name,
                        "transition_year": _year,
                        "area_year": _next_year,
                        "check": "next_year_area",
                        "comparison_available": True,
                        "reconciliation_status": "pass"
                        if _max_abs_diff <= tolerance_m2
                        else "fail",
                        "max_abs_diff_m2": _max_abs_diff,
                        "sum_abs_diff_m2": float(_diff.sum()),
                        "worst_label": str(_diff.idxmax()),
                        "reference_area_m2": float(_end_from_area.sum()),
                    }
                )
            else:
                _expected_edge = (
                    _max_area_year is not None
                    and _max_transition_year is not None
                    and _year == _max_transition_year
                    and _next_year == _max_area_year + 1
                )
                _rows.append(
                    {
                        "zone": zone_name,
                        "transition_year": _year,
                        "area_year": _next_year,
                        "check": "next_year_area",
                        "comparison_available": False,
                        "reconciliation_status": (
                            "expected_historical_edge"
                            if _expected_edge
                            else "missing_next_area_year"
                        ),
                        "max_abs_diff_m2": np.nan,
                        "sum_abs_diff_m2": np.nan,
                        "worst_label": "",
                        "reference_area_m2": np.nan,
                    }
                )

        return _rows

    "reconciliation helper defined"
    return (build_reconciliation_rows,)


@app.cell
def reconcile_tables(
    RECONCILIATION_ABS_TOLERANCE_M2,
    area_contract_checks,
    build_reconciliation_rows,
    loaded_area_tables,
    loaded_transition_tables,
    loaded_zone_names,
    pd,
    transition_contract_checks,
):
    _area_fail_zones = set(
        area_contract_checks.loc[
            area_contract_checks["area_hard_failure_count"] > 0,
            "zone",
        ]
    )
    _transition_fail_zones = set(
        transition_contract_checks.loc[
            transition_contract_checks["transition_hard_failure_count"] > 0,
            "zone",
        ]
    )
    zones_ready_for_reconciliation = tuple(
        _zone_name
        for _zone_name in loaded_zone_names
        if _zone_name not in _area_fail_zones
        and _zone_name not in _transition_fail_zones
    )

    _reconciliation_rows = []
    for _zone_name in zones_ready_for_reconciliation:
        _reconciliation_rows.extend(
            build_reconciliation_rows(
                _zone_name,
                loaded_area_tables[_zone_name],
                loaded_transition_tables[_zone_name],
                RECONCILIATION_ABS_TOLERANCE_M2,
            )
        )

    reconciliation_diagnostics = pd.DataFrame(_reconciliation_rows)
    if reconciliation_diagnostics.empty:
        reconciliation_summary_by_check = pd.DataFrame()
        reconciliation_failures = pd.DataFrame()
    else:
        _available_reconciliations = reconciliation_diagnostics.loc[
            reconciliation_diagnostics["comparison_available"]
        ]
        reconciliation_summary_by_check = (
            _available_reconciliations.groupby("check", dropna=False)
            .agg(
                comparisons=("zone", "count"),
                failed_comparisons=(
                    "reconciliation_status",
                    lambda _status: int((_status == "fail").sum()),
                ),
                max_class_abs_diff_m2=("max_abs_diff_m2", "max"),
                median_class_abs_diff_m2=("max_abs_diff_m2", "median"),
                max_sum_abs_diff_m2=("sum_abs_diff_m2", "max"),
            )
            .reset_index()
        )
        reconciliation_failures = reconciliation_diagnostics.loc[
            reconciliation_diagnostics["reconciliation_status"].isin(
                ["fail", "missing_area_year", "missing_next_area_year"]
            )
        ].sort_values(["zone", "transition_year", "check"])

    reconciliation_summary_by_check
    return reconciliation_diagnostics, reconciliation_failures


@app.cell
def show_reconciliation_diagnostics(reconciliation_diagnostics):
    reconciliation_diagnostics
    return


@app.cell
def show_reconciliation_failures(reconciliation_failures):
    reconciliation_failures
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-Zone Status

    The final status table rolls schema, value, load, and reconciliation findings up
    to the zone level. `fail` means the zone should not be used by later notebooks
    until repaired. `warn` means the zone is usable for the next stage, but one or
    more expected notes should remain visible.
    """)
    return


@app.cell
def summarize_zone_status(
    area_contract_checks,
    complete_zone_names,
    load_errors,
    np,
    pd,
    reconciliation_diagnostics,
    transition_contract_checks,
):
    _area_checks_by_zone = (
        area_contract_checks.set_index("zone")
        if not area_contract_checks.empty
        else pd.DataFrame()
    )
    _transition_checks_by_zone = (
        transition_contract_checks.set_index("zone")
        if not transition_contract_checks.empty
        else pd.DataFrame()
    )
    _load_errors_by_zone = (
        load_errors.groupby("zone").size().to_dict() if not load_errors.empty else {}
    )

    _zone_status_rows = []
    for _zone_name in complete_zone_names:
        _load_error_count = int(_load_errors_by_zone.get(_zone_name, 0))
        if _zone_name in _area_checks_by_zone.index:
            _area_row = _area_checks_by_zone.loc[_zone_name]
            _area_hard_count = int(_area_row["area_hard_failure_count"])
            _area_note_count = int(_area_row["area_note_count"])
            _area_issue_summary = str(_area_row["area_issue_summary"])
            _has_2020_area = bool(_area_row["has_2020_area"])
        else:
            _area_hard_count = 1
            _area_note_count = 0
            _area_issue_summary = "area_table did not load"
            _has_2020_area = False

        if _zone_name in _transition_checks_by_zone.index:
            _transition_row = _transition_checks_by_zone.loc[_zone_name]
            _transition_hard_count = int(
                _transition_row["transition_hard_failure_count"]
            )
            _transition_note_count = int(_transition_row["transition_note_count"])
            _transition_issue_summary = str(_transition_row["transition_issue_summary"])
        else:
            _transition_hard_count = 1
            _transition_note_count = 0
            _transition_issue_summary = "transition_table did not load"

        if not reconciliation_diagnostics.empty:
            _zone_reconciliation = reconciliation_diagnostics.loc[
                reconciliation_diagnostics["zone"] == _zone_name
            ]
            _reconciliation_failure_count = int(
                _zone_reconciliation["reconciliation_status"]
                .isin(["fail", "missing_area_year", "missing_next_area_year"])
                .sum()
            )
            _expected_edge_count = int(
                (
                    _zone_reconciliation["reconciliation_status"]
                    == "expected_historical_edge"
                ).sum()
            )
            _max_reconciliation_diff_m2 = (
                float(_zone_reconciliation["max_abs_diff_m2"].max())
                if not _zone_reconciliation["max_abs_diff_m2"].dropna().empty
                else np.nan
            )
        else:
            _reconciliation_failure_count = 0
            _expected_edge_count = 0
            _max_reconciliation_diff_m2 = np.nan

        _hard_failure_count = (
            _load_error_count
            + _area_hard_count
            + _transition_hard_count
            + _reconciliation_failure_count
        )
        _note_count = _area_note_count + _transition_note_count
        _contract_status = (
            "fail" if _hard_failure_count else "warn" if _note_count else "pass"
        )

        _issue_parts = [
            _part
            for _part in [_area_issue_summary, _transition_issue_summary]
            if _part and _part != "nan"
        ]
        if _reconciliation_failure_count:
            _issue_parts.append(
                f"reconciliation failures: {_reconciliation_failure_count}"
            )
        if _expected_edge_count:
            _issue_parts.append(
                f"expected 2021 -> 2022 edge comparisons: {_expected_edge_count}"
            )

        _zone_status_rows.append(
            {
                "zone": _zone_name,
                "contract_status": _contract_status,
                "usable_for_next_notebook": _hard_failure_count == 0 and _has_2020_area,
                "hard_failure_count": _hard_failure_count,
                "note_count": _note_count,
                "expected_edge_count": _expected_edge_count,
                "max_reconciliation_diff_m2": _max_reconciliation_diff_m2,
                "issue_summary": "; ".join(_issue_parts),
            }
        )

    zone_contract_status = pd.DataFrame(_zone_status_rows)
    zone_contract_status["_status_order"] = zone_contract_status["contract_status"].map(
        {"fail": 0, "warn": 1, "pass": 2}
    )
    zone_contract_status = (
        zone_contract_status.sort_values(["_status_order", "zone"])
        .drop(columns=["_status_order"])
        .reset_index(drop=True)
    )

    safe_historical_zone_names = tuple(
        zone_contract_status.loc[
            zone_contract_status["usable_for_next_notebook"],
            "zone",
        ]
    )

    zone_contract_status
    return safe_historical_zone_names, zone_contract_status


@app.cell
def show_safe_historical_zones(pd, safe_historical_zone_names):
    safe_historical_zones_table = pd.DataFrame({"zone": safe_historical_zone_names})

    safe_historical_zones_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary And Next Step

    The conclusion below is the handoff to `02_chen_2020_compatibility.py`. Zones
    marked usable have loaded historical tables, pass hard schema and value checks,
    and reconcile within the documented tolerance wherever both compared area years
    exist.
    """)
    return


@app.cell
def build_contract_conclusion(
    mo,
    np,
    reconciliation_diagnostics,
    safe_historical_zone_names,
    zone_contract_status,
):
    _status_counts = (
        zone_contract_status["contract_status"]
        .value_counts()
        .reindex(["pass", "warn", "fail"], fill_value=0)
    )
    _safe_zone_count = len(safe_historical_zone_names)
    _first_safe_zones = (
        ", ".join(safe_historical_zone_names[:10])
        if safe_historical_zone_names
        else "none"
    )
    _failure_count = int(_status_counts["fail"])
    _max_diff = (
        reconciliation_diagnostics.loc[
            reconciliation_diagnostics["comparison_available"],
            "max_abs_diff_m2",
        ].max()
        if not reconciliation_diagnostics.empty
        else np.nan
    )

    contract_check_conclusion = mo.md(
        f"""
    ### Contract Check Result

    - Pass zones: `{int(_status_counts["pass"])}`
    - Warn zones: `{int(_status_counts["warn"])}`
    - Fail zones: `{_failure_count}`
    - Usable zones for `02_chen_2020_compatibility.py`: `{_safe_zone_count}`
    - First usable zones: `{_first_safe_zones}`
    - Largest available reconciliation difference: `{_max_diff:.6g} m²`

    The expected `2021 -> 2022` transition edge is reported but does not make a
    zone fail. This notebook produced diagnostic tables only; it did not write a
    durable zone manifest.
    """
    )

    contract_check_conclusion
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Limitations

    These checks validate aggregate table contracts, not spatial semantics. They do
    not compare Chen urban pixels with GLC settlements and do not decide whether
    Chen projections are compatible with the historical baseline. They only tell the
    next notebook which historical zones can be trusted as table inputs.
    """)
    return


if __name__ == "__main__":
    app.run()
