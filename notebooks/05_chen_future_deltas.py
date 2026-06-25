import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415
    from itertools import pairwise  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import ee  # noqa: PLC0415
    import marimo as mo  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    import seaborn as sns  # noqa: PLC0415
    from dagster_components.partitions import zone_partitions  # noqa: PLC0415

    from nu_afolu.constants import (  # noqa: PLC0415
        CHEN_COLLECTION_ID,
        CHEN_URBAN_VALUE,
        CHEN_YEARS,
        LABEL_LIST,
        SSP_NAMES,
    )

    return (
        CHEN_COLLECTION_ID,
        CHEN_URBAN_VALUE,
        CHEN_YEARS,
        LABEL_LIST,
        Path,
        SSP_NAMES,
        ee,
        json,
        mo,
        np,
        os,
        pairwise,
        pd,
        plt,
        sns,
        zone_partitions,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 05 Chen Future Deltas

    This notebook converts Chen SSP urban-area projections into decadal settlement
    growth demand by zone and scenario. It makes the future demand signal explicit
    before any source-class allocation is attempted.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal, Inputs, And Assumptions

    This stage uses Chen projected urban area, GLC-derived 2020 `settlements`, and
    the compatibility thresholds established in `02_chen_2020_compatibility.py`.

    The default stitching rule comes from `analysis_contracts.md`: use GLC 2020
    `settlements` as the baseline for future pseudo-tables, and use Chen decadal
    deltas as growth signals. A negative Chen delta is not converted into
    de-urbanization in this first diagnostic workflow; it is clipped to zero demand
    and recorded as unresolved scenario mismatch.
    """)
    return


@app.cell
def _(Path, os, pd):
    PROJECT_ROOT = Path.cwd()
    OUT_PATH_KEY = "OUT_PATH"
    BASELINE_YEAR = 2020
    CHEN_SCALE_M = 1000
    RATIO_DENOMINATOR_FLOOR_M2 = 1_000_000.0
    MANUAL_REVIEW_ABS_DIFF_HA = 5_000.0
    MANUAL_REVIEW_ABS_RATIO_ERROR = 1.0
    DEMAND_BASELINE_CHOICE = "glc_2020_settlements"
    DEMAND_CALIBRATION_CHOICE = "glc_2020_plus_clipped_chen_deltas"
    NEGATIVE_DELTA_POLICY = "clip_to_zero_and_record"


    def read_dotenv_value(path: Path, key: str) -> str | None:
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
        _out_path_raw = read_dotenv_value(PROJECT_ROOT / ".env", OUT_PATH_KEY)
        OUT_PATH_SOURCE = ".env file" if _out_path_raw else "missing"

    OUT_PATH = Path(_out_path_raw).expanduser() if _out_path_raw else None
    OUT_PATH_EXISTS = bool(OUT_PATH and OUT_PATH.exists())

    configuration_summary = pd.DataFrame(
        [
            {"setting": OUT_PATH_KEY, "value": str(OUT_PATH) if OUT_PATH else "not configured", "source": OUT_PATH_SOURCE},
            {"setting": "BASELINE_YEAR", "value": BASELINE_YEAR, "source": "Chen/GLC compatibility contract"},
            {"setting": "CHEN_SCALE_M", "value": CHEN_SCALE_M, "source": "Chen approximate resolution"},
            {"setting": "DEMAND_BASELINE_CHOICE", "value": DEMAND_BASELINE_CHOICE, "source": "analysis_contracts.md"},
            {"setting": "DEMAND_CALIBRATION_CHOICE", "value": DEMAND_CALIBRATION_CHOICE, "source": "notebook default"},
            {"setting": "NEGATIVE_DELTA_POLICY", "value": NEGATIVE_DELTA_POLICY, "source": "analysis_contracts.md"},
        ]
    )

    configuration_summary
    return (
        BASELINE_YEAR,
        CHEN_SCALE_M,
        DEMAND_BASELINE_CHOICE,
        DEMAND_CALIBRATION_CHOICE,
        MANUAL_REVIEW_ABS_DIFF_HA,
        MANUAL_REVIEW_ABS_RATIO_ERROR,
        NEGATIVE_DELTA_POLICY,
        OUT_PATH,
        RATIO_DENOMINATOR_FLOOR_M2,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Projection Years And Interval Semantics

    Chen provides decadal images from 2020 through 2100. The interval rows produced
    below use `start_year` and `end_year` to mean the change implied between two
    adjacent Chen images. These are decadal scenario intervals, not historical
    annual `Y -> Y + 1` transitions.
    """)
    return


@app.cell
def _(
    CHEN_COLLECTION_ID,
    CHEN_URBAN_VALUE,
    CHEN_YEARS,
    SSP_NAMES,
    pairwise,
    pd,
):
    chen_contract_summary = pd.DataFrame(
        [
            {"field": "collection", "value": CHEN_COLLECTION_ID},
            {"field": "urban pixel value", "value": CHEN_URBAN_VALUE},
            {"field": "projection years", "value": ", ".join(str(_year) for _year in CHEN_YEARS)},
            {"field": "scenarios", "value": ", ".join(SSP_NAMES)},
        ]
    )

    projection_intervals = pd.DataFrame(
        [
            {
                "start_year": int(_start_year),
                "end_year": int(_end_year),
                "interval_years": int(_end_year - _start_year),
                "interval_semantics": f"{_start_year} to {_end_year}",
            }
            for _start_year, _end_year in pairwise(CHEN_YEARS)
        ]
    )

    projection_intervals
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baseline Choice

    The future demand table keeps the baseline and trajectory concepts separate.
    Chen totals are treated as a projected urban-area signal, while GLC 2020
    `settlements` remains the baseline settlement area for later pseudo-`area_table`
    construction. This avoids silently replacing the observed 2020 state with Chen.
    """)
    return


@app.cell
def _(OUT_PATH, Path, pd, zone_partitions):
    REQUIRED_ARTIFACT_SPECS = {
        "bbox_ee": {"relative_dir": Path("bbox") / "ee", "extension": ".json"},
        "area_table": {"relative_dir": Path("area_table"), "extension": ".parquet"},
        "transition_table": {"relative_dir": Path("transition_table"), "extension": ".nc"},
    }

    partition_zone_names = tuple(zone_partitions.get_partition_keys())

    artifact_zone_sets = {}
    for _artifact_name, _spec in REQUIRED_ARTIFACT_SPECS.items():
        _artifact_dir = OUT_PATH / _spec["relative_dir"] if OUT_PATH else None
        if _artifact_dir and _artifact_dir.exists():
            artifact_zone_sets[_artifact_name] = {
                _path.stem for _path in _artifact_dir.glob(f"*{_spec['extension']}")
            }
        else:
            artifact_zone_sets[_artifact_name] = set()

    _zone_rows = []
    for _zone_name in partition_zone_names:
        _row = {"zone": _zone_name}
        for _artifact_name in REQUIRED_ARTIFACT_SPECS:
            _row[_artifact_name] = _zone_name in artifact_zone_sets[_artifact_name]
        _row["complete_required_inputs"] = all(
            _row[_artifact_name] for _artifact_name in REQUIRED_ARTIFACT_SPECS
        )
        _zone_rows.append(_row)

    input_zone_inventory = pd.DataFrame(_zone_rows)
    candidate_zone_names = tuple(
        input_zone_inventory.loc[input_zone_inventory["complete_required_inputs"], "zone"]
    )

    input_zone_summary = pd.DataFrame(
        [
            {"metric": "canonical partition zones", "value": len(partition_zone_names)},
            {"metric": "zones with required local inputs", "value": len(candidate_zone_names)},
            {"metric": "bbox/ee files discovered", "value": len(artifact_zone_sets["bbox_ee"])},
            {"metric": "area_table files discovered", "value": len(artifact_zone_sets["area_table"])},
            {"metric": "transition_table files discovered", "value": len(artifact_zone_sets["transition_table"])},
        ]
    )

    input_zone_summary
    return (candidate_zone_names,)


@app.cell
def _(
    BASELINE_YEAR,
    LABEL_LIST,
    OUT_PATH,
    RATIO_DENOMINATOR_FLOOR_M2,
    candidate_zone_names,
    np,
    pd,
):
    _baseline_rows = []
    _baseline_error_rows = []

    for _zone_name in candidate_zone_names:
        _area_path = OUT_PATH / "area_table" / f"{_zone_name}.parquet"
        try:
            _area_table = pd.read_parquet(_area_path)
            _area_table.index = _area_table.index.astype(int)
            _normalized_area = (
                _area_table.reindex(columns=list(LABEL_LIST))
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
            )
            if BASELINE_YEAR not in _normalized_area.index:
                _baseline_error_rows.append(
                    {"zone": _zone_name, "error": f"missing {BASELINE_YEAR} area row"}
                )
                continue
            _settlement_area_m2 = float(_normalized_area.loc[BASELINE_YEAR, "settlements"])
            if not np.isfinite(_settlement_area_m2) or _settlement_area_m2 < 0:
                _baseline_error_rows.append(
                    {"zone": _zone_name, "error": "invalid settlements area"}
                )
                continue
            _baseline_rows.append(
                {
                    "zone": _zone_name,
                    "glc_settlements_2020_m2": _settlement_area_m2,
                    "glc_settlements_2020_ha": _settlement_area_m2 / 10_000.0,
                    "near_zero_glc_baseline": _settlement_area_m2 < RATIO_DENOMINATOR_FLOOR_M2,
                    "area_year_min": int(_normalized_area.index.min()),
                    "area_year_max": int(_normalized_area.index.max()),
                    "has_observed_area_after_2020": bool((_normalized_area.index > BASELINE_YEAR).any()),
                }
            )
        except Exception as _exc:  # noqa: BLE001
            _baseline_error_rows.append({"zone": _zone_name, "error": str(_exc)})

    glc_settlement_baseline = pd.DataFrame(_baseline_rows)
    baseline_load_errors = pd.DataFrame(_baseline_error_rows, columns=["zone", "error"])
    safe_historical_zone_names = tuple(glc_settlement_baseline["zone"])

    baseline_summary = pd.DataFrame(
        [
            {"metric": "candidate zones", "value": len(candidate_zone_names)},
            {"metric": "zones with usable GLC 2020 settlements", "value": len(safe_historical_zone_names)},
            {"metric": "near-zero GLC baselines", "value": int(glc_settlement_baseline["near_zero_glc_baseline"].sum())},
            {"metric": "zones with observed area after 2020", "value": int(glc_settlement_baseline["has_observed_area_after_2020"].sum())},
            {"metric": "baseline load errors", "value": len(baseline_load_errors)},
        ]
    )

    baseline_summary
    return (
        baseline_load_errors,
        glc_settlement_baseline,
        safe_historical_zone_names,
    )


@app.cell
def _(baseline_load_errors):
    baseline_load_errors
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chen Urban Trajectories

    The Chen collection is reduced over the same zone geometries used by the
    historical artifacts. The collection does not expose year metadata in this
    environment, so image order is interpreted through the local `CHEN_YEARS`
    contract.

    For efficiency, each year/SSP urban mask is converted into an area band and
    stacked into a single image. Each zone is then reduced once across all stacked
    bands.
    """)
    return


@app.cell
def _(CHEN_COLLECTION_ID, CHEN_YEARS, SSP_NAMES, ee, pd):
    def short_error(exc: Exception) -> str:
        message = str(exc).replace("\n", " ")
        return message[:500] + ("..." if len(message) > 500 else "")


    try:
        ee.Initialize()
        chen_collection = ee.ImageCollection(CHEN_COLLECTION_ID)
        chen_collection_size = int(chen_collection.size().getInfo())
        _first_image = ee.Image(chen_collection.toList(chen_collection_size).get(0))
        chen_first_image_band_names = tuple(_first_image.bandNames().getInfo())
        chen_source_ready = chen_collection_size >= len(CHEN_YEARS) and set(SSP_NAMES).issubset(chen_first_image_band_names)
        chen_source_error = ""
    except Exception as _exc:  # noqa: BLE001
        chen_collection = None
        chen_collection_size = None
        chen_first_image_band_names = ()
        chen_source_ready = False
        chen_source_error = short_error(_exc)

    chen_source_summary = pd.DataFrame(
        [
            {
                "collection_size": chen_collection_size,
                "expected_year_count": len(CHEN_YEARS),
                "first_image_band_names": ", ".join(chen_first_image_band_names),
                "expected_ssp_bands_present": set(SSP_NAMES).issubset(chen_first_image_band_names),
                "source_ready": chen_source_ready,
                "error": chen_source_error,
            }
        ]
    )

    chen_source_summary
    return (
        chen_collection,
        chen_collection_size,
        chen_source_error,
        chen_source_ready,
        short_error,
    )


@app.cell
def _(
    CHEN_URBAN_VALUE,
    CHEN_YEARS,
    Path,
    SSP_NAMES,
    chen_collection,
    chen_collection_size,
    chen_source_ready,
    ee,
    json,
):
    def load_ee_geometry(path: Path):
        with path.open(encoding="utf-8") as file:
            return ee.Geometry(ee.deserializer.decode(json.load(file)))


    def build_chen_area_stack():
        collection_list = chen_collection.toList(chen_collection_size)
        bands = []
        band_keys = []
        for year_index, year in enumerate(CHEN_YEARS):
            year_image = ee.Image(collection_list.get(year_index))
            for ssp in SSP_NAMES:
                band_key = f"{ssp}_{year}"
                bands.append(
                    year_image.select(ssp)
                    .eq(ee.Number(CHEN_URBAN_VALUE))
                    .multiply(ee.Image.pixelArea())
                    .rename(band_key)
                )
                band_keys.append((ssp, int(year), band_key))
        return ee.Image.cat(bands), tuple(band_keys)


    if chen_source_ready:
        chen_area_stack, chen_area_band_keys = build_chen_area_stack()
    else:
        chen_area_stack = None
        chen_area_band_keys = ()

    "Chen reduction helpers defined"
    return chen_area_band_keys, chen_area_stack, load_ee_geometry


@app.cell
def _(
    CHEN_SCALE_M,
    OUT_PATH,
    chen_area_band_keys,
    chen_area_stack,
    chen_source_error,
    chen_source_ready,
    ee,
    load_ee_geometry,
    pd,
    safe_historical_zone_names,
    short_error,
):
    _trajectory_rows = []
    _chen_error_rows = []

    if chen_source_ready and chen_area_stack is not None:
        for _zone_name in safe_historical_zone_names:
            try:
                _geometry = load_ee_geometry(OUT_PATH / "bbox" / "ee" / f"{_zone_name}.json")
                _result = (
                    chen_area_stack.reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=_geometry,
                        scale=CHEN_SCALE_M,
                        maxPixels=int(1e10),
                        tileScale=4,
                    ).getInfo()
                )
                for _ssp, _year, _band_key in chen_area_band_keys:
                    _area_m2 = float(_result.get(_band_key) or 0.0)
                    _trajectory_rows.append(
                        {
                            "zone": _zone_name,
                            "ssp": _ssp,
                            "year": _year,
                            "chen_urban_area_m2": _area_m2,
                            "chen_urban_area_ha": _area_m2 / 10_000.0,
                        }
                    )
            except Exception as _exc:  # noqa: BLE001
                _chen_error_rows.append({"zone": _zone_name, "error": short_error(_exc)})
    else:
        _chen_error_rows.extend(
            {"zone": _zone_name, "error": chen_source_error}
            for _zone_name in safe_historical_zone_names
        )

    chen_urban_trajectory = pd.DataFrame(
        _trajectory_rows,
        columns=["zone", "ssp", "year", "chen_urban_area_m2", "chen_urban_area_ha"],
    )
    chen_reduction_errors = pd.DataFrame(_chen_error_rows, columns=["zone", "error"])

    chen_reduction_summary = pd.DataFrame(
        [
            {"metric": "zones requested", "value": len(safe_historical_zone_names)},
            {"metric": "zones reduced successfully", "value": chen_urban_trajectory["zone"].nunique() if not chen_urban_trajectory.empty else 0},
            {"metric": "trajectory rows", "value": len(chen_urban_trajectory)},
            {"metric": "zone reduction errors", "value": len(chen_reduction_errors)},
        ]
    )

    chen_reduction_summary
    return chen_reduction_errors, chen_urban_trajectory


@app.cell
def _(chen_reduction_errors):
    chen_reduction_errors
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compatibility Decisions Carried Forward

    Notebook `02` treated 2020 Chen/GLC mismatch as a decision point rather than a
    hard validation. This section recomputes the same zone-level decision flags so
    the demand table can carry compatibility context forward into allocation.
    """)
    return


@app.cell
def _(
    BASELINE_YEAR,
    MANUAL_REVIEW_ABS_DIFF_HA,
    MANUAL_REVIEW_ABS_RATIO_ERROR,
    RATIO_DENOMINATOR_FLOOR_M2,
    chen_urban_trajectory,
    glc_settlement_baseline,
    np,
    pd,
):
    chen_2020_compatibility = chen_urban_trajectory.loc[
        chen_urban_trajectory["year"] == BASELINE_YEAR
    ].merge(
        glc_settlement_baseline[
            [
                "zone",
                "glc_settlements_2020_m2",
                "glc_settlements_2020_ha",
                "near_zero_glc_baseline",
            ]
        ],
        on="zone",
        how="inner",
    )

    chen_2020_compatibility = chen_2020_compatibility.assign(
        signed_difference_m2=lambda _df: _df["chen_urban_area_m2"] - _df["glc_settlements_2020_m2"],
        absolute_difference_m2=lambda _df: _df["signed_difference_m2"].abs(),
        signed_difference_ha=lambda _df: _df["signed_difference_m2"] / 10_000.0,
        absolute_difference_ha=lambda _df: _df["absolute_difference_m2"] / 10_000.0,
        ratio_denominator_is_stable=lambda _df: _df["glc_settlements_2020_m2"] >= RATIO_DENOMINATOR_FLOOR_M2,
    )

    _stable_denominator = chen_2020_compatibility["ratio_denominator_is_stable"]
    chen_2020_compatibility = chen_2020_compatibility.assign(
        chen_to_glc_ratio=np.where(
            _stable_denominator,
            chen_2020_compatibility["chen_urban_area_m2"] / chen_2020_compatibility["glc_settlements_2020_m2"],
            np.nan,
        )
    )
    chen_2020_compatibility = chen_2020_compatibility.assign(
        ratio_error=lambda _df: _df["chen_to_glc_ratio"] - 1.0,
        absolute_ratio_error=lambda _df: _df["ratio_error"].abs(),
    )


    def _assign_compatibility_decision(row: pd.Series) -> str:
        _ratio_flag = (
            pd.notna(row["max_abs_ratio_error"])
            and row["max_abs_ratio_error"] > MANUAL_REVIEW_ABS_RATIO_ERROR
        )
        _absolute_flag = row["max_abs_difference_ha"] > MANUAL_REVIEW_ABS_DIFF_HA
        if row["ratio_unstable_rows"] > 0:
            return "manual_review_small_baseline"
        if _ratio_flag and _absolute_flag:
            return "manual_review_large_absolute_and_ratio_mismatch"
        if _absolute_flag:
            return "manual_review_large_absolute_mismatch"
        if _ratio_flag:
            return "manual_review_large_ratio_mismatch"
        return "carry_forward_with_documented_mismatch"


    zone_compatibility_decisions = (
        chen_2020_compatibility.groupby("zone", dropna=False)
        .agg(
            glc_settlements_2020_ha=("glc_settlements_2020_ha", "first"),
            median_chen_urban_2020_ha=("chen_urban_area_ha", "median"),
            max_abs_difference_ha=("absolute_difference_ha", "max"),
            max_abs_ratio_error=("absolute_ratio_error", "max"),
            ratio_unstable_rows=("ratio_denominator_is_stable", lambda _series: int((~_series).sum())),
        )
        .reset_index()
    )
    zone_compatibility_decisions = zone_compatibility_decisions.assign(
        compatibility_decision=lambda _df: _df.apply(_assign_compatibility_decision, axis=1),
        needs_manual_review=lambda _df: _df["compatibility_decision"].str.startswith("manual_review"),
    )

    compatibility_decision_summary = (
        zone_compatibility_decisions["compatibility_decision"]
        .value_counts()
        .rename_axis("compatibility_decision")
        .reset_index(name="zone_count")
    )

    manual_review_zones = zone_compatibility_decisions.loc[
        zone_compatibility_decisions["needs_manual_review"]
    ].sort_values(["max_abs_difference_ha", "max_abs_ratio_error"], ascending=[False, False])

    compatibility_decision_summary
    return (
        chen_2020_compatibility,
        manual_review_zones,
        zone_compatibility_decisions,
    )


@app.cell
def _(manual_review_zones):
    manual_review_zones
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chen Urban Trajectory Summary

    The total trajectory plot is not used as an allocation table. It is a quick
    scenario-level check: if a scenario declines or stalls in aggregate, the
    negative-delta diagnostics below should explain where that behavior comes from.
    """)
    return


@app.cell
def _(BASELINE_YEAR, CHEN_YEARS, chen_urban_trajectory):
    trajectory_summary_by_ssp_year = (
        chen_urban_trajectory.groupby(["ssp", "year"], dropna=False)
        .agg(
            zones=("zone", "nunique"),
            total_chen_urban_area_m2=("chen_urban_area_m2", "sum"),
            median_zone_chen_urban_area_m2=("chen_urban_area_m2", "median"),
            max_zone_chen_urban_area_m2=("chen_urban_area_m2", "max"),
        )
        .reset_index()
        .assign(
            total_chen_urban_area_ha=lambda _df: _df["total_chen_urban_area_m2"] / 10_000.0,
            median_zone_chen_urban_area_ha=lambda _df: _df["median_zone_chen_urban_area_m2"] / 10_000.0,
        )
    )

    _trajectory_start = trajectory_summary_by_ssp_year.loc[
        trajectory_summary_by_ssp_year["year"] == BASELINE_YEAR,
        ["ssp", "total_chen_urban_area_m2"],
    ].rename(columns={"total_chen_urban_area_m2": "total_chen_urban_2020_m2"})
    _trajectory_end = trajectory_summary_by_ssp_year.loc[
        trajectory_summary_by_ssp_year["year"] == max(CHEN_YEARS),
        ["ssp", "total_chen_urban_area_m2"],
    ].rename(columns={"total_chen_urban_area_m2": "total_chen_urban_2100_m2"})

    trajectory_summary_by_ssp = _trajectory_start.merge(_trajectory_end, on="ssp", how="inner")
    trajectory_summary_by_ssp = trajectory_summary_by_ssp.assign(
        net_chen_change_2020_2100_m2=lambda _df: _df["total_chen_urban_2100_m2"] - _df["total_chen_urban_2020_m2"],
        total_chen_urban_2020_ha=lambda _df: _df["total_chen_urban_2020_m2"] / 10_000.0,
        total_chen_urban_2100_ha=lambda _df: _df["total_chen_urban_2100_m2"] / 10_000.0,
        net_chen_change_2020_2100_ha=lambda _df: _df["net_chen_change_2020_2100_m2"] / 10_000.0,
    )

    trajectory_summary_by_ssp
    return (trajectory_summary_by_ssp_year,)


@app.cell
def _(plt, sns, trajectory_summary_by_ssp_year):
    sns.set_theme(style="whitegrid")

    _trajectory_fig, _trajectory_ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        data=trajectory_summary_by_ssp_year,
        x="year",
        y="total_chen_urban_area_ha",
        hue="ssp",
        marker="o",
        ax=_trajectory_ax,
    )
    _trajectory_ax.set_xlabel("Chen projection year")
    _trajectory_ax.set_ylabel("Total Chen urban area (ha)")
    _trajectory_ax.set_title("Total Chen urban trajectory by SSP")
    _trajectory_ax.legend(title="SSP", bbox_to_anchor=(1.02, 1), loc="upper left")
    _trajectory_fig.tight_layout()

    total_trajectory_plot = _trajectory_fig

    total_trajectory_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decadal Deltas And Diagnostic Demand

    The raw Chen delta is the difference in Chen urban area between adjacent
    projection years. The first diagnostic allocation method only consumes positive
    settlement growth, so negative raw deltas are clipped to zero demand and the
    clipped area is retained in separate columns.
    """)
    return


@app.cell
def _(
    DEMAND_BASELINE_CHOICE,
    DEMAND_CALIBRATION_CHOICE,
    NEGATIVE_DELTA_POLICY,
    chen_2020_compatibility,
    chen_urban_trajectory,
    np,
    zone_compatibility_decisions,
):
    _trajectory_sorted = chen_urban_trajectory.sort_values(["zone", "ssp", "year"]).copy()
    _trajectory_sorted["start_year"] = _trajectory_sorted.groupby(["zone", "ssp"])["year"].shift()
    _trajectory_sorted["start_chen_urban_area_m2"] = _trajectory_sorted.groupby(["zone", "ssp"])[
        "chen_urban_area_m2"
    ].shift()
    _trajectory_sorted["start_chen_urban_area_ha"] = _trajectory_sorted["start_chen_urban_area_m2"] / 10_000.0

    chen_settlement_demand_table = (
        _trajectory_sorted.loc[_trajectory_sorted["start_year"].notna()]
        .rename(
            columns={
                "year": "end_year",
                "chen_urban_area_m2": "end_chen_urban_area_m2",
                "chen_urban_area_ha": "end_chen_urban_area_ha",
            }
        )
        .copy()
    )

    chen_settlement_demand_table["start_year"] = chen_settlement_demand_table["start_year"].astype(int)
    chen_settlement_demand_table["end_year"] = chen_settlement_demand_table["end_year"].astype(int)
    chen_settlement_demand_table = chen_settlement_demand_table.assign(
        interval_years=lambda _df: _df["end_year"] - _df["start_year"],
        raw_delta_m2=lambda _df: _df["end_chen_urban_area_m2"] - _df["start_chen_urban_area_m2"],
    )
    chen_settlement_demand_table = chen_settlement_demand_table.assign(
        raw_delta_ha=lambda _df: _df["raw_delta_m2"] / 10_000.0,
        delta_status=np.select(
            [
                chen_settlement_demand_table["raw_delta_m2"] < 0,
                chen_settlement_demand_table["raw_delta_m2"] == 0,
                chen_settlement_demand_table["raw_delta_m2"] > 0,
            ],
            ["negative", "zero", "positive"],
            default="unknown",
        ),
        demand_m2=lambda _df: _df["raw_delta_m2"].clip(lower=0.0),
        clipped_negative_delta_m2=lambda _df: np.where(_df["raw_delta_m2"] < 0, -_df["raw_delta_m2"], 0.0),
    )
    chen_settlement_demand_table = chen_settlement_demand_table.assign(
        demand_ha=lambda _df: _df["demand_m2"] / 10_000.0,
        clipped_negative_delta_ha=lambda _df: _df["clipped_negative_delta_m2"] / 10_000.0,
    )

    chen_settlement_demand_table = chen_settlement_demand_table.merge(
        chen_2020_compatibility[
            [
                "zone",
                "ssp",
                "glc_settlements_2020_m2",
                "glc_settlements_2020_ha",
                "signed_difference_m2",
                "absolute_difference_ha",
                "chen_to_glc_ratio",
            ]
        ].rename(
            columns={
                "signed_difference_m2": "chen_2020_minus_glc_2020_m2",
                "absolute_difference_ha": "chen_glc_2020_abs_difference_ha",
            }
        ),
        on=["zone", "ssp"],
        how="left",
    )
    chen_settlement_demand_table = chen_settlement_demand_table.merge(
        zone_compatibility_decisions[
            ["zone", "compatibility_decision", "needs_manual_review"]
        ],
        on="zone",
        how="left",
    )

    chen_settlement_demand_table = chen_settlement_demand_table.sort_values(
        ["zone", "ssp", "start_year", "end_year"]
    ).reset_index(drop=True)
    chen_settlement_demand_table["cumulative_raw_delta_from_2020_m2"] = chen_settlement_demand_table.groupby(
        ["zone", "ssp"]
    )["raw_delta_m2"].cumsum()
    chen_settlement_demand_table["cumulative_demand_from_2020_m2"] = chen_settlement_demand_table.groupby(
        ["zone", "ssp"]
    )["demand_m2"].cumsum()
    chen_settlement_demand_table = chen_settlement_demand_table.assign(
        glc_plus_raw_delta_area_m2=lambda _df: _df["glc_settlements_2020_m2"] + _df["cumulative_raw_delta_from_2020_m2"],
        glc_plus_clipped_demand_area_m2=lambda _df: _df["glc_settlements_2020_m2"] + _df["cumulative_demand_from_2020_m2"],
        baseline_choice=DEMAND_BASELINE_CHOICE,
        calibration_choice=DEMAND_CALIBRATION_CHOICE,
        negative_delta_policy=NEGATIVE_DELTA_POLICY,
    )

    chen_settlement_demand_table = chen_settlement_demand_table[
        [
            "zone",
            "ssp",
            "start_year",
            "end_year",
            "interval_years",
            "start_chen_urban_area_m2",
            "end_chen_urban_area_m2",
            "raw_delta_m2",
            "raw_delta_ha",
            "delta_status",
            "demand_m2",
            "demand_ha",
            "clipped_negative_delta_m2",
            "clipped_negative_delta_ha",
            "glc_settlements_2020_m2",
            "chen_2020_minus_glc_2020_m2",
            "chen_to_glc_ratio",
            "glc_plus_raw_delta_area_m2",
            "glc_plus_clipped_demand_area_m2",
            "compatibility_decision",
            "needs_manual_review",
            "baseline_choice",
            "calibration_choice",
            "negative_delta_policy",
        ]
    ]

    chen_settlement_demand_table
    return (chen_settlement_demand_table,)


@app.cell
def _(chen_settlement_demand_table):
    demand_summary_by_interval_ssp = (
        chen_settlement_demand_table.groupby(["start_year", "end_year", "ssp"], dropna=False)
        .agg(
            zones=("zone", "nunique"),
            total_raw_delta_ha=("raw_delta_ha", "sum"),
            total_demand_ha=("demand_ha", "sum"),
            total_clipped_negative_delta_ha=("clipped_negative_delta_ha", "sum"),
            positive_delta_rows=("delta_status", lambda _series: int((_series == "positive").sum())),
            zero_delta_rows=("delta_status", lambda _series: int((_series == "zero").sum())),
            negative_delta_rows=("delta_status", lambda _series: int((_series == "negative").sum())),
            manual_review_zone_rows=("needs_manual_review", "sum"),
        )
        .reset_index()
    )

    delta_status_summary = (
        chen_settlement_demand_table.groupby(["ssp", "delta_status"], dropna=False)
        .agg(
            interval_rows=("zone", "count"),
            zones=("zone", "nunique"),
            total_raw_delta_ha=("raw_delta_ha", "sum"),
            total_demand_ha=("demand_ha", "sum"),
            total_clipped_negative_delta_ha=("clipped_negative_delta_ha", "sum"),
        )
        .reset_index()
    )

    demand_summary_by_interval_ssp
    return (demand_summary_by_interval_ssp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Negative And Non-Monotonic Deltas

    Negative Chen deltas indicate that the projected urban mask gets smaller for a
    zone, SSP, and decade. This workflow does not infer which AFOLU class would
    receive de-urbanized area, so those deltas are treated as unresolved mismatch
    rather than settlement-loss transitions.
    """)
    return


@app.cell
def _(chen_settlement_demand_table):
    negative_delta_diagnostics = (
        chen_settlement_demand_table.loc[chen_settlement_demand_table["raw_delta_m2"] < 0]
        .sort_values("clipped_negative_delta_m2", ascending=False)
        .reset_index(drop=True)
    )

    large_negative_delta_diagnostics = negative_delta_diagnostics.loc[
        :,
        [
            "zone",
            "ssp",
            "start_year",
            "end_year",
            "raw_delta_ha",
            "clipped_negative_delta_ha",
            "compatibility_decision",
            "needs_manual_review",
        ],
    ].head(25)

    large_negative_delta_diagnostics
    return (negative_delta_diagnostics,)


@app.cell
def _(demand_summary_by_interval_ssp, negative_delta_diagnostics, plt, sns):
    _plot_demand_summary = demand_summary_by_interval_ssp.copy()
    _plot_demand_summary["interval"] = (
        _plot_demand_summary["start_year"].astype(str)
        + "-"
        + _plot_demand_summary["end_year"].astype(str)
    )
    _negative_by_ssp = (
        negative_delta_diagnostics.groupby("ssp", dropna=False)
        .agg(total_clipped_negative_delta_ha=("clipped_negative_delta_ha", "sum"))
        .reset_index()
    )

    _demand_fig, _demand_axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(
        data=_plot_demand_summary,
        x="interval",
        y="total_demand_ha",
        hue="ssp",
        ax=_demand_axes[0],
    )
    _demand_axes[0].set_xlabel("Chen interval")
    _demand_axes[0].set_ylabel("Clipped diagnostic demand (ha)")
    _demand_axes[0].set_title("Positive settlement-growth demand")
    _demand_axes[0].tick_params(axis="x", rotation=45)
    _demand_axes[0].legend(title="SSP", fontsize="small")

    sns.barplot(
        data=_negative_by_ssp,
        x="ssp",
        y="total_clipped_negative_delta_ha",
        color="#d65f5f",
        ax=_demand_axes[1],
    )
    _demand_axes[1].set_xlabel("SSP")
    _demand_axes[1].set_ylabel("Recorded clipped negative delta (ha)")
    _demand_axes[1].set_title("Unresolved negative-delta area")
    _demand_fig.tight_layout()

    demand_delta_plots = _demand_fig

    demand_delta_plots
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calibration Options For Allocation

    The demand table keeps enough columns to support several later choices. The
    recommended first allocation path is the conservative one: begin from GLC 2020
    settlements and add only non-negative Chen decadal demand. Raw Chen totals and
    raw cumulative deltas remain visible for diagnostics.
    """)
    return


@app.cell
def _(DEMAND_BASELINE_CHOICE, NEGATIVE_DELTA_POLICY, pd):
    baseline_calibration_summary = pd.DataFrame(
        [
            {
                "choice": "baseline settlement area",
                "selected_value": DEMAND_BASELINE_CHOICE,
                "interpretation": "Start future pseudo-area tables from observed GLC 2020 settlements.",
            },
            {
                "choice": "growth signal",
                "selected_value": "Chen adjacent-year urban-area deltas",
                "interpretation": "Use Chen as a decadal expansion signal, not as a wholesale replacement for observed 2020.",
            },
            {
                "choice": "negative deltas",
                "selected_value": NEGATIVE_DELTA_POLICY,
                "interpretation": "Do not create de-urbanization transitions in the first diagnostic allocation.",
            },
            {
                "choice": "allocation readiness",
                "selected_value": "carry compatibility_decision and needs_manual_review columns",
                "interpretation": "Notebook 06 can filter, stratify, or route flagged zones to pooled/review workflows.",
            },
        ]
    )

    baseline_calibration_summary
    return


@app.cell
def _(
    chen_settlement_demand_table,
    chen_urban_trajectory,
    mo,
    negative_delta_diagnostics,
    zone_compatibility_decisions,
):
    _total_demand_by_ssp = (
        chen_settlement_demand_table.groupby("ssp", dropna=False)["demand_ha"]
        .sum()
        .sort_values(ascending=False)
    )
    _total_clipped_by_ssp = (
        chen_settlement_demand_table.groupby("ssp", dropna=False)["clipped_negative_delta_ha"]
        .sum()
        .sort_values(ascending=False)
    )
    _negative_interval_rows = int((chen_settlement_demand_table["delta_status"] == "negative").sum())
    _zero_interval_rows = int((chen_settlement_demand_table["delta_status"] == "zero").sum())
    _positive_interval_rows = int((chen_settlement_demand_table["delta_status"] == "positive").sum())
    _manual_review_zone_count = int(zone_compatibility_decisions["needs_manual_review"].sum())
    _max_negative_delta_ha = float(negative_delta_diagnostics["clipped_negative_delta_ha"].max()) if not negative_delta_diagnostics.empty else 0.0
    _top_demand_ssp = _total_demand_by_ssp.index[0] if not _total_demand_by_ssp.empty else "none"
    _top_demand_ha = float(_total_demand_by_ssp.iloc[0]) if not _total_demand_by_ssp.empty else 0.0
    _top_clipped_ssp = _total_clipped_by_ssp.index[0] if not _total_clipped_by_ssp.empty else "none"
    _top_clipped_ha = float(_total_clipped_by_ssp.iloc[0]) if not _total_clipped_by_ssp.empty else 0.0

    final_demand_readout = mo.md(
        f"""
    ### Demand Readout

    - Zones with Chen trajectories: `{chen_urban_trajectory['zone'].nunique()}`
    - Decadal demand rows: `{len(chen_settlement_demand_table)}`
    - Positive delta rows: `{_positive_interval_rows}`
    - Zero delta rows: `{_zero_interval_rows}`
    - Negative delta rows: `{_negative_interval_rows}`
    - Largest single clipped negative delta: `{_max_negative_delta_ha:,.1f} ha`
    - Zones flagged by 2020 compatibility decisions: `{_manual_review_zone_count}`
    - Highest total clipped demand SSP: `{_top_demand_ssp}` (`{_top_demand_ha:,.1f} ha`)
    - Highest recorded negative-delta SSP: `{_top_clipped_ssp}` (`{_top_clipped_ha:,.1f} ha`)

    The reusable output from this notebook is `chen_settlement_demand_table`.
    It is explicit about baseline choice, raw Chen deltas, clipped diagnostic demand,
    negative-delta area, and compatibility flags. No source-class allocation happens
    here; that begins in `06_pseudo_transition_allocation.py`.
    """
    )

    final_demand_readout
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Limitations And Next Step

    This notebook does not write durable scenario artifacts and does not allocate
    future settlement demand across AFOLU source classes. Chen remains a binary
    urban/non-urban projection, so the demand table cannot identify which land-use
    class should become `settlements`.

    Notebook `06_pseudo_transition_allocation.py` should consume
    `chen_settlement_demand_table`, combine it with the historical source priors from
    notebook `04`, and run explicit mass-balance checks before emitting any
    diagnostic pseudo-`area_table` or pseudo-`transition_table` artifacts.
    """)
    return


if __name__ == "__main__":
    app.run()
