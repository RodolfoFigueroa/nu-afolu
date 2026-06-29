import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def import_dependencies():
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415
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
        pd,
        plt,
        sns,
        zone_partitions,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 02 Chen 2020 Compatibility

    This notebook compares Chen 2020 urban area against the GLC-derived 2020
    `settlements` baseline at the zone level. The purpose is to decide whether Chen
    is close enough to the historical settlement baseline to support later scenario
    construction, calibration, or filtering.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal, Inputs, And Assumptions

    The compatibility check uses zones that passed the historical table checks in
    the previous stage. This notebook recomputes the usable input list from local
    artifacts so it can be rerun independently, then compares:

    - GLC-derived `area_table.loc[2020, "settlements"]`
    - Chen 2020 urban area reduced over the same zone geometry for each SSP

    The comparison is diagnostic. Chen urban pixels and GLC `settlements` are not
    treated as semantically identical products.
    """)
    return


@app.cell
def configure_compatibility_check(Path, os, pd):
    PROJECT_ROOT = Path.cwd()
    OUT_PATH_KEY = "OUT_PATH"
    COMPARISON_YEAR = 2020
    CHEN_SCALE_M = 1000
    RATIO_DENOMINATOR_FLOOR_M2 = 1_000_000.0
    WORST_ZONE_COUNT = 15

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
                "value": str(OUT_PATH) if OUT_PATH else "not configured",
                "source": OUT_PATH_SOURCE,
            },
            {
                "setting": "COMPARISON_YEAR",
                "value": COMPARISON_YEAR,
                "source": "notebook default",
            },
            {
                "setting": "CHEN_SCALE_M",
                "value": CHEN_SCALE_M,
                "source": "Chen contract",
            },
            {
                "setting": "RATIO_DENOMINATOR_FLOOR_M2",
                "value": RATIO_DENOMINATOR_FLOOR_M2,
                "source": "one Chen-pixel guardrail",
            },
        ]
    )

    configuration_summary
    return (
        CHEN_SCALE_M,
        COMPARISON_YEAR,
        OUT_PATH,
        RATIO_DENOMINATOR_FLOOR_M2,
        WORST_ZONE_COUNT,
    )


@app.cell(hide_code=True)
def _(CHEN_COLLECTION_ID, COMPARISON_YEAR, mo):
    mo.md(f"""
    ## Why {COMPARISON_YEAR}

    The local Chen Earth Engine collection is `{CHEN_COLLECTION_ID}` and the project
    contract lists `{COMPARISON_YEAR}` as the first Chen year. The collection images
    do not expose year metadata in this environment, so this notebook uses the image
    order defined by `CHEN_YEARS` and treats the first image as `{COMPARISON_YEAR}`.
    """)
    return


@app.cell
def show_chen_contract(
    CHEN_COLLECTION_ID,
    CHEN_URBAN_VALUE,
    CHEN_YEARS,
    SSP_NAMES,
    pd,
):
    chen_contract_summary = pd.DataFrame(
        [
            {"field": "collection", "value": CHEN_COLLECTION_ID},
            {"field": "urban pixel value", "value": CHEN_URBAN_VALUE},
            {"field": "years", "value": ", ".join(str(_year) for _year in CHEN_YEARS)},
            {"field": "scenarios", "value": ", ".join(SSP_NAMES)},
        ]
    )

    chen_contract_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loaded Analysis Zones

    The previous notebook established that historical table artifacts are usable.
    Here, the input zone list is rebuilt from the required local artifacts and then
    filtered to zones with a valid 2020 settlement baseline.
    """)
    return


@app.cell
def discover_input_zones(OUT_PATH, Path, pd, zone_partitions):
    REQUIRED_ARTIFACT_SPECS = {
        "bbox_ee": {"relative_dir": Path("bbox") / "ee", "extension": ".json"},
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
        input_zone_inventory.loc[
            input_zone_inventory["complete_required_inputs"], "zone"
        ]
    )

    input_zone_summary = pd.DataFrame(
        [
            {"metric": "canonical partition zones", "value": len(partition_zone_names)},
            {
                "metric": "zones with required local inputs",
                "value": len(candidate_zone_names),
            },
            {
                "metric": "bbox/ee files discovered",
                "value": len(artifact_zone_sets["bbox_ee"]),
            },
            {
                "metric": "area_table files discovered",
                "value": len(artifact_zone_sets["area_table"]),
            },
        ]
    )

    input_zone_summary
    return (candidate_zone_names,)


@app.cell
def load_glc_settlement_baseline(
    COMPARISON_YEAR,
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
            if COMPARISON_YEAR not in _normalized_area.index:
                _baseline_error_rows.append(
                    {"zone": _zone_name, "error": f"missing {COMPARISON_YEAR} area row"}
                )
                continue
            _settlement_area_m2 = float(
                _normalized_area.loc[COMPARISON_YEAR, "settlements"]
            )
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
                    "near_zero_glc_baseline": _settlement_area_m2
                    < RATIO_DENOMINATOR_FLOOR_M2,
                    "area_year_min": int(_normalized_area.index.min()),
                    "area_year_max": int(_normalized_area.index.max()),
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
            {
                "metric": "zones with usable GLC 2020 settlements",
                "value": len(safe_historical_zone_names),
            },
            {
                "metric": "near-zero GLC baselines",
                "value": int(glc_settlement_baseline["near_zero_glc_baseline"].sum()),
            },
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
def show_glc_settlement_baseline(glc_settlement_baseline):
    glc_settlement_baseline
    return


@app.cell
def show_baseline_load_errors(baseline_load_errors):
    baseline_load_errors
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chen 2020 Urban Area

    Chen is reduced over the same zone geometries used by the historical artifacts.
    Each scenario band is converted to an urban mask using `CHEN_URBAN_VALUE`, then
    multiplied by `ee.Image.pixelArea()` and summed at the Chen analysis scale.

    This cell performs Earth Engine reductions, so it may take longer than the
    purely local checks above.
    """)
    return


@app.cell
def prepare_chen_2020_image(
    CHEN_COLLECTION_ID,
    COMPARISON_YEAR,
    SSP_NAMES,
    ee,
    pd,
):
    def short_error(exc: Exception) -> str:
        _message = str(exc).replace("\n", " ")
        return _message[:500] + ("..." if len(_message) > 500 else "")

    try:
        ee.Initialize()
        chen_collection = ee.ImageCollection(CHEN_COLLECTION_ID)
        chen_collection_size = int(chen_collection.size().getInfo())
        chen_2020_image = ee.Image(chen_collection.toList(chen_collection_size).get(0))
        chen_2020_band_names = tuple(chen_2020_image.bandNames().getInfo())
        chen_source_ready = True
        chen_source_error = ""
    except Exception as _exc:  # noqa: BLE001
        chen_collection = None
        chen_collection_size = None
        chen_2020_image = None
        chen_2020_band_names = ()
        chen_source_ready = False
        chen_source_error = short_error(_exc)

    chen_source_summary = pd.DataFrame(
        [
            {
                "collection_size": chen_collection_size,
                "selected_image_year": COMPARISON_YEAR,
                "band_names": ", ".join(chen_2020_band_names),
                "expected_ssp_bands_present": set(SSP_NAMES).issubset(
                    chen_2020_band_names
                ),
                "source_ready": chen_source_ready,
                "error": chen_source_error,
            }
        ]
    )

    chen_source_summary
    return chen_2020_image, chen_source_error, chen_source_ready, short_error


@app.cell
def define_chen_reduction_helpers(
    CHEN_SCALE_M,
    CHEN_URBAN_VALUE,
    OUT_PATH,
    Path,
    SSP_NAMES,
    chen_2020_image,
    ee,
    json,
):
    def load_ee_geometry(path: Path):
        with path.open(encoding="utf-8") as _file:
            return ee.Geometry(ee.deserializer.decode(json.load(_file)))

    def reduce_chen_urban_area_2020(zone_name: str) -> dict[str, object]:
        _geometry = load_ee_geometry(OUT_PATH / "bbox" / "ee" / f"{zone_name}.json")
        _result = (
            chen_2020_image.eq(ee.Number(CHEN_URBAN_VALUE))
            .multiply(ee.Image.pixelArea())
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=_geometry,
                scale=CHEN_SCALE_M,
                maxPixels=int(1e10),
            )
            .getInfo()
        )
        _row = {"zone": zone_name}
        for _ssp in SSP_NAMES:
            _row[_ssp] = float(_result.get(_ssp) or 0.0)
        return _row

    "Chen reduction helpers defined"
    return (reduce_chen_urban_area_2020,)


@app.cell
def reduce_chen_2020_by_zone(
    SSP_NAMES,
    chen_source_error,
    chen_source_ready,
    pd,
    reduce_chen_urban_area_2020,
    safe_historical_zone_names,
    short_error,
):
    _chen_rows = []
    _chen_error_rows = []

    if chen_source_ready:
        for _zone_name in safe_historical_zone_names:
            try:
                _chen_rows.append(reduce_chen_urban_area_2020(_zone_name))
            except Exception as _exc:  # noqa: BLE001
                _chen_error_rows.append(
                    {"zone": _zone_name, "error": short_error(_exc)}
                )
    else:
        _chen_error_rows.extend(
            {"zone": _zone_name, "error": chen_source_error}
            for _zone_name in safe_historical_zone_names
        )

    chen_urban_2020_wide = pd.DataFrame(_chen_rows)
    chen_reduction_errors = pd.DataFrame(_chen_error_rows, columns=["zone", "error"])

    if chen_urban_2020_wide.empty:
        chen_urban_2020_long = pd.DataFrame(
            columns=["zone", "ssp", "chen_urban_2020_m2", "chen_urban_2020_ha"]
        )
    else:
        chen_urban_2020_long = chen_urban_2020_wide.melt(
            id_vars="zone",
            value_vars=list(SSP_NAMES),
            var_name="ssp",
            value_name="chen_urban_2020_m2",
        ).assign(chen_urban_2020_ha=lambda _df: _df["chen_urban_2020_m2"] / 10_000.0)

    chen_reduction_summary = pd.DataFrame(
        [
            {"metric": "zones requested", "value": len(safe_historical_zone_names)},
            {
                "metric": "zones reduced successfully",
                "value": chen_urban_2020_wide["zone"].nunique()
                if "zone" in chen_urban_2020_wide
                else 0,
            },
            {"metric": "zone reduction errors", "value": len(chen_reduction_errors)},
            {"metric": "scenario rows", "value": len(chen_urban_2020_long)},
        ]
    )

    chen_reduction_summary
    return chen_reduction_errors, chen_urban_2020_long, chen_urban_2020_wide


@app.cell
def show_chen_urban_2020(chen_urban_2020_long):
    chen_urban_2020_long
    return


@app.cell
def show_chen_reduction_errors(chen_reduction_errors):
    chen_reduction_errors
    return


@app.cell(hide_code=True)
def _(RATIO_DENOMINATOR_FLOOR_M2, mo):
    mo.md(f"""
    ## Compatibility Metrics

    The table below compares Chen 2020 urban area with GLC 2020 `settlements` for
    every zone and SSP. Ratio metrics are only computed when the GLC settlement
    baseline is at least `{RATIO_DENOMINATOR_FLOOR_M2:,.0f} m²`, roughly one Chen
    pixel. Smaller denominators are flagged because ratios can explode for tiny
    baseline areas.
    """)
    return


@app.cell
def compute_compatibility_metrics(
    RATIO_DENOMINATOR_FLOOR_M2,
    chen_urban_2020_long,
    glc_settlement_baseline,
    np,
):
    compatibility_metrics = chen_urban_2020_long.merge(
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

    compatibility_metrics = compatibility_metrics.assign(
        signed_difference_m2=lambda _df: (
            _df["chen_urban_2020_m2"] - _df["glc_settlements_2020_m2"]
        ),
        absolute_difference_m2=lambda _df: _df["signed_difference_m2"].abs(),
        signed_difference_ha=lambda _df: _df["signed_difference_m2"] / 10_000.0,
        absolute_difference_ha=lambda _df: _df["absolute_difference_m2"] / 10_000.0,
        ratio_denominator_is_stable=lambda _df: (
            _df["glc_settlements_2020_m2"] >= RATIO_DENOMINATOR_FLOOR_M2
        ),
    )

    _stable_denominator = compatibility_metrics["ratio_denominator_is_stable"]
    compatibility_metrics = compatibility_metrics.assign(
        chen_to_glc_ratio=np.where(
            _stable_denominator,
            compatibility_metrics["chen_urban_2020_m2"]
            / compatibility_metrics["glc_settlements_2020_m2"],
            np.nan,
        )
    )
    compatibility_metrics = compatibility_metrics.assign(
        ratio_error=lambda _df: _df["chen_to_glc_ratio"] - 1.0,
        absolute_ratio_error=lambda _df: _df["ratio_error"].abs(),
    )

    compatibility_metrics
    return (compatibility_metrics,)


@app.cell
def summarize_compatibility_by_ssp(compatibility_metrics):
    stable_ratio_metrics = compatibility_metrics.loc[
        compatibility_metrics["ratio_denominator_is_stable"]
    ]

    compatibility_summary_by_ssp = (
        compatibility_metrics.groupby("ssp", dropna=False)
        .agg(
            zones=("zone", "nunique"),
            ratio_unstable_rows=(
                "ratio_denominator_is_stable",
                lambda _series: int((~_series).sum()),
            ),
            median_abs_difference_ha=("absolute_difference_ha", "median"),
            p90_abs_difference_ha=(
                "absolute_difference_ha",
                lambda _series: float(_series.quantile(0.9)),
            ),
            max_abs_difference_ha=("absolute_difference_ha", "max"),
            median_signed_difference_ha=("signed_difference_ha", "median"),
        )
        .reset_index()
    )

    _ratio_summary = (
        stable_ratio_metrics.groupby("ssp", dropna=False)
        .agg(
            stable_ratio_rows=("zone", "count"),
            median_ratio=("chen_to_glc_ratio", "median"),
            median_abs_ratio_error=("absolute_ratio_error", "median"),
            p90_abs_ratio_error=(
                "absolute_ratio_error",
                lambda _series: float(_series.quantile(0.9)),
            ),
            max_abs_ratio_error=("absolute_ratio_error", "max"),
        )
        .reset_index()
    )

    compatibility_summary_by_ssp = compatibility_summary_by_ssp.merge(
        _ratio_summary,
        on="ssp",
        how="left",
    )

    compatibility_summary_by_ssp
    return compatibility_summary_by_ssp, stable_ratio_metrics


@app.cell
def summarize_compatibility_by_zone(compatibility_metrics):
    zone_compatibility_summary = (
        compatibility_metrics.groupby("zone", dropna=False)
        .agg(
            glc_settlements_2020_ha=("glc_settlements_2020_ha", "first"),
            median_chen_urban_2020_ha=("chen_urban_2020_ha", "median"),
            max_abs_difference_ha=("absolute_difference_ha", "max"),
            max_abs_ratio_error=("absolute_ratio_error", "max"),
            ratio_unstable_rows=(
                "ratio_denominator_is_stable",
                lambda _series: int((~_series).sum()),
            ),
        )
        .reset_index()
    )

    zone_compatibility_summary = zone_compatibility_summary.assign(
        needs_manual_review=lambda _df: (
            (_df["ratio_unstable_rows"] > 0)
            | (_df["max_abs_ratio_error"] > 1.0)
            | (_df["max_abs_difference_ha"] > 5_000.0)
        )
    )

    zone_compatibility_summary
    return (zone_compatibility_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Worst-Zone Diagnostics

    The first table ranks absolute mismatch in hectares. The second ranks ratio
    error only for zones whose GLC settlement denominator is large enough for the
    ratio to be meaningful under the guardrail above.
    """)
    return


@app.cell
def rank_worst_zone_diagnostics(WORST_ZONE_COUNT, compatibility_metrics):
    _diagnostic_columns = [
        "zone",
        "ssp",
        "glc_settlements_2020_ha",
        "chen_urban_2020_ha",
        "signed_difference_ha",
        "absolute_difference_ha",
        "chen_to_glc_ratio",
        "absolute_ratio_error",
    ]

    worst_absolute_mismatch = (
        compatibility_metrics.sort_values("absolute_difference_m2", ascending=False)
        .loc[:, _diagnostic_columns]
        .head(WORST_ZONE_COUNT)
        .reset_index(drop=True)
    )

    worst_ratio_mismatch = (
        compatibility_metrics.loc[compatibility_metrics["ratio_denominator_is_stable"]]
        .sort_values("absolute_ratio_error", ascending=False)
        .loc[:, _diagnostic_columns]
        .head(WORST_ZONE_COUNT)
        .reset_index(drop=True)
    )

    worst_absolute_mismatch
    return (worst_ratio_mismatch,)


@app.cell
def show_worst_ratio_mismatch(worst_ratio_mismatch):
    worst_ratio_mismatch
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visual Comparison

    The scatterplot compares GLC settlements with Chen urban area. Points near the
    diagonal are more compatible in absolute terms. The distribution plots show how
    absolute mismatch and ratio error vary across SSPs.
    """)
    return


@app.cell
def plot_glc_chen_scatter(compatibility_metrics, plt, sns):
    sns.set_theme(style="whitegrid")

    _scatter_fig, _scatter_ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=compatibility_metrics,
        x="glc_settlements_2020_ha",
        y="chen_urban_2020_ha",
        hue="ssp",
        alpha=0.75,
        s=42,
        ax=_scatter_ax,
    )
    _axis_max = float(
        max(
            compatibility_metrics["glc_settlements_2020_ha"].max(),
            compatibility_metrics["chen_urban_2020_ha"].max(),
        )
    )
    _scatter_ax.plot(
        [0, _axis_max],
        [0, _axis_max],
        color="black",
        linewidth=1,
        linestyle="--",
        label="1:1",
    )
    _scatter_ax.set_xlabel("GLC 2020 settlements (ha)")
    _scatter_ax.set_ylabel("Chen 2020 urban (ha)")
    _scatter_ax.set_title("Chen urban area versus GLC settlements by zone")
    _scatter_ax.legend(title="SSP", bbox_to_anchor=(1.02, 1), loc="upper left")
    _scatter_fig.tight_layout()

    glc_chen_scatter_plot = _scatter_fig

    glc_chen_scatter_plot
    return


@app.cell
def plot_mismatch_distributions(
    compatibility_metrics,
    plt,
    sns,
    stable_ratio_metrics,
):
    _distribution_fig, _distribution_axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.boxplot(
        data=compatibility_metrics,
        x="ssp",
        y="absolute_difference_ha",
        ax=_distribution_axes[0],
    )
    _distribution_axes[0].set_xlabel("SSP")
    _distribution_axes[0].set_ylabel("Absolute difference (ha)")
    _distribution_axes[0].set_title("Absolute mismatch")

    sns.boxplot(
        data=stable_ratio_metrics,
        x="ssp",
        y="ratio_error",
        ax=_distribution_axes[1],
    )
    _distribution_axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
    _distribution_axes[1].set_xlabel("SSP")
    _distribution_axes[1].set_ylabel("Chen / GLC - 1")
    _distribution_axes[1].set_title("Ratio error, stable denominators only")

    _distribution_fig.tight_layout()

    mismatch_distribution_plots = _distribution_fig

    mismatch_distribution_plots
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation And Decision Points

    The conclusion distinguishes median behavior from high-risk outliers. This is a
    compatibility screen, not a validation claim: systematic differences can reflect
    product semantics, resolution mismatch, boundary effects, or true land-cover
    disagreement.
    """)
    return


@app.cell
def build_compatibility_conclusion(
    chen_reduction_errors,
    chen_urban_2020_wide,
    compatibility_metrics,
    compatibility_summary_by_ssp,
    mo,
    zone_compatibility_summary,
):
    _successful_zone_count = (
        int(chen_urban_2020_wide["zone"].nunique())
        if "zone" in chen_urban_2020_wide
        else 0
    )
    _ratio_unstable_row_count = int(
        (~compatibility_metrics["ratio_denominator_is_stable"]).sum()
    )
    _review_zone_count = int(zone_compatibility_summary["needs_manual_review"].sum())
    _max_median_ratio_error = float(
        compatibility_summary_by_ssp["median_abs_ratio_error"].max()
    )
    _max_p90_ratio_error = float(
        compatibility_summary_by_ssp["p90_abs_ratio_error"].max()
    )
    _max_abs_difference_ha = float(
        compatibility_metrics["absolute_difference_ha"].max()
    )

    if chen_reduction_errors.empty and _successful_zone_count:
        if _review_zone_count == 0 and _max_median_ratio_error <= 0.5:
            _recommendation = "Proceed without an immediate zone filter, while preserving calibration diagnostics."
        elif _max_median_ratio_error <= 0.5:
            _recommendation = "Proceed with calibration and manual review of high-mismatch zones before allocation."
        else:
            _recommendation = "Do not use uncalibrated Chen areas globally; proceed only with calibration or a reviewed zone subset."
    else:
        _recommendation = (
            "Do not proceed until Chen reductions succeed for the required zones."
        )

    compatibility_conclusion = mo.md(
        f"""
    ### Compatibility Readout

    - Zones compared: `{_successful_zone_count}`
    - Scenario comparison rows: `{len(compatibility_metrics)}`
    - Ratio-unstable rows: `{_ratio_unstable_row_count}`
    - Zones flagged for manual review: `{_review_zone_count}`
    - Largest absolute mismatch: `{_max_abs_difference_ha:,.1f} ha`
    - Largest SSP median absolute ratio error: `{_max_median_ratio_error:.3f}`
    - Largest SSP p90 absolute ratio error: `{_max_p90_ratio_error:.3f}`

    Recommendation: **{_recommendation}**

    This result does not establish that Chen urban pixels are equivalent to GLC
    `settlements`. It only quantifies whether the 2020 baseline mismatch is small
    enough to carry forward explicitly into the scenario-construction notebooks.
    """
    )

    compatibility_conclusion
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Limitations And Next Step

    This notebook compares zone-level area totals only. It does not diagnose
    within-zone spatial alignment, boundary sensitivity, or Chen-grid aggregation
    effects. Those questions belong in `03_spatial_resolution_diagnostics.py`.

    No durable artifact is written here. Later notebooks should carry forward the
    compatibility metrics, the unstable-ratio flag, and the manual-review zones if
    they choose to proceed.
    """)
    return


if __name__ == "__main__":
    app.run()
