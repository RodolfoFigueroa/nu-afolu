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
        LABEL_LIST,
    )

    return (
        CHEN_COLLECTION_ID,
        CHEN_URBAN_VALUE,
        LABEL_LIST,
        Path,
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
    # 03 Spatial Resolution Diagnostics

    This notebook diagnoses whether the zone-level Chen/GLC compatibility results
    hide important effects from comparing 30 m historical GLC-derived artifacts with
    approximately 1 km Chen urban projections.

    The analysis is diagnostic. It does not validate Chen pixels as equivalent to
    GLC `settlements`, and it does not create pseudo-transition artifacts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal, Inputs, And Assumptions

    The previous notebook compared zone-level area totals. This stage looks at the
    same question on the Chen grid:

    - aggregate the observed 2020 GLC settlement mask to Chen-grid settlement
      fractions;
    - compare those fractions with Chen 2020 urban pixels;
    - flag zones where a small number of Chen pixels, boundary effects, or spatial
      disagreement could dominate the zone-level result.

    The first implementation uses `SSP2` as the representative Chen 2020 mask for
    grid-level diagnostics. The selection of diagnostic zones still considers all
    SSPs when ranking zone-level mismatch.
    """)
    return


@app.cell
def configure_spatial_diagnostics(LABEL_LIST, Path, os, pd):
    PROJECT_ROOT = Path.cwd()
    OUT_PATH_KEY = "OUT_PATH"
    COMPARISON_YEAR = 2020
    DIAGNOSTIC_SSP = "SSP2"
    CHEN_SCALE_M = 1000
    CHEN_PIXEL_AREA_M2 = CHEN_SCALE_M**2
    RATIO_DENOMINATOR_FLOOR_M2 = CHEN_PIXEL_AREA_M2
    LOW_SETTLEMENT_FRACTION = 0.10
    HIGH_SETTLEMENT_FRACTION = 0.50
    MAX_DIAGNOSTIC_ZONES = 12


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
    SETTLEMENT_RASTER_ID = LABEL_LIST.index("settlements") + 1

    configuration_summary = pd.DataFrame(
        [
            {"setting": OUT_PATH_KEY, "value": str(OUT_PATH) if OUT_PATH else "not configured", "source": OUT_PATH_SOURCE},
            {"setting": "COMPARISON_YEAR", "value": COMPARISON_YEAR, "source": "notebook default"},
            {"setting": "DIAGNOSTIC_SSP", "value": DIAGNOSTIC_SSP, "source": "representative Chen mask"},
            {"setting": "CHEN_SCALE_M", "value": CHEN_SCALE_M, "source": "Chen contract"},
            {"setting": "RATIO_DENOMINATOR_FLOOR_M2", "value": RATIO_DENOMINATOR_FLOOR_M2, "source": "one Chen pixel"},
        ]
    )

    configuration_summary
    return (
        CHEN_PIXEL_AREA_M2,
        CHEN_SCALE_M,
        COMPARISON_YEAR,
        DIAGNOSTIC_SSP,
        HIGH_SETTLEMENT_FRACTION,
        LOW_SETTLEMENT_FRACTION,
        MAX_DIAGNOSTIC_ZONES,
        OUT_PATH,
        RATIO_DENOMINATOR_FLOOR_M2,
        SETTLEMENT_RASTER_ID,
    )


@app.cell(hide_code=True)
def _(MAX_DIAGNOSTIC_ZONES, mo):
    mo.md(f"""
    ## Diagnostic Zone Selection

    The notebook recomputes the zone-level compatibility screen so it can select a
    small but informative set of zones. The diagnostic set combines worst absolute
    mismatch, worst stable ratio mismatch, ratio-unstable small-baseline zones, and
    typical zones near the median mismatch. The target size is
    `{MAX_DIAGNOSTIC_ZONES}` zones.
    """)
    return


@app.cell
def discover_candidate_zones(OUT_PATH, Path, pd, zone_partitions):
    REQUIRED_ARTIFACT_SPECS = {
        "bbox_ee": {"relative_dir": Path("bbox") / "ee", "extension": ".json"},
        "area_raster": {"relative_dir": Path("area_raster"), "extension": ".json"},
        "area_table": {"relative_dir": Path("area_table"), "extension": ".parquet"},
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

    zone_rows = []
    for _zone_name in partition_zone_names:
        _row = {"zone": _zone_name}
        for _artifact_name in REQUIRED_ARTIFACT_SPECS:
            _row[_artifact_name] = _zone_name in artifact_zone_sets[_artifact_name]
        _row["complete_required_inputs"] = all(
            _row[_artifact_name] for _artifact_name in REQUIRED_ARTIFACT_SPECS
        )
        zone_rows.append(_row)

    input_zone_inventory = pd.DataFrame(zone_rows)
    candidate_zone_names = tuple(
        input_zone_inventory.loc[input_zone_inventory["complete_required_inputs"], "zone"]
    )

    input_zone_summary = pd.DataFrame(
        [
            {"metric": "canonical partition zones", "value": len(partition_zone_names)},
            {"metric": "zones with required spatial inputs", "value": len(candidate_zone_names)},
            {"metric": "bbox/ee files discovered", "value": len(artifact_zone_sets["bbox_ee"])},
            {"metric": "area_raster files discovered", "value": len(artifact_zone_sets["area_raster"])},
            {"metric": "area_table files discovered", "value": len(artifact_zone_sets["area_table"])},
        ]
    )

    input_zone_summary
    return (candidate_zone_names,)


@app.cell
def load_glc_baseline(
    COMPARISON_YEAR,
    LABEL_LIST,
    OUT_PATH,
    RATIO_DENOMINATOR_FLOOR_M2,
    candidate_zone_names,
    pd,
):
    baseline_rows = []
    baseline_error_rows = []

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
                baseline_error_rows.append({"zone": _zone_name, "error": f"missing {COMPARISON_YEAR} area row"})
                continue
            _zone_area_m2 = float(_normalized_area.loc[COMPARISON_YEAR].sum())
            _settlement_area_m2 = float(_normalized_area.loc[COMPARISON_YEAR, "settlements"])
            baseline_rows.append(
                {
                    "zone": _zone_name,
                    "zone_area_table_2020_m2": _zone_area_m2,
                    "glc_settlements_2020_m2": _settlement_area_m2,
                    "glc_settlements_2020_ha": _settlement_area_m2 / 10_000.0,
                    "ratio_denominator_is_stable": _settlement_area_m2 >= RATIO_DENOMINATOR_FLOOR_M2,
                }
            )
        except Exception as _exc:  # noqa: BLE001
            baseline_error_rows.append({"zone": _zone_name, "error": str(_exc)})

    glc_baseline = pd.DataFrame(baseline_rows)
    baseline_errors = pd.DataFrame(baseline_error_rows, columns=["zone", "error"])
    baseline_zone_names = tuple(glc_baseline["zone"])

    baseline_summary = pd.DataFrame(
        [
            {"metric": "candidate zones", "value": len(candidate_zone_names)},
            {"metric": "zones with usable 2020 baseline", "value": len(baseline_zone_names)},
            {"metric": "small-baseline zones below one Chen pixel", "value": int((~glc_baseline["ratio_denominator_is_stable"]).sum())},
            {"metric": "baseline errors", "value": len(baseline_errors)},
        ]
    )

    baseline_summary
    return baseline_zone_names, glc_baseline


@app.cell
def prepare_chen_collection(
    CHEN_COLLECTION_ID,
    COMPARISON_YEAR,
    DIAGNOSTIC_SSP,
    ee,
    pd,
):
    def short_error(exc: Exception) -> str:
        message = str(exc).replace("\n", " ")
        return message[:500] + ("..." if len(message) > 500 else "")


    try:
        ee.Initialize()
        chen_collection = ee.ImageCollection(CHEN_COLLECTION_ID)
        chen_collection_size = int(chen_collection.size().getInfo())
        chen_2020_image = ee.Image(chen_collection.toList(chen_collection_size).get(0))
        chen_2020_band_names = tuple(chen_2020_image.bandNames().getInfo())
        chen_2020_diagnostic_band = chen_2020_image.select(DIAGNOSTIC_SSP)
        chen_source_ready = True
        chen_source_error = ""
    except Exception as _exc:  # noqa: BLE001
        chen_collection = None
        chen_collection_size = None
        chen_2020_image = None
        chen_2020_band_names = ()
        chen_2020_diagnostic_band = None
        chen_source_ready = False
        chen_source_error = short_error(_exc)

    chen_source_summary = pd.DataFrame(
        [
            {
                "collection_size": chen_collection_size,
                "selected_image_year": COMPARISON_YEAR,
                "diagnostic_ssp": DIAGNOSTIC_SSP,
                "band_names": ", ".join(chen_2020_band_names),
                "source_ready": chen_source_ready,
                "error": chen_source_error,
            }
        ]
    )

    chen_source_summary
    return (
        chen_2020_band_names,
        chen_2020_diagnostic_band,
        chen_2020_image,
        chen_source_error,
        chen_source_ready,
        short_error,
    )


@app.cell
def define_zone_level_chen_helpers(
    CHEN_SCALE_M,
    CHEN_URBAN_VALUE,
    OUT_PATH,
    Path,
    chen_2020_band_names,
    chen_2020_image,
    ee,
    json,
):
    def load_ee_geometry(path: Path):
        with path.open(encoding="utf-8") as file:
            return ee.Geometry(ee.deserializer.decode(json.load(file)))


    def reduce_chen_urban_area_2020(zone_name: str) -> dict[str, object]:
        geometry = load_ee_geometry(OUT_PATH / "bbox" / "ee" / f"{zone_name}.json")
        result = (
            chen_2020_image.eq(ee.Number(CHEN_URBAN_VALUE))
            .multiply(ee.Image.pixelArea())
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=geometry,
                scale=CHEN_SCALE_M,
                maxPixels=int(1e10),
            )
            .getInfo()
        )
        row = {"zone": zone_name}
        for ssp_name in chen_2020_band_names:
            row[ssp_name] = float(result.get(ssp_name) or 0.0)
        return row


    "zone-level Chen helpers defined"
    return load_ee_geometry, reduce_chen_urban_area_2020


@app.cell
def reduce_zone_level_chen(
    baseline_zone_names,
    chen_2020_band_names,
    chen_source_error,
    chen_source_ready,
    pd,
    reduce_chen_urban_area_2020,
    short_error,
):
    chen_area_rows = []
    chen_area_error_rows = []

    if chen_source_ready:
        for _zone_name in baseline_zone_names:
            try:
                chen_area_rows.append(reduce_chen_urban_area_2020(_zone_name))
            except Exception as _exc:  # noqa: BLE001
                chen_area_error_rows.append({"zone": _zone_name, "error": short_error(_exc)})
    else:
        chen_area_error_rows.extend(
            {"zone": _zone_name, "error": chen_source_error}
            for _zone_name in baseline_zone_names
        )

    chen_area_wide = pd.DataFrame(chen_area_rows)
    chen_area_errors = pd.DataFrame(chen_area_error_rows, columns=["zone", "error"])

    if chen_area_wide.empty:
        chen_area_long = pd.DataFrame(columns=["zone", "ssp", "chen_urban_2020_m2"])
    else:
        chen_area_long = chen_area_wide.melt(
            id_vars="zone",
            value_vars=list(chen_2020_band_names),
            var_name="ssp",
            value_name="chen_urban_2020_m2",
        )

    chen_area_summary = pd.DataFrame(
        [
            {"metric": "zones requested", "value": len(baseline_zone_names)},
            {"metric": "zones reduced successfully", "value": chen_area_wide["zone"].nunique() if "zone" in chen_area_wide else 0},
            {"metric": "zone reduction errors", "value": len(chen_area_errors)},
            {"metric": "scenario rows", "value": len(chen_area_long)},
        ]
    )

    chen_area_summary
    return (chen_area_long,)


@app.cell
def compute_zone_level_compatibility(chen_area_long, glc_baseline, np, pd):
    compatibility_metrics = chen_area_long.merge(
        glc_baseline[
            [
                "zone",
                "zone_area_table_2020_m2",
                "glc_settlements_2020_m2",
                "glc_settlements_2020_ha",
                "ratio_denominator_is_stable",
            ]
        ],
        on="zone",
        how="inner",
    )

    compatibility_metrics = compatibility_metrics.assign(
        chen_urban_2020_ha=lambda _df: _df["chen_urban_2020_m2"] / 10_000.0,
        signed_difference_m2=lambda _df: _df["chen_urban_2020_m2"] - _df["glc_settlements_2020_m2"],
    )
    compatibility_metrics = compatibility_metrics.assign(
        absolute_difference_m2=lambda _df: _df["signed_difference_m2"].abs(),
        signed_difference_ha=lambda _df: _df["signed_difference_m2"] / 10_000.0,
        absolute_difference_ha=lambda _df: _df["absolute_difference_m2"] / 10_000.0,
    )
    compatibility_metrics = compatibility_metrics.assign(
        chen_to_glc_ratio=np.where(
            compatibility_metrics["ratio_denominator_is_stable"],
            compatibility_metrics["chen_urban_2020_m2"] / compatibility_metrics["glc_settlements_2020_m2"],
            np.nan,
        )
    )
    compatibility_metrics = compatibility_metrics.assign(
        ratio_error=lambda _df: _df["chen_to_glc_ratio"] - 1.0,
        absolute_ratio_error=lambda _df: _df["ratio_error"].abs(),
    )

    zone_compatibility_summary = (
        compatibility_metrics.groupby("zone", dropna=False)
        .agg(
            glc_settlements_2020_ha=("glc_settlements_2020_ha", "first"),
            median_chen_urban_2020_ha=("chen_urban_2020_ha", "median"),
            max_abs_difference_ha=("absolute_difference_ha", "max"),
            max_abs_ratio_error=("absolute_ratio_error", "max"),
            ratio_denominator_is_stable=("ratio_denominator_is_stable", "first"),
        )
        .reset_index()
    )

    compatibility_selection_summary = pd.DataFrame(
        [
            {"metric": "compatibility rows", "value": len(compatibility_metrics)},
            {"metric": "zones compared", "value": compatibility_metrics["zone"].nunique()},
            {"metric": "ratio-unstable zones", "value": int((~zone_compatibility_summary["ratio_denominator_is_stable"]).sum())},
            {"metric": "maximum absolute mismatch ha", "value": float(zone_compatibility_summary["max_abs_difference_ha"].max())},
        ]
    )

    compatibility_selection_summary
    return (zone_compatibility_summary,)


@app.cell
def select_diagnostic_zones(
    MAX_DIAGNOSTIC_ZONES,
    pd,
    zone_compatibility_summary,
):
    selected_zone_rows = []
    _selected_zone_set: set[str] = set()


    def _add_zone(zone: str, reason: str) -> None:
        if zone in _selected_zone_set or len(_selected_zone_set) >= MAX_DIAGNOSTIC_ZONES:
            return
        _selected_zone_set.add(zone)
        selected_zone_rows.append({"zone": zone, "selection_reason": reason})


    _worst_absolute_zones = (
        zone_compatibility_summary.sort_values("max_abs_difference_ha", ascending=False)
        .head(4)["zone"]
        .tolist()
    )
    for _zone in _worst_absolute_zones:
        _add_zone(_zone, "worst absolute mismatch")

    _worst_ratio_zones = (
        zone_compatibility_summary.loc[zone_compatibility_summary["ratio_denominator_is_stable"]]
        .sort_values("max_abs_ratio_error", ascending=False)
        .head(4)["zone"]
        .tolist()
    )
    for _zone in _worst_ratio_zones:
        _add_zone(_zone, "worst stable ratio mismatch")

    _small_baseline_zones = (
        zone_compatibility_summary.loc[~zone_compatibility_summary["ratio_denominator_is_stable"], "zone"]
        .tolist()
    )
    for _zone in _small_baseline_zones:
        _add_zone(_zone, "small GLC baseline below one Chen pixel")

    _median_abs_difference = zone_compatibility_summary["max_abs_difference_ha"].median()
    _typical_zones = (
        zone_compatibility_summary.assign(
            distance_from_median_abs=lambda _df: (
                _df["max_abs_difference_ha"] - _median_abs_difference
            ).abs()
        )
        .sort_values("distance_from_median_abs")
        .head(MAX_DIAGNOSTIC_ZONES)["zone"]
        .tolist()
    )
    for _zone in _typical_zones:
        _add_zone(_zone, "near median absolute mismatch")

    diagnostic_zone_table = (
        pd.DataFrame(selected_zone_rows)
        .merge(zone_compatibility_summary, on="zone", how="left")
        .reset_index(drop=True)
    )
    diagnostic_zone_names = tuple(diagnostic_zone_table["zone"])

    diagnostic_zone_table
    return diagnostic_zone_names, diagnostic_zone_table


@app.cell(hide_code=True)
def _(diagnostic_zone_names, mo):
    mo.md(f"""
    ## Selected Diagnostic Zones

    `{len(diagnostic_zone_names)}` zones were selected for grid-level diagnostics.
    The selection is intentionally small because the next step performs heavier
    Earth Engine reductions over resampled settlement masks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Settlement Fractions On The Chen Grid

    The observed 2020 settlement mask is aggregated to the Chen grid using a mean
    reducer. A Chen-scale cell with value `0.25` therefore means that roughly one
    quarter of the contributing 30 m observed pixels were classified as
    `settlements`.

    The diagnostics below compare those fractions with the Chen 2020 urban mask for
    the representative `SSP2` band.
    """)
    return


@app.cell
def define_spatial_reduction_helpers(
    COMPARISON_YEAR,
    Path,
    SETTLEMENT_RASTER_ID,
    ee,
    json,
    np,
):
    def load_ee_image(path: Path):
        with path.open(encoding="utf-8") as file:
            return ee.Image(ee.deserializer.decode(json.load(file)))


    def build_observed_settlement_fraction(area_raster, target_projection):
        observed_settlement_mask = (
            area_raster.select(str(COMPARISON_YEAR))
            .eq(ee.Number(SETTLEMENT_RASTER_ID))
            .unmask(0)
            .rename("observed_settlement_mask")
        )
        return (
            observed_settlement_mask.reduceResolution(
                reducer=ee.Reducer.mean(),
                maxPixels=4096,
            )
            .reproject(target_projection)
            .rename("observed_settlement_fraction")
        )


    def ratio_or_nan(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else np.nan


    "spatial reduction helpers defined"
    return build_observed_settlement_fraction, load_ee_image


@app.cell
def define_zone_spatial_diagnostic(
    CHEN_SCALE_M,
    CHEN_URBAN_VALUE,
    HIGH_SETTLEMENT_FRACTION,
    LOW_SETTLEMENT_FRACTION,
    OUT_PATH,
    build_observed_settlement_fraction,
    chen_2020_diagnostic_band,
    ee,
    load_ee_geometry,
    load_ee_image,
    np,
):
    def reduce_zone_spatial_diagnostic(zone: str) -> dict[str, object]:
        geometry = load_ee_geometry(OUT_PATH / "bbox" / "ee" / f"{zone}.json")
        area_raster = load_ee_image(OUT_PATH / "area_raster" / f"{zone}.json").clip(geometry)
        target_projection = chen_2020_diagnostic_band.projection()
        observed_fraction = build_observed_settlement_fraction(area_raster, target_projection).clip(geometry)

        chen_band = chen_2020_diagnostic_band.clip(geometry)
        chen_urban_mask = chen_band.eq(ee.Number(CHEN_URBAN_VALUE))
        chen_nonurban_mask = chen_band.neq(ee.Number(CHEN_URBAN_VALUE))
        pixel_area = ee.Image.pixelArea().reproject(target_projection)

        area_image = ee.Image.cat(
            [
                pixel_area.rename("chen_grid_area_m2"),
                pixel_area.updateMask(chen_urban_mask).rename("chen_urban_area_m2"),
                observed_fraction.multiply(pixel_area).rename("observed_fraction_area_m2"),
                pixel_area.updateMask(
                    chen_urban_mask.And(observed_fraction.lt(LOW_SETTLEMENT_FRACTION))
                ).rename("chen_urban_low_observed_area_m2"),
                pixel_area.updateMask(
                    chen_nonurban_mask.And(observed_fraction.gt(HIGH_SETTLEMENT_FRACTION))
                ).rename("high_observed_nonurban_area_m2"),
                pixel_area.updateMask(
                    observed_fraction.gt(HIGH_SETTLEMENT_FRACTION)
                ).rename("high_observed_fraction_area_m2"),
            ]
        )

        mean_image = ee.Image.cat(
            [
                observed_fraction.updateMask(chen_urban_mask).rename(
                    "mean_observed_fraction_on_chen_urban"
                ),
                observed_fraction.updateMask(chen_nonurban_mask).rename(
                    "mean_observed_fraction_on_chen_nonurban"
                ),
            ]
        )

        area_result = area_image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=CHEN_SCALE_M,
            maxPixels=int(1e10),
        ).getInfo()
        mean_result = mean_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=CHEN_SCALE_M,
            maxPixels=int(1e10),
        ).getInfo()

        row = {"zone": zone}
        for key, value in area_result.items():
            row[key] = float(value or 0.0)
        for key, value in mean_result.items():
            row[key] = float(value) if value is not None else np.nan
        return row


    "zone spatial diagnostic reducer defined"
    return (reduce_zone_spatial_diagnostic,)


@app.cell
def reduce_selected_spatial_diagnostics(
    diagnostic_zone_names,
    pd,
    reduce_zone_spatial_diagnostic,
    short_error,
):
    spatial_diagnostic_rows = []
    spatial_diagnostic_error_rows = []

    for _zone_name in diagnostic_zone_names:
        try:
            spatial_diagnostic_rows.append(reduce_zone_spatial_diagnostic(_zone_name))
        except Exception as _exc:  # noqa: BLE001
            spatial_diagnostic_error_rows.append({"zone": _zone_name, "error": short_error(_exc)})

    spatial_diagnostics_raw = pd.DataFrame(spatial_diagnostic_rows)
    spatial_diagnostic_errors = pd.DataFrame(
        spatial_diagnostic_error_rows,
        columns=["zone", "error"],
    )

    spatial_reduction_summary = pd.DataFrame(
        [
            {"metric": "diagnostic zones requested", "value": len(diagnostic_zone_names)},
            {"metric": "diagnostic zones reduced", "value": spatial_diagnostics_raw["zone"].nunique() if "zone" in spatial_diagnostics_raw else 0},
            {"metric": "spatial reduction errors", "value": len(spatial_diagnostic_errors)},
        ]
    )

    spatial_reduction_summary
    return spatial_diagnostic_errors, spatial_diagnostics_raw


@app.cell
def show_spatial_reduction_errors(spatial_diagnostic_errors):
    spatial_diagnostic_errors
    return


@app.cell
def build_spatial_diagnostic_table(
    CHEN_PIXEL_AREA_M2,
    diagnostic_zone_table,
    glc_baseline,
    np,
    spatial_diagnostics_raw,
):
    spatial_diagnostics = spatial_diagnostics_raw.merge(
        diagnostic_zone_table[
            [
                "zone",
                "selection_reason",
                "glc_settlements_2020_ha",
                "median_chen_urban_2020_ha",
                "max_abs_difference_ha",
                "max_abs_ratio_error",
                "ratio_denominator_is_stable",
            ]
        ],
        on="zone",
        how="left",
    ).merge(
        glc_baseline[["zone", "zone_area_table_2020_m2", "glc_settlements_2020_m2"]],
        on="zone",
        how="left",
    )

    spatial_diagnostics = spatial_diagnostics.assign(
        chen_grid_equivalent_pixels=lambda _df: _df["chen_grid_area_m2"] / CHEN_PIXEL_AREA_M2,
        chen_urban_equivalent_pixels=lambda _df: _df["chen_urban_area_m2"] / CHEN_PIXEL_AREA_M2,
        chen_urban_low_observed_share=lambda _df: _df["chen_urban_low_observed_area_m2"]
        / _df["chen_urban_area_m2"].replace(0, np.nan),
        high_observed_nonurban_share=lambda _df: _df["high_observed_nonurban_area_m2"]
        / _df["high_observed_fraction_area_m2"].replace(0, np.nan),
        observed_fraction_area_ha=lambda _df: _df["observed_fraction_area_m2"] / 10_000.0,
        chen_grid_to_area_table_ratio=lambda _df: _df["chen_grid_area_m2"]
        / _df["zone_area_table_2020_m2"].replace(0, np.nan),
        observed_fraction_to_area_table_ratio=lambda _df: _df["observed_fraction_area_m2"]
        / _df["glc_settlements_2020_m2"].replace(0, np.nan),
    )

    spatial_diagnostics
    return (spatial_diagnostics,)


@app.cell(hide_code=True)
def _(LOW_SETTLEMENT_FRACTION, RATIO_DENOMINATOR_FLOOR_M2, mo):
    mo.md(f"""
    ## Risk Flag Definitions

    The risk flags are screening diagnostics, not automatic exclusion rules. A
    single flag means the zone needs context; multiple flags mean the zone-level
    Chen/GLC comparison is probably sensitive to resolution, boundary support, or
    product semantics.

    The thresholds are intentionally simple and visible:

    - `small_baseline_risk` checks whether the GLC 2020 settlement baseline is
      smaller than one Chen-grid pixel (`{RATIO_DENOMINATOR_FLOOR_M2:,.0f} m2`).
    - `few_chen_urban_pixels_risk` checks whether fewer than 50 Chen-equivalent
      urban pixels drive the Chen signal.
    - `weak_overlap_risk` checks whether Chen urban cells have low observed GLC
      settlement fractions on average.
    - `chen_urban_low_observed_dominates` checks whether most Chen urban area falls
      where observed settlement fraction is below `{LOW_SETTLEMENT_FRACTION:.2f}`.
    - `high_observed_missed_risk` checks whether concentrated observed settlement
      cells are mostly outside Chen urban cells.
    - `boundary_area_risk` checks whether the Chen-grid reduction support differs
      materially from the 2020 area-table support.
    """)
    return


@app.cell
def show_risk_flag_definitions(
    HIGH_SETTLEMENT_FRACTION,
    LOW_SETTLEMENT_FRACTION,
    RATIO_DENOMINATOR_FLOOR_M2,
    pd,
):
    risk_flag_definitions = pd.DataFrame(
        [
            {
                "risk_flag": "small_baseline_risk",
                "condition": f"GLC settlements < {RATIO_DENOMINATOR_FLOOR_M2:,.0f} m2",
                "why_it_matters": "Ratio errors become unstable when the observed settlement denominator is smaller than one Chen pixel.",
                "interpretation": "Use absolute area diagnostics or manual review instead of ratio ranking alone.",
            },
            {
                "risk_flag": "few_chen_urban_pixels_risk",
                "condition": "Chen urban equivalent pixels < 50",
                "why_it_matters": "A small number of coarse cells can dominate the zone-level Chen area total.",
                "interpretation": "The result is sensitive to grid alignment and individual Chen-cell classification.",
            },
            {
                "risk_flag": "weak_overlap_risk",
                "condition": "Mean observed settlement fraction inside Chen urban cells < 0.25",
                "why_it_matters": "Chen urban cells mostly cover locations that are not settlement-dense in the 30 m observed product.",
                "interpretation": "Treat Chen 2020 as poorly aligned with the GLC settlement baseline for that zone.",
            },
            {
                "risk_flag": "chen_urban_low_observed_dominates",
                "condition": f"> 50% of Chen urban area has observed fraction < {LOW_SETTLEMENT_FRACTION:.2f}",
                "why_it_matters": "Most Chen urban area is supported by very little observed settlement area after resampling.",
                "interpretation": "This is a stronger version of weak overlap and should trigger calibration or manual map review.",
            },
            {
                "risk_flag": "high_observed_missed_risk",
                "condition": f"> 25% of high-observed-fraction area is Chen non-urban; high fraction > {HIGH_SETTLEMENT_FRACTION:.2f}",
                "why_it_matters": "Observed settlement concentrations exist in places Chen does not classify as urban.",
                "interpretation": "Chen may under-cover or shift the observed settlement footprint within the zone.",
            },
            {
                "risk_flag": "boundary_area_risk",
                "condition": "Chen-grid support / area-table support is outside 0.90 to 1.10",
                "why_it_matters": "The two reductions may be summarizing materially different effective zone areas.",
                "interpretation": "Inspect geometry, boundary pixels, and reduction scale before using zone totals.",
            },
            {
                "risk_flag": "high_spatial_risk",
                "condition": "Two or more risk flags are true",
                "why_it_matters": "Compound issues are harder to explain with a single tolerance or calibration choice.",
                "interpretation": "Carry forward as a manual-review, exclusion, or lower-confidence zone.",
            },
        ]
    )

    risk_flag_definitions
    return


@app.cell
def flag_spatial_risks(pd, spatial_diagnostics):
    spatial_risk_table = spatial_diagnostics.assign(
        small_baseline_risk=lambda _df: ~_df["ratio_denominator_is_stable"],
        few_chen_urban_pixels_risk=lambda _df: _df["chen_urban_equivalent_pixels"] < 50,
        weak_overlap_risk=lambda _df: _df["mean_observed_fraction_on_chen_urban"] < 0.25,
        chen_urban_low_observed_dominates=lambda _df: _df["chen_urban_low_observed_share"] > 0.50,
        high_observed_missed_risk=lambda _df: _df["high_observed_nonurban_share"] > 0.25,
        boundary_area_risk=lambda _df: ~_df["chen_grid_to_area_table_ratio"].between(0.90, 1.10),
    )

    _risk_columns = [
        "small_baseline_risk",
        "few_chen_urban_pixels_risk",
        "weak_overlap_risk",
        "chen_urban_low_observed_dominates",
        "high_observed_missed_risk",
        "boundary_area_risk",
    ]

    spatial_risk_table = spatial_risk_table.assign(
        spatial_risk_count=lambda _df: _df[_risk_columns].sum(axis=1),
        high_spatial_risk=lambda _df: _df["spatial_risk_count"] >= 2,
    )

    high_risk_spatial_zones = spatial_risk_table.loc[
        spatial_risk_table["high_spatial_risk"],
        [
            "zone",
            "selection_reason",
            "spatial_risk_count",
            *_risk_columns,
            "mean_observed_fraction_on_chen_urban",
            "chen_urban_low_observed_share",
            "high_observed_nonurban_share",
            "chen_urban_equivalent_pixels",
            "max_abs_difference_ha",
        ],
    ].sort_values(["spatial_risk_count", "max_abs_difference_ha"], ascending=False)

    spatial_risk_summary = pd.DataFrame(
        [
            {"metric": "diagnostic zones evaluated", "value": len(spatial_risk_table)},
            {"metric": "high spatial-risk zones", "value": int(spatial_risk_table["high_spatial_risk"].sum())},
            {"metric": "weak Chen/GLC overlap zones", "value": int(spatial_risk_table["weak_overlap_risk"].sum())},
            {"metric": "few Chen urban-pixel zones", "value": int(spatial_risk_table["few_chen_urban_pixels_risk"].sum())},
            {"metric": "high observed settlement missed zones", "value": int(spatial_risk_table["high_observed_missed_risk"].sum())},
            {"metric": "boundary area-risk zones", "value": int(spatial_risk_table["boundary_area_risk"].sum())},
        ]
    )

    spatial_risk_summary
    return high_risk_spatial_zones, spatial_risk_table


@app.cell
def show_high_risk_spatial_zones(high_risk_spatial_zones):
    high_risk_spatial_zones
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spatial Examples

    The next plots summarize the selected diagnostic zones. They are not maps, but
    they make the grid-level behavior visible: how much of Chen urban area overlaps
    observed settlement fractions, how many Chen-equivalent urban pixels drive the
    comparison, and which zones accumulated multiple risk flags.
    """)
    return


@app.cell
def plot_observed_fraction_on_chen_urban(plt, sns, spatial_risk_table):
    sns.set_theme(style="whitegrid")

    _fraction_fig, _fraction_ax = plt.subplots(figsize=(10, 5))
    _fraction_data = spatial_risk_table.sort_values(
        "mean_observed_fraction_on_chen_urban",
        ascending=True,
    )
    sns.barplot(
        data=_fraction_data,
        x="mean_observed_fraction_on_chen_urban",
        y="zone",
        hue="selection_reason",
        dodge=False,
        ax=_fraction_ax,
    )
    _fraction_ax.axvline(0.25, color="black", linestyle="--", linewidth=1)
    _fraction_ax.set_xlabel("Mean observed settlement fraction inside Chen urban cells")
    _fraction_ax.set_ylabel("Zone")
    _fraction_ax.set_title("Observed settlement fraction on the Chen urban grid")
    _fraction_ax.legend(title="Selection reason", bbox_to_anchor=(1.02, 1), loc="upper left")
    _fraction_fig.tight_layout()

    observed_fraction_plot = _fraction_fig

    observed_fraction_plot
    return


@app.cell
def plot_grid_risk_relationships(plt, sns, spatial_risk_table):
    _risk_fig, _risk_axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.scatterplot(
        data=spatial_risk_table,
        x="chen_urban_equivalent_pixels",
        y="chen_urban_low_observed_share",
        hue="high_spatial_risk",
        size="max_abs_difference_ha",
        sizes=(40, 220),
        ax=_risk_axes[0],
    )
    _risk_axes[0].axhline(0.50, color="black", linestyle="--", linewidth=1)
    _risk_axes[0].axvline(50, color="black", linestyle=":", linewidth=1)
    _risk_axes[0].set_xlabel("Chen urban equivalent pixels")
    _risk_axes[0].set_ylabel("Share of Chen urban area with observed fraction < 0.10")
    _risk_axes[0].set_title("Chen urban concentration and low observed overlap")

    sns.barplot(
        data=spatial_risk_table.sort_values("spatial_risk_count", ascending=False),
        x="spatial_risk_count",
        y="zone",
        hue="high_spatial_risk",
        dodge=False,
        ax=_risk_axes[1],
    )
    _risk_axes[1].set_xlabel("Number of spatial risk flags")
    _risk_axes[1].set_ylabel("Zone")
    _risk_axes[1].set_title("Risk flag count by diagnostic zone")

    _risk_fig.tight_layout()

    grid_risk_plots = _risk_fig

    grid_risk_plots
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation

    These diagnostics identify where a zone-level-only workflow is fragile. A zone
    can have a moderate area mismatch but still be risky if the Chen urban cells
    cover places with little observed settlement fraction, or if observed settlement
    concentrations mostly sit outside Chen urban cells.
    """)
    return


@app.cell
def build_spatial_conclusion(high_risk_spatial_zones, mo, spatial_risk_table):
    _diagnostic_count = len(spatial_risk_table)
    _high_risk_count = int(spatial_risk_table["high_spatial_risk"].sum())
    _weak_overlap_count = int(spatial_risk_table["weak_overlap_risk"].sum())
    _missed_observed_count = int(spatial_risk_table["high_observed_missed_risk"].sum())
    _few_urban_pixels_count = int(spatial_risk_table["few_chen_urban_pixels_risk"].sum())
    _high_risk_names = ", ".join(high_risk_spatial_zones["zone"].head(8))

    if _high_risk_count:
        _recommendation = (
            "Carry these zones forward only with calibration, exclusion, or manual "
            "spatial review before allocation."
        )
    else:
        _recommendation = "The selected diagnostics do not show a severe spatial reason to stop."

    spatial_conclusion = mo.md(
        f"""
    ### Spatial Diagnostic Readout

    - Diagnostic zones evaluated: `{_diagnostic_count}`
    - High spatial-risk zones: `{_high_risk_count}`
    - Weak Chen/GLC overlap zones: `{_weak_overlap_count}`
    - Zones with high observed settlement fractions outside Chen urban cells: `{_missed_observed_count}`
    - Zones where fewer than 50 Chen-equivalent urban pixels drive the Chen signal: `{_few_urban_pixels_count}`
    - High-risk examples: `{_high_risk_names or 'none'}`

    Recommendation: **{_recommendation}**

    These outputs are diagnostic rather than validating. The next notebook should
    carry forward the high-risk zone list when deciding whether historical
    settlement-source priors should be pooled, filtered, or interpreted with lower
    confidence.
    """
    )

    spatial_conclusion
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Limitations And Next Step

    This notebook summarizes Chen-grid fractions and risk flags, but it does not
    produce durable rasters or maps. It also uses `SSP2` as a representative 2020
    Chen mask for spatial diagnostics. If the workflow moves toward publication or
    formal review, selected high-risk zones should be inspected with map views and
    the diagnostics should be repeated across SSP bands where relevant.

    Next, `04_historical_settlement_transition_priors.py` should summarize observed
    transitions into `settlements`, while keeping the high-risk spatial zones in
    view.
    """)
    return


if __name__ == "__main__":
    app.run()
