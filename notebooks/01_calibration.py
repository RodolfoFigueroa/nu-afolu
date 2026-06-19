import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")

with app.setup:
    import json
    import os
    from pathlib import Path

    import ee
    import ee.deserializer
    import leafmap.foliumap as leafmap
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from dagster_components.partitions import zone_partitions

    from nu_afolu.chen import (
        CHEN_COLLECTION_ID,
        CHEN_URBAN_VALUE,
        CHEN_YEARS,
        GeoManager,
        SSP_NAMES,
        Zone,
        chen_urban_mask,
    )
    from nu_afolu.constants import LABEL_LIST
    from nu_afolu.utils import safe_ratio

    ee.Initialize()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Chen SSP Settlement Calibration Analysis

    This notebook evaluates Chen's SSP urban-land projections against the project's observed 2020 settlement layer. It calibrates Chen's 2020 urban baseline, checks sensitivity to observed-settlement thresholds, and writes calibration artifacts for the downstream transition-closure notebook.

    The historical pipeline estimates full year-to-year transitions from 30m GLC-FCS30D classes. Chen is coarser and only forecasts urban expansion, so transition construction and readiness screening now live in `02_transition_closure.py`.
    """)
    return


@app.cell
def _():
    LABEL_MAP = dict(enumerate(LABEL_LIST, start=1))
    LABEL_ID_BY_NAME = {label: idx for idx, label in LABEL_MAP.items()}
    SETTLEMENT_IDX = LABEL_ID_BY_NAME["settlements"]

    SOURCE_YEAR = 2020
    CORRECTION_FACTOR_BOUNDS = (0.25, 4.0)
    MIN_OBSERVED_SETTLEMENT_AREA_M2 = 1_000_000
    HIGH_IOU_THRESHOLD = 0.25
    MEDIUM_IOU_THRESHOLD = 0.10
    return (
        CORRECTION_FACTOR_BOUNDS,
        HIGH_IOU_THRESHOLD,
        MEDIUM_IOU_THRESHOLD,
        MIN_OBSERVED_SETTLEMENT_AREA_M2,
        SETTLEMENT_IDX,
        SOURCE_YEAR,
    )


@app.cell
def _():
    SCALE_SENSITIVITY_THRESHOLDS = [0.10, 0.25, 0.50]
    return (SCALE_SENSITIVITY_THRESHOLDS,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Paths
    """)
    return


@app.cell
def _():
    out_path = Path(os.environ["OUT_PATH"])
    return (out_path,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Image collections
    """)
    return


@app.cell
def _():
    col_chen = ee.ImageCollection(CHEN_COLLECTION_ID)
    return (col_chen,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Load raster data
    """)
    return


@app.cell
def _(SETTLEMENT_IDX, col_chen, out_path):
    manager = GeoManager()

    for zone in zone_partitions.get_partition_keys():
        try:
            with (out_path / "bbox" / "ee" / f"{zone}.json").open() as f:
                bbox: ee.Geometry = ee.deserializer.decode(json.load(f))

            with (out_path / "area_raster" / f"{zone}.json").open() as f:
                area_raster: ee.Image = ee.deserializer.decode(json.load(f)).clip(bbox)
                area_raster = area_raster.updateMask(area_raster.neq(ee.Number(0)))

            with (out_path / "transition_raster" / f"{zone}.json").open() as f:
                transition_raster: ee.Image = ee.deserializer.decode(json.load(f)).clip(
                    bbox
                )
                transition_raster = transition_raster.updateMask(
                    transition_raster.neq(ee.Number(0))
                )

            area_df = pd.read_parquet(out_path / "area_table" / f"{zone}.parquet")
        except FileNotFoundError:
            continue

        manager[zone] = Zone(
            bbox=bbox,
            area_raster=area_raster,
            transition_raster=transition_raster,
            area_df=area_df,
            chen_collection=col_chen,
            settlement_idx=SETTLEMENT_IDX,
        )

    manager.area_df.head(5)
    return (manager,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Grid alignment

    Chen's SSP projections are 1km equal-area rasters, while the historical GLC-FCS30D-derived land-use rasters are 30m. The calibration and scenario logic therefore treats Chen as a coarse expansion signal rather than as a future 30m land-use map.

    For calibration, the observed 2020 settlement mask is aggregated onto Chen's grid. The aggregation uses a mean reducer, so each Chen pixel receives a fractional observed-settlement value between 0 and 1. For example, a value of 0.35 means that about 35% of the 30m pixels inside that Chen cell are observed as settlements. This lets the notebook compare Chen and observed settlements in area terms without pretending the 1km Chen pixels identify exact 30m settlement locations.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Calibration process

    Calibration answers one question before any future scenario is used: if Chen says a place is urban in 2020, how well does that agree with this project's observed 2020 settlement layer?

    For each zone and SSP, the notebook builds two 2020 surfaces on Chen's 1km grid:

    1. `chen_mask`: Chen's 2020 urban indicator, where urban pixels are 1 and non-urban pixels are 0.
    2. `observed_fraction`: the 30m settlement mask aggregated to Chen's grid, where values range from 0 to 1.

    Both are multiplied by `ee.Image.pixelArea()` on Chen's projection, giving comparable areas in square meters:

    - `chen_area_m2 = chen_mask * pixel_area`
    - `observed_area_m2 = observed_fraction * pixel_area`
    - `tp_area_m2 = chen_mask * observed_fraction * pixel_area`

    The true-positive area is fractional because a Chen urban pixel can contain only some observed settlement at 30m. False positives and false negatives are derived from those same areas:

    - `fp_area_m2 = chen_area_m2 - tp_area_m2`
    - `fn_area_m2 = observed_area_m2 - tp_area_m2`

    This makes the spatial metrics comparable across zones even though the source datasets have different resolutions.
    """)
    return


@app.cell
def _(
    CORRECTION_FACTOR_BOUNDS,
    HIGH_IOU_THRESHOLD,
    MEDIUM_IOU_THRESHOLD,
    MIN_OBSERVED_SETTLEMENT_AREA_M2,
    SOURCE_YEAR,
):
    def reliability_label(
        observed_area_m2: float,
        chen_area_m2: float,
        area_bias: float,
        iou: float,
    ) -> str:
        if observed_area_m2 < MIN_OBSERVED_SETTLEMENT_AREA_M2:
            return "low"
        if chen_area_m2 <= 0 or not np.isfinite(area_bias) or not np.isfinite(iou):
            return "low"
        if 0.5 <= area_bias <= 2.0 and iou >= HIGH_IOU_THRESHOLD:
            return "high"
        if iou >= MEDIUM_IOU_THRESHOLD:
            return "medium"
        return "low"


    def observed_settlement_fraction_image(zone: Zone, scenario: str) -> ee.Image:
        chen_projection = zone.ssp_images[scenario].select(str(SOURCE_YEAR)).projection()
        return (
            zone.settlement_mask.select(str(SOURCE_YEAR))
            .unmask(0)
            .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=2048)
            .reproject(chen_projection)
            .rename("observed_settlement_fraction")
        )


    def calibration_row_from_metrics(
        zone_name: str,
        scenario: str,
        metrics: dict[str, float],
    ) -> dict[str, object]:
        observed_area_m2 = float(metrics.get(f"{scenario}_observed_area_m2") or 0)
        chen_area_m2 = float(metrics.get(f"{scenario}_chen_area_m2") or 0)
        tp_area_m2 = max(float(metrics.get(f"{scenario}_tp_area_m2") or 0), 0)
        fp_area_m2 = max(float(metrics.get(f"{scenario}_fp_area_m2") or 0), 0)
        fn_area_m2 = max(float(metrics.get(f"{scenario}_fn_area_m2") or 0), 0)

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
            and np.isfinite(correction_factor_raw)
        )
        correction_factor = (
            float(np.clip(correction_factor_raw, *CORRECTION_FACTOR_BOUNDS))
            if calibration_valid
            else 1.0
        )

        return {
            "zone": zone_name,
            "scenario": scenario,
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
            "reliability": reliability_label(
                observed_area_m2,
                chen_area_m2,
                area_bias,
                iou,
            ),
        }


    def calibrate_zone(zone_name: str, zone: Zone) -> list[dict[str, object]]:
        metric_images: list[ee.Image] = []
        for scenario in SSP_NAMES:
            chen_projection = zone.ssp_images[scenario].select(str(SOURCE_YEAR)).projection()
            pixel_area = ee.Image.pixelArea().reproject(chen_projection)
            observed_fraction = observed_settlement_fraction_image(zone, scenario)
            chen_mask = chen_urban_mask(zone, scenario, SOURCE_YEAR).reproject(
                chen_projection
            )

            observed_area = observed_fraction.multiply(pixel_area).rename(
                f"{scenario}_observed_area_m2"
            )
            chen_area = chen_mask.multiply(pixel_area).rename(f"{scenario}_chen_area_m2")
            tp_area = observed_fraction.multiply(chen_mask).multiply(pixel_area).rename(
                f"{scenario}_tp_area_m2"
            )
            fp_area = chen_area.subtract(tp_area).rename(f"{scenario}_fp_area_m2")
            fn_area = observed_area.subtract(tp_area).rename(f"{scenario}_fn_area_m2")
            metric_images.extend([observed_area, chen_area, tp_area, fp_area, fn_area])

        metrics = (
            ee.Image.cat(metric_images)
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=zone.bbox,
                scale=1000,
                maxPixels=int(1e10),
            )
            .getInfo()
        ) or {}

        return [calibration_row_from_metrics(zone_name, scenario, metrics) for scenario in SSP_NAMES]


    def calibrate_zone_scenario(zone_name: str, zone: Zone, scenario: str) -> dict[str, object]:
        rows = calibrate_zone(zone_name, zone)
        return next(row for row in rows if row["scenario"] == scenario)


    def build_calibration_table(manager: GeoManager) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for zone_name, zone in manager:
            rows.extend(calibrate_zone(zone_name, zone))
        return pd.DataFrame(rows)


    def summarize_calibration(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.assign(
                abs_error_km2=lambda frame: frame["area_error_m2"].abs().div(1e6),
                squared_error_km4=lambda frame: frame["area_error_m2"].div(1e6) ** 2,
            )
            .groupby("scenario")
            .agg(
                zones=("zone", "count"),
                valid_calibrations=("calibration_valid", "sum"),
                observed_total_km2=("observed_area_m2", lambda series: series.sum() / 1e6),
                chen_total_km2=("chen_area_m2", lambda series: series.sum() / 1e6),
                mae_km2=("abs_error_km2", "mean"),
                rmse_km2=("squared_error_km4", lambda series: np.sqrt(series.mean())),
                median_ape_pct=("ape", lambda series: np.nanmedian(series) * 100),
                mean_precision=("precision", "mean"),
                mean_recall=("recall", "mean"),
                mean_iou=("iou", "mean"),
            )
            .round(3)
        )


    def reliability_counts(df: pd.DataFrame) -> pd.DataFrame:
        return pd.crosstab(df["scenario"], df["reliability"]).reindex(
            columns=["high", "medium", "low"],
            fill_value=0,
        )

    return (
        build_calibration_table,
        observed_settlement_fraction_image,
        reliability_counts,
        summarize_calibration,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Calibration helper functions

    The helper functions below do three things:

    1. Create the comparable Chen and observed 2020 surfaces.
    2. Reduce those surfaces to per-zone calibration metrics.
    3. Convert the metrics into reliability labels and correction factors.

    The implementation stacks all five SSP calibration bands before calling Earth Engine. That keeps the expensive work to one `reduceRegion` call per zone instead of one call per zone per SSP. The resulting dataframe still has one row per `zone, scenario` so later cells can use SSP-specific correction factors.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Calibration metrics and reliability flags

    The calibration table has one row per zone and SSP. The key columns are:

    - `area_bias`: Chen 2020 urban area divided by observed 2020 settlement area. Values above 1 mean Chen overestimates area; values below 1 mean Chen underestimates area.
    - `ape`: absolute percent error in area, expressed as a fraction before summary formatting.
    - `precision`: of the area Chen marks urban, the fraction that overlaps observed settlement.
    - `recall`: of the observed settlement area, the fraction captured by Chen.
    - `iou`: intersection over union, a compact spatial-overlap score.
    - `correction_factor`: observed area divided by Chen area, clipped to a conservative range before it is used in the calibrated scenario.
    - `reliability`: a high, medium, or low label based on observed area, area bias, and IoU.

    Reliability is a diagnostic flag, not a filter. Low-reliability zones remain in the future transition tables so uncertainty is visible instead of hidden.
    """)
    return


@app.cell
def _(
    build_calibration_table,
    manager,
    reliability_counts,
    summarize_calibration,
):
    df_calibration = build_calibration_table(manager)
    df_calibration_summary = summarize_calibration(df_calibration)
    df_reliability_counts = reliability_counts(df_calibration)

    mo.vstack(
        [
            mo.md("### Calibration summary by SSP"),
            df_calibration_summary,
            mo.md("### Reliability counts by SSP"),
            df_reliability_counts,
        ]
    )
    return (df_calibration,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reading the calibration result

    The summary table aggregates error and overlap metrics by SSP. The reliability table shows how many zones are high, medium, or low confidence for each scenario.

    The correction factor is only an area correction. It does not improve Chen's spatial allocation. Later, the raw scenario keeps Chen's unadjusted expansion area, while the calibrated scenario scales the resulting `source class -> settlements` areas by this correction factor. When the factor is greater than 1, the extra calibrated area should be interpreted as an aggregate sensitivity adjustment rather than additional precisely located pixels.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Calibration diagnostics

    The first scatter plot compares observed 2020 settlement area to Chen 2020 urban area. Points near the 1:1 line have good area agreement. The second scatter plot applies the clipped correction factor, showing how well the calibrated baseline matches observed area by construction. The histogram shows how much correction is being applied and whether many zones are hitting the clipping bounds.

    These plots should be read together with precision, recall, and IoU. A zone can have good total area agreement but poor spatial agreement, which would make source-class allocation less trustworthy.
    """)
    return


@app.cell
def _(df_calibration):
    _plot_df = df_calibration.assign(
        observed_area_km2=lambda frame: frame["observed_area_m2"].div(1e6),
        chen_area_km2=lambda frame: frame["chen_area_m2"].div(1e6),
        calibrated_chen_area_km2=lambda frame: frame["chen_area_m2"]
        .mul(frame["correction_factor"])
        .div(1e6),
    )

    _fig, _axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.scatterplot(
        data=_plot_df,
        x="observed_area_km2",
        y="chen_area_km2",
        hue="scenario",
        style="reliability",
        s=35,
        ax=_axes[0],
    )
    _limit = max(_plot_df["observed_area_km2"].max(), _plot_df["chen_area_km2"].max())
    _axes[0].plot([0, _limit], [0, _limit], "k--", linewidth=1)
    _axes[0].set_title("Observed vs Chen 2020")
    _axes[0].set_xlabel("Observed settlements (km^2)")
    _axes[0].set_ylabel("Chen urban (km^2)")

    sns.scatterplot(
        data=_plot_df,
        x="observed_area_km2",
        y="calibrated_chen_area_km2",
        hue="scenario",
        style="reliability",
        s=35,
        legend=False,
        ax=_axes[1],
    )
    _limit = max(
        _plot_df["observed_area_km2"].max(),
        _plot_df["calibrated_chen_area_km2"].max(),
    )
    _axes[1].plot([0, _limit], [0, _limit], "k--", linewidth=1)
    _axes[1].set_title("Observed vs calibrated Chen 2020")
    _axes[1].set_xlabel("Observed settlements (km^2)")
    _axes[1].set_ylabel("Calibrated Chen urban (km^2)")

    sns.histplot(
        data=_plot_df,
        x="correction_factor",
        hue="scenario",
        element="step",
        fill=False,
        bins=20,
        ax=_axes[2],
    )
    _axes[2].axvline(1, color="k", linestyle="--", linewidth=1)
    _axes[2].set_title("Clipped correction factors")
    _axes[2].set_xlabel("Observed 2020 / Chen 2020")

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Scale-sensitivity checks

    The calibration section uses fractional observed settlement area on Chen's 1km grid. That is the most area-preserving comparison, but it is useful to know whether conclusions are fragile to the definition of an observed settlement cell.

    This section thresholds the observed settlement fraction at 10%, 25%, and 50%. A Chen grid cell is treated as observed settlement if at least that fraction of its 30m pixels are settlements. The resulting precision, recall, and IoU are compared across thresholds.

    If a zone changes adequacy dramatically across thresholds, the Chen-vs-observed agreement is scale-sensitive and should receive manual review even if the default fractional calibration looked acceptable.
    """)
    return


@app.cell
def _(
    SCALE_SENSITIVITY_THRESHOLDS,
    SOURCE_YEAR,
    manager,
    observed_settlement_fraction_image,
):
    def threshold_label(threshold: float) -> str:
        return f"t{int(round(threshold * 100)):02d}"


    def scale_sensitivity_row_from_metrics(
        zone_name: str,
        scenario: str,
        threshold: float,
        metrics: dict[str, float],
    ) -> dict[str, object]:
        label = threshold_label(threshold)
        observed_area_m2 = float(metrics.get(f"{scenario}_{label}_observed_area_m2") or 0)
        chen_area_m2 = float(metrics.get(f"{scenario}_{label}_chen_area_m2") or 0)
        tp_area_m2 = max(float(metrics.get(f"{scenario}_{label}_tp_area_m2") or 0), 0)
        fp_area_m2 = max(float(metrics.get(f"{scenario}_{label}_fp_area_m2") or 0), 0)
        fn_area_m2 = max(float(metrics.get(f"{scenario}_{label}_fn_area_m2") or 0), 0)
        union_area_m2 = chen_area_m2 + observed_area_m2 - tp_area_m2
        return {
            "zone": zone_name,
            "scenario": scenario,
            "threshold": threshold,
            "observed_area_m2": observed_area_m2,
            "chen_area_m2": chen_area_m2,
            "tp_area_m2": tp_area_m2,
            "fp_area_m2": fp_area_m2,
            "fn_area_m2": fn_area_m2,
            "precision": safe_ratio(tp_area_m2, chen_area_m2),
            "recall": safe_ratio(tp_area_m2, observed_area_m2),
            "iou": safe_ratio(tp_area_m2, union_area_m2),
            "area_bias": safe_ratio(chen_area_m2, observed_area_m2),
        }


    def scale_sensitivity_rows_for_zone(zone_name: str, zone: Zone) -> list[dict[str, object]]:
        metric_images: list[ee.Image] = []
        for scenario in SSP_NAMES:
            chen_projection = zone.ssp_images[scenario].select(str(SOURCE_YEAR)).projection()
            pixel_area = ee.Image.pixelArea().reproject(chen_projection)
            observed_fraction = observed_settlement_fraction_image(zone, scenario)
            chen_mask = chen_urban_mask(zone, scenario, SOURCE_YEAR).reproject(chen_projection)
            for threshold in SCALE_SENSITIVITY_THRESHOLDS:
                label = threshold_label(threshold)
                observed_mask = observed_fraction.gte(ee.Number(threshold))
                observed_area = observed_mask.multiply(pixel_area).rename(f"{scenario}_{label}_observed_area_m2")
                chen_area = chen_mask.multiply(pixel_area).rename(f"{scenario}_{label}_chen_area_m2")
                tp_area = observed_mask.multiply(chen_mask).multiply(pixel_area).rename(f"{scenario}_{label}_tp_area_m2")
                fp_area = chen_area.subtract(tp_area).rename(f"{scenario}_{label}_fp_area_m2")
                fn_area = observed_area.subtract(tp_area).rename(f"{scenario}_{label}_fn_area_m2")
                metric_images.extend([observed_area, chen_area, tp_area, fp_area, fn_area])

        metrics = (
            ee.Image.cat(metric_images)
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=zone.bbox,
                scale=1000,
                maxPixels=int(1e10),
            )
            .getInfo()
        ) or {}

        rows: list[dict[str, object]] = []
        for scenario in SSP_NAMES:
            for threshold in SCALE_SENSITIVITY_THRESHOLDS:
                rows.append(scale_sensitivity_row_from_metrics(zone_name, scenario, threshold, metrics))
        return rows


    def build_scale_sensitivity_table(manager: GeoManager) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for zone_name, zone in manager:
            rows.extend(scale_sensitivity_rows_for_zone(zone_name, zone))
        return pd.DataFrame(rows)


    df_scale_sensitivity = build_scale_sensitivity_table(manager)

    assert df_scale_sensitivity.shape[0] == len(manager.zones) * len(SSP_NAMES) * len(SCALE_SENSITIVITY_THRESHOLDS)
    assert df_scale_sensitivity["threshold"].isin(SCALE_SENSITIVITY_THRESHOLDS).all()
    assert (df_scale_sensitivity[["observed_area_m2", "chen_area_m2", "tp_area_m2", "fp_area_m2", "fn_area_m2"]] >= -1e-6).all().all()

    mo.vstack(
        [
            mo.md("### Scale-sensitivity summary"),
            df_scale_sensitivity.groupby(["scenario", "threshold"])[["precision", "recall", "iou"]].mean().round(3),
            df_scale_sensitivity.head(10),
        ]
    )
    return (df_scale_sensitivity,)


@app.cell
def _(df_scale_sensitivity):
    _fig, _axes = plt.subplots(1, 3, figsize=(16, 5))
    for _axis, _metric in zip(_axes, ["precision", "recall", "iou"], strict=True):
        sns.lineplot(
            data=df_scale_sensitivity,
            x="threshold",
            y=_metric,
            hue="scenario",
            marker="o",
            ax=_axis,
        )
        _axis.set_title(f"{_metric.title()} by observed-settlement threshold")
        _axis.set_xlabel("Observed settlement fraction threshold")
        _axis.set_ylabel(_metric.title())

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(df_calibration, df_scale_sensitivity, out_path):
    _chen_artifact_dir = out_path / "chen"
    _chen_artifact_dir.mkdir(parents=True, exist_ok=True)

    _calibration_path = _chen_artifact_dir / "calibration.parquet"
    _scale_sensitivity_path = _chen_artifact_dir / "scale_sensitivity.parquet"

    df_calibration.to_parquet(_calibration_path, index=False)
    df_scale_sensitivity.to_parquet(_scale_sensitivity_path, index=False)

    mo.vstack(
        [
            mo.md("### Calibration artifacts"),
            pd.DataFrame(
                [
                    {
                        "artifact": "calibration",
                        "path": str(_calibration_path),
                        "rows": len(df_calibration),
                    },
                    {
                        "artifact": "scale_sensitivity",
                        "path": str(_scale_sensitivity_path),
                        "rows": len(df_scale_sensitivity),
                    },
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Handoff to transition closure

    This notebook stops at calibration and scale-sensitivity diagnostics. The persisted calibration table at `OUT_PATH/chen/calibration.parquet` is the handoff to `02_transition_closure.py`, which builds diagnostic future settlement transitions, readiness screens, and review candidates.
    """)
    return


if __name__ == "__main__":
    app.run()
