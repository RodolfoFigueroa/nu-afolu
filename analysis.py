import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")

with app.setup:
    import json
    import os
    from collections import UserDict
    from collections.abc import Callable, Iterator
    from functools import cached_property
    from pathlib import Path

    import ee
    import marimo as mo
    import ee.deserializer
    import leafmap.foliumap as leafmap
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import xarray as xr
    from dagster_components.partitions import zone_partitions

    from nu_afolu.constants import LABEL_LIST

    ee.Initialize()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Chen SSP Settlement Expansion Analysis

    This notebook evaluates whether Chen's SSP urban-land projections can serve as a future settlement-expansion signal for the AFOLU carbon model.

    The historical pipeline estimates full year-to-year transitions from 30m GLC-FCS30D classes. Chen is coarser and only forecasts urban expansion, so this notebook builds a narrower scenario: future `source class -> settlements` transitions. The notebook first calibrates Chen's 2020 urban baseline against observed 2020 settlements, then uses decadal Chen increments to estimate raw and bias-corrected settlement transitions.
    """)
    return


@app.cell
def _():
    LABEL_MAP = dict(enumerate(LABEL_LIST, start=1))
    LABEL_ID_BY_NAME = {label: idx for idx, label in LABEL_MAP.items()}
    SETTLEMENT_IDX = LABEL_ID_BY_NAME["settlements"]

    SSP_NAMES = [f"SSP{suffix}" for suffix in range(1, 6)]
    CHEN_YEARS = list(range(2020, 2101, 10))
    FUTURE_YEARS = CHEN_YEARS[1:]
    SOURCE_YEAR = 2020
    SOURCE_CLASSES = [label for label in LABEL_LIST if label != "settlements"]

    CORRECTION_FACTOR_BOUNDS = (0.25, 4.0)
    MIN_OBSERVED_SETTLEMENT_AREA_M2 = 1_000_000
    HIGH_IOU_THRESHOLD = 0.25
    MEDIUM_IOU_THRESHOLD = 0.10
    return (
        CORRECTION_FACTOR_BOUNDS,
        FUTURE_YEARS,
        HIGH_IOU_THRESHOLD,
        LABEL_MAP,
        MEDIUM_IOU_THRESHOLD,
        MIN_OBSERVED_SETTLEMENT_AREA_M2,
        SETTLEMENT_IDX,
        SOURCE_CLASSES,
        SOURCE_YEAR,
        SSP_NAMES,
    )


@app.cell
def _():
    CHEN_URBAN_VALUE = 2

    HISTORICAL_GROWTH_PERIODS = [(2000, 2010), (2010, 2020), (2000, 2020)]
    RECENT_GROWTH_PERIOD = (2010, 2020)
    MIN_HISTORICAL_GROWTH_AREA_M2 = 1_000_000

    SENSITIVE_CLASSES = ["forests_primary", "forests_mangroves", "wetlands"]
    WATCH_CLASSES = ["forests_secondary"]
    SCALE_SENSITIVITY_THRESHOLDS = [0.10, 0.25, 0.50]
    return (
        CHEN_URBAN_VALUE,
        HISTORICAL_GROWTH_PERIODS,
        MIN_HISTORICAL_GROWTH_AREA_M2,
        RECENT_GROWTH_PERIOD,
        SCALE_SENSITIVITY_THRESHOLDS,
        SENSITIVE_CLASSES,
        WATCH_CLASSES,
    )


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
    col_chen = ee.ImageCollection(
        "projects/sat-io/open-datasets/FUTURE-URBAN-LAND/CHEN_2020_2100"
    )
    return (col_chen,)


@app.cell
def _(CHEN_URBAN_VALUE):
    def reduce_ssp_col(
        col: ee.ImageCollection, *, geometry: ee.Geometry, scale: float
    ) -> pd.DataFrame:
        reduced: ee.FeatureCollection = col.map(
            lambda img: ee.Feature(
                geometry,
                img.eq(ee.Number(CHEN_URBAN_VALUE))
                .multiply(ee.Image.pixelArea())
                .reduceRegion(ee.Reducer.sum(), geometry=geometry, scale=scale),
            )
        )

        collected: dict[str, list[float]] = {}
        for suffix in range(1, 6):
            property_name = f"SSP{suffix}"
            arr = reduced.aggregate_array(property_name).getInfo()
            if not isinstance(arr, list):
                err = f"Expected list for property {property_name}, got {type(arr)}"
                raise TypeError(err)
            collected[property_name] = arr

        return pd.DataFrame(collected, index=list(range(2020, 2101, 10)))

    return (reduce_ssp_col,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Manager classes
    """)
    return


@app.cell
def _(CHEN_URBAN_VALUE, SETTLEMENT_IDX, col_chen, reduce_ssp_col):
    class Zone:
        def __init__(
            self,
            bbox: ee.Geometry,
            area_raster: ee.Image,
            transition_raster: ee.Image,
            area_df: pd.DataFrame,
        ) -> None:
            self._bbox: ee.Geometry = bbox
            self.area_raster: ee.Image = area_raster
            self.transition_raster: ee.Image = transition_raster
            self.area_df: pd.DataFrame = area_df
            self._fields_to_invalidate: list[str] = ["ssp_images", "area_chen"]

        @property
        def bbox(self) -> ee.Geometry:
            return self._bbox

        @bbox.setter
        def bbox(self, value: ee.Geometry) -> None:
            self._bbox = value
            for field in self._fields_to_invalidate:
                if field in self.__dict__:
                    del self.__dict__[field]

        @bbox.deleter
        def bbox(self) -> None:
            del self._bbox
            for field in self._fields_to_invalidate:
                if field in self.__dict__:
                    del self.__dict__[field]

        @cached_property
        def ssp_images(self) -> dict[str, ee.Image]:
            ssp_images: dict[str, ee.Image] = {}
            for suffix in range(1, 6):
                name = f"SSP{suffix}"
                ssp_images[name] = (
                    col_chen.select(name)
                    .toBands()
                    .rename([str(year) for year in range(2020, 2101, 10)])
                    .eq(ee.Number(CHEN_URBAN_VALUE))
                    .clip(self.bbox)
                    .selfMask()
                )
            return ssp_images

        @cached_property
        def area_chen(self) -> pd.DataFrame:
            return reduce_ssp_col(col_chen, geometry=self.bbox, scale=1000)

        @cached_property
        def settlement_mask(self) -> ee.Image:
            return self.area_raster.eq(ee.Number(SETTLEMENT_IDX))

        def resample_settlement_mask(self, ssp: str) -> ee.Image:
            return self.settlement_mask.reduceResolution(
                reducer=ee.Reducer.mean(), maxPixels=2048
            ).reproject(self.ssp_images[ssp].projection())

    class ObservableDict(UserDict):
        def __init__(self, on_change: Callable, *args, **kwargs) -> None:
            self._on_change = on_change
            super().__init__(*args, **kwargs)

        def __setitem__(self, key: str, item: Zone) -> None:
            super().__setitem__(key, item)
            self._on_change()

        def __delitem__(self, key: str) -> None:
            super().__delitem__(key)
            self._on_change()


    class GeoManager:
        def __init__(self) -> None:
            self._zones: ObservableDict = ObservableDict(self._invalidate)
            self._area_df: pd.DataFrame | None = None
            self._area_arr: xr.DataArray | None = None

        def _invalidate(self) -> None:
            self._area_df = None
            self._area_arr = None

        def __getitem__(self, key: str) -> Zone:
            return self._zones[key]

        def __setitem__(self, key: str, value: Zone) -> None:
            self._zones[key] = value

        def __delitem__(self, key: str) -> None:
            del self._zones[key]

        def __iter__(self) -> Iterator[tuple[str, Zone]]:
            return iter(self._zones.items())

        @property
        def zones(self) -> ObservableDict[str, Zone]:
            return self._zones

        @property
        def area_df(self) -> pd.DataFrame:
            if self._area_df is None:
                self._area_df = (
                    pd.concat(
                        [zone.area_df.assign(zone=key) for key, zone in self._zones.items()]
                    )
                    .reset_index(names="year")
                    .assign(year=lambda df: df["year"].astype(int))
                    .set_index(["zone", "year"])
                )
            return self._area_df

        @property
        def area_arr(self) -> xr.DataArray:
            if self._area_arr is None:
                out = self.area_df.rename_axis(columns="category").stack().to_xarray()
                out.name = "area"
                if not isinstance(out, xr.DataArray):
                    err = (
                        f"Expected area_arr to be an xarray.DataArray, but got {type(out)}"
                    )
                    raise TypeError(err)
                self._area_arr = out
            return self._area_arr

    return GeoManager, Zone


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Load raster data
    """)
    return


@app.cell
def _(GeoManager, Zone, out_path):
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
    GeoManager,
    HIGH_IOU_THRESHOLD,
    MEDIUM_IOU_THRESHOLD,
    MIN_OBSERVED_SETTLEMENT_AREA_M2,
    SOURCE_YEAR,
    SSP_NAMES,
    Zone,
):
    def safe_ratio(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return np.nan
        return float(numerator / denominator)


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


    def chen_urban_mask(zone: Zone, scenario: str, year: int) -> ee.Image:
        return zone.ssp_images[scenario].select(str(year)).unmask(0).rename("chen_urban")


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
        chen_urban_mask,
        observed_settlement_fraction_image,
        reliability_counts,
        safe_ratio,
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
    ## Future settlement expansion scenarios

    Chen is used only as a future settlement-expansion signal. This notebook does not try to estimate a full future land-use transition matrix. It estimates a narrower set of transitions where the destination class is `settlements`.

    The key modeling choice is to use Chen as a source of new urban area after 2020, not as a replacement for the observed 2020 settlement baseline. For each SSP, the notebook walks through Chen's decadal maps and asks: which cells are urban now that were not already urban in 2020 or in any earlier future decade?

    That gives each Chen expansion cell a first expansion year:

    - urban in 2020: not counted as future expansion.
    - first urban in 2030: counted in the 2020-2030 period.
    - first urban in 2040: counted in the 2030-2040 period.
    - already counted in 2030: not counted again in 2040 or later.

    This monotonic accounting avoids double-counting persistent urban pixels across decades.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Expansion and allocation helper functions

    The helper functions below separate two tasks:

    1. `first_expansion_year_image` creates an Earth Engine image whose pixel value is the first future year when Chen marks that pixel as urban. Pixels already urban in 2020 are masked out.
    2. `reduce_zone_expansion_sources` overlays those future expansion pixels on the observed 2020 land-use raster, then sums expansion area by source class.

    The reducer works at the 30m historical land-use resolution because the question is not only "how much Chen expansion occurred?" but also "which observed class would that expansion replace?" The Chen mask chooses the coarse future expansion footprint; the 30m `area_raster` supplies the source-class composition within that footprint.

    The implementation stacks all SSP and decade bands before reduction. For each zone, that means one grouped Earth Engine reduction produces all source-class areas for all SSP-decade combinations. This keeps the notebook tractable while preserving the full output shape.
    """)
    return


@app.cell
def _(
    FUTURE_YEARS,
    GeoManager,
    LABEL_MAP,
    SOURCE_CLASSES,
    SOURCE_YEAR,
    SSP_NAMES,
    Zone,
    chen_urban_mask,
):
    def first_expansion_year_image(zone: Zone, scenario: str) -> ee.Image:
        prior_urban = chen_urban_mask(zone, scenario, SOURCE_YEAR)
        first_year = ee.Image(0).rename("first_expansion_year").int16()

        for future_year in FUTURE_YEARS:
            current_urban = chen_urban_mask(zone, scenario, future_year)
            new_urban = current_urban.And(prior_urban.Not())
            first_year = first_year.where(new_urban, ee.Number(future_year))
            prior_urban = prior_urban.Or(current_urban)

        return first_year.selfMask().rename("first_expansion_year")


    def expansion_area_bands(zone: Zone) -> tuple[list[ee.Image], list[dict[str, object]]]:
        area_bands: list[ee.Image] = []
        band_specs: list[dict[str, object]] = []
        pixel_area = ee.Image.pixelArea()

        for scenario in SSP_NAMES:
            first_year = first_expansion_year_image(zone, scenario)
            for future_year in FUTURE_YEARS:
                band_name = f"{scenario}_{future_year}"
                area_bands.append(
                    pixel_area.updateMask(first_year.eq(ee.Number(future_year))).rename(
                        band_name
                    )
                )
                band_specs.append(
                    {
                        "band_name": band_name,
                        "scenario": scenario,
                        "year": future_year,
                        "period_start_year": future_year - 10,
                    }
                )

        return area_bands, band_specs


    def reduce_zone_expansion_sources(
        zone_name: str,
        zone: Zone,
    ) -> list[dict[str, object]]:
        area_bands, band_specs = expansion_area_bands(zone)
        source_label = zone.area_raster.select(str(SOURCE_YEAR)).unmask(0).rename(
            "source_label"
        )

        reducer = ee.Reducer.sum().repeat(len(area_bands)).group(
            groupField=len(area_bands),
            groupName="source_label",
        )
        result = (
            ee.Image.cat(area_bands)
            .addBands(source_label)
            .updateMask(source_label.gt(0))
            .reduceRegion(
                reducer=reducer,
                geometry=zone.bbox,
                scale=30,
                maxPixels=int(1e10),
                tileScale=4,
            )
            .getInfo()
        ) or {}

        rows: list[dict[str, object]] = []
        for group in result.get("groups", []):
            source_label_idx = int(group["source_label"])
            source_class = LABEL_MAP.get(source_label_idx, f"unknown_{source_label_idx}")
            area_values = group.get("sum", [])
            if not isinstance(area_values, list):
                area_values = [area_values]
            for spec, area_value in zip(band_specs, area_values, strict=False):
                area_m2 = float(area_value or 0)
                if area_m2 <= 0:
                    continue
                rows.append(
                    {
                        "zone": zone_name,
                        "scenario": spec["scenario"],
                        "year": spec["year"],
                        "period_start_year": spec["period_start_year"],
                        "source_label": source_label_idx,
                        "from_class": source_class,
                        "area_m2": area_m2,
                    }
                )
        return rows


    def reduce_expansion_sources(
        zone_name: str,
        zone: Zone,
        scenario: str,
    ) -> list[dict[str, object]]:
        return [
            row
            for row in reduce_zone_expansion_sources(zone_name, zone)
            if row["scenario"] == scenario
        ]


    def build_chen_transition_tables(
        manager: GeoManager,
        df_calibration: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        calibration_lookup = df_calibration.set_index(["zone", "scenario"]).to_dict("index")
        expansion_rows: list[dict[str, object]] = []
        transition_rows: list[dict[str, object]] = []

        for zone_name, zone in manager:
            raw_rows = reduce_zone_expansion_sources(zone_name, zone)
            area_lookup = {
                (str(row["scenario"]), int(row["year"]), str(row["from_class"])): float(
                    row["area_m2"]
                )
                for row in raw_rows
            }

            for scenario in SSP_NAMES:
                calibration = calibration_lookup[(zone_name, scenario)]
                correction_factor = float(calibration["correction_factor"])
                reliability = str(calibration["reliability"])

                for future_year in FUTURE_YEARS:
                    existing_settlement_area_m2 = area_lookup.get(
                        (scenario, future_year, "settlements"),
                        0.0,
                    )
                    source_area_by_class = {
                        source_class: area_lookup.get((scenario, future_year, source_class), 0.0)
                        for source_class in SOURCE_CLASSES
                    }
                    nonsettlement_source_area_m2 = sum(source_area_by_class.values())
                    chen_new_area_m2 = (
                        nonsettlement_source_area_m2 + existing_settlement_area_m2
                    )

                    expansion_rows.append(
                        {
                            "zone": zone_name,
                            "scenario": scenario,
                            "period_start_year": future_year - 10,
                            "year": future_year,
                            "chen_new_area_m2": chen_new_area_m2,
                            "nonsettlement_source_area_m2": nonsettlement_source_area_m2,
                            "existing_settlement_area_m2": existing_settlement_area_m2,
                            "correction_factor": correction_factor,
                            "reliability": reliability,
                        }
                    )

                    for source_class, raw_area_m2 in source_area_by_class.items():
                        base_row = {
                            "zone": zone_name,
                            "scenario": scenario,
                            "period_start_year": future_year - 10,
                            "year": future_year,
                            "from_class": source_class,
                            "to_class": "settlements",
                            "correction_factor": correction_factor,
                            "reliability": reliability,
                        }
                        transition_rows.append(
                            {
                                **base_row,
                                "calibration": "raw",
                                "area_m2": raw_area_m2,
                                "scaled_up_area_only": False,
                            }
                        )
                        transition_rows.append(
                            {
                                **base_row,
                                "calibration": "calibrated",
                                "area_m2": raw_area_m2 * correction_factor,
                                "scaled_up_area_only": correction_factor > 1,
                            }
                        )

        return pd.DataFrame(expansion_rows), pd.DataFrame(transition_rows)

    return (build_chen_transition_tables,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Source-class allocation and scenario outputs

    For each Chen expansion cell, the observed 2020 30m land-use classes underneath that cell are treated as the source classes for future settlement expansion. The notebook deliberately uses conservative allocation:

    - observed non-settlement classes become candidate `from_class -> settlements` transitions.
    - observed `settlements` inside a Chen expansion cell are reported separately as `existing_settlement_area_m2`.
    - existing settlement area is not redistributed across other classes.

    This means the raw transition total can be smaller than Chen's coarse new-urban area when part of the Chen expansion footprint was already settlement in 2020. That is intentional: it avoids creating artificial emissions by forcing already-settled pixels into a land conversion.

    Two outputs are created:

    - `df_chen_expansion`: one row per `zone, scenario, decade`, with total Chen expansion, non-settlement source area, existing settlement area, correction factor, and reliability.
    - `df_chen_transitions`: one row per `zone, scenario, decade, source class, calibration`, with `to_class` fixed to `settlements`.

    The `raw` rows use the directly observed source-class area. The `calibrated` rows multiply the raw area by the clipped 2020 correction factor from the calibration section. If `scaled_up_area_only` is true, the calibrated area is larger than the spatially observed source area and should be treated as an aggregate sensitivity adjustment, not as newly located pixels.
    """)
    return


@app.cell
def _(
    CORRECTION_FACTOR_BOUNDS,
    build_chen_transition_tables,
    df_calibration,
    manager,
):
    df_chen_expansion, df_chen_transitions = build_chen_transition_tables(
        manager,
        df_calibration,
    )

    assert (df_chen_expansion["chen_new_area_m2"] >= -1e-6).all()
    assert (df_chen_transitions["area_m2"] >= -1e-6).all()
    assert set(df_chen_transitions["to_class"]) == {"settlements"}
    assert set(df_chen_transitions["reliability"]).issubset({"high", "medium", "low"})
    assert set(df_chen_expansion["reliability"]).issubset({"high", "medium", "low"})

    _bounds = df_calibration["correction_factor"].between(*CORRECTION_FACTOR_BOUNDS)
    _invalid_one = (~df_calibration["calibration_valid"]) & df_calibration[
        "correction_factor"
    ].eq(1.0)
    assert (_bounds | _invalid_one).all()

    _raw_totals = df_chen_transitions[df_chen_transitions["calibration"].eq("raw")].groupby(
        ["zone", "scenario", "year"]
    )["area_m2"].sum()
    _nonsettlement_totals = df_chen_expansion.set_index(["zone", "scenario", "year"])[
        "nonsettlement_source_area_m2"
    ]
    assert _raw_totals.le(_nonsettlement_totals.reindex(_raw_totals.index).fillna(0) + 1e-6).all()

    mo.vstack(
        [
            mo.md("### Expansion summary"),
            df_chen_expansion.head(10),
            mo.md("### Transition table"),
            df_chen_transitions.head(20),
        ]
    )
    return df_chen_expansion, df_chen_transitions


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reading the scenario tables

    The assertions in the output cell are sanity checks for the scenario construction:

    - transition areas must be non-negative.
    - every transition must end in `settlements`.
    - reliability labels must stay within the expected high, medium, and low categories.
    - correction factors must either be valid clipped factors or exactly 1.0 for invalid calibration cases.
    - raw transition totals cannot exceed the non-settlement source area reported in `df_chen_expansion`.

    The head of each table is shown only as a preview. The useful downstream object is `df_chen_transitions`, which is already shaped like a partial transition table for a carbon model: zone, SSP, period, source class, destination class, calibration type, and area.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Scenario diagnostics

    These plots summarize the future transition scenario after calibration and allocation.

    The line plot shows total projected `class -> settlements` area by SSP and future decade. The raw line is the direct Chen expansion footprint allocated onto observed 2020 source classes. The calibrated line applies the 2020 area correction factor, so the gap between raw and calibrated totals is a quick visual measure of how much baseline bias correction changes the emissions-relevant transition area.

    The bar plot shows which observed source classes contribute most to raw projected settlement expansion. This is the main carbon-modeling bridge: if future settlements mostly replace croplands and pastures, the carbon implications differ from a scenario where they replace forests, wetlands, or mangroves.

    These plots aggregate across all zones. Zone-level reliability remains available in the output tables and should be used before interpreting local results.
    """)
    return


@app.cell
def _(df_chen_transitions):
    _transition_totals = (
        df_chen_transitions.groupby(["scenario", "year", "calibration"], as_index=False)[
            "area_m2"
        ]
        .sum()
        .assign(area_km2=lambda frame: frame["area_m2"].div(1e6))
    )
    _source_totals = (
        df_chen_transitions[df_chen_transitions["calibration"].eq("raw")]
        .groupby(["scenario", "from_class"], as_index=False)["area_m2"]
        .sum()
        .assign(area_km2=lambda frame: frame["area_m2"].div(1e6))
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.lineplot(
        data=_transition_totals,
        x="year",
        y="area_km2",
        hue="scenario",
        style="calibration",
        marker="o",
        ax=_axes[0],
    )
    _axes[0].set_title("Projected settlement transition area")
    _axes[0].set_xlabel("Chen scenario year")
    _axes[0].set_ylabel("Area (km^2)")

    sns.barplot(
        data=_source_totals,
        x="area_km2",
        y="from_class",
        hue="scenario",
        ax=_axes[1],
    )
    _axes[1].set_title("Raw projected source classes")
    _axes[1].set_xlabel("Area (km^2)")
    _axes[1].set_ylabel("Source class")

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Dataset semantics and source assumptions

    Before interpreting the diagnostics, the notebook records the exact Chen dataset semantics being used. The [GEE Community Catalog page](https://gee-community-catalog.org/projects/urban_projection/) describes the Chen et al. projection as a binary 1km urban/non-urban dataset for SSP1-SSP5 at 10-year intervals from 2020 through 2100. In this Earth Engine asset, pixel value `2` is urban and pixel value `1` is non-urban.

    The Chen paper, [Global projections of future urban land expansion under shared socioeconomic pathways](https://www.nature.com/articles/s41467-020-14386-x), frames these maps as future urban land expansion scenarios, not as a complete future land-use product. The paper also treats conversion from urban back to non-urban as constrained under declining urban demand, which is consistent with this notebook's monotonic first-expansion-year accounting.

    For this analysis, the working assumptions are:

    - Chen `urban` is used as the closest available future analogue for this project's `settlements` class.
    - Chen provides urban extent by decade, not annual transitions and not probabilities.
    - The notebook estimates only future `source class -> settlements` transitions.
    - The observed 2020 GLC-FCS30D-derived land-use raster remains the source of baseline land classes.
    - All outputs are diagnostic. Nothing here is treated as carbon-model-ready until the adequacy checks below are reviewed.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Historical settlement-growth plausibility

    A calibrated 2020 baseline is not enough. Chen could match observed 2020 settlement area and still imply future growth rates that are implausible for the zones in this project.

    This section compares Chen's projected decadal settlement expansion against observed historical settlement growth from the GLC-FCS30D-derived area tables. The main reference period is 2010-2020, because it is the most recent full decade available in the observed data. The 2000-2010 and 2000-2020 periods are kept as context.

    The resulting plausibility labels are intentionally conservative:

    - `consistent`: Chen decadal growth is between 0.25x and 4x the observed 2010-2020 growth.
    - `low_growth`: Chen growth is below 0.25x the recent observed decade.
    - `high_growth`: Chen growth is above 4x and up to 8x the recent observed decade.
    - `extreme_growth`: Chen growth is above 8x the recent observed decade.
    - `insufficient_history`: observed 2010-2020 settlement growth is smaller than 1 km^2, so ratios would be unstable.
    """)
    return


@app.cell
def _(
    GeoManager,
    HISTORICAL_GROWTH_PERIODS,
    MIN_HISTORICAL_GROWTH_AREA_M2,
    RECENT_GROWTH_PERIOD,
    df_chen_expansion,
    manager,
):
    def classify_growth_plausibility(
        ratio: float,
        historical_growth_area_m2: float,
    ) -> str:
        if historical_growth_area_m2 < MIN_HISTORICAL_GROWTH_AREA_M2:
            return "insufficient_history"
        if not np.isfinite(ratio):
            return "insufficient_history"
        if ratio < 0.25:
            return "low_growth"
        if ratio <= 4:
            return "consistent"
        if ratio <= 8:
            return "high_growth"
        return "extreme_growth"


    def build_historical_settlement_growth(manager: GeoManager) -> pd.DataFrame:
        settlement_area = manager.area_df["settlements"].unstack("year")
        rows = []
        for start_year, end_year in HISTORICAL_GROWTH_PERIODS:
            start_area = settlement_area[start_year]
            end_area = settlement_area[end_year]
            delta_area = end_area - start_area
            rows.append(
                pd.DataFrame(
                    {
                        "zone": settlement_area.index,
                        "period_start_year": start_year,
                        "year": end_year,
                        "start_settlement_area_m2": start_area.to_numpy(),
                        "end_settlement_area_m2": end_area.to_numpy(),
                        "historical_growth_area_m2": delta_area.to_numpy(),
                        "historical_growth_pct": delta_area.div(start_area.replace(0, np.nan)).to_numpy(),
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)


    def build_chen_growth_plausibility(
        df_expansion: pd.DataFrame,
        df_historical_growth: pd.DataFrame,
    ) -> pd.DataFrame:
        recent_start, recent_end = RECENT_GROWTH_PERIOD
        recent_growth = (
            df_historical_growth[
                df_historical_growth["period_start_year"].eq(recent_start)
                & df_historical_growth["year"].eq(recent_end)
            ][["zone", "historical_growth_area_m2", "historical_growth_pct"]]
            .rename(
                columns={
                    "historical_growth_area_m2": "recent_growth_area_m2",
                    "historical_growth_pct": "recent_growth_pct",
                }
            )
        )
        out = df_expansion.merge(recent_growth, on="zone", how="left")
        out["chen_to_recent_growth_ratio"] = out["chen_new_area_m2"].div(
            out["recent_growth_area_m2"].replace(0, np.nan)
        )
        out["growth_plausibility"] = [
            classify_growth_plausibility(ratio, growth)
            for ratio, growth in zip(
                out["chen_to_recent_growth_ratio"],
                out["recent_growth_area_m2"],
                strict=True,
            )
        ]
        return out


    def summarize_growth_plausibility(df: pd.DataFrame) -> pd.DataFrame:
        label_rank = {
            "consistent": 0,
            "low_growth": 1,
            "high_growth": 2,
            "extreme_growth": 3,
            "insufficient_history": 4,
        }
        rank_label = {rank: label for label, rank in label_rank.items()}
        summary = (
            df.assign(_rank=lambda frame: frame["growth_plausibility"].map(label_rank))
            .groupby(["zone", "scenario"], as_index=False)
            .agg(
                recent_growth_area_m2=("recent_growth_area_m2", "first"),
                max_chen_new_area_m2=("chen_new_area_m2", "max"),
                max_chen_to_recent_growth_ratio=("chen_to_recent_growth_ratio", "max"),
                worst_growth_rank=("_rank", "max"),
            )
        )
        summary["worst_growth_plausibility"] = summary["worst_growth_rank"].map(rank_label)
        return summary.drop(columns="worst_growth_rank")


    df_historical_settlement_growth = build_historical_settlement_growth(manager)
    df_chen_growth_plausibility = build_chen_growth_plausibility(
        df_chen_expansion,
        df_historical_settlement_growth,
    )
    df_growth_plausibility_summary = summarize_growth_plausibility(
        df_chen_growth_plausibility
    )

    assert df_historical_settlement_growth.shape[0] == len(manager.zones) * len(
        HISTORICAL_GROWTH_PERIODS
    )
    assert set(df_chen_growth_plausibility["growth_plausibility"]).issubset(
        {"consistent", "low_growth", "high_growth", "extreme_growth", "insufficient_history"}
    )

    mo.vstack(
        [
            mo.md("### Historical settlement growth"),
            df_historical_settlement_growth.head(10),
            mo.md("### Chen growth plausibility"),
            pd.crosstab(
                df_chen_growth_plausibility["scenario"],
                df_chen_growth_plausibility["growth_plausibility"],
            ),
        ]
    )
    return (
        df_chen_growth_plausibility,
        df_growth_plausibility_summary,
        df_historical_settlement_growth,
    )


@app.cell
def _(df_chen_growth_plausibility, df_historical_settlement_growth):
    _history_plot_df = df_historical_settlement_growth.assign(
        historical_growth_km2=lambda frame: frame["historical_growth_area_m2"].div(1e6),
        period=lambda frame: frame["period_start_year"].astype(str) + "-" + frame["year"].astype(str),
    )
    _growth_plot_df = df_chen_growth_plausibility.assign(
        chen_new_area_km2=lambda frame: frame["chen_new_area_m2"].div(1e6),
        recent_growth_km2=lambda frame: frame["recent_growth_area_m2"].div(1e6),
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.boxplot(data=_history_plot_df, x="period", y="historical_growth_km2", ax=_axes[0])
    _axes[0].axhline(0, color="k", linewidth=1)
    _axes[0].set_title("Observed settlement growth by period")
    _axes[0].set_xlabel("Historical period")
    _axes[0].set_ylabel("Growth (km^2)")

    sns.scatterplot(
        data=_growth_plot_df,
        x="recent_growth_km2",
        y="chen_new_area_km2",
        hue="growth_plausibility",
        style="scenario",
        s=30,
        ax=_axes[1],
    )
    _limit = max(_growth_plot_df["recent_growth_km2"].max(), _growth_plot_df["chen_new_area_km2"].max())
    _axes[1].plot([0, _limit], [0, _limit], "k--", linewidth=1)
    _axes[1].set_title("Chen expansion vs recent observed growth")
    _axes[1].set_xlabel("Observed 2010-2020 growth (km^2)")
    _axes[1].set_ylabel("Chen decadal expansion (km^2)")

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Sensitive source-class inspection

    The carbon relevance of Chen expansion depends heavily on which observed classes are replaced by future settlements. Before any carbon-model run, the notebook explicitly checks whether projected settlement expansion draws from sensitive classes.

    The strict sensitive group is `forests_primary`, `forests_mangroves`, and `wetlands`. `forests_secondary` is tracked separately as a watch class: it can be carbon-relevant, but it is broader and more common in the current classification.

    The sensitive flag is based on the share of transition area from the strict sensitive group:

    - `low`: below 5%.
    - `watch`: from 5% through 10%.
    - `high`: above 10%.
    """)
    return


@app.cell
def _(SENSITIVE_CLASSES, WATCH_CLASSES, df_chen_transitions):
    def build_sensitive_transition_summary(df_transitions: pd.DataFrame) -> pd.DataFrame:
        totals = (
            df_transitions.groupby(
                ["zone", "scenario", "year", "calibration"],
                as_index=False,
            )["area_m2"]
            .sum()
            .rename(columns={"area_m2": "total_transition_area_m2"})
        )
        grouped = (
            df_transitions.assign(
                sensitivity_group=lambda frame: np.select(
                    [
                        frame["from_class"].isin(SENSITIVE_CLASSES),
                        frame["from_class"].isin(WATCH_CLASSES),
                    ],
                    ["sensitive", "watch"],
                    default="other",
                )
            )
            .groupby(
                ["zone", "scenario", "year", "calibration", "sensitivity_group"],
                as_index=False,
            )["area_m2"]
            .sum()
        )
        pivot = (
            grouped.pivot_table(
                index=["zone", "scenario", "year", "calibration"],
                columns="sensitivity_group",
                values="area_m2",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
            .rename_axis(columns=None)
        )
        for column in ["sensitive", "watch", "other"]:
            if column not in pivot:
                pivot[column] = 0.0
        out = totals.merge(pivot, on=["zone", "scenario", "year", "calibration"], how="left")
        out[["sensitive", "watch", "other"]] = out[["sensitive", "watch", "other"]].fillna(0)
        out = out.rename(
            columns={
                "sensitive": "sensitive_area_m2",
                "watch": "watch_area_m2",
                "other": "other_area_m2",
            }
        )
        out["sensitive_share"] = out["sensitive_area_m2"].div(
            out["total_transition_area_m2"].replace(0, np.nan)
        ).fillna(0)
        out["watch_share"] = out["watch_area_m2"].div(
            out["total_transition_area_m2"].replace(0, np.nan)
        ).fillna(0)
        out["sensitive_flag"] = np.select(
            [out["sensitive_share"].gt(0.10), out["sensitive_share"].ge(0.05)],
            ["high", "watch"],
            default="low",
        )
        return out


    def summarize_sensitive_flags(df: pd.DataFrame) -> pd.DataFrame:
        flag_rank = {"low": 0, "watch": 1, "high": 2}
        rank_flag = {rank: flag for flag, rank in flag_rank.items()}
        summary = (
            df[df["calibration"].eq("raw")]
            .assign(_rank=lambda frame: frame["sensitive_flag"].map(flag_rank))
            .groupby(["zone", "scenario"], as_index=False)
            .agg(
                max_sensitive_share=("sensitive_share", "max"),
                max_watch_share=("watch_share", "max"),
                max_sensitive_area_m2=("sensitive_area_m2", "max"),
                worst_sensitive_rank=("_rank", "max"),
            )
        )
        summary["worst_sensitive_flag"] = summary["worst_sensitive_rank"].map(rank_flag)
        return summary.drop(columns="worst_sensitive_rank")


    df_sensitive_transition_summary = build_sensitive_transition_summary(df_chen_transitions)
    df_sensitive_flag_summary = summarize_sensitive_flags(df_sensitive_transition_summary)

    assert df_sensitive_transition_summary["sensitive_share"].between(0, 1).all()
    assert df_sensitive_transition_summary["watch_share"].between(0, 1).all()
    assert set(df_sensitive_transition_summary["sensitive_flag"]).issubset({"low", "watch", "high"})

    mo.vstack(
        [
            mo.md("### Sensitive transition shares"),
            pd.crosstab(
                df_sensitive_transition_summary["scenario"],
                df_sensitive_transition_summary["sensitive_flag"],
            ),
            df_sensitive_transition_summary.head(10),
        ]
    )
    return df_sensitive_flag_summary, df_sensitive_transition_summary


@app.cell
def _(df_sensitive_transition_summary):
    _sensitive_plot_df = (
        df_sensitive_transition_summary[df_sensitive_transition_summary["calibration"].eq("raw")]
        .groupby(["scenario", "year"], as_index=False)[["sensitive_area_m2", "watch_area_m2", "total_transition_area_m2"]]
        .sum()
        .assign(
            sensitive_share=lambda frame: frame["sensitive_area_m2"].div(frame["total_transition_area_m2"].replace(0, np.nan)).fillna(0),
            watch_share=lambda frame: frame["watch_area_m2"].div(frame["total_transition_area_m2"].replace(0, np.nan)).fillna(0),
            sensitive_area_km2=lambda frame: frame["sensitive_area_m2"].div(1e6),
            watch_area_km2=lambda frame: frame["watch_area_m2"].div(1e6),
        )
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.lineplot(data=_sensitive_plot_df, x="year", y="sensitive_share", hue="scenario", marker="o", ax=_axes[0])
    _axes[0].axhline(0.05, color="k", linestyle="--", linewidth=1)
    _axes[0].axhline(0.10, color="k", linestyle=":", linewidth=1)
    _axes[0].set_title("Sensitive-class share of raw transitions")
    _axes[0].set_xlabel("Chen scenario year")
    _axes[0].set_ylabel("Sensitive share")

    _sens_long = _sensitive_plot_df.melt(
        id_vars=["scenario", "year"],
        value_vars=["sensitive_area_km2", "watch_area_km2"],
        var_name="group",
        value_name="area_km2",
    )
    sns.lineplot(data=_sens_long, x="year", y="area_km2", hue="scenario", style="group", marker="o", ax=_axes[1])
    _axes[1].set_title("Sensitive and watch transition area")
    _axes[1].set_xlabel("Chen scenario year")
    _axes[1].set_ylabel("Area (km^2)")

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Area, spatial, and readiness screens

    The notebook reports separate diagnostic screens instead of collapsing the land estimates into a single pass/fail metric.

    The main fields are:

    - `calibration_adequacy`: do Chen and observed 2020 settlements agree well enough in area and space?
    - `growth_risk`: does Chen's future expansion look too high relative to recent observed settlement growth?
    - `sensitive_class_risk`: is projected expansion drawing substantially from primary forests, mangroves, or wetlands?
    - `land_estimate_readiness`: the primary readiness label for deciding what should be reviewed next.
    - `manual_review_priority`: which cases deserve visual inspection first.

    Low Chen growth is not treated as a failure. It can be a conservative or low-expansion SSP outcome. High and extreme growth remain review concerns.
    """)
    return


@app.cell
def _(
    GeoManager,
    SOURCE_YEAR,
    df_calibration,
    df_growth_plausibility_summary,
    df_sensitive_flag_summary,
    manager,
):
    SENSITIVE_AREA_RISK_THRESHOLD_M2 = 1_000_000
    WATCH_AREA_RISK_THRESHOLD_M2 = 5_000_000


    def classify_area_adequacy(area_bias: float, ape: float) -> str:
        if not np.isfinite(area_bias) or not np.isfinite(ape):
            return "poor"
        if 0.5 <= area_bias <= 2 and ape <= 0.5:
            return "good"
        if 0.25 <= area_bias <= 4 and ape <= 1.5:
            return "moderate"
        return "poor"


    def classify_spatial_adequacy(iou: float, precision: float) -> str:
        if not np.isfinite(iou) or not np.isfinite(precision):
            return "poor"
        if iou >= 0.25 and precision >= 0.5:
            return "good"
        if iou >= 0.10 and precision >= 0.25:
            return "moderate"
        return "poor"


    def classify_calibration_adequacy(row: pd.Series) -> str:
        if row["area_adequacy"] == "poor" or row["spatial_adequacy"] == "poor":
            return "poor"
        if row["area_adequacy"] == "good" and row["spatial_adequacy"] == "good" and row["reliability"] == "high":
            return "good"
        return "moderate"


    def classify_growth_risk(plausibility: str) -> str:
        if plausibility in {"consistent", "low_growth"}:
            return "low"
        if plausibility == "high_growth":
            return "watch"
        if plausibility == "extreme_growth":
            return "high"
        return "review"


    def classify_sensitive_class_risk(
        sensitive_share: float,
        sensitive_area_m2: float,
        watch_share: float,
    ) -> str:
        if sensitive_area_m2 >= SENSITIVE_AREA_RISK_THRESHOLD_M2 and sensitive_share > 0.10:
            return "high"
        if sensitive_area_m2 >= SENSITIVE_AREA_RISK_THRESHOLD_M2 and sensitive_share >= 0.05:
            return "watch"
        if watch_share >= 0.25 or sensitive_share >= 0.05:
            return "watch"
        return "low"


    def classify_land_estimate_readiness(row: pd.Series) -> str:
        if row["calibration_adequacy"] == "poor" or row["growth_risk"] == "high" or row["sensitive_class_risk"] == "high":
            return "not_ready"
        if row["calibration_adequacy"] == "good" and row["growth_risk"] == "low" and row["sensitive_class_risk"] == "low":
            return "ready_for_manual_review"
        return "needs_targeted_review"


    def classify_manual_review_priority(row: pd.Series) -> str:
        if row["land_estimate_readiness"] == "not_ready":
            return "high"
        if row["growth_risk"] in {"watch", "review"} or row["sensitive_class_risk"] == "watch" or row["calibration_adequacy"] == "moderate":
            return "medium"
        return "low"


    def build_zone_context(manager: GeoManager) -> pd.DataFrame:
        area_2020 = manager.area_df.xs(SOURCE_YEAR, level="year")
        out = pd.DataFrame(
            {
                "zone": area_2020.index,
                "observed_settlement_area_2020_m2": area_2020["settlements"],
                "observed_total_area_2020_m2": area_2020.sum(axis=1),
            }
        ).reset_index(drop=True)
        out["observed_settlement_fraction_2020"] = out[
            "observed_settlement_area_2020_m2"
        ].div(out["observed_total_area_2020_m2"].replace(0, np.nan))
        return out


    df_zone_context = build_zone_context(manager)
    df_land_estimation_assessment = (
        df_calibration.merge(df_zone_context, on="zone", how="left")
        .merge(df_growth_plausibility_summary, on=["zone", "scenario"], how="left")
        .merge(df_sensitive_flag_summary, on=["zone", "scenario"], how="left")
    )

    df_land_estimation_assessment["area_adequacy"] = [
        classify_area_adequacy(area_bias, ape)
        for area_bias, ape in zip(
            df_land_estimation_assessment["area_bias"],
            df_land_estimation_assessment["ape"],
            strict=True,
        )
    ]
    df_land_estimation_assessment["spatial_adequacy"] = [
        classify_spatial_adequacy(iou, precision)
        for iou, precision in zip(
            df_land_estimation_assessment["iou"],
            df_land_estimation_assessment["precision"],
            strict=True,
        )
    ]
    df_land_estimation_assessment["calibration_adequacy"] = df_land_estimation_assessment.apply(
        classify_calibration_adequacy,
        axis=1,
    )
    df_land_estimation_assessment["growth_risk"] = df_land_estimation_assessment[
        "worst_growth_plausibility"
    ].map(classify_growth_risk)
    df_land_estimation_assessment["sensitive_class_risk"] = [
        classify_sensitive_class_risk(sensitive_share, sensitive_area_m2, watch_share)
        for sensitive_share, sensitive_area_m2, watch_share in zip(
            df_land_estimation_assessment["max_sensitive_share"].fillna(0),
            df_land_estimation_assessment["max_sensitive_area_m2"].fillna(0),
            df_land_estimation_assessment["max_watch_share"].fillna(0),
            strict=True,
        )
    ]
    df_land_estimation_assessment["land_estimate_readiness"] = df_land_estimation_assessment.apply(
        classify_land_estimate_readiness,
        axis=1,
    )
    df_land_estimation_assessment["manual_review_priority"] = df_land_estimation_assessment.apply(
        classify_manual_review_priority,
        axis=1,
    )
    # Compatibility alias for older downstream cells in this notebook.
    df_land_estimation_assessment["overall_assessment"] = df_land_estimation_assessment[
        "land_estimate_readiness"
    ]

    assert df_land_estimation_assessment.shape[0] == df_calibration.shape[0]
    assert set(df_land_estimation_assessment["area_adequacy"]).issubset({"good", "moderate", "poor"})
    assert set(df_land_estimation_assessment["spatial_adequacy"]).issubset({"good", "moderate", "poor"})
    assert set(df_land_estimation_assessment["calibration_adequacy"]).issubset({"good", "moderate", "poor"})
    assert set(df_land_estimation_assessment["growth_risk"]).issubset({"low", "watch", "high", "review"})
    assert set(df_land_estimation_assessment["sensitive_class_risk"]).issubset({"low", "watch", "high"})
    assert set(df_land_estimation_assessment["land_estimate_readiness"]).issubset(
        {"ready_for_manual_review", "needs_targeted_review", "not_ready"}
    )
    assert set(df_land_estimation_assessment["manual_review_priority"]).issubset({"low", "medium", "high"})

    mo.vstack(
        [
            mo.md("### Land-estimate readiness counts"),
            pd.crosstab(
                df_land_estimation_assessment["scenario"],
                df_land_estimation_assessment["land_estimate_readiness"],
            ),
            mo.md("### Review priority"),
            pd.crosstab(
                df_land_estimation_assessment["scenario"],
                df_land_estimation_assessment["manual_review_priority"],
            ),
            df_land_estimation_assessment.head(10),
        ]
    )
    return (df_land_estimation_assessment,)


@app.cell
def _(df_land_estimation_assessment):
    _assessment_plot_df = df_land_estimation_assessment.assign(
        observed_settlement_area_2020_km2=lambda frame: frame["observed_settlement_area_2020_m2"].div(1e6),
    )

    _fig, _axes = plt.subplots(1, 3, figsize=(17, 5))

    sns.scatterplot(
        data=_assessment_plot_df,
        x="observed_settlement_area_2020_km2",
        y="ape",
        hue="land_estimate_readiness",
        style="scenario",
        s=30,
        ax=_axes[0],
    )
    _axes[0].set_xscale("log")
    _axes[0].set_title("Area error vs settlement size")
    _axes[0].set_xlabel("Observed 2020 settlements (km^2)")
    _axes[0].set_ylabel("APE")

    sns.scatterplot(
        data=_assessment_plot_df,
        x="observed_settlement_fraction_2020",
        y="iou",
        hue="land_estimate_readiness",
        style="scenario",
        s=30,
        legend=False,
        ax=_axes[1],
    )
    _axes[1].set_title("Spatial overlap vs settlement fraction")
    _axes[1].set_xlabel("2020 settlement fraction")
    _axes[1].set_ylabel("IoU")

    sns.scatterplot(
        data=_assessment_plot_df,
        x="max_chen_to_recent_growth_ratio",
        y="max_sensitive_share",
        hue="manual_review_priority",
        style="scenario",
        s=30,
        ax=_axes[2],
    )
    _axes[2].set_xscale("log")
    _axes[2].axhline(0.05, color="k", linestyle="--", linewidth=1)
    _axes[2].axhline(0.10, color="k", linestyle=":", linewidth=1)
    _axes[2].set_title("Growth ratio vs sensitive share")
    _axes[2].set_xlabel("Max Chen / recent growth ratio")
    _axes[2].set_ylabel("Max sensitive share")

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
    GeoManager,
    SCALE_SENSITIVITY_THRESHOLDS,
    SOURCE_YEAR,
    SSP_NAMES,
    Zone,
    chen_urban_mask,
    manager,
    observed_settlement_fraction_image,
    safe_ratio,
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
def _(df_land_estimation_assessment, df_scale_sensitivity):
    _scale_plot_df = df_scale_sensitivity.merge(
        df_land_estimation_assessment[["zone", "scenario", "overall_assessment"]],
        on=["zone", "scenario"],
        how="left",
    )

    _fig, _axes = plt.subplots(1, 3, figsize=(16, 5))
    for _axis, _metric in zip(_axes, ["precision", "recall", "iou"], strict=True):
        sns.lineplot(data=_scale_plot_df, x="threshold", y=_metric, hue="scenario", marker="o", ax=_axis)
        _axis.set_title(f"{_metric.title()} by observed-settlement threshold")
        _axis.set_xlabel("Observed settlement fraction threshold")
        _axis.set_ylabel(_metric.title())

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Review candidates and provisional acceptance policy

    The notebook treats land-estimate adequacy as a diagnostic workflow rather than a pass/fail gate.

    Recommended interpretation:

    - `ready_for_manual_review`: calibration is good, growth risk is low, and sensitive-class risk is low. These are the first candidates for map inspection.
    - `needs_targeted_review`: at least one dimension needs attention, but the result is not automatically unusable.
    - `not_ready`: calibration is poor, growth is extreme, or sensitive-class risk is high. Do not use these for carbon modeling without method changes or external justification.

    The review-candidate table intentionally includes multiple reasons so manual review does not focus only on the worst failures.
    """)
    return


@app.cell
def _(df_land_estimation_assessment):
    def build_review_candidates(df_assessment: pd.DataFrame) -> pd.DataFrame:
        base = df_assessment.copy()
        base["max_sensitive_share"] = base["max_sensitive_share"].fillna(0)
        selections = []
        for _scenario, group in base.groupby("scenario"):
            selections.append(group.nsmallest(3, "iou").assign(review_reason="lowest_iou"))
            selections.append(group.nlargest(3, "correction_factor").assign(review_reason="largest_correction_factor"))
            growth_group = group.replace([np.inf, -np.inf], np.nan).dropna(
                subset=["max_chen_to_recent_growth_ratio"]
            )
            if not growth_group.empty:
                selections.append(
                    growth_group.nlargest(3, "max_chen_to_recent_growth_ratio").assign(
                        review_reason="largest_growth_ratio"
                    )
                )
            selections.append(group.nlargest(3, "max_sensitive_share").assign(review_reason="largest_sensitive_share"))
            for readiness in ["ready_for_manual_review", "needs_targeted_review", "not_ready"]:
                readiness_group = group[group["land_estimate_readiness"].eq(readiness)]
                if not readiness_group.empty:
                    selections.append(
                        readiness_group.iloc[[len(readiness_group) // 2]].assign(
                            review_reason=f"representative_{readiness}"
                        )
                    )
        out = pd.concat(selections, ignore_index=True)
        return (
            out[
                [
                    "zone",
                    "scenario",
                    "review_reason",
                    "land_estimate_readiness",
                    "manual_review_priority",
                    "calibration_adequacy",
                    "growth_risk",
                    "sensitive_class_risk",
                    "reliability",
                    "area_adequacy",
                    "spatial_adequacy",
                    "iou",
                    "area_bias",
                    "correction_factor",
                    "worst_growth_plausibility",
                    "max_chen_to_recent_growth_ratio",
                    "worst_sensitive_flag",
                    "max_sensitive_share",
                    "max_sensitive_area_m2",
                ]
            ]
            .drop_duplicates()
            .sort_values(["scenario", "review_reason", "zone"])
            .reset_index(drop=True)
        )


    df_review_candidates = build_review_candidates(df_land_estimation_assessment)

    assert not df_review_candidates.empty

    mo.vstack(
        [
            mo.md("### Review candidates"),
            df_review_candidates,
            mo.md("### Land-estimate readiness counts"),
            pd.crosstab(
                df_land_estimation_assessment["scenario"],
                df_land_estimation_assessment["land_estimate_readiness"],
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Spatial sanity check

    The map provides a visual check for one representative zone. It overlays Chen's 2020 urban layer with the observed 2020 settlement mask:

    - red: Chen SSP1 urban pixels in 2020.
    - green: observed settlements from the project's 2020 land-use raster.

    Use `df_review_candidates` to choose zones for inspection. Start with candidates flagged for `lowest_iou`, `largest_correction_factor`, or `largest_sensitive_share`, then compare the map with the zone's calibration and assessment rows.

    This map is not evidence that Chen can locate 30m future transitions. It is a sanity check that the calibration metrics correspond to a believable spatial pattern.
    """)
    return


@app.cell
def _(SETTLEMENT_IDX, SOURCE_YEAR, manager):
    _example_zone_name = "01.1.01"
    _example_zone = manager[_example_zone_name]

    m = leafmap.Map()
    m.add_ee_layer(
        _example_zone.ssp_images["SSP1"].select(str(SOURCE_YEAR)),
        {"min": 0, "max": 1, "palette": ["red"]},
        name=f"Chen SSP1 {SOURCE_YEAR} urban",
    )
    m.add_ee_layer(
        _example_zone.area_raster.select(str(SOURCE_YEAR))
        .eq(ee.Number(SETTLEMENT_IDX))
        .selfMask(),
        {"min": 0, "max": 1, "palette": ["green"]},
        name=f"Observed {SOURCE_YEAR} settlements",
    )
    m.add_layer_control()
    m
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Final recommendation and decision log

    This section turns the diagnostic tables into a short decision record. The goal is not to approve Chen-derived transitions for the carbon model yet; it is to identify which zone-scenario pairs are reasonable enough for manual spatial review.

    Recommended workflow:

    1. Start with `ready_for_manual_review` rows, especially those with low manual review priority.
    2. Inspect `df_review_candidates` and the map for representative ready, targeted-review, and not-ready cases.
    3. Treat `needs_targeted_review` rows as conditionally usable only after the specific issue is understood.
    4. Do not use `not_ready` rows downstream without changing the method or adding external justification.
    5. Once manual review is complete, create a separate approved subset for any future carbon-model input.
    """)
    return


@app.cell
def _(df_land_estimation_assessment):
    df_final_readiness_summary = (
        pd.crosstab(
            df_land_estimation_assessment["scenario"],
            df_land_estimation_assessment["land_estimate_readiness"],
        )
        .reindex(
            columns=["ready_for_manual_review", "needs_targeted_review", "not_ready"],
            fill_value=0,
        )
        .assign(
            total=lambda frame: frame.sum(axis=1),
            ready_share=lambda frame: frame["ready_for_manual_review"].div(frame["total"]),
            not_ready_share=lambda frame: frame["not_ready"].div(frame["total"]),
        )
    )

    df_final_ssp_summary = (
        df_land_estimation_assessment.groupby("scenario")
        .agg(
            ready_for_manual_review=(
                "land_estimate_readiness",
                lambda series: series.eq("ready_for_manual_review").sum(),
            ),
            needs_targeted_review=(
                "land_estimate_readiness",
                lambda series: series.eq("needs_targeted_review").sum(),
            ),
            not_ready=(
                "land_estimate_readiness",
                lambda series: series.eq("not_ready").sum(),
            ),
            high_review_priority=(
                "manual_review_priority",
                lambda series: series.eq("high").sum(),
            ),
            median_iou=("iou", "median"),
            median_area_bias=("area_bias", "median"),
            median_correction_factor=("correction_factor", "median"),
            max_sensitive_share=("max_sensitive_share", "max"),
        )
        .assign(
            ready_share=lambda frame: frame["ready_for_manual_review"].div(
                frame[["ready_for_manual_review", "needs_targeted_review", "not_ready"]].sum(axis=1)
            ),
            not_ready_share=lambda frame: frame["not_ready"].div(
                frame[["ready_for_manual_review", "needs_targeted_review", "not_ready"]].sum(axis=1)
            ),
        )
        .sort_values(
            ["ready_share", "not_ready_share", "median_iou"],
            ascending=[False, True, False],
        )
        .round(3)
    )

    df_final_problematic_zones = (
        df_land_estimation_assessment.assign(
            is_not_ready=lambda frame: frame["land_estimate_readiness"].eq("not_ready"),
            is_high_priority=lambda frame: frame["manual_review_priority"].eq("high"),
        )
        .groupby("zone")
        .agg(
            not_ready_scenarios=("is_not_ready", "sum"),
            high_priority_scenarios=("is_high_priority", "sum"),
            min_iou=("iou", "min"),
            max_correction_factor=("correction_factor", "max"),
            max_growth_ratio=("max_chen_to_recent_growth_ratio", "max"),
            max_sensitive_share=("max_sensitive_share", "max"),
        )
        .query("not_ready_scenarios > 0 or high_priority_scenarios > 0")
        .sort_values(
            ["not_ready_scenarios", "high_priority_scenarios", "min_iou"],
            ascending=[False, False, True],
        )
        .round(3)
    )

    df_final_ready_zones = (
        df_land_estimation_assessment.assign(
            is_ready=lambda frame: frame["land_estimate_readiness"].eq("ready_for_manual_review"),
        )
        .groupby("zone")
        .agg(
            ready_scenarios=("is_ready", "sum"),
            median_iou=("iou", "median"),
            median_area_bias=("area_bias", "median"),
            max_sensitive_share=("max_sensitive_share", "max"),
        )
        .query("ready_scenarios > 0")
        .sort_values(["ready_scenarios", "median_iou"], ascending=[False, False])
        .round(3)
    )

    _best_scenario = df_final_ssp_summary.index[0]
    _ready_total = int(df_land_estimation_assessment["land_estimate_readiness"].eq("ready_for_manual_review").sum())
    _review_total = int(df_land_estimation_assessment["land_estimate_readiness"].eq("needs_targeted_review").sum())
    _not_ready_total = int(df_land_estimation_assessment["land_estimate_readiness"].eq("not_ready").sum())

    mo.vstack(
        [
            mo.md(
                f"""
    ### Decision summary

    Across all SSPs and zones, `{_ready_total}` zone-scenario pairs are ready for manual review, `{_review_total}` need targeted review, and `{_not_ready_total}` are not ready. By the current diagnostic screen, `{_best_scenario}` has the strongest overall readiness profile.

    The next step should be manual inspection, not carbon-model integration. Use `df_review_candidates` to choose map examples, then use `df_final_ready_zones` and `df_final_problematic_zones` to decide which zone-scenario pairs are worth carrying forward.
    """
            ),
            mo.md("### Readiness by SSP"),
            df_final_readiness_summary,
            mo.md("### SSP usability summary"),
            df_final_ssp_summary,
            mo.md("### Consistently problematic zones"),
            df_final_problematic_zones.head(15),
            mo.md("### Zones with at least one ready scenario"),
            df_final_ready_zones.head(15),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
