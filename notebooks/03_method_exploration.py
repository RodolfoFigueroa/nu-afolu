import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")

with app.setup:
    import os
    from pathlib import Path

    import ee
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from dagster_components.partitions import zone_partitions

    from nu_afolu.artifact_validation import (
        raise_for_validation_errors,
        validate_calibration_artifacts,
        validate_exploration_artifacts,
    )
    from nu_afolu.chen import (
        CHEN_COLLECTION_ID,
        GeoManager,
        SSP_NAMES,
        Zone,
        chen_urban_mask,
        load_chen_manager,
        observed_settlement_fraction_image,
    )
    from nu_afolu.constants import LABEL_LIST
    from nu_afolu.metrics import add_area_calibration_fields
    from nu_afolu.utils import safe_ratio

    ee.Initialize()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Chen SSP Method Exploration

    This notebook compares candidate calibration methods for Chen's 2020 urban baseline against the current canonical calibration from `01_calibration.py`. It is exploratory: it writes separate method-comparison artifacts and does not alter the calibration handoff consumed by `02_transition_closure.py`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Provenance scope

    This exploration notebook reuses the same baseline evidence as the calibration and transition notebooks. It does not introduce a new observed settlement source.

    - Observed settlement is still the 2020 `settlements` class from the upstream GLC-FCS30D-derived `area_raster`; the expected historical decision window remains 2000 through 2020.
    - Canonical calibration inputs are read from `OUT_PATH/chen/calibration.parquet` and `OUT_PATH/chen/scale_sensitivity.parquet`.
    - New method-comparison artifacts are written under `OUT_PATH/chen/exploration/` only. They are exploratory diagnostics for stress-testing adequacy, not a replacement for the canonical calibration handoff.

    The full provenance and artifact contract is documented in `docs/data_provenance.md`.
    """)
    return


@app.cell
def _():
    LABEL_MAP = dict(enumerate(LABEL_LIST, start=1))
    LABEL_ID_BY_NAME = {label: idx for idx, label in LABEL_MAP.items()}
    SETTLEMENT_IDX = LABEL_ID_BY_NAME["settlements"]

    SOURCE_YEAR = 2020
    CORRECTION_FACTOR_BOUNDS = (0.25, 4.0)
    SCALE_SENSITIVITY_THRESHOLDS = [0.10, 0.25, 0.50]
    BUFFER_OBSERVED_THRESHOLD = 0.25
    BUFFER_DISTANCES_M = [1000, 2000]

    METHOD_COLUMNS = [
        "zone",
        "scenario",
        "method",
        "observed_threshold",
        "buffer_m",
        "observed_area_m2",
        "chen_area_m2",
        "area_error_m2",
        "area_bias",
        "ape",
        "correction_factor_raw",
        "correction_factor",
        "calibration_valid",
        "precision",
        "recall",
        "iou",
        "buffered_precision",
        "buffered_recall",
        "buffered_f1",
        "spatial_metric_name",
        "spatial_score",
        "valid_comparison",
    ]
    return (
        BUFFER_DISTANCES_M,
        BUFFER_OBSERVED_THRESHOLD,
        CORRECTION_FACTOR_BOUNDS,
        METHOD_COLUMNS,
        SCALE_SENSITIVITY_THRESHOLDS,
        SETTLEMENT_IDX,
        SOURCE_YEAR,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Inputs
    """)
    return


@app.cell
def _():
    out_path = Path(os.environ["OUT_PATH"])
    chen_artifact_dir = out_path / "chen"
    exploration_artifact_dir = chen_artifact_dir / "exploration"
    col_chen = ee.ImageCollection(CHEN_COLLECTION_ID)
    return chen_artifact_dir, col_chen, exploration_artifact_dir, out_path


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load zone rasters

    The exploration notebook reloads the same historical rasters and Chen collection as the calibration and transition notebooks. It keeps all new outputs under `OUT_PATH/chen/exploration/` so the canonical handoff stays unchanged.
    """)
    return


@app.cell
def _(SETTLEMENT_IDX, col_chen, out_path):
    manager, df_missing_zones = load_chen_manager(
        out_path,
        zone_partitions.get_partition_keys(),
        col_chen,
        SETTLEMENT_IDX,
    )

    mo.vstack(
        [
            mo.md("### Loaded zones"),
            pd.DataFrame(
                [
                    {
                        "loaded_zones": len(manager.zones),
                        "missing_zones": len(df_missing_zones),
                    }
                ]
            ),
            df_missing_zones.head(10),
        ]
    )
    return (manager,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load canonical calibration artifacts

    `01_calibration.py` remains the owner of the canonical calibration handoff. This notebook reads those outputs as baselines, then writes only exploration-specific comparisons.
    """)
    return


@app.cell
def _(SCALE_SENSITIVITY_THRESHOLDS, chen_artifact_dir, manager):
    calibration_path = chen_artifact_dir / "calibration.parquet"
    scale_sensitivity_path = chen_artifact_dir / "scale_sensitivity.parquet"

    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Missing calibration artifact: {calibration_path}. Run 01_calibration.py first."
        )
    if not scale_sensitivity_path.exists():
        raise FileNotFoundError(
            f"Missing scale-sensitivity artifact: {scale_sensitivity_path}. Run 01_calibration.py first."
        )

    df_calibration = pd.read_parquet(calibration_path)
    df_scale_sensitivity = pd.read_parquet(scale_sensitivity_path)

    _validation_report = validate_calibration_artifacts(
        df_calibration,
        df_scale_sensitivity,
        zone_names=manager.zones,
        thresholds=SCALE_SENSITIVITY_THRESHOLDS,
    )
    raise_for_validation_errors(_validation_report)
    _validation_summary = (
        _validation_report
        if not _validation_report.empty
        else pd.DataFrame(
            [
                {
                    "artifact": "canonical_calibration_inputs",
                    "check": "validation",
                    "severity": "pass",
                    "message": "Canonical calibration artifact checks passed.",
                    "rows": 0,
                }
            ]
        )
    )

    mo.vstack(
        [
            mo.md("### Input artifact validation"),
            _validation_summary,
            mo.md("### Input artifacts"),
            pd.DataFrame(
                [
                    {
                        "artifact": "calibration",
                        "path": str(calibration_path),
                        "rows": len(df_calibration),
                    },
                    {
                        "artifact": "scale_sensitivity",
                        "path": str(scale_sensitivity_path),
                        "rows": len(df_scale_sensitivity),
                    },
                ]
            ),
        ]
    )
    return df_calibration, df_scale_sensitivity


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Method tables

    The comparison table normalizes each candidate into one row per `zone, scenario, method`. Fractional and threshold methods use ordinary IoU as their spatial score. Buffered methods use a tolerance-aware F1 score built from directional buffered precision and recall.
    """)
    return


@app.cell
def _(CORRECTION_FACTOR_BOUNDS, METHOD_COLUMNS):
    def threshold_method_name(threshold: float) -> str:
        return f"threshold_{int(round(threshold * 100)):02d}"


    def build_canonical_method_table(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["method"] = "fractional_current"
        out["observed_threshold"] = np.nan
        out["buffer_m"] = np.nan
        out["buffered_precision"] = np.nan
        out["buffered_recall"] = np.nan
        out["buffered_f1"] = np.nan
        out["spatial_metric_name"] = "iou"
        out["spatial_score"] = out["iou"]
        out["valid_comparison"] = out["calibration_valid"] & np.isfinite(
            out["spatial_score"]
        )
        return out[METHOD_COLUMNS]


    def build_threshold_method_table(df: pd.DataFrame) -> pd.DataFrame:
        out = add_area_calibration_fields(
            df,
            correction_factor_bounds=CORRECTION_FACTOR_BOUNDS,
        )
        out["method"] = out["threshold"].map(threshold_method_name)
        out["observed_threshold"] = out["threshold"]
        out["buffer_m"] = np.nan
        out["buffered_precision"] = np.nan
        out["buffered_recall"] = np.nan
        out["buffered_f1"] = np.nan
        out["spatial_metric_name"] = "iou"
        out["spatial_score"] = out["iou"]
        out["valid_comparison"] = out["calibration_valid"] & np.isfinite(
            out["spatial_score"]
        )
        return out[METHOD_COLUMNS]


    threshold_method_name
    return (
        build_canonical_method_table,
        build_threshold_method_table,
        threshold_method_name,
    )


@app.cell
def _(
    build_canonical_method_table,
    build_threshold_method_table,
    df_calibration,
    df_scale_sensitivity,
):
    df_canonical_method = build_canonical_method_table(df_calibration)
    df_threshold_methods = build_threshold_method_table(df_scale_sensitivity)

    assert set(df_canonical_method["method"]) == {"fractional_current"}
    assert set(df_threshold_methods["method"]) == {
        "threshold_10",
        "threshold_25",
        "threshold_50",
    }

    mo.vstack(
        [
            mo.md("### Canonical and threshold methods"),
            pd.concat([df_canonical_method, df_threshold_methods], ignore_index=True)
            .groupby("method")
            .agg(
                rows=("zone", "count"),
                median_iou=("iou", "median"),
                median_area_bias=("area_bias", "median"),
                median_correction_factor=("correction_factor", "median"),
            )
            .round(3),
        ]
    )
    return df_canonical_method, df_threshold_methods


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Buffered agreement methods

    Buffered methods keep the 25% observed-settlement threshold but allow spatial tolerance on Chen's 1 km grid. They report two directional quantities: the share of Chen urban area near observed settlement, and the share of observed settlement area near Chen urban. Their harmonic mean is the method's spatial score.
    """)
    return


@app.cell
def _(
    BUFFER_DISTANCES_M,
    BUFFER_OBSERVED_THRESHOLD,
    CORRECTION_FACTOR_BOUNDS,
    METHOD_COLUMNS,
    SOURCE_YEAR,
    threshold_method_name,
):
    def buffered_method_name(threshold: float, buffer_m: int) -> str:
        return f"{threshold_method_name(threshold)}_buffer_{int(buffer_m)}m"


    def buffered_method_row_from_metrics(
        zone_name: str,
        scenario: str,
        threshold: float,
        buffer_m: int,
        metrics: dict[str, float],
    ) -> dict[str, object]:
        prefix = f"{scenario}_b{int(buffer_m)}m"
        observed_area_m2 = float(metrics.get(f"{prefix}_observed_area_m2") or 0)
        chen_area_m2 = float(metrics.get(f"{prefix}_chen_area_m2") or 0)
        chen_near_observed_area_m2 = float(
            metrics.get(f"{prefix}_chen_near_observed_area_m2") or 0
        )
        observed_near_chen_area_m2 = float(
            metrics.get(f"{prefix}_observed_near_chen_area_m2") or 0
        )
        buffered_precision = safe_ratio(chen_near_observed_area_m2, chen_area_m2)
        buffered_recall = safe_ratio(observed_near_chen_area_m2, observed_area_m2)
        buffered_f1 = (
            safe_ratio(
                2 * buffered_precision * buffered_recall,
                buffered_precision + buffered_recall,
            )
            if np.isfinite(buffered_precision) and np.isfinite(buffered_recall)
            else np.nan
        )

        return {
            "zone": zone_name,
            "scenario": scenario,
            "method": buffered_method_name(threshold, buffer_m),
            "observed_threshold": threshold,
            "buffer_m": buffer_m,
            "observed_area_m2": observed_area_m2,
            "chen_area_m2": chen_area_m2,
            "buffered_precision": buffered_precision,
            "buffered_recall": buffered_recall,
            "buffered_f1": buffered_f1,
            "spatial_metric_name": "buffered_f1",
            "spatial_score": buffered_f1,
        }


    def buffered_method_rows_for_zone(
        zone_name: str,
        zone: Zone,
    ) -> list[dict[str, object]]:
        metric_images: list[ee.Image] = []

        for scenario in SSP_NAMES:
            chen_projection = zone.ssp_images[scenario].select(str(SOURCE_YEAR)).projection()
            pixel_area = ee.Image.pixelArea().reproject(chen_projection)
            observed_mask = (
                observed_settlement_fraction_image(zone, scenario, SOURCE_YEAR)
                .gte(ee.Number(BUFFER_OBSERVED_THRESHOLD))
                .reproject(chen_projection)
                .rename("observed_mask")
            )
            chen_mask = chen_urban_mask(zone, scenario, SOURCE_YEAR).reproject(
                chen_projection
            )

            observed_area = observed_mask.multiply(pixel_area)
            chen_area = chen_mask.multiply(pixel_area)

            for buffer_m in BUFFER_DISTANCES_M:
                prefix = f"{scenario}_b{int(buffer_m)}m"
                observed_buffer = (
                    observed_mask.focal_max(radius=buffer_m, units="meters")
                    .reproject(chen_projection)
                    .rename("observed_buffer")
                )
                chen_buffer = (
                    chen_mask.focal_max(radius=buffer_m, units="meters")
                    .reproject(chen_projection)
                    .rename("chen_buffer")
                )
                metric_images.extend(
                    [
                        observed_area.rename(f"{prefix}_observed_area_m2"),
                        chen_area.rename(f"{prefix}_chen_area_m2"),
                        chen_mask
                        .multiply(observed_buffer)
                        .multiply(pixel_area)
                        .rename(f"{prefix}_chen_near_observed_area_m2"),
                        observed_mask
                        .multiply(chen_buffer)
                        .multiply(pixel_area)
                        .rename(f"{prefix}_observed_near_chen_area_m2"),
                    ]
                )

        metrics = (
            ee.Image.cat(metric_images)
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=zone.bbox,
                scale=1000,
                maxPixels=int(1e10),
                tileScale=4,
            )
            .getInfo()
        ) or {}

        rows: list[dict[str, object]] = []
        for scenario in SSP_NAMES:
            for buffer_m in BUFFER_DISTANCES_M:
                rows.append(
                    buffered_method_row_from_metrics(
                        zone_name,
                        scenario,
                        BUFFER_OBSERVED_THRESHOLD,
                        buffer_m,
                        metrics,
                    )
                )
        return rows


    def build_buffered_method_table(
        manager: GeoManager,
        df_threshold_methods: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for zone_name, zone in manager:
            rows.extend(buffered_method_rows_for_zone(zone_name, zone))

        out = add_area_calibration_fields(
            pd.DataFrame(rows),
            correction_factor_bounds=CORRECTION_FACTOR_BOUNDS,
        )
        threshold_baseline = df_threshold_methods[
            df_threshold_methods["method"].eq(threshold_method_name(BUFFER_OBSERVED_THRESHOLD))
        ][["zone", "scenario", "precision", "recall", "iou"]]
        out = out.merge(threshold_baseline, on=["zone", "scenario"], how="left")
        out["valid_comparison"] = out["calibration_valid"] & np.isfinite(
            out["spatial_score"]
        )
        return out[METHOD_COLUMNS]

    return (build_buffered_method_table,)


@app.cell
def _(
    BUFFER_DISTANCES_M,
    build_buffered_method_table,
    df_threshold_methods,
    manager,
):
    df_buffered_methods = build_buffered_method_table(manager, df_threshold_methods)

    assert set(df_buffered_methods["method"]) == {
        "threshold_25_buffer_1000m",
        "threshold_25_buffer_2000m",
    }
    assert df_buffered_methods.shape[0] == len(manager.zones) * len(SSP_NAMES) * len(
        BUFFER_DISTANCES_M
    )

    mo.vstack(
        [
            mo.md("### Buffered methods"),
            df_buffered_methods.groupby("method")[
                ["buffered_precision", "buffered_recall", "buffered_f1"]
            ]
            .median()
            .round(3),
            df_buffered_methods.head(10),
        ]
    )
    return (df_buffered_methods,)


@app.cell
def _(
    CORRECTION_FACTOR_BOUNDS,
    df_buffered_methods,
    df_canonical_method,
    df_threshold_methods,
    manager,
):
    df_method_comparison = pd.concat(
        [df_canonical_method, df_threshold_methods, df_buffered_methods],
        ignore_index=True,
    )

    _expected_methods = {
        "fractional_current",
        "threshold_10",
        "threshold_25",
        "threshold_50",
        "threshold_25_buffer_1000m",
        "threshold_25_buffer_2000m",
    }
    assert set(df_method_comparison["method"]) == _expected_methods
    assert df_method_comparison.shape[0] == len(manager.zones) * len(SSP_NAMES) * len(
        _expected_methods
    )
    assert (df_method_comparison[["observed_area_m2", "chen_area_m2"]] >= -1e-6).all().all()
    assert df_method_comparison["correction_factor"].between(
        *CORRECTION_FACTOR_BOUNDS
    ).all() or df_method_comparison["correction_factor"].eq(1.0).any()

    df_method_comparison.head(10)
    return (df_method_comparison,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Comparison diagnostics
    """)
    return


@app.cell
def _(df_method_comparison):
    _plot_df = df_method_comparison.replace([np.inf, -np.inf], np.nan)
    _fig, _axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.scatterplot(
        data=_plot_df.dropna(subset=["area_bias", "spatial_score"]),
        x="area_bias",
        y="spatial_score",
        hue="method",
        style="spatial_metric_name",
        ax=_axes[0],
    )
    _axes[0].axvline(1, color="black", linestyle="--", linewidth=1)
    _axes[0].set_xscale("log")
    _axes[0].set_title("Spatial score vs area bias")
    _axes[0].set_xlabel("Chen / observed area")
    _axes[0].set_ylabel("Method spatial score")

    sns.boxplot(
        data=_plot_df,
        x="method",
        y="correction_factor",
        ax=_axes[1],
    )
    _axes[1].axhline(1, color="black", linestyle="--", linewidth=1)
    _axes[1].set_title("Correction-factor distribution")
    _axes[1].set_xlabel("")
    _axes[1].set_ylabel("Correction factor")
    _axes[1].tick_params(axis="x", rotation=45)

    _scenario_summary_plot = (
        _plot_df.groupby(["scenario", "method"], as_index=False)
        .agg(
            median_spatial_score=("spatial_score", "median"),
            median_ape=("ape", "median"),
        )
        .assign(median_ape_pct=lambda frame: frame["median_ape"] * 100)
    )
    sns.scatterplot(
        data=_scenario_summary_plot,
        x="median_ape_pct",
        y="median_spatial_score",
        hue="method",
        style="scenario",
        ax=_axes[2],
    )
    _axes[2].set_title("Scenario-level method tradeoff")
    _axes[2].set_xlabel("Median absolute percentage error (%)")
    _axes[2].set_ylabel("Median spatial score")

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(df_method_comparison, manager):
    def build_method_summary(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.replace([np.inf, -np.inf], np.nan)
            .assign(ape_pct=lambda frame: frame["ape"] * 100)
            .groupby("method", as_index=False)
            .agg(
                rows=("zone", "count"),
                valid_comparisons=("valid_comparison", "sum"),
                median_spatial_score=("spatial_score", "median"),
                mean_spatial_score=("spatial_score", "mean"),
                median_ape_pct=("ape_pct", "median"),
                median_area_bias=("area_bias", "median"),
                median_correction_factor=("correction_factor", "median"),
            )
            .assign(
                valid_share=lambda frame: frame["valid_comparisons"].div(frame["rows"])
            )
            .sort_values(
                ["median_spatial_score", "median_ape_pct"],
                ascending=[False, True],
            )
            .round(3)
        )


    def build_scenario_method_summary(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.replace([np.inf, -np.inf], np.nan)
            .assign(ape_pct=lambda frame: frame["ape"] * 100)
            .groupby(["scenario", "method"], as_index=False)
            .agg(
                rows=("zone", "count"),
                valid_comparisons=("valid_comparison", "sum"),
                median_spatial_score=("spatial_score", "median"),
                median_ape_pct=("ape_pct", "median"),
                median_area_bias=("area_bias", "median"),
            )
            .round(3)
        )


    def build_method_recommendation_candidates(df: pd.DataFrame) -> pd.DataFrame:
        out = df.replace([np.inf, -np.inf], np.nan).copy()
        out["invalid_rank"] = (~out["valid_comparison"]).astype(int)
        out["spatial_rank"] = out["spatial_score"].fillna(-np.inf)
        out["ape_rank"] = out["ape"].fillna(np.inf)
        out["area_bias_distance"] = np.abs(
            np.log(out["area_bias"].where(out["area_bias"].gt(0)))
        )
        return (
            out.sort_values(
                [
                    "zone",
                    "scenario",
                    "invalid_rank",
                    "spatial_rank",
                    "ape_rank",
                    "area_bias_distance",
                ],
                ascending=[True, True, True, False, True, True],
            )
            .groupby(["zone", "scenario"], as_index=False)
            .head(1)[
                [
                    "zone",
                    "scenario",
                    "method",
                    "spatial_metric_name",
                    "spatial_score",
                    "ape",
                    "area_bias",
                    "correction_factor",
                    "observed_threshold",
                    "buffer_m",
                    "valid_comparison",
                ]
            ]
            .reset_index(drop=True)
        )


    df_method_summary = build_method_summary(df_method_comparison)
    df_scenario_method_summary = build_scenario_method_summary(df_method_comparison)
    df_method_recommendation_candidates = build_method_recommendation_candidates(
        df_method_comparison
    )

    assert df_method_recommendation_candidates.shape[0] == len(manager.zones) * len(SSP_NAMES)

    mo.vstack(
        [
            mo.md("### Method summary"),
            df_method_summary,
            mo.md("### Scenario-method summary"),
            df_scenario_method_summary.head(20),
            mo.md("### Recommended method candidates by zone and scenario"),
            df_method_recommendation_candidates.head(20),
        ]
    )
    return df_method_recommendation_candidates, df_method_summary


@app.cell(hide_code=True)
def _(df_method_summary):
    _current = df_method_summary[df_method_summary["method"].eq("fractional_current")].iloc[0]
    _best = df_method_summary.iloc[0]

    _interpretation_table = df_method_summary[
        [
            "method",
            "median_spatial_score",
            "median_ape_pct",
            "median_area_bias",
            "median_correction_factor",
            "valid_share",
        ]
    ]

    mo.vstack(
        [
            mo.md(
                rf"""
                # Interpretation

                `{_best["method"]}` has the highest median spatial score (`{_best["median_spatial_score"]}`), compared with `{_current["median_spatial_score"]}` for `fractional_current`.

                That does not automatically make it the calibration method to promote. The buffered methods answer a different question: whether Chen and observed settlement are near each other within a spatial tolerance. They improve spatial agreement, but they may preserve area less well than the current fractional calibration.

                In this run, `fractional_current` remains the best canonical baseline for area-preserving correction, while the buffered methods are useful as tolerance-aware adequacy diagnostics.
                """
            ),
            _interpretation_table,
            mo.md(
                r"""
                The next comparison should focus on zone-scenario pairs where the preferred exploratory method differs from `fractional_current`, especially cases where buffered agreement is high but area error remains large.
                """
            ),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    # Disagreement typology

    The method comparison can make buffered methods look like easy winners because their score asks whether Chen and observed settlement are near each other within a tolerance. This diagnostic keeps that tolerance-aware signal, but puts it beside the current fractional calibration and the strict 50% observed-settlement threshold so zone-scenario pairs with hidden area mismatch stay visible.
    """)
    return


@app.cell
def _(
    BUFFER_DISTANCES_M,
    BUFFER_OBSERVED_THRESHOLD,
    df_method_comparison,
    manager,
):
    def build_disagreement_typology(
        df: pd.DataFrame,
        *,
        canonical_method: str = "fractional_current",
        strict_method: str = "threshold_50",
        buffered_method: str | None = None,
        low_iou_threshold: float = 0.35,
        high_buffered_threshold: float = 0.80,
        low_buffered_threshold: float = 0.70,
        high_ape_threshold: float = 0.35,
        moderate_ape_threshold: float = 0.20,
        method_gain_threshold: float = 0.10,
        ape_tolerance: float = 0.05,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if buffered_method is None:
            buffered_method = (
                f"threshold_{int(BUFFER_OBSERVED_THRESHOLD * 100)}_buffer_"
                f"{max(BUFFER_DISTANCES_M)}m"
            )

        method_names = [canonical_method, strict_method, buffered_method]
        metric_names = [
            "spatial_score",
            "ape",
            "area_bias",
            "correction_factor",
            "valid_comparison",
        ]
        wide = (
            df[df["method"].isin(method_names)]
            .pivot_table(
                index=["zone", "scenario"],
                columns="method",
                values=metric_names,
                aggfunc="first",
            )
            .sort_index(axis=1)
        )
        wide.columns = [f"{metric}__{method}" for metric, method in wide.columns]
        out = wide.reset_index()

        required_columns = {
            f"spatial_score__{canonical_method}",
            f"ape__{canonical_method}",
            f"area_bias__{canonical_method}",
            f"correction_factor__{canonical_method}",
            f"valid_comparison__{canonical_method}",
            f"spatial_score__{strict_method}",
            f"ape__{strict_method}",
            f"spatial_score__{buffered_method}",
        }
        missing_columns = sorted(required_columns.difference(out.columns))
        if missing_columns:
            message = f"Missing method diagnostics: {missing_columns}"
            raise ValueError(message)

        out = out.rename(
            columns={
                f"spatial_score__{canonical_method}": "current_iou",
                f"ape__{canonical_method}": "current_ape",
                f"area_bias__{canonical_method}": "current_area_bias",
                f"correction_factor__{canonical_method}": "current_correction_factor",
                f"valid_comparison__{canonical_method}": "current_valid",
                f"spatial_score__{strict_method}": "strict_iou",
                f"ape__{strict_method}": "strict_ape",
                f"spatial_score__{buffered_method}": "buffered_f1_widest",
            }
        )
        numeric_columns = [
            "current_iou",
            "current_ape",
            "current_area_bias",
            "current_correction_factor",
            "strict_iou",
            "strict_ape",
            "buffered_f1_widest",
        ]
        out[numeric_columns] = out[numeric_columns].apply(pd.to_numeric, errors="coerce")
        out["current_valid"] = (
            out["current_valid"].astype("boolean").fillna(False).astype(bool)
        )
        out["buffered_gain_over_current"] = out["buffered_f1_widest"] - out["current_iou"]
        out["strict_iou_gain_over_current"] = out["strict_iou"] - out["current_iou"]
        out["strict_ape_delta"] = out["strict_ape"] - out["current_ape"]

        out["area_error_class"] = np.select(
            [
                out["current_ape"].ge(high_ape_threshold),
                out["current_ape"].ge(moderate_ape_threshold),
            ],
            ["large_area_mismatch", "moderate_area_mismatch"],
            default="area_close",
        )
        out["spatial_agreement_class"] = np.select(
            [
                out["current_iou"].ge(0.50),
                out["current_iou"].ge(low_iou_threshold),
            ],
            ["strong_current_overlap", "moderate_current_overlap"],
            default="weak_current_overlap",
        )
        out["diagnostic_type"] = np.select(
            [
                ~out["current_valid"],
                out["buffered_f1_widest"].ge(high_buffered_threshold)
                & out["current_ape"].ge(high_ape_threshold),
                out["buffered_f1_widest"].lt(low_buffered_threshold)
                & out["current_iou"].lt(low_iou_threshold),
                out["strict_iou_gain_over_current"].ge(method_gain_threshold)
                & out["strict_ape_delta"].le(ape_tolerance),
                out["current_iou"].ge(low_iou_threshold)
                & out["current_ape"].le(moderate_ape_threshold),
            ],
            [
                "invalid_current_calibration",
                "tolerance_masks_area_mismatch",
                "weak_even_with_tolerance",
                "strict_threshold_improves_overlap",
                "stable_current_candidate",
            ],
            default="needs_targeted_method_review",
        )
        out["review_score"] = (
            out["current_ape"].fillna(1.0).clip(lower=0, upper=2) * 2
            + (1 - out["current_iou"].fillna(0.0)).clip(lower=0, upper=1)
            + out["buffered_gain_over_current"].fillna(0.0).clip(lower=0, upper=1)
            + (~out["current_valid"]).astype(int)
        )

        display_columns = [
            "zone",
            "scenario",
            "diagnostic_type",
            "area_error_class",
            "spatial_agreement_class",
            "current_iou",
            "current_ape",
            "current_area_bias",
            "current_correction_factor",
            "strict_iou",
            "strict_ape",
            "buffered_f1_widest",
            "buffered_gain_over_current",
            "strict_iou_gain_over_current",
            "strict_ape_delta",
            "current_valid",
            "review_score",
        ]
        out = out[display_columns].sort_values(
            ["review_score", "current_ape", "current_iou"],
            ascending=[False, False, True],
        )

        summary = (
            out.groupby("diagnostic_type", as_index=False)
            .agg(
                rows=("zone", "count"),
                median_current_iou=("current_iou", "median"),
                median_current_ape=("current_ape", "median"),
                median_buffered_gain=("buffered_gain_over_current", "median"),
                median_strict_iou_gain=("strict_iou_gain_over_current", "median"),
                max_review_score=("review_score", "max"),
            )
            .assign(share=lambda frame: frame["rows"].div(len(out)))
            .sort_values(["rows", "max_review_score"], ascending=[False, False])
            .round(3)
        )
        return out.reset_index(drop=True), summary


    df_disagreement_typology, df_disagreement_summary = build_disagreement_typology(
        df_method_comparison
    )

    _expected_disagreement_rows = len(manager.zones) * len(SSP_NAMES)
    if df_disagreement_typology.shape[0] != _expected_disagreement_rows:
        message = (
            "Expected one disagreement row per zone and scenario, got "
            f"{df_disagreement_typology.shape[0]} rows."
        )
        raise ValueError(message)
    if set(df_disagreement_typology["scenario"]) != set(SSP_NAMES):
        message = "Disagreement typology scenarios do not match SSP_NAMES."
        raise ValueError(message)

    df_disagreement_typology.head(20)
    return df_disagreement_summary, df_disagreement_typology


@app.cell
def _(df_disagreement_summary, df_disagreement_typology):
    mo.vstack(
        [
            mo.md("### Disagreement typology summary"),
            df_disagreement_summary,
            mo.md("### Highest-priority disagreement cases"),
            df_disagreement_typology.head(25),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Export exploration artifacts

    These files are exploratory products only. Promoting any method into the canonical calibration workflow should happen by editing `01_calibration.py` in a later step.
    """)
    return


@app.cell
def _(
    df_disagreement_summary,
    df_disagreement_typology,
    df_method_comparison,
    df_method_recommendation_candidates,
    df_method_summary,
    exploration_artifact_dir,
    manager,
):
    exploration_artifact_dir.mkdir(parents=True, exist_ok=True)

    _exploration_artifacts = {
        "method_comparison": (
            df_method_comparison,
            exploration_artifact_dir / "method_comparison.parquet",
        ),
        "method_summary": (
            df_method_summary,
            exploration_artifact_dir / "method_summary.parquet",
        ),
        "method_recommendation_candidates": (
            df_method_recommendation_candidates,
            exploration_artifact_dir / "method_recommendation_candidates.parquet",
        ),
        "disagreement_typology": (
            df_disagreement_typology,
            exploration_artifact_dir / "disagreement_typology.parquet",
        ),
        "disagreement_summary": (
            df_disagreement_summary,
            exploration_artifact_dir / "disagreement_summary.parquet",
        ),
    }

    _validation_report = validate_exploration_artifacts(
        df_method_comparison,
        df_method_summary,
        df_method_recommendation_candidates,
        df_disagreement_typology,
        df_disagreement_summary,
        zone_names=manager.zones,
    )
    raise_for_validation_errors(_validation_report)
    _validation_summary = (
        _validation_report
        if not _validation_report.empty
        else pd.DataFrame(
            [
                {
                    "artifact": "exploration_outputs",
                    "check": "validation",
                    "severity": "pass",
                    "message": "All artifact validation checks passed.",
                    "rows": 0,
                }
            ]
        )
    )

    _export_rows: list[dict[str, object]] = []
    for _artifact_name, (_frame, _path) in _exploration_artifacts.items():
        _frame.to_parquet(_path, index=False)
        _export_rows.append(
            {
                "artifact": _artifact_name,
                "path": str(_path),
                "rows": len(_frame),
            }
        )

    df_exploration_artifacts = pd.DataFrame(_export_rows)
    mo.vstack(
        [
            mo.md("### Exploration artifact validation"),
            _validation_summary,
            mo.md("### Exploration artifacts"),
            df_exploration_artifacts,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
