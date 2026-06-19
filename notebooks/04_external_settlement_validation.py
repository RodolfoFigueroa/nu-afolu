import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import os  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import ee  # noqa: PLC0415
    import marimo as mo  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    import seaborn as sns  # noqa: PLC0415
    from dagster_components.partitions import zone_partitions  # noqa: PLC0415

    from nu_afolu.artifact_validation import (  # noqa: PLC0415
        raise_for_validation_errors,
        validate_external_validation_artifacts,
        validate_transition_closure_artifacts,
    )
    from nu_afolu.chen import (  # noqa: PLC0415
        CHEN_COLLECTION_ID,
        SSP_NAMES,
        ChenAnalysisZone,
        ChenAnalysisZoneCollection,
        chen_urban_mask,
        load_chen_analysis_zones,
        observed_settlement_fraction_image,
    )
    from nu_afolu.constants import LABEL_LIST  # noqa: PLC0415
    from nu_afolu.external_validation import (  # noqa: PLC0415
        classify_baseline_comparator_support,
        classify_external_advisory,
        classify_growth_alignment,
        combine_baseline_support,
    )
    from nu_afolu.metrics import agreement_metrics_from_areas  # noqa: PLC0415

    ee.Initialize()
    return (
        CHEN_COLLECTION_ID,
        ChenAnalysisZone,
        ChenAnalysisZoneCollection,
        LABEL_LIST,
        Path,
        SSP_NAMES,
        agreement_metrics_from_areas,
        chen_urban_mask,
        classify_baseline_comparator_support,
        classify_external_advisory,
        classify_growth_alignment,
        combine_baseline_support,
        ee,
        load_chen_analysis_zones,
        mo,
        np,
        observed_settlement_fraction_image,
        os,
        pd,
        plt,
        raise_for_validation_errors,
        sns,
        validate_external_validation_artifacts,
        validate_transition_closure_artifacts,
        zone_partitions,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # External Settlement Validation

    This notebook adds an independent GHSL built-up-surface check around the Chen workflow. It does not replace the GLC-FCS30D-derived observed baseline and it does not produce carbon-model inputs. Its role is advisory: test whether Chen's 2020 baseline and first future decade are credible enough, relative to independent built-up evidence, to support manual review or further calibration.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Provenance scope

    External validation is evidence, not a new baseline.

    - Observed settlement remains the 2020 `settlements` class from the upstream GLC-FCS30D-derived `area_raster`.
    - Chen remains the only long-range SSP-style future settlement-expansion signal in this workflow.
    - GHSL is used here as an independent built-up-surface anchor for 2020 baseline agreement and 2020-2030 near-term plausibility.
    - GHSL 2030 is treated as near-term plausibility evidence, not observed truth, because the product is spatially-temporally interpolated or extrapolated through 2030.
    - Outputs are written under `OUT_PATH/chen/external_validation/` and leave all existing readiness labels unchanged.

    The full provenance and artifact contract is documented in `docs/data_provenance.md`.
    """)
    return


@app.cell
def _(LABEL_LIST):
    LABEL_MAP = dict(enumerate(LABEL_LIST, start=1))
    LABEL_ID_BY_NAME = {label: idx for idx, label in LABEL_MAP.items()}
    SETTLEMENT_IDX = LABEL_ID_BY_NAME["settlements"]

    SOURCE_YEAR = 2020
    VALIDATION_YEAR = 2030
    GHSL_COLLECTION_ID = "JRC/GHSL/P2023A/GHS_BUILT_S"
    GHSL_DATASET_NAME = "ghsl_built_surface"
    GHSL_BUILT_SURFACE_BAND = "built_surface"
    GHSL_SCALE_M = 100
    CHEN_SCALE_M = 1000

    BASELINE_COMPARATORS = {
        "glc_settlements_2020": "GLC-FCS30D settlements, 2020",
        "chen_urban_2020": "Chen urban baseline, 2020",
    }
    return (
        BASELINE_COMPARATORS,
        CHEN_SCALE_M,
        GHSL_BUILT_SURFACE_BAND,
        GHSL_COLLECTION_ID,
        GHSL_DATASET_NAME,
        GHSL_SCALE_M,
        SETTLEMENT_IDX,
        SOURCE_YEAR,
        VALIDATION_YEAR,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Inputs
    """)
    return


@app.cell
def _(CHEN_COLLECTION_ID, GHSL_COLLECTION_ID, Path, ee, os):
    out_path = Path(os.environ["OUT_PATH"])
    chen_artifact_dir = out_path / "chen"
    external_validation_dir = chen_artifact_dir / "external_validation"
    col_chen = ee.ImageCollection(CHEN_COLLECTION_ID)
    col_ghsl = ee.ImageCollection(GHSL_COLLECTION_ID)
    return chen_artifact_dir, col_chen, external_validation_dir, out_path


@app.cell
def _(mo):
    mo.md(r"""
    ## Load zone rasters

    The external validation notebook reloads the same zone bounding boxes and GLC-FCS30D-derived rasters as the other Chen notebooks. This keeps the comparison independent of live kernel state from `01`, `02`, or `03`.
    """)
    return


@app.cell
def _(
    SETTLEMENT_IDX,
    col_chen,
    load_chen_analysis_zones,
    mo,
    out_path,
    pd,
    zone_partitions,
):
    manager, df_missing_zones = load_chen_analysis_zones(
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


@app.cell
def _(mo):
    mo.md(r"""
    ## Load diagnostic handoff artifacts

    This notebook reads the validated closure artifacts produced by `02_transition_closure.py`. It uses the current readiness and feasibility labels as context only; external validation writes separate advisory labels instead of modifying the closure decision.
    """)
    return


@app.cell
def _(
    chen_artifact_dir,
    manager,
    mo,
    pd,
    raise_for_validation_errors,
    validate_transition_closure_artifacts,
):
    _artifact_paths = {
        "calibration": chen_artifact_dir / "calibration.parquet",
        "chen_expansion": chen_artifact_dir / "chen_expansion.parquet",
        "chen_transitions": chen_artifact_dir / "chen_transitions.parquet",
        "transition_feasibility": chen_artifact_dir / "transition_feasibility.parquet",
        "historical_growth_diagnostics": chen_artifact_dir / "historical_growth_diagnostics.parquet",
        "land_estimation_assessment": chen_artifact_dir / "land_estimation_assessment.parquet",
        "review_candidates": chen_artifact_dir / "review_candidates.parquet",
    }
    _missing_artifacts = [
        str(path) for path in _artifact_paths.values() if not path.exists()
    ]
    if _missing_artifacts:
        _message = (
            "Missing closure artifacts. Run 01_calibration.py and 02_transition_closure.py first: "
            f"{_missing_artifacts}"
        )
        raise FileNotFoundError(_message)

    df_calibration = pd.read_parquet(_artifact_paths["calibration"])
    df_chen_expansion = pd.read_parquet(_artifact_paths["chen_expansion"])
    df_chen_transitions = pd.read_parquet(_artifact_paths["chen_transitions"])
    df_transition_feasibility = pd.read_parquet(_artifact_paths["transition_feasibility"])
    df_historical_growth_diagnostics = pd.read_parquet(
        _artifact_paths["historical_growth_diagnostics"]
    )
    df_land_estimation_assessment = pd.read_parquet(
        _artifact_paths["land_estimation_assessment"]
    )
    df_review_candidates = pd.read_parquet(_artifact_paths["review_candidates"])

    _validation_report = validate_transition_closure_artifacts(
        df_calibration,
        df_chen_expansion,
        df_chen_transitions,
        df_transition_feasibility,
        df_historical_growth_diagnostics,
        df_land_estimation_assessment,
        df_review_candidates,
        zone_names=manager.zones,
    )
    raise_for_validation_errors(_validation_report)
    _validation_summary = (
        _validation_report
        if not _validation_report.empty
        else pd.DataFrame(
            [
                {
                    "artifact": "transition_closure_inputs",
                    "check": "validation",
                    "severity": "pass",
                    "message": "Transition-closure handoff validation checks passed.",
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
                        "artifact": artifact,
                        "path": str(path),
                        "rows": len(pd.read_parquet(path)),
                    }
                    for artifact, path in _artifact_paths.items()
                ]
            ),
        ]
    )
    return df_chen_expansion, df_land_estimation_assessment


@app.cell
def _(mo):
    mo.md(r"""
    # GHSL baseline agreement

    The baseline comparison uses GHSL 2020 built-up surface as the external reference. For each Chen 1km cell, GHSL 100m built-up surface is summed onto Chen's grid. GLC settlement area and Chen urban area are also represented as m2 per Chen cell, and overlap is approximated as the cellwise minimum of GHSL built surface and comparator area.
    """)
    return


@app.cell
def _(
    CHEN_SCALE_M,
    ChenAnalysisZone,
    GHSL_BUILT_SURFACE_BAND,
    GHSL_COLLECTION_ID,
    SOURCE_YEAR,
    agreement_metrics_from_areas,
    chen_urban_mask,
    ee,
    observed_settlement_fraction_image,
):
    def ghsl_built_surface_image(year: int) -> ee.Image:
        return ee.Image(f"{GHSL_COLLECTION_ID}/{year}").select(
            GHSL_BUILT_SURFACE_BAND
        )


    def chen_projection(zone: ChenAnalysisZone, scenario: str) -> ee.Projection:
        return zone.chen_urban_masks_by_scenario[scenario].select(str(SOURCE_YEAR)).projection()


    def chen_grid_pixel_area(zone: ChenAnalysisZone, scenario: str) -> ee.Image:
        return ee.Image.pixelArea().reproject(chen_projection(zone, scenario))


    def ghsl_built_area_on_chen_grid(
        zone: ChenAnalysisZone,
        scenario: str,
        year: int,
    ) -> ee.Image:
        return (
            ghsl_built_surface_image(year)
            .clip(zone.bbox)
            .reduceResolution(reducer=ee.Reducer.sum(), maxPixels=4096)
            .reproject(chen_projection(zone, scenario))
            .rename("external_area_m2")
            .unmask(0)
        )


    def glc_settlement_area_on_chen_grid(
        zone: ChenAnalysisZone,
        scenario: str,
    ) -> ee.Image:
        return (
            observed_settlement_fraction_image(zone, scenario, SOURCE_YEAR)
            .multiply(chen_grid_pixel_area(zone, scenario))
            .rename("comparator_area_m2")
            .unmask(0)
        )


    def chen_urban_area_on_chen_grid(
        zone: ChenAnalysisZone,
        scenario: str,
        year: int,
    ) -> ee.Image:
        return (
            chen_urban_mask(zone, scenario, year)
            .multiply(chen_grid_pixel_area(zone, scenario))
            .rename("comparator_area_m2")
            .unmask(0)
        )


    def agreement_stack(
        external_area: ee.Image,
        comparator_area: ee.Image,
    ) -> ee.Image:
        tp_area = external_area.min(comparator_area).rename("tp_area_m2")
        return external_area.addBands([comparator_area, tp_area])


    def reduce_agreement_stack(image: ee.Image, geometry: ee.Geometry) -> dict[str, float]:
        reduced = image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=CHEN_SCALE_M,
            maxPixels=1e13,
        ).getInfo()
        external_area = float(reduced.get("external_area_m2") or 0.0)
        comparator_area = float(reduced.get("comparator_area_m2") or 0.0)
        tp_area = float(reduced.get("tp_area_m2") or 0.0)
        metrics = agreement_metrics_from_areas(
            observed_area_m2=external_area,
            chen_area_m2=comparator_area,
            tp_area_m2=tp_area,
        )
        return {
            "external_area_m2": metrics["observed_area_m2"],
            "comparator_area_m2": metrics["chen_area_m2"],
            "tp_area_m2": metrics["tp_area_m2"],
            "fp_area_m2": metrics["fp_area_m2"],
            "fn_area_m2": metrics["fn_area_m2"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "iou": metrics["iou"],
            "area_bias": metrics["area_bias"],
            "ape": metrics["ape"],
        }

    return (
        agreement_stack,
        chen_urban_area_on_chen_grid,
        ghsl_built_area_on_chen_grid,
        ghsl_built_surface_image,
        glc_settlement_area_on_chen_grid,
        reduce_agreement_stack,
    )


@app.cell
def _(
    BASELINE_COMPARATORS,
    ChenAnalysisZoneCollection,
    GHSL_DATASET_NAME,
    SOURCE_YEAR,
    SSP_NAMES,
    agreement_stack,
    chen_urban_area_on_chen_grid,
    classify_baseline_comparator_support,
    ghsl_built_area_on_chen_grid,
    glc_settlement_area_on_chen_grid,
    manager,
    pd,
    reduce_agreement_stack,
):
    def build_external_baseline_agreement_table(
        manager: ChenAnalysisZoneCollection,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for zone_name, zone in manager:
            for scenario in SSP_NAMES:
                external_area = ghsl_built_area_on_chen_grid(zone, scenario, SOURCE_YEAR)
                comparators = {
                    "glc_settlements_2020": glc_settlement_area_on_chen_grid(
                        zone,
                        scenario,
                    ),
                    "chen_urban_2020": chen_urban_area_on_chen_grid(
                        zone,
                        scenario,
                        SOURCE_YEAR,
                    ),
                }
                for comparator_name, comparator_area in comparators.items():
                    metrics = reduce_agreement_stack(
                        agreement_stack(external_area, comparator_area),
                        zone.bbox,
                    )
                    rows.append(
                        {
                            "zone": zone_name,
                            "scenario": scenario,
                            "comparator": comparator_name,
                            "external_dataset": GHSL_DATASET_NAME,
                            "external_year": SOURCE_YEAR,
                            **metrics,
                            "comparator_support": classify_baseline_comparator_support(
                                external_area_m2=metrics["external_area_m2"],
                                comparator_area_m2=metrics["comparator_area_m2"],
                                iou=metrics["iou"],
                                ape=metrics["ape"],
                            ),
                        }
                    )
        return pd.DataFrame(rows)


    df_external_baseline_agreement = build_external_baseline_agreement_table(manager)

    _expected_baseline_rows = len(manager.zones) * len(SSP_NAMES) * len(
        BASELINE_COMPARATORS
    )
    if df_external_baseline_agreement.shape[0] != _expected_baseline_rows:
        _message = (
            "Expected one external baseline row per zone, scenario, and comparator; "
            f"found {df_external_baseline_agreement.shape[0]} rows."
        )
        raise ValueError(_message)

    df_external_baseline_agreement.head(10)
    return (df_external_baseline_agreement,)


@app.cell
def _(df_external_baseline_agreement, mo):
    _baseline_summary = (
        df_external_baseline_agreement.groupby(
            ["comparator", "scenario", "comparator_support"],
            as_index=False,
        )
        .agg(
            rows=("zone", "count"),
            median_iou=("iou", "median"),
            median_ape=("ape", "median"),
            median_area_bias=("area_bias", "median"),
        )
        .round(3)
    )

    mo.vstack(
        [
            mo.md("### Baseline agreement against GHSL 2020"),
            _baseline_summary,
            mo.md("### Baseline rows"),
            df_external_baseline_agreement.head(20),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # GHSL 2020-2030 growth alignment

    The growth alignment compares GHSL 2020-2030 built-up-surface change against Chen's first decadal expansion. GHSL growth is independent of SSP, while Chen growth is SSP-specific. Both raw and calibrated Chen growth are kept; the calibrated row is used for the advisory review flag.
    """)
    return


@app.cell
def _(
    ChenAnalysisZoneCollection,
    GHSL_DATASET_NAME,
    GHSL_SCALE_M,
    SOURCE_YEAR,
    SSP_NAMES,
    VALIDATION_YEAR,
    classify_growth_alignment,
    df_chen_expansion,
    ee,
    ghsl_built_surface_image,
    manager,
    np,
    pd,
):
    def ghsl_growth_image(start_year: int, end_year: int) -> ee.Image:
        return (
            ghsl_built_surface_image(end_year)
            .subtract(ghsl_built_surface_image(start_year))
            .max(0)
            .rename("ghsl_growth_area_m2")
        )


    def reduce_ghsl_growth_by_zone(
        manager: ChenAnalysisZoneCollection,
        *,
        start_year: int,
        end_year: int,
    ) -> pd.DataFrame:
        growth = ghsl_growth_image(start_year, end_year)
        rows: list[dict[str, object]] = []
        for zone_name, zone in manager:
            reduced = growth.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=zone.bbox,
                scale=GHSL_SCALE_M,
                maxPixels=1e13,
            ).getInfo()
            rows.append(
                {
                    "zone": zone_name,
                    "ghsl_growth_area_m2": float(
                        reduced.get("ghsl_growth_area_m2") or 0.0
                    ),
                }
            )
        return pd.DataFrame(rows)


    def build_external_growth_alignment_table(
        df_chen_expansion: pd.DataFrame,
        df_ghsl_growth: pd.DataFrame,
    ) -> pd.DataFrame:
        chen_2030 = df_chen_expansion[df_chen_expansion["year"].eq(VALIDATION_YEAR)][
            ["zone", "scenario", "nonsettlement_source_area_m2", "correction_factor"]
        ].merge(df_ghsl_growth, on="zone", how="left")

        rows: list[dict[str, object]] = []
        for row in chen_2030.itertuples(index=False):
            raw_growth = float(row.nonsettlement_source_area_m2)
            calibrated_growth = raw_growth * float(row.correction_factor)
            for calibration, chen_growth in (
                ("raw", raw_growth),
                ("calibrated", calibrated_growth),
            ):
                ratio = (
                    chen_growth / row.ghsl_growth_area_m2
                    if row.ghsl_growth_area_m2 > 0
                    else np.nan
                )
                rows.append(
                    {
                        "zone": row.zone,
                        "scenario": row.scenario,
                        "calibration": calibration,
                        "external_dataset": GHSL_DATASET_NAME,
                        "period_start_year": SOURCE_YEAR,
                        "year": VALIDATION_YEAR,
                        "ghsl_growth_area_m2": row.ghsl_growth_area_m2,
                        "chen_growth_area_m2": chen_growth,
                        "chen_to_external_growth_ratio": ratio,
                        "growth_alignment": classify_growth_alignment(
                            external_growth_area_m2=row.ghsl_growth_area_m2,
                            chen_growth_area_m2=chen_growth,
                        ),
                    }
                )
        return pd.DataFrame(rows)


    df_ghsl_growth_by_zone = reduce_ghsl_growth_by_zone(
        manager,
        start_year=SOURCE_YEAR,
        end_year=VALIDATION_YEAR,
    )
    df_external_growth_alignment = build_external_growth_alignment_table(
        df_chen_expansion,
        df_ghsl_growth_by_zone,
    )

    _expected_growth_rows = len(manager.zones) * len(SSP_NAMES) * 2
    if df_external_growth_alignment.shape[0] != _expected_growth_rows:
        _message = (
            "Expected one external growth row per zone, scenario, and calibration; "
            f"found {df_external_growth_alignment.shape[0]} rows."
        )
        raise ValueError(_message)

    df_external_growth_alignment.head(10)
    return (df_external_growth_alignment,)


@app.cell
def _(df_external_growth_alignment, mo):
    _growth_summary = (
        df_external_growth_alignment.groupby(
            ["scenario", "calibration", "growth_alignment"],
            as_index=False,
        )
        .agg(
            rows=("zone", "count"),
            median_ghsl_growth_area_m2=("ghsl_growth_area_m2", "median"),
            median_chen_growth_area_m2=("chen_growth_area_m2", "median"),
            median_ratio=("chen_to_external_growth_ratio", "median"),
        )
        .round(3)
    )

    mo.vstack(
        [
            mo.md("### Growth alignment summary"),
            _growth_summary,
            mo.md("### Growth alignment rows"),
            df_external_growth_alignment.head(20),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # External review flags

    The review flag joins GHSL baseline support, GHSL growth alignment, and the existing transition-closure readiness context. These labels are advisory only; they do not demote or promote the closure notebook's readiness labels.
    """)
    return


@app.cell
def _(
    SSP_NAMES,
    classify_external_advisory,
    combine_baseline_support,
    df_external_baseline_agreement,
    df_external_growth_alignment,
    df_land_estimation_assessment,
    manager,
    pd,
):
    def build_external_review_flags(
        df_baseline: pd.DataFrame,
        df_growth: pd.DataFrame,
        df_assessment: pd.DataFrame,
    ) -> pd.DataFrame:
        baseline = (
            df_baseline.pivot_table(
                index=["zone", "scenario"],
                columns="comparator",
                values="comparator_support",
                aggfunc="first",
            )
            .reset_index()
            .rename(
                columns={
                    "glc_settlements_2020": "glc_baseline_support",
                    "chen_urban_2020": "chen_baseline_support",
                }
            )
        )
        baseline["external_baseline_validation"] = [
            combine_baseline_support(glc, chen)
            for glc, chen in zip(
                baseline["glc_baseline_support"],
                baseline["chen_baseline_support"],
                strict=True,
            )
        ]

        calibrated_growth = df_growth[df_growth["calibration"].eq("calibrated")][
            [
                "zone",
                "scenario",
                "ghsl_growth_area_m2",
                "chen_growth_area_m2",
                "chen_to_external_growth_ratio",
                "growth_alignment",
            ]
        ].rename(
            columns={
                "chen_growth_area_m2": "calibrated_chen_growth_area_m2",
                "chen_to_external_growth_ratio": (
                    "calibrated_chen_to_external_growth_ratio"
                ),
                "growth_alignment": "calibrated_growth_alignment",
            }
        )

        assessment_context = df_assessment[
            [
                "zone",
                "scenario",
                "land_estimate_readiness",
                "manual_review_priority",
                "overall_assessment",
                "transition_feasibility",
                "reliability",
                "iou",
                "ape",
            ]
        ]

        out = (
            baseline.merge(calibrated_growth, on=["zone", "scenario"], how="left")
            .merge(assessment_context, on=["zone", "scenario"], how="left")
        )
        out["external_advisory"] = [
            classify_external_advisory(baseline_label, growth_label)
            for baseline_label, growth_label in zip(
                out["external_baseline_validation"],
                out["calibrated_growth_alignment"],
                strict=True,
            )
        ]
        return out[
            [
                "zone",
                "scenario",
                "glc_baseline_support",
                "chen_baseline_support",
                "external_baseline_validation",
                "calibrated_growth_alignment",
                "external_advisory",
                "land_estimate_readiness",
                "manual_review_priority",
                "overall_assessment",
                "transition_feasibility",
                "reliability",
                "iou",
                "ape",
                "ghsl_growth_area_m2",
                "calibrated_chen_growth_area_m2",
                "calibrated_chen_to_external_growth_ratio",
            ]
        ]


    df_external_review_flags = build_external_review_flags(
        df_external_baseline_agreement,
        df_external_growth_alignment,
        df_land_estimation_assessment,
    )

    _expected_review_flag_rows = len(manager.zones) * len(SSP_NAMES)
    if df_external_review_flags.shape[0] != _expected_review_flag_rows:
        _message = (
            "Expected one external review flag per zone and scenario; "
            f"found {df_external_review_flags.shape[0]} rows."
        )
        raise ValueError(_message)

    df_external_review_flags.head(10)
    return (df_external_review_flags,)


@app.cell
def _(df_external_review_flags, mo, pd):
    def build_external_validation_summary(df_flags: pd.DataFrame) -> pd.DataFrame:
        out = (
            df_flags.groupby(
                [
                    "scenario",
                    "external_advisory",
                    "external_baseline_validation",
                    "calibrated_growth_alignment",
                ],
                as_index=False,
            )
            .agg(
                rows=("zone", "count"),
                median_ghsl_growth_area_m2=("ghsl_growth_area_m2", "median"),
                median_calibrated_chen_to_external_growth_ratio=(
                    "calibrated_chen_to_external_growth_ratio",
                    "median",
                ),
            )
            .sort_values(["scenario", "external_advisory"])
        )
        out["share"] = out["rows"].div(out.groupby("scenario")["rows"].transform("sum"))
        return out[
            [
                "scenario",
                "external_advisory",
                "external_baseline_validation",
                "calibrated_growth_alignment",
                "rows",
                "share",
                "median_ghsl_growth_area_m2",
                "median_calibrated_chen_to_external_growth_ratio",
            ]
        ].round(3)


    df_external_validation_summary = build_external_validation_summary(
        df_external_review_flags
    )

    mo.vstack(
        [
            mo.md("### External validation summary"),
            df_external_validation_summary,
            mo.md("### Highest-conflict review flags"),
            df_external_review_flags.sort_values(
                ["external_advisory", "ape", "iou"],
                ascending=[True, False, True],
            ).head(20),
        ]
    )
    return (df_external_validation_summary,)


@app.cell
def _(df_external_review_flags, np, plt, sns):
    _plot_df = df_external_review_flags.replace([np.inf, -np.inf], np.nan)
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.countplot(
        data=_plot_df,
        x="scenario",
        hue="external_advisory",
        ax=_axes[0],
    )
    _axes[0].set_title("External advisory labels by SSP")
    _axes[0].set_xlabel("Scenario")
    _axes[0].set_ylabel("Zone count")

    sns.scatterplot(
        data=_plot_df,
        x="ghsl_growth_area_m2",
        y="calibrated_chen_growth_area_m2",
        hue="external_advisory",
        ax=_axes[1],
    )
    _growth_limit = float(
        np.nanmax(
            [
                _plot_df["ghsl_growth_area_m2"].max(),
                _plot_df["calibrated_chen_growth_area_m2"].max(),
            ]
        )
    )
    _axes[1].plot([0, _growth_limit], [0, _growth_limit], color="black", linewidth=1)
    _axes[1].set_title("Calibrated Chen growth vs GHSL growth")
    _axes[1].set_xlabel("GHSL 2020-2030 growth area (m2)")
    _axes[1].set_ylabel("Calibrated Chen 2020-2030 growth area (m2)")

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Export external validation artifacts

    These outputs are advisory validation products only. They can support manual review decisions, but they do not alter canonical calibration, transition closure, or production model inputs.
    """)
    return


@app.cell
def _(
    df_external_baseline_agreement,
    df_external_growth_alignment,
    df_external_review_flags,
    df_external_validation_summary,
    df_land_estimation_assessment,
    external_validation_dir,
    manager,
    mo,
    pd,
    raise_for_validation_errors,
    validate_external_validation_artifacts,
):
    _validation_report = validate_external_validation_artifacts(
        df_external_baseline_agreement,
        df_external_growth_alignment,
        df_external_review_flags,
        df_external_validation_summary,
        df_land_estimation_assessment,
        zone_names=manager.zones,
    )
    raise_for_validation_errors(_validation_report)
    _validation_summary = (
        _validation_report
        if not _validation_report.empty
        else pd.DataFrame(
            [
                {
                    "artifact": "external_validation_outputs",
                    "check": "validation",
                    "severity": "pass",
                    "message": "All external validation artifact checks passed.",
                    "rows": 0,
                }
            ]
        )
    )

    external_validation_dir.mkdir(parents=True, exist_ok=True)
    _external_validation_artifacts = {
        "external_baseline_agreement": (
            df_external_baseline_agreement,
            external_validation_dir / "external_baseline_agreement.parquet",
        ),
        "external_growth_alignment": (
            df_external_growth_alignment,
            external_validation_dir / "external_growth_alignment.parquet",
        ),
        "external_review_flags": (
            df_external_review_flags,
            external_validation_dir / "external_review_flags.parquet",
        ),
        "external_validation_summary": (
            df_external_validation_summary,
            external_validation_dir / "external_validation_summary.parquet",
        ),
    }

    _export_rows: list[dict[str, object]] = []
    for _artifact_name, (_frame, _path) in _external_validation_artifacts.items():
        _frame.to_parquet(_path, index=False)
        _export_rows.append(
            {
                "artifact": _artifact_name,
                "path": str(_path),
                "rows": len(_frame),
            }
        )

    df_external_validation_artifacts = pd.DataFrame(_export_rows)
    mo.vstack(
        [
            mo.md("### External validation artifact validation"),
            _validation_summary,
            mo.md("### External validation artifacts"),
            df_external_validation_artifacts,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
