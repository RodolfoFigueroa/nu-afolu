import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415
    from datetime import UTC, datetime  # noqa: PLC0415
    from itertools import pairwise  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import ee  # noqa: PLC0415
    import marimo as mo  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    import seaborn as sns  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415
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
        UTC,
        datetime,
        ee,
        json,
        mo,
        np,
        os,
        pairwise,
        pd,
        plt,
        sns,
        xr,
        zone_partitions,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 06 Pseudo-Transition Allocation

    This notebook turns Chen-derived settlement-growth demand into diagnostic
    pseudo-`area_table` and pseudo-`transition_table` artifacts. The outputs are
    scenario-construction experiments, not approved carbon-model inputs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal, Inputs, And Guardrails

    The allocation step combines three inputs: GLC 2020 baseline area by AFOLU
    class, Chen decadal settlement demand from notebook `05`, and historical
    settlement-source priors from notebook `04`.

    The first implementation only models settlement expansion. Negative Chen deltas
    remain recorded as unresolved mismatch, and no non-settlement-to-non-settlement
    transitions are invented. Every emitted transition table must be explicit about
    its decadal interval semantics.
    """)
    return


@app.cell
def _(LABEL_LIST, Path, UTC, datetime, os, pd):
    PROJECT_ROOT = Path.cwd()
    OUT_PATH_KEY = "OUT_PATH"
    BASELINE_YEAR = 2020
    CHEN_SCALE_M = 1000
    VALIDATION_TOLERANCE_M2 = 1.0
    EPSILON_M2 = 1e-6
    CHEN_PIXEL_AREA_M2 = 1_000_000.0
    RATIO_DENOMINATOR_FLOOR_M2 = 1_000_000.0
    MANUAL_REVIEW_ABS_DIFF_HA = 5_000.0
    MANUAL_REVIEW_ABS_RATIO_ERROR = 1.0
    MIN_ZONE_PRIOR_AREA_M2 = 100_000.0
    MIN_ACTIVE_YEARS_FOR_ZONE_PRIOR = 3
    MAX_DOMINANT_SOURCE_SHARE = 0.90
    COMMON_SOURCE_SHARE_THRESHOLD = 0.05
    DEMAND_BASELINE_CHOICE = "glc_2020_settlements"
    DEMAND_CALIBRATION_CHOICE = "glc_2020_plus_clipped_chen_deltas"
    NEGATIVE_DELTA_POLICY = "clip_to_zero_and_record"
    INTERVAL_SEMANTICS = "decadal_start_year"
    SAVE_DIAGNOSTIC_ARTIFACTS = True
    OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "chen_pseudo_tables"
    DIAGNOSTIC_RUN_LABEL = "latest"

    # Explicitly carried forward from 03_spatial_resolution_diagnostics.py.
    HIGH_SPATIAL_RISK_ZONE_NAMES = (
        "02.2.02",
        "02.1.01",
        "02.2.03",
        "03.2.01",
        "26.1.01",
    )

    ALLOCATION_METHODS = (
        "historical_shares",
        "availability_constrained",
        "priority_ranking",
    )

    PRIOR_ZONE_SPECIFIC_STATUSES = (
        "zone_specific_candidate",
        "zone_specific_with_dominance_warning",
        "zone_specific_with_availability_warning",
    )

    PRIORITY_SOURCE_ORDER = (
        "croplands",
        "pastures",
        "grasslands",
        "shrublands",
        "other",
        "forests_secondary",
        "forests_primary",
        "forests_mangroves",
        "wetlands",
        "flooded",
    )
    NON_SETTLEMENT_LABELS = tuple(_label for _label in LABEL_LIST if _label != "settlements")


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
    GENERATED_AT_UTC = datetime.now(tz=UTC).isoformat()

    configuration_summary = pd.DataFrame(
        [
            {"setting": OUT_PATH_KEY, "value": str(OUT_PATH) if OUT_PATH else "not configured", "source": OUT_PATH_SOURCE},
            {"setting": "BASELINE_YEAR", "value": BASELINE_YEAR, "source": "analysis contract"},
            {"setting": "ALLOCATION_METHODS", "value": ", ".join(ALLOCATION_METHODS), "source": "pseudo_transition_methods.md"},
            {"setting": "OUTPUT_ROOT", "value": str(OUTPUT_ROOT), "source": "repo-local diagnostic output"},
            {"setting": "SAVE_DIAGNOSTIC_ARTIFACTS", "value": SAVE_DIAGNOSTIC_ARTIFACTS, "source": "notebook default"},
            {"setting": "VALIDATION_TOLERANCE_M2", "value": VALIDATION_TOLERANCE_M2, "source": "notebook default"},
        ]
    )

    configuration_summary
    return (
        ALLOCATION_METHODS,
        BASELINE_YEAR,
        CHEN_PIXEL_AREA_M2,
        CHEN_SCALE_M,
        COMMON_SOURCE_SHARE_THRESHOLD,
        DEMAND_BASELINE_CHOICE,
        DEMAND_CALIBRATION_CHOICE,
        DIAGNOSTIC_RUN_LABEL,
        EPSILON_M2,
        GENERATED_AT_UTC,
        HIGH_SPATIAL_RISK_ZONE_NAMES,
        INTERVAL_SEMANTICS,
        MANUAL_REVIEW_ABS_DIFF_HA,
        MANUAL_REVIEW_ABS_RATIO_ERROR,
        MAX_DOMINANT_SOURCE_SHARE,
        MIN_ACTIVE_YEARS_FOR_ZONE_PRIOR,
        MIN_ZONE_PRIOR_AREA_M2,
        NEGATIVE_DELTA_POLICY,
        NON_SETTLEMENT_LABELS,
        OUTPUT_ROOT,
        OUT_PATH,
        PRIORITY_SOURCE_ORDER,
        PRIOR_ZONE_SPECIFIC_STATUSES,
        RATIO_DENOMINATOR_FLOOR_M2,
        SAVE_DIAGNOSTIC_ARTIFACTS,
        VALIDATION_TOLERANCE_M2,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Artifact Contract Being Reproduced

    The pseudo-`area_table` keeps the historical schema: rows are years, columns are
    the full AFOLU `LABEL_LIST`, and values are square meters. The
    pseudo-`transition_table` keeps dimensions `year`, `start`, and `end`, but here
    `year=2020` means the decadal interval `2020 -> 2030`, not an annual transition.
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
            {"metric": "zones with required inputs", "value": len(candidate_zone_names)},
            {"metric": "bbox/ee files discovered", "value": len(artifact_zone_sets["bbox_ee"])},
            {"metric": "area_table files discovered", "value": len(artifact_zone_sets["area_table"])},
            {"metric": "transition_table files discovered", "value": len(artifact_zone_sets["transition_table"])},
        ]
    )

    input_zone_summary
    return (candidate_zone_names,)


@app.cell
def _(BASELINE_YEAR, LABEL_LIST, OUT_PATH, candidate_zone_names, np, pd, xr):
    def normalize_area_table(area_table: pd.DataFrame) -> pd.DataFrame:
        normalized = area_table.copy()
        normalized.index = normalized.index.astype(int)
        normalized.index.name = "year"
        normalized = normalized.reindex(columns=list(LABEL_LIST))
        normalized = normalized.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return normalized.sort_index()


    loaded_area_tables = {}
    loaded_transition_tables = {}
    _load_error_rows = []

    for _zone_name in candidate_zone_names:
        _area_path = OUT_PATH / "area_table" / f"{_zone_name}.parquet"
        _transition_path = OUT_PATH / "transition_table" / f"{_zone_name}.nc"
        try:
            loaded_area_tables[_zone_name] = normalize_area_table(pd.read_parquet(_area_path))
        except Exception as _exc:  # noqa: BLE001
            _load_error_rows.append({"zone": _zone_name, "artifact": "area_table", "error": str(_exc)})

        try:
            loaded_transition_tables[_zone_name] = xr.load_dataarray(_transition_path)
        except Exception as _exc:  # noqa: BLE001
            _load_error_rows.append({"zone": _zone_name, "artifact": "transition_table", "error": str(_exc)})

    loaded_zone_names = tuple(
        _zone_name
        for _zone_name in candidate_zone_names
        if _zone_name in loaded_area_tables and _zone_name in loaded_transition_tables
    )
    load_errors = pd.DataFrame(_load_error_rows, columns=["zone", "artifact", "error"])

    _baseline_rows = []
    _baseline_error_rows = []
    for _zone_name in loaded_zone_names:
        _area = loaded_area_tables[_zone_name]
        if BASELINE_YEAR not in _area.index:
            _baseline_error_rows.append({"zone": _zone_name, "error": f"missing {BASELINE_YEAR} area row"})
            continue
        _baseline = _area.loc[BASELINE_YEAR].astype(float)
        if not np.isfinite(_baseline.to_numpy()).all() or (_baseline < 0).any():
            _baseline_error_rows.append({"zone": _zone_name, "error": "invalid baseline values"})
            continue
        _baseline_rows.append({"zone": _zone_name, **{_label: float(_baseline[_label]) for _label in LABEL_LIST}})

    baseline_area_by_zone = pd.DataFrame(_baseline_rows)
    baseline_load_errors = pd.DataFrame(_baseline_error_rows, columns=["zone", "error"])
    allocation_zone_names = tuple(baseline_area_by_zone["zone"])

    load_summary = pd.DataFrame(
        [
            {"metric": "candidate zones", "value": len(candidate_zone_names)},
            {"metric": "zones with both historical tables loaded", "value": len(loaded_zone_names)},
            {"metric": "zones with usable 2020 baseline", "value": len(allocation_zone_names)},
            {"metric": "table load errors", "value": len(load_errors)},
            {"metric": "baseline load errors", "value": len(baseline_load_errors)},
        ]
    )

    load_summary
    return (
        allocation_zone_names,
        baseline_area_by_zone,
        baseline_load_errors,
        load_errors,
        loaded_transition_tables,
    )


@app.cell
def _(baseline_load_errors, load_errors, pd):
    pd.concat(
        [
            load_errors.assign(error_type="table_load"),
            baseline_load_errors.assign(artifact="area_table", error_type="baseline_load"),
        ],
        ignore_index=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Historical Settlement-Source Priors

    The allocation methods need source-class weights. This section rebuilds the
    historical priors from notebook `04`: zone-specific shares where the evidence is
    strong enough, and a pooled excluding-high-spatial-risk prior as fallback.
    """)
    return


@app.cell
def _(
    NON_SETTLEMENT_LABELS,
    allocation_zone_names,
    loaded_transition_tables,
    pd,
    xr,
):
    def transition_slice_to_frame(
        zone_name: str,
        transition_table: xr.DataArray,
        *,
        start_labels: tuple[str, ...],
    ) -> pd.DataFrame:
        frame = (
            transition_table.sel(end="settlements", start=list(start_labels))
            .to_series()
            .rename("area_m2")
            .reset_index()
        )
        return frame.assign(zone=zone_name).loc[:, ["zone", "year", "start", "area_m2"]]


    _new_transition_frames = [
        transition_slice_to_frame(
            _zone_name,
            loaded_transition_tables[_zone_name],
            start_labels=NON_SETTLEMENT_LABELS,
        )
        for _zone_name in allocation_zone_names
    ]

    new_settlement_transitions = pd.concat(_new_transition_frames, ignore_index=True)
    new_settlement_transitions = new_settlement_transitions.assign(
        area_ha=lambda _df: _df["area_m2"] / 10_000.0,
        has_new_settlement_transition=lambda _df: _df["area_m2"] > 0,
    )

    settlement_transition_summary = pd.DataFrame(
        [
            {"metric": "zones", "value": new_settlement_transitions["zone"].nunique()},
            {"metric": "new transition rows", "value": len(new_settlement_transitions)},
            {"metric": "rows with positive new settlement area", "value": int(new_settlement_transitions["has_new_settlement_transition"].sum())},
            {"metric": "total new settlement area ha", "value": float(new_settlement_transitions["area_ha"].sum())},
        ]
    )

    settlement_transition_summary
    return (new_settlement_transitions,)


@app.cell
def _(
    CHEN_PIXEL_AREA_M2,
    COMMON_SOURCE_SHARE_THRESHOLD,
    HIGH_SPATIAL_RISK_ZONE_NAMES,
    MAX_DOMINANT_SOURCE_SHARE,
    MIN_ACTIVE_YEARS_FOR_ZONE_PRIOR,
    MIN_ZONE_PRIOR_AREA_M2,
    NON_SETTLEMENT_LABELS,
    baseline_area_by_zone,
    new_settlement_transitions,
    np,
    pd,
):
    source_area_by_zone = (
        new_settlement_transitions.groupby(["zone", "start"], dropna=False)["area_m2"]
        .sum()
        .reset_index()
        .rename(columns={"start": "source_class", "area_m2": "new_settlement_area_m2"})
    )
    source_area_by_zone["zone_total_new_settlement_m2"] = source_area_by_zone.groupby("zone")[
        "new_settlement_area_m2"
    ].transform("sum")
    source_area_by_zone = source_area_by_zone.assign(
        source_share=np.where(
            source_area_by_zone["zone_total_new_settlement_m2"] > 0,
            source_area_by_zone["new_settlement_area_m2"] / source_area_by_zone["zone_total_new_settlement_m2"],
            0.0,
        ),
        high_spatial_risk=lambda _df: _df["zone"].isin(HIGH_SPATIAL_RISK_ZONE_NAMES),
    )

    historical_source_share_by_zone = source_area_by_zone.sort_values(
        ["zone", "source_share"],
        ascending=[True, False],
    ).reset_index(drop=True)

    _non_risk_transitions = new_settlement_transitions.loc[
        ~new_settlement_transitions["zone"].isin(HIGH_SPATIAL_RISK_ZONE_NAMES)
    ]
    pooled_source_prior_excluding_high_risk = (
        _non_risk_transitions.groupby("start", dropna=False)["area_m2"]
        .sum()
        .reset_index()
        .rename(columns={"start": "source_class", "area_m2": "new_settlement_area_m2"})
    )
    pooled_non_risk_total_new_settlement_m2 = float(
        pooled_source_prior_excluding_high_risk["new_settlement_area_m2"].sum()
    )
    pooled_source_prior_excluding_high_risk = pooled_source_prior_excluding_high_risk.assign(
        source_share=lambda _df: _df["new_settlement_area_m2"] / pooled_non_risk_total_new_settlement_m2
        if pooled_non_risk_total_new_settlement_m2
        else 0.0,
        prior_scope="excluding_high_spatial_risk_zones",
    ).sort_values("source_share", ascending=False)

    source_availability_2020 = baseline_area_by_zone.melt(
        id_vars="zone",
        value_vars=list(NON_SETTLEMENT_LABELS),
        var_name="source_class",
        value_name="baseline_source_area_m2",
    )

    source_availability_with_shares = historical_source_share_by_zone.merge(
        source_availability_2020,
        on=["zone", "source_class"],
        how="left",
    )
    source_availability_with_shares = source_availability_with_shares.assign(
        common_historical_source=lambda _df: _df["source_share"] >= COMMON_SOURCE_SHARE_THRESHOLD,
        scarce_in_baseline=lambda _df: _df["baseline_source_area_m2"].fillna(0) < CHEN_PIXEL_AREA_M2,
    )
    scarce_common_sources = source_availability_with_shares.loc[
        source_availability_with_shares["common_historical_source"]
        & source_availability_with_shares["scarce_in_baseline"]
        & (source_availability_with_shares["new_settlement_area_m2"] > 0)
    ]

    zone_prior_base = (
        new_settlement_transitions.groupby("zone", dropna=False)
        .agg(
            total_new_settlement_m2=("area_m2", "sum"),
            active_year_count=("area_m2", lambda _series: int((_series > 0).groupby(new_settlement_transitions.loc[_series.index, "year"]).any().sum())),
            positive_source_year_rows=("has_new_settlement_transition", "sum"),
        )
        .reset_index()
    )

    dominant_source_by_zone = (
        historical_source_share_by_zone.sort_values(["zone", "source_share"], ascending=[True, False])
        .groupby("zone", as_index=False)
        .first()
        .rename(columns={"source_class": "dominant_source_class", "source_share": "dominant_source_share"})
        .loc[:, ["zone", "dominant_source_class", "dominant_source_share"]]
    )
    scarce_common_count_by_zone = (
        scarce_common_sources.groupby("zone")
        .size()
        .rename("scarce_common_source_count")
        .reset_index()
    )

    zone_prior_quality = (
        zone_prior_base.merge(dominant_source_by_zone, on="zone", how="left")
        .merge(scarce_common_count_by_zone, on="zone", how="left")
        .assign(
            scarce_common_source_count=lambda _df: _df["scarce_common_source_count"].fillna(0).astype(int),
            high_spatial_risk=lambda _df: _df["zone"].isin(HIGH_SPATIAL_RISK_ZONE_NAMES),
        )
    )


    def assign_prior_quality_status(row: pd.Series) -> str:
        if row["total_new_settlement_m2"] <= 0:
            return "pooled_required_no_history"
        if row["high_spatial_risk"]:
            return "review_or_pooled_spatial_risk"
        if (
            row["total_new_settlement_m2"] < MIN_ZONE_PRIOR_AREA_M2
            or row["active_year_count"] < MIN_ACTIVE_YEARS_FOR_ZONE_PRIOR
        ):
            return "pooled_preferred_sparse_history"
        if row["dominant_source_share"] > MAX_DOMINANT_SOURCE_SHARE:
            return "zone_specific_with_dominance_warning"
        if row["scarce_common_source_count"] > 0:
            return "zone_specific_with_availability_warning"
        return "zone_specific_candidate"


    zone_prior_quality = zone_prior_quality.assign(
        total_new_settlement_ha=lambda _df: _df["total_new_settlement_m2"] / 10_000.0,
        prior_quality_status=lambda _df: _df.apply(assign_prior_quality_status, axis=1),
    )

    prior_quality_summary = (
        zone_prior_quality["prior_quality_status"]
        .value_counts()
        .rename_axis("prior_quality_status")
        .reset_index(name="zone_count")
    )

    prior_quality_summary
    return (
        historical_source_share_by_zone,
        pooled_source_prior_excluding_high_risk,
        zone_prior_quality,
    )


@app.cell
def _(pooled_source_prior_excluding_high_risk):
    pooled_source_prior_excluding_high_risk
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Settlement Demand From Chen

    The allocation notebook recomputes the notebook `05` demand table so it can be
    run independently. Chen images are interpreted by the local `CHEN_YEARS`
    contract, reduced over each zone geometry, then converted into adjacent decadal
    deltas.
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
    allocation_zone_names,
    chen_area_band_keys,
    chen_area_stack,
    chen_source_error,
    chen_source_ready,
    ee,
    load_ee_geometry,
    pd,
    short_error,
):
    _trajectory_rows = []
    _chen_error_rows = []

    if chen_source_ready and chen_area_stack is not None:
        for _zone_name in allocation_zone_names:
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
            for _zone_name in allocation_zone_names
        )

    chen_urban_trajectory = pd.DataFrame(
        _trajectory_rows,
        columns=["zone", "ssp", "year", "chen_urban_area_m2", "chen_urban_area_ha"],
    )
    chen_reduction_errors = pd.DataFrame(_chen_error_rows, columns=["zone", "error"])

    chen_reduction_summary = pd.DataFrame(
        [
            {"metric": "zones requested", "value": len(allocation_zone_names)},
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


@app.cell
def _(
    BASELINE_YEAR,
    DEMAND_BASELINE_CHOICE,
    DEMAND_CALIBRATION_CHOICE,
    MANUAL_REVIEW_ABS_DIFF_HA,
    MANUAL_REVIEW_ABS_RATIO_ERROR,
    NEGATIVE_DELTA_POLICY,
    RATIO_DENOMINATOR_FLOOR_M2,
    baseline_area_by_zone,
    chen_urban_trajectory,
    np,
    pd,
):
    glc_settlement_baseline = baseline_area_by_zone.loc[
        :,
        ["zone", "settlements"],
    ].rename(columns={"settlements": "glc_settlements_2020_m2"})
    glc_settlement_baseline = glc_settlement_baseline.assign(
        glc_settlements_2020_ha=lambda _df: _df["glc_settlements_2020_m2"] / 10_000.0,
        near_zero_glc_baseline=lambda _df: _df["glc_settlements_2020_m2"] < RATIO_DENOMINATOR_FLOOR_M2,
    )

    chen_2020_compatibility = chen_urban_trajectory.loc[
        chen_urban_trajectory["year"] == BASELINE_YEAR
    ].merge(
        glc_settlement_baseline,
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


    def assign_compatibility_decision(row: pd.Series) -> str:
        ratio_flag = (
            pd.notna(row["max_abs_ratio_error"])
            and row["max_abs_ratio_error"] > MANUAL_REVIEW_ABS_RATIO_ERROR
        )
        absolute_flag = row["max_abs_difference_ha"] > MANUAL_REVIEW_ABS_DIFF_HA
        if row["ratio_unstable_rows"] > 0:
            return "manual_review_small_baseline"
        if ratio_flag and absolute_flag:
            return "manual_review_large_absolute_and_ratio_mismatch"
        if absolute_flag:
            return "manual_review_large_absolute_mismatch"
        if ratio_flag:
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
        compatibility_decision=lambda _df: _df.apply(assign_compatibility_decision, axis=1),
        needs_manual_review=lambda _df: _df["compatibility_decision"].str.startswith("manual_review"),
    )

    _trajectory_sorted = chen_urban_trajectory.sort_values(["zone", "ssp", "year"]).copy()
    _trajectory_sorted["start_year"] = _trajectory_sorted.groupby(["zone", "ssp"])["year"].shift()
    _trajectory_sorted["start_chen_urban_area_m2"] = _trajectory_sorted.groupby(["zone", "ssp"])[
        "chen_urban_area_m2"
    ].shift()

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
        glc_settlement_baseline,
        on="zone",
        how="left",
    ).merge(
        zone_compatibility_decisions[["zone", "compatibility_decision", "needs_manual_review"]],
        on="zone",
        how="left",
    )

    chen_settlement_demand_table = chen_settlement_demand_table.sort_values(
        ["zone", "ssp", "start_year", "end_year"]
    ).reset_index(drop=True)
    chen_settlement_demand_table["cumulative_demand_from_2020_m2"] = chen_settlement_demand_table.groupby(
        ["zone", "ssp"]
    )["demand_m2"].cumsum()
    chen_settlement_demand_table = chen_settlement_demand_table.assign(
        glc_plus_clipped_demand_area_m2=lambda _df: _df["glc_settlements_2020_m2"] + _df["cumulative_demand_from_2020_m2"],
        baseline_choice=DEMAND_BASELINE_CHOICE,
        calibration_choice=DEMAND_CALIBRATION_CHOICE,
        negative_delta_policy=NEGATIVE_DELTA_POLICY,
    )

    chen_settlement_demand_table
    return chen_settlement_demand_table, zone_compatibility_decisions


@app.cell
def _(
    chen_settlement_demand_table,
    chen_urban_trajectory,
    pd,
    zone_compatibility_decisions,
):
    demand_summary_by_method_input = (
        chen_settlement_demand_table.groupby(["ssp", "delta_status"], dropna=False)
        .agg(
            interval_rows=("zone", "count"),
            zones=("zone", "nunique"),
            total_demand_ha=("demand_ha", "sum"),
            total_clipped_negative_delta_ha=("clipped_negative_delta_ha", "sum"),
        )
        .reset_index()
    )

    demand_readiness_summary = pd.DataFrame(
        [
            {"metric": "zones with Chen trajectories", "value": chen_urban_trajectory["zone"].nunique()},
            {"metric": "demand rows", "value": len(chen_settlement_demand_table)},
            {"metric": "positive demand rows", "value": int((chen_settlement_demand_table["delta_status"] == "positive").sum())},
            {"metric": "zero demand rows", "value": int((chen_settlement_demand_table["delta_status"] == "zero").sum())},
            {"metric": "negative demand rows", "value": int((chen_settlement_demand_table["delta_status"] == "negative").sum())},
            {"metric": "compatibility manual-review zones", "value": int(zone_compatibility_decisions["needs_manual_review"].sum())},
        ]
    )

    demand_readiness_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Allocation Methods Included

    Three candidate methods are implemented. `historical_shares` uses zone-specific
    historical priors when the prior-quality table says they are usable and falls
    back to the pooled excluding-high-risk prior otherwise. `availability_constrained`
    weights those same priors by available source area at the start of each
    interval. `priority_ranking` allocates in the explicit priority order from
    `pseudo_transition_methods.md`.
    """)
    return


@app.cell
def _(
    ALLOCATION_METHODS,
    LABEL_LIST,
    NON_SETTLEMENT_LABELS,
    PRIORITY_SOURCE_ORDER,
    baseline_area_by_zone,
    chen_settlement_demand_table,
    historical_source_share_by_zone,
    pd,
    pooled_source_prior_excluding_high_risk,
    zone_prior_quality,
):
    baseline_area_lookup = (
        baseline_area_by_zone.set_index("zone")
        .loc[:, list(LABEL_LIST)]
        .astype(float)
    )

    zone_prior_quality_lookup = zone_prior_quality.set_index("zone")["prior_quality_status"].to_dict()

    zone_source_share_matrix = (
        historical_source_share_by_zone.pivot_table(
            index="zone",
            columns="source_class",
            values="source_share",
            fill_value=0.0,
        )
        .reindex(columns=list(NON_SETTLEMENT_LABELS), fill_value=0.0)
    )

    pooled_source_share_series = (
        pooled_source_prior_excluding_high_risk.set_index("source_class")["source_share"]
        .reindex(list(NON_SETTLEMENT_LABELS))
        .fillna(0.0)
        .astype(float)
    )
    if float(pooled_source_share_series.sum()) > 0:
        pooled_source_share_series = pooled_source_share_series / float(pooled_source_share_series.sum())

    priority_rank_lookup = {
        _source_class: _rank for _rank, _source_class in enumerate(PRIORITY_SOURCE_ORDER, start=1)
    }

    allocation_input_summary = pd.DataFrame(
        [
            {"metric": "baseline zones", "value": len(baseline_area_lookup)},
            {"metric": "demand rows", "value": len(chen_settlement_demand_table)},
            {"metric": "allocation methods", "value": len(ALLOCATION_METHODS)},
            {"metric": "non-settlement source classes", "value": len(NON_SETTLEMENT_LABELS)},
            {"metric": "priority source classes", "value": len(PRIORITY_SOURCE_ORDER)},
        ]
    )

    allocation_input_summary
    return (
        baseline_area_lookup,
        pooled_source_share_series,
        priority_rank_lookup,
        zone_prior_quality_lookup,
        zone_source_share_matrix,
    )


@app.cell
def _(
    NON_SETTLEMENT_LABELS,
    PRIOR_ZONE_SPECIFIC_STATUSES,
    pd,
    pooled_source_share_series,
    zone_prior_quality_lookup,
    zone_source_share_matrix,
):
    def source_prior_for_zone(zone_name: str) -> tuple[pd.Series, str, str]:
        prior_quality_status = zone_prior_quality_lookup.get(zone_name, "pooled_required_no_history")
        if (
            prior_quality_status in PRIOR_ZONE_SPECIFIC_STATUSES
            and zone_name in zone_source_share_matrix.index
            and float(zone_source_share_matrix.loc[zone_name].sum()) > 0
        ):
            prior = zone_source_share_matrix.loc[zone_name].astype(float)
            prior_scope = "zone_specific"
        else:
            prior = pooled_source_share_series.copy()
            prior_scope = "pooled_excluding_high_spatial_risk"

        prior = prior.reindex(list(NON_SETTLEMENT_LABELS)).fillna(0.0).clip(lower=0.0)
        if float(prior.sum()) > 0:
            prior = prior / float(prior.sum())
        return prior, prior_scope, prior_quality_status


    "source prior helper defined"
    return (source_prior_for_zone,)


@app.cell
def _(
    ALLOCATION_METHODS,
    SSP_NAMES,
    VALIDATION_TOLERANCE_M2,
    allocation_zone_names,
    build_scenario_tables,
    chen_settlement_demand_table,
    pd,
):
    pseudo_area_tables = {}
    pseudo_transition_tables = {}
    _interval_report_rows = []
    _source_report_rows = []

    for _method in ALLOCATION_METHODS:
        for _ssp in SSP_NAMES:
            _ssp_demand = chen_settlement_demand_table.loc[
                chen_settlement_demand_table["ssp"] == _ssp
            ]
            for _zone_name in allocation_zone_names:
                _demand_rows = _ssp_demand.loc[_ssp_demand["zone"] == _zone_name]
                _area_table, _transition_table, _interval_records, _source_records = build_scenario_tables(
                    _method,
                    _zone_name,
                    _ssp,
                    _demand_rows,
                )
                pseudo_area_tables[(_method, _ssp, _zone_name)] = _area_table
                pseudo_transition_tables[(_method, _ssp, _zone_name)] = _transition_table
                _interval_report_rows.extend(_interval_records)
                _source_report_rows.extend(_source_records)

    allocation_interval_report = pd.DataFrame(_interval_report_rows)
    allocation_source_report = pd.DataFrame(_source_report_rows)

    allocation_summary_by_method_ssp = (
        allocation_interval_report.groupby(["method", "ssp"], dropna=False)
        .agg(
            zones=("zone", "nunique"),
            intervals=("start_year", "count"),
            total_demand_ha=("demand_m2", lambda _series: float(_series.sum() / 10_000.0)),
            total_allocated_ha=("allocated_m2", lambda _series: float(_series.sum() / 10_000.0)),
            total_unresolved_demand_ha=("unresolved_demand_m2", lambda _series: float(_series.sum() / 10_000.0)),
            intervals_with_unresolved_demand=("unresolved_demand_m2", lambda _series: int((_series > VALIDATION_TOLERANCE_M2).sum())),
            manual_review_intervals=("needs_manual_review", "sum"),
        )
        .reset_index()
    )

    allocation_summary_by_method_ssp
    return (
        allocation_interval_report,
        allocation_source_report,
        allocation_summary_by_method_ssp,
        pseudo_area_tables,
        pseudo_transition_tables,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Allocation Diagnostics

    The summary table reports how much positive Chen demand each method can allocate
    after source availability caps are enforced. Unresolved demand is not hidden; it
    means the method ran out of eligible source area in that zone and interval.
    """)
    return


@app.cell
def _(EPSILON_M2, allocation_source_report):
    source_allocation_summary = (
        allocation_source_report.groupby(["method", "source_class"], dropna=False)
        .agg(
            allocated_ha=("allocated_m2", lambda _series: float(_series.sum() / 10_000.0)),
            exhausted_interval_count=("source_exhausted", "sum"),
            positive_allocation_rows=("allocated_m2", lambda _series: int((_series > EPSILON_M2).sum())),
        )
        .reset_index()
        .sort_values(["method", "allocated_ha"], ascending=[True, False])
    )

    source_allocation_summary
    return (source_allocation_summary,)


@app.cell
def _(plt, sns, source_allocation_summary):
    sns.set_theme(style="whitegrid")

    _top_source_plot_data = source_allocation_summary.loc[
        source_allocation_summary["allocated_ha"] > 0
    ]

    _source_fig, _source_ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=_top_source_plot_data,
        x="allocated_ha",
        y="source_class",
        hue="method",
        ax=_source_ax,
    )
    _source_ax.set_xlabel("Allocated future settlement source area (ha)")
    _source_ax.set_ylabel("Source class")
    _source_ax.set_title("Allocated source classes by method")
    _source_ax.legend(title="Method", loc="lower right")
    _source_fig.tight_layout()

    source_allocation_plot = _source_fig

    source_allocation_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mass-Balance Checks

    Before anything is written to disk, every method/SSP/zone pseudo-table is
    checked against the historical table contract. The checks cover label sets,
    dimensions, finite non-negative values, transition row and column sums, source
    over-allocation, and reconciliation between clipped Chen demand and allocated
    settlement gain.
    """)
    return


@app.cell
def _(
    CHEN_YEARS,
    LABEL_LIST,
    NON_SETTLEMENT_LABELS,
    VALIDATION_TOLERANCE_M2,
    allocation_interval_report,
    allocation_source_report,
    np,
    pd,
    pseudo_area_tables,
    pseudo_transition_tables,
):
    allocation_source_report_with_checks = allocation_source_report.assign(
        source_overallocated_m2=lambda _df: (
            _df["allocated_m2"] - _df["source_available_start_m2"]
        ).clip(lower=0.0)
    )

    _source_overallocation_lookup = (
        allocation_source_report_with_checks.groupby(["method", "ssp", "zone"], dropna=False)[
            "source_overallocated_m2"
        ]
        .max()
        .to_dict()
    )

    _validation_rows = []
    for (_method, _ssp, _zone_name), _area_table in pseudo_area_tables.items():
        _transition_table = pseudo_transition_tables[(_method, _ssp, _zone_name)]
        _intervals = allocation_interval_report.loc[
            (allocation_interval_report["method"] == _method)
            & (allocation_interval_report["ssp"] == _ssp)
            & (allocation_interval_report["zone"] == _zone_name)
        ].sort_values(["start_year", "end_year"])

        _area_values = _area_table.to_numpy(dtype=float)
        _transition_values = _transition_table.to_numpy()
        _area_labels_valid = list(_area_table.columns) == list(LABEL_LIST)
        _area_years_valid = tuple(int(_year) for _year in _area_table.index) == tuple(CHEN_YEARS)
        _transition_dims_valid = tuple(_transition_table.dims) == ("year", "start", "end")
        _transition_coords_valid = (
            tuple(str(_label) for _label in _transition_table.coords["start"].to_numpy()) == tuple(LABEL_LIST)
            and tuple(str(_label) for _label in _transition_table.coords["end"].to_numpy()) == tuple(LABEL_LIST)
            and tuple(int(_year) for _year in _transition_table.coords["year"].to_numpy()) == tuple(CHEN_YEARS[:-1])
        )
        _area_values_valid = bool(np.isfinite(_area_values).all() and (_area_values >= -VALIDATION_TOLERANCE_M2).all())
        _transition_values_valid = bool(np.isfinite(_transition_values).all() and (_transition_values >= -VALIDATION_TOLERANCE_M2).all())

        _max_start_mass_diff_m2 = 0.0
        _max_end_mass_diff_m2 = 0.0
        _max_settlement_gain_diff_m2 = 0.0
        for _interval in _intervals.itertuples(index=False):
            _start_year = int(_interval.start_year)
            _end_year = int(_interval.end_year)
            _start_area = _area_table.loc[_start_year].astype(float)
            _end_area = _area_table.loc[_end_year].astype(float)
            _transition_slice = _transition_table.sel(year=_start_year)
            _start_sum = _transition_slice.sum(dim="end").to_series().reindex(list(LABEL_LIST)).astype(float)
            _end_sum = _transition_slice.sum(dim="start").to_series().reindex(list(LABEL_LIST)).astype(float)
            _max_start_mass_diff_m2 = max(
                _max_start_mass_diff_m2,
                float((_start_sum - _start_area).abs().max()),
            )
            _max_end_mass_diff_m2 = max(
                _max_end_mass_diff_m2,
                float((_end_sum - _end_area).abs().max()),
            )
            _settlement_gain_from_transition_m2 = float(
                _transition_slice.sel(
                    start=list(NON_SETTLEMENT_LABELS),
                    end="settlements",
                ).sum()
            )
            _max_settlement_gain_diff_m2 = max(
                _max_settlement_gain_diff_m2,
                abs(_settlement_gain_from_transition_m2 - float(_interval.allocated_m2)),
            )

        _demand_reconciliation_diff_m2 = float(
            (
                _intervals["demand_m2"]
                - _intervals["allocated_m2"]
                - _intervals["unresolved_demand_m2"]
            )
            .abs()
            .max()
        )
        _source_overallocated_m2 = float(
            _source_overallocation_lookup.get((_method, _ssp, _zone_name), 0.0)
        )
        _unresolved_demand_m2 = float(_intervals["unresolved_demand_m2"].sum())
        _clipped_negative_delta_m2 = float(_intervals["clipped_negative_delta_m2"].sum())

        _validation_passed = bool(
            _area_labels_valid
            and _area_years_valid
            and _transition_dims_valid
            and _transition_coords_valid
            and _area_values_valid
            and _transition_values_valid
            and _max_start_mass_diff_m2 <= VALIDATION_TOLERANCE_M2
            and _max_end_mass_diff_m2 <= VALIDATION_TOLERANCE_M2
            and _max_settlement_gain_diff_m2 <= VALIDATION_TOLERANCE_M2
            and _demand_reconciliation_diff_m2 <= VALIDATION_TOLERANCE_M2
            and _source_overallocated_m2 <= VALIDATION_TOLERANCE_M2
        )

        _validation_rows.append(
            {
                "method": _method,
                "ssp": _ssp,
                "zone": _zone_name,
                "area_labels_valid": _area_labels_valid,
                "area_years_valid": _area_years_valid,
                "transition_dims_valid": _transition_dims_valid,
                "transition_coords_valid": _transition_coords_valid,
                "area_values_valid": _area_values_valid,
                "transition_values_valid": _transition_values_valid,
                "max_start_mass_diff_m2": _max_start_mass_diff_m2,
                "max_end_mass_diff_m2": _max_end_mass_diff_m2,
                "max_settlement_gain_diff_m2": _max_settlement_gain_diff_m2,
                "max_demand_reconciliation_diff_m2": _demand_reconciliation_diff_m2,
                "max_source_overallocated_m2": _source_overallocated_m2,
                "unresolved_demand_m2": _unresolved_demand_m2,
                "clipped_negative_delta_m2": _clipped_negative_delta_m2,
                "validation_passed": _validation_passed,
            }
        )

    validation_report = pd.DataFrame(_validation_rows)
    validation_summary_by_method_ssp = (
        validation_report.groupby(["method", "ssp"], dropna=False)
        .agg(
            zones=("zone", "nunique"),
            validation_passed=("validation_passed", "all"),
            failed_zone_count=("validation_passed", lambda _series: int((~_series).sum())),
            max_start_mass_diff_m2=("max_start_mass_diff_m2", "max"),
            max_end_mass_diff_m2=("max_end_mass_diff_m2", "max"),
            max_settlement_gain_diff_m2=("max_settlement_gain_diff_m2", "max"),
            max_demand_reconciliation_diff_m2=("max_demand_reconciliation_diff_m2", "max"),
            max_source_overallocated_m2=("max_source_overallocated_m2", "max"),
            unresolved_demand_ha=("unresolved_demand_m2", lambda _series: float(_series.sum() / 10_000.0)),
        )
        .reset_index()
    )

    validation_summary_by_method_ssp
    return (
        allocation_source_report_with_checks,
        validation_report,
        validation_summary_by_method_ssp,
    )


@app.cell
def _(validation_report):
    validation_failures = validation_report.loc[
        ~validation_report["validation_passed"]
    ].sort_values(["method", "ssp", "zone"])

    validation_failures
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Output Artifacts And Provenance

    When all validation checks pass, the notebook writes diagnostic artifacts under
    `outputs/chen_pseudo_tables/latest/`. The directory hierarchy records method,
    baseline, calibration, and SSP. A manifest and validation reports are written
    alongside the table artifacts.
    """)
    return


@app.cell
def _(
    ALLOCATION_METHODS,
    DEMAND_BASELINE_CHOICE,
    DEMAND_CALIBRATION_CHOICE,
    DIAGNOSTIC_RUN_LABEL,
    GENERATED_AT_UTC,
    INTERVAL_SEMANTICS,
    NEGATIVE_DELTA_POLICY,
    OUTPUT_ROOT,
    SAVE_DIAGNOSTIC_ARTIFACTS,
    SSP_NAMES,
    VALIDATION_TOLERANCE_M2,
    allocation_interval_report,
    allocation_source_report_with_checks,
    allocation_zone_names,
    json,
    pd,
    pseudo_area_tables,
    pseudo_transition_tables,
    validation_report,
):
    all_validation_passed = bool(validation_report["validation_passed"].all())
    saved_artifact_rows = []
    save_error_rows = []

    if SAVE_DIAGNOSTIC_ARTIFACTS and all_validation_passed:
        _reports_dir = OUTPUT_ROOT / DIAGNOSTIC_RUN_LABEL / "reports"
        _reports_dir.mkdir(parents=True, exist_ok=True)

        for (_method, _ssp, _zone_name), _area_table in pseudo_area_tables.items():
            _transition_table = pseudo_transition_tables[(_method, _ssp, _zone_name)]
            _scenario_dir = (
                OUTPUT_ROOT
                / DIAGNOSTIC_RUN_LABEL
                / _method
                / "baseline_glc_2020"
                / "calibration_glc_2020_plus_clipped_chen_deltas"
                / _ssp
            )
            _area_dir = _scenario_dir / "area_table"
            _transition_dir = _scenario_dir / "transition_table"
            _area_dir.mkdir(parents=True, exist_ok=True)
            _transition_dir.mkdir(parents=True, exist_ok=True)
            _area_path = _area_dir / f"{_zone_name}.parquet"
            _transition_path = _transition_dir / f"{_zone_name}.nc"
            try:
                _area_table.to_parquet(_area_path)
                _transition_table.to_netcdf(_transition_path)
                saved_artifact_rows.append(
                    {
                        "method": _method,
                        "ssp": _ssp,
                        "zone": _zone_name,
                        "area_table_path": str(_area_path),
                        "transition_table_path": str(_transition_path),
                    }
                )
            except Exception as _exc:  # noqa: BLE001
                save_error_rows.append(
                    {
                        "method": _method,
                        "ssp": _ssp,
                        "zone": _zone_name,
                        "error": str(_exc),
                    }
                )

        saved_artifact_index = pd.DataFrame(saved_artifact_rows)
        save_errors = pd.DataFrame(save_error_rows, columns=["method", "ssp", "zone", "error"])

        allocation_interval_report.to_parquet(_reports_dir / "allocation_interval_report.parquet")
        allocation_source_report_with_checks.to_parquet(_reports_dir / "allocation_source_report.parquet")
        validation_report.to_parquet(_reports_dir / "validation_report.parquet")
        saved_artifact_index.to_parquet(_reports_dir / "saved_artifact_index.parquet")

        output_manifest = {
            "generated_at_utc": GENERATED_AT_UTC,
            "run_label": DIAGNOSTIC_RUN_LABEL,
            "output_root": str(OUTPUT_ROOT / DIAGNOSTIC_RUN_LABEL),
            "artifact_status": "written" if save_errors.empty else "write_errors",
            "methods": list(ALLOCATION_METHODS),
            "ssps": list(SSP_NAMES),
            "zone_count": len(allocation_zone_names),
            "baseline_choice": DEMAND_BASELINE_CHOICE,
            "calibration_choice": DEMAND_CALIBRATION_CHOICE,
            "negative_delta_policy": NEGATIVE_DELTA_POLICY,
            "interval_semantics": INTERVAL_SEMANTICS,
            "validation_tolerance_m2": VALIDATION_TOLERANCE_M2,
            "all_validation_passed": all_validation_passed,
            "artifact_count": len(saved_artifact_index),
            "report_files": {
                "allocation_interval_report": str(_reports_dir / "allocation_interval_report.parquet"),
                "allocation_source_report": str(_reports_dir / "allocation_source_report.parquet"),
                "validation_report": str(_reports_dir / "validation_report.parquet"),
                "saved_artifact_index": str(_reports_dir / "saved_artifact_index.parquet"),
            },
        }
        with (_reports_dir / "provenance_manifest.json").open("w", encoding="utf-8") as _file:
            json.dump(output_manifest, _file, indent=2)
    else:
        saved_artifact_index = pd.DataFrame(
            columns=["method", "ssp", "zone", "area_table_path", "transition_table_path"]
        )
        save_errors = pd.DataFrame(columns=["method", "ssp", "zone", "error"])
        output_manifest = {
            "generated_at_utc": GENERATED_AT_UTC,
            "run_label": DIAGNOSTIC_RUN_LABEL,
            "output_root": str(OUTPUT_ROOT / DIAGNOSTIC_RUN_LABEL),
            "artifact_status": "skipped_validation_failed"
            if SAVE_DIAGNOSTIC_ARTIFACTS
            else "skipped_by_configuration",
            "all_validation_passed": all_validation_passed,
        }

    save_summary = pd.DataFrame(
        [
            {"metric": "save requested", "value": SAVE_DIAGNOSTIC_ARTIFACTS},
            {"metric": "all validation passed", "value": all_validation_passed},
            {"metric": "artifacts written", "value": len(saved_artifact_index)},
            {"metric": "write errors", "value": len(save_errors)},
            {"metric": "output root", "value": str(OUTPUT_ROOT / DIAGNOSTIC_RUN_LABEL)},
        ]
    )

    save_summary
    return save_errors, saved_artifact_index


@app.cell
def _(save_errors):
    save_errors
    return


@app.cell
def _(
    allocation_summary_by_method_ssp,
    plt,
    sns,
    validation_summary_by_method_ssp,
):
    _validation_plot_data = validation_summary_by_method_ssp.copy()
    _validation_plot_data["unresolved_demand_ha"] = _validation_plot_data["unresolved_demand_ha"].round(8)

    _validation_fig, _validation_axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(
        data=allocation_summary_by_method_ssp,
        x="ssp",
        y="total_allocated_ha",
        hue="method",
        ax=_validation_axes[0],
    )
    _validation_axes[0].set_xlabel("SSP")
    _validation_axes[0].set_ylabel("Allocated settlement expansion (ha)")
    _validation_axes[0].set_title("Allocated demand by method and SSP")
    _validation_axes[0].legend(title="Method", fontsize="small")

    sns.barplot(
        data=_validation_plot_data,
        x="ssp",
        y="unresolved_demand_ha",
        hue="method",
        ax=_validation_axes[1],
    )
    _validation_axes[1].set_xlabel("SSP")
    _validation_axes[1].set_ylabel("Unresolved demand (ha)")
    _validation_axes[1].set_title("Unresolved demand after allocation caps")
    _validation_axes[1].legend(title="Method", fontsize="small")
    _validation_fig.tight_layout()

    validation_summary_plot = _validation_fig

    validation_summary_plot
    return


@app.cell
def _(
    allocation_interval_report,
    allocation_source_report,
    mo,
    pseudo_area_tables,
    saved_artifact_index,
    validation_report,
    zone_compatibility_decisions,
):
    _scenario_count = len(pseudo_area_tables)
    _failed_validation_count = int((~validation_report["validation_passed"]).sum())
    _total_unresolved_ha = float(allocation_interval_report["unresolved_demand_m2"].sum() / 10_000.0)
    _total_allocated_ha = float(allocation_interval_report["allocated_m2"].sum() / 10_000.0)
    _max_start_diff = float(validation_report["max_start_mass_diff_m2"].max())
    _max_end_diff = float(validation_report["max_end_mass_diff_m2"].max())
    _manual_review_zone_count = int(zone_compatibility_decisions["needs_manual_review"].sum())

    allocation_readout = mo.md(
        f"""
    ### Allocation Readout

    - Scenario table sets constructed: `{_scenario_count}`
    - Allocation interval rows: `{len(allocation_interval_report)}`
    - Source allocation rows: `{len(allocation_source_report)}`
    - Total allocated across method sensitivity cases: `{_total_allocated_ha:,.1f} ha`
    - Total unresolved demand across method sensitivity cases: `{_total_unresolved_ha:,.6f} ha`
    - Failed validation rows: `{_failed_validation_count}`
    - Max transition start mass-balance difference: `{_max_start_diff:.6g} m2`
    - Max transition end mass-balance difference: `{_max_end_diff:.6g} m2`
    - Compatibility manual-review zones carried forward: `{_manual_review_zone_count}`
    - Artifacts written: `{len(saved_artifact_index)}`

    These outputs are diagnostic scenario artifacts. Notebook `07` should only run
    the carbon model on artifacts whose validation report remains passing, and it
    should preserve the method, SSP, baseline, calibration, and interval semantics
    from the manifest.
    """
    )

    allocation_readout
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Limitations And Next Step

    The methods here are aggregate, zone-level allocation assumptions. They do not
    encode spatial suitability, adjacency to existing settlements, infrastructure,
    protected areas, or within-zone Chen-pixel placement. Priority ranking is an
    explicit sensitivity case, not an empirical model.

    Next, `07_carbon_model_scenario_runs.py` should load only validated diagnostic
    artifacts, run the carbon model without changing model internals unless decadal
    interval semantics require it, and label all outputs as Chen-derived scenario
    diagnostics.
    """)
    return


@app.cell
def _(
    BASELINE_YEAR,
    DEMAND_BASELINE_CHOICE,
    DEMAND_CALIBRATION_CHOICE,
    EPSILON_M2,
    INTERVAL_SEMANTICS,
    LABEL_LIST,
    NEGATIVE_DELTA_POLICY,
    NON_SETTLEMENT_LABELS,
    VALIDATION_TOLERANCE_M2,
    allocate_interval,
    baseline_area_lookup,
    np,
    pd,
    priority_rank_lookup,
    xr,
):
    def build_transition_matrix(start_area: pd.Series, allocations: pd.Series) -> pd.DataFrame:
        matrix = pd.DataFrame(0.0, index=list(LABEL_LIST), columns=list(LABEL_LIST))
        for label in LABEL_LIST:
            matrix.loc[label, label] = float(start_area[label])
        for source_class, allocation_value_m2 in allocations.items():
            allocation_amount_m2 = float(allocation_value_m2)
            if allocation_amount_m2 <= EPSILON_M2:
                continue
            matrix.loc[source_class, source_class] -= allocation_amount_m2
            matrix.loc[source_class, "settlements"] += allocation_amount_m2
        return matrix.clip(lower=0.0)


    def build_scenario_tables(
        method: str,
        zone_name: str,
        ssp: str,
        demand_rows: pd.DataFrame,
    ) -> tuple[pd.DataFrame, xr.DataArray, list[dict[str, object]], list[dict[str, object]]]:
        current_area = baseline_area_lookup.loc[zone_name].astype(float).copy()
        area_records = [{"year": BASELINE_YEAR, **{_label: float(current_area[_label]) for _label in LABEL_LIST}}]
        transition_matrices = []
        transition_years = []
        interval_records = []
        source_records = []

        for demand_row in demand_rows.sort_values(["start_year", "end_year"]).itertuples(index=False):
            start_year = int(demand_row.start_year)
            end_year = int(demand_row.end_year)
            start_area = current_area.copy()
            demand_m2 = float(demand_row.demand_m2)
            clipped_negative_delta_m2 = float(demand_row.clipped_negative_delta_m2)
            allocations, weights, unresolved_m2, prior_scope, prior_quality_status = allocate_interval(
                method,
                zone_name,
                current_area,
                demand_m2,
            )
            allocated_total_m2 = float(allocations.sum())
            transition_matrix = build_transition_matrix(start_area, allocations)

            current_area = start_area.copy()
            for source_class, allocation_value_m2 in allocations.items():
                current_area.loc[source_class] = max(float(current_area.loc[source_class]) - float(allocation_value_m2), 0.0)
            current_area.loc["settlements"] = float(current_area.loc["settlements"]) + allocated_total_m2

            area_records.append({"year": end_year, **{_label: float(current_area[_label]) for _label in LABEL_LIST}})
            transition_matrices.append(transition_matrix.to_numpy(dtype=float))
            transition_years.append(start_year)

            interval_records.append(
                {
                    "method": method,
                    "zone": zone_name,
                    "ssp": ssp,
                    "start_year": start_year,
                    "end_year": end_year,
                    "interval_years": int(demand_row.interval_years),
                    "demand_m2": demand_m2,
                    "allocated_m2": allocated_total_m2,
                    "unresolved_demand_m2": max(unresolved_m2, 0.0),
                    "clipped_negative_delta_m2": clipped_negative_delta_m2,
                    "prior_scope": prior_scope,
                    "prior_quality_status": prior_quality_status,
                    "compatibility_decision": str(demand_row.compatibility_decision),
                    "needs_manual_review": bool(demand_row.needs_manual_review),
                }
            )

            available = start_area.reindex(list(NON_SETTLEMENT_LABELS)).astype(float).clip(lower=0.0)
            interval_weights = weights.reindex(list(NON_SETTLEMENT_LABELS)).fillna(0.0)
            source_records.extend(
                {
                    "method": method,
                    "zone": zone_name,
                    "ssp": ssp,
                    "start_year": start_year,
                    "end_year": end_year,
                    "source_class": source_class,
                    "source_available_start_m2": float(available[source_class]),
                    "allocation_weight": float(interval_weights[source_class]),
                    "allocated_m2": float(allocations[source_class]),
                    "priority_rank": priority_rank_lookup.get(source_class, np.nan),
                    "source_exhausted": bool(
                        available[source_class] > EPSILON_M2
                        and available[source_class] - allocations[source_class] <= VALIDATION_TOLERANCE_M2
                    ),
                }
                for source_class in NON_SETTLEMENT_LABELS
            )

        area_table = (
            pd.DataFrame(area_records)
            .drop_duplicates(subset="year", keep="last")
            .set_index("year")
            .reindex(columns=list(LABEL_LIST))
            .astype(float)
        )
        transition_table = xr.DataArray(
            np.stack(transition_matrices, axis=0),
            dims=("year", "start", "end"),
            coords={
                "year": transition_years,
                "start": list(LABEL_LIST),
                "end": list(LABEL_LIST),
            },
            name="area_m2",
            attrs={
                "interval_semantics": INTERVAL_SEMANTICS,
                "baseline_choice": DEMAND_BASELINE_CHOICE,
                "calibration_choice": DEMAND_CALIBRATION_CHOICE,
                "negative_delta_policy": NEGATIVE_DELTA_POLICY,
                "method": method,
                "ssp": ssp,
                "zone": zone_name,
            },
        )
        area_table.index.name = "year"
        return area_table, transition_table, interval_records, source_records


    "scenario table builders defined"
    return (build_scenario_tables,)


@app.cell
def _(EPSILON_M2, NON_SETTLEMENT_LABELS, np, pd):
    def allocate_proportionally(
        demand_m2: float,
        available_by_source: pd.Series,
        weight_by_source: pd.Series,
    ) -> tuple[pd.Series, float]:
        allocations = pd.Series(0.0, index=list(NON_SETTLEMENT_LABELS))
        remaining_demand_m2 = max(float(demand_m2), 0.0)
        remaining_available = (
            available_by_source.reindex(list(NON_SETTLEMENT_LABELS))
            .fillna(0.0)
            .astype(float)
            .clip(lower=0.0)
        )
        weights = (
            weight_by_source.reindex(list(NON_SETTLEMENT_LABELS))
            .fillna(0.0)
            .astype(float)
            .clip(lower=0.0)
        )

        for _iteration in range(len(NON_SETTLEMENT_LABELS) + 2):
            if remaining_demand_m2 <= EPSILON_M2:
                break
            eligible = remaining_available > EPSILON_M2
            if not bool(eligible.any()):
                break
            effective_weights = weights.where(eligible, 0.0)
            if float(effective_weights.sum()) <= EPSILON_M2:
                effective_weights = remaining_available.where(eligible, 0.0)
            if float(effective_weights.sum()) <= EPSILON_M2:
                break

            proposed = remaining_demand_m2 * effective_weights / float(effective_weights.sum())
            capped = pd.Series(
                np.minimum(proposed.to_numpy(), remaining_available.to_numpy()),
                index=list(NON_SETTLEMENT_LABELS),
            )
            allocated_this_round = float(capped.sum())
            if allocated_this_round <= EPSILON_M2:
                break
            allocations = allocations + capped
            remaining_available = (remaining_available - capped).clip(lower=0.0)
            remaining_demand_m2 -= allocated_this_round

        return allocations.clip(lower=0.0), max(remaining_demand_m2, 0.0)


    "proportional allocator defined"
    return (allocate_proportionally,)


@app.cell
def _(
    EPSILON_M2,
    NON_SETTLEMENT_LABELS,
    allocate_by_priority,
    allocate_proportionally,
    pd,
    priority_rank_lookup,
    source_prior_for_zone,
):
    def allocate_interval(
        method: str,
        zone_name: str,
        current_area: pd.Series,
        demand_m2: float,
    ) -> tuple[pd.Series, pd.Series, float, str, str]:
        prior, prior_scope, prior_quality_status = source_prior_for_zone(zone_name)
        available = current_area.reindex(list(NON_SETTLEMENT_LABELS)).astype(float).clip(lower=0.0)

        if method == "historical_shares":
            weights = prior
            allocations, unresolved_m2 = allocate_proportionally(demand_m2, available, weights)
        elif method == "availability_constrained":
            weights = prior * available
            if float(weights.sum()) <= EPSILON_M2:
                weights = available
            allocations, unresolved_m2 = allocate_proportionally(demand_m2, available, weights)
            prior_scope = f"{prior_scope}_availability_weighted"
        elif method == "priority_ranking":
            weights = pd.Series(
                {
                    _source_class: 1.0 / priority_rank_lookup[_source_class]
                    if _source_class in priority_rank_lookup
                    else 0.0
                    for _source_class in NON_SETTLEMENT_LABELS
                }
            )
            allocations, unresolved_m2 = allocate_by_priority(demand_m2, available)
            prior_scope = "priority_order"
        else:
            error_message = f"Unknown allocation method: {method}"
            raise ValueError(error_message)

        return allocations, weights, unresolved_m2, prior_scope, prior_quality_status


    "interval allocator defined"
    return (allocate_interval,)


@app.cell
def _(EPSILON_M2, NON_SETTLEMENT_LABELS, PRIORITY_SOURCE_ORDER, pd):
    def allocate_by_priority(
        demand_m2: float,
        available_by_source: pd.Series,
    ) -> tuple[pd.Series, float]:
        allocations = pd.Series(0.0, index=list(NON_SETTLEMENT_LABELS))
        remaining_demand_m2 = max(float(demand_m2), 0.0)
        remaining_available = (
            available_by_source.reindex(list(NON_SETTLEMENT_LABELS))
            .fillna(0.0)
            .astype(float)
            .clip(lower=0.0)
        )

        for source_class in PRIORITY_SOURCE_ORDER:
            if remaining_demand_m2 <= EPSILON_M2:
                break
            available_m2 = float(remaining_available.get(source_class, 0.0))
            allocated_m2 = min(remaining_demand_m2, available_m2)
            allocations.loc[source_class] = allocated_m2
            remaining_available.loc[source_class] = max(available_m2 - allocated_m2, 0.0)
            remaining_demand_m2 -= allocated_m2

        return allocations, max(remaining_demand_m2, 0.0)


    "priority allocator defined"
    return (allocate_by_priority,)


if __name__ == "__main__":
    app.run()
