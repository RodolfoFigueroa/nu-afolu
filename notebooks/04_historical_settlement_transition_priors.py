import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def import_dependencies():
    import os  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import marimo as mo  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    import seaborn as sns  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415
    from dagster_components.partitions import zone_partitions  # noqa: PLC0415

    from nu_afolu.constants import LABEL_LIST  # noqa: PLC0415

    return LABEL_LIST, Path, mo, np, os, pd, plt, sns, xr, zone_partitions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 04 Historical Settlement Transition Priors

    This notebook summarizes observed historical transitions into `settlements`.
    The output is an empirical prior for later pseudo-transition allocation methods:
    when Chen implies future settlement growth, which historical AFOLU source
    classes have supplied settlement expansion?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal, Inputs, And Assumptions

    This stage uses historical `transition_table` and `area_table` artifacts only.
    It separates settlement persistence from new settlement expansion, computes
    source-class shares for non-settlement classes that became `settlements`, and
    assesses whether those shares are strong enough to use by zone or should be
    pooled.

    The priors are empirical summaries, not spatial suitability models. They should
    guide diagnostic allocation choices, not imply that historical source patterns
    must hold under every future SSP scenario.
    """)
    return


@app.cell
def configure_transition_prior_analysis(LABEL_LIST, Path, os, pd):
    PROJECT_ROOT = Path.cwd()
    OUT_PATH_KEY = "OUT_PATH"
    COMPARISON_YEAR = 2020
    CHEN_PIXEL_AREA_M2 = 1_000_000.0
    MIN_ZONE_PRIOR_AREA_M2 = 100_000.0
    MIN_ACTIVE_YEARS_FOR_ZONE_PRIOR = 3
    MAX_DOMINANT_SOURCE_SHARE = 0.90
    COMMON_SOURCE_SHARE_THRESHOLD = 0.05

    # Explicitly carried forward from 03_spatial_resolution_diagnostics.py.
    HIGH_SPATIAL_RISK_ZONE_NAMES = (
        "02.2.02",
        "02.1.01",
        "02.2.03",
        "03.2.01",
        "26.1.01",
    )

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
    NON_SETTLEMENT_LABELS = tuple(
        _label for _label in LABEL_LIST if _label != "settlements"
    )

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
                "source": "baseline availability year",
            },
            {
                "setting": "MIN_ZONE_PRIOR_AREA_M2",
                "value": MIN_ZONE_PRIOR_AREA_M2,
                "source": "notebook default",
            },
            {
                "setting": "MIN_ACTIVE_YEARS_FOR_ZONE_PRIOR",
                "value": MIN_ACTIVE_YEARS_FOR_ZONE_PRIOR,
                "source": "notebook default",
            },
            {
                "setting": "HIGH_SPATIAL_RISK_ZONE_NAMES",
                "value": ", ".join(HIGH_SPATIAL_RISK_ZONE_NAMES),
                "source": "notebook 03 readout",
            },
        ]
    )

    configuration_summary
    return (
        CHEN_PIXEL_AREA_M2,
        COMMON_SOURCE_SHARE_THRESHOLD,
        COMPARISON_YEAR,
        HIGH_SPATIAL_RISK_ZONE_NAMES,
        MAX_DOMINANT_SOURCE_SHARE,
        MIN_ACTIVE_YEARS_FOR_ZONE_PRIOR,
        MIN_ZONE_PRIOR_AREA_M2,
        NON_SETTLEMENT_LABELS,
        OUT_PATH,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Historical Tables Loaded

    The notebook rebuilds the zone list from local artifacts so it can be rerun
    independently. It loads only zones with both table artifacts available.
    """)
    return


@app.cell
def discover_table_zones(
    HIGH_SPATIAL_RISK_ZONE_NAMES,
    OUT_PATH,
    Path,
    pd,
    zone_partitions,
):
    REQUIRED_TABLE_SPECS = {
        "area_table": {"relative_dir": Path("area_table"), "extension": ".parquet"},
        "transition_table": {
            "relative_dir": Path("transition_table"),
            "extension": ".nc",
        },
    }

    partition_zone_names = tuple(zone_partitions.get_partition_keys())
    table_zone_sets = {}
    for _artifact_name, _spec in REQUIRED_TABLE_SPECS.items():
        _artifact_dir = OUT_PATH / _spec["relative_dir"] if OUT_PATH else None
        if _artifact_dir and _artifact_dir.exists():
            table_zone_sets[_artifact_name] = {
                _path.stem for _path in _artifact_dir.glob(f"*{_spec['extension']}")
            }
        else:
            table_zone_sets[_artifact_name] = set()

    _zone_rows = []
    for _zone_name in partition_zone_names:
        _row = {"zone": _zone_name}
        for _artifact_name in REQUIRED_TABLE_SPECS:
            _row[_artifact_name] = _zone_name in table_zone_sets[_artifact_name]
        _row["complete_table_set"] = all(
            _row[_artifact_name] for _artifact_name in REQUIRED_TABLE_SPECS
        )
        _zone_rows.append(_row)

    table_zone_inventory = pd.DataFrame(_zone_rows)
    candidate_zone_names = tuple(
        table_zone_inventory.loc[table_zone_inventory["complete_table_set"], "zone"]
    )

    table_zone_summary = pd.DataFrame(
        [
            {"metric": "canonical partition zones", "value": len(partition_zone_names)},
            {
                "metric": "zones with area and transition tables",
                "value": len(candidate_zone_names),
            },
            {
                "metric": "high spatial-risk zones present",
                "value": len(
                    set(candidate_zone_names) & set(HIGH_SPATIAL_RISK_ZONE_NAMES)
                ),
            },
        ]
    )

    table_zone_summary
    return (candidate_zone_names,)


@app.cell
def load_historical_tables(OUT_PATH, candidate_zone_names, pd, xr):
    loaded_area_tables = {}
    loaded_transition_tables = {}
    _load_error_rows = []

    for _zone_name in candidate_zone_names:
        _area_path = OUT_PATH / "area_table" / f"{_zone_name}.parquet"
        _transition_path = OUT_PATH / "transition_table" / f"{_zone_name}.nc"
        try:
            _area_table = pd.read_parquet(_area_path)
            _area_table.index = _area_table.index.astype(int)
            loaded_area_tables[_zone_name] = _area_table
        except Exception as _exc:  # noqa: BLE001
            _load_error_rows.append(
                {"zone": _zone_name, "artifact": "area_table", "error": str(_exc)}
            )

        try:
            loaded_transition_tables[_zone_name] = xr.load_dataarray(_transition_path)
        except Exception as _exc:  # noqa: BLE001
            _load_error_rows.append(
                {"zone": _zone_name, "artifact": "transition_table", "error": str(_exc)}
            )

    loaded_zone_names = tuple(
        _zone_name
        for _zone_name in candidate_zone_names
        if _zone_name in loaded_area_tables and _zone_name in loaded_transition_tables
    )

    load_errors = pd.DataFrame(_load_error_rows, columns=["zone", "artifact", "error"])
    load_summary = pd.DataFrame(
        [
            {"metric": "candidate zones", "value": len(candidate_zone_names)},
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
    ## Settlement Transitions

    The transition table records a full start/end matrix for each historical start
    year. For prior estimation, only transitions with `end == "settlements"` are
    used as evidence for settlement expansion. The cell below keeps settlement
    persistence separate from new settlement transitions.
    """)
    return


@app.cell
def define_transition_extractors(
    LABEL_LIST,
    NON_SETTLEMENT_LABELS,
    np,
    pd,
    xr,
):
    def normalize_area_table(area_table: pd.DataFrame) -> pd.DataFrame:
        normalized = area_table.copy()
        normalized.index = normalized.index.astype(int)
        normalized.index.name = "year"
        normalized = normalized.reindex(columns=list(LABEL_LIST))
        normalized = normalized.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return normalized.sort_index()

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

    def build_settlement_reconciliation_rows(
        zone_name: str,
        area_table: pd.DataFrame,
        transition_table: xr.DataArray,
    ) -> list[dict[str, object]]:
        area = normalize_area_table(area_table)
        area_years = {int(_year) for _year in area.index}
        rows = []
        for _year in [
            int(_year) for _year in transition_table.coords["year"].to_numpy()
        ]:
            new_settlement_m2 = float(
                transition_table.sel(
                    year=_year,
                    start=list(NON_SETTLEMENT_LABELS),
                    end="settlements",
                ).sum()
            )
            settlement_loss_m2 = float(
                transition_table.sel(
                    year=_year,
                    start="settlements",
                    end=list(NON_SETTLEMENT_LABELS),
                ).sum()
            )
            net_from_transitions_m2 = new_settlement_m2 - settlement_loss_m2
            next_year = _year + 1
            if _year in area_years and next_year in area_years:
                area_change_m2 = float(
                    area.loc[next_year, "settlements"] - area.loc[_year, "settlements"]
                )
                abs_diff_m2 = abs(net_from_transitions_m2 - area_change_m2)
                comparison_status = "available"
            else:
                area_change_m2 = np.nan
                abs_diff_m2 = np.nan
                comparison_status = "missing_next_area_year"
            rows.append(
                {
                    "zone": zone_name,
                    "year": _year,
                    "new_settlement_m2": new_settlement_m2,
                    "settlement_loss_m2": settlement_loss_m2,
                    "net_from_transitions_m2": net_from_transitions_m2,
                    "area_table_settlement_change_m2": area_change_m2,
                    "net_change_abs_diff_m2": abs_diff_m2,
                    "comparison_status": comparison_status,
                }
            )
        return rows

    "transition extraction helpers defined"
    return (
        build_settlement_reconciliation_rows,
        normalize_area_table,
        transition_slice_to_frame,
    )


@app.cell
def extract_settlement_transitions(
    NON_SETTLEMENT_LABELS,
    build_settlement_reconciliation_rows,
    loaded_area_tables,
    loaded_transition_tables,
    loaded_zone_names,
    pd,
    transition_slice_to_frame,
):
    _new_transition_frames = []
    _persistence_frames = []
    _reconciliation_rows = []

    for _zone_name in loaded_zone_names:
        _transition_table = loaded_transition_tables[_zone_name]
        _new_transition_frames.append(
            transition_slice_to_frame(
                _zone_name,
                _transition_table,
                start_labels=NON_SETTLEMENT_LABELS,
            )
        )
        _persistence_frames.append(
            transition_slice_to_frame(
                _zone_name,
                _transition_table,
                start_labels=("settlements",),
            ).rename(columns={"area_m2": "persistence_area_m2"})
        )
        _reconciliation_rows.extend(
            build_settlement_reconciliation_rows(
                _zone_name,
                loaded_area_tables[_zone_name],
                _transition_table,
            )
        )

    new_settlement_transitions = pd.concat(_new_transition_frames, ignore_index=True)
    settlement_persistence = pd.concat(_persistence_frames, ignore_index=True)
    settlement_reconciliation = pd.DataFrame(_reconciliation_rows)

    new_settlement_transitions = new_settlement_transitions.assign(
        area_ha=lambda _df: _df["area_m2"] / 10_000.0,
        has_new_settlement_transition=lambda _df: _df["area_m2"] > 0,
    )

    settlement_transition_summary = pd.DataFrame(
        [
            {"metric": "zones", "value": new_settlement_transitions["zone"].nunique()},
            {"metric": "new transition rows", "value": len(new_settlement_transitions)},
            {
                "metric": "rows with positive new settlement area",
                "value": int(
                    new_settlement_transitions["has_new_settlement_transition"].sum()
                ),
            },
            {
                "metric": "total new settlement area ha",
                "value": float(new_settlement_transitions["area_ha"].sum()),
            },
            {
                "metric": "max net reconciliation diff m2",
                "value": float(
                    settlement_reconciliation["net_change_abs_diff_m2"].max(skipna=True)
                ),
            },
        ]
    )

    settlement_transition_summary
    return new_settlement_transitions, settlement_reconciliation


@app.cell
def show_settlement_reconciliation(settlement_reconciliation):
    settlement_reconciliation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Source-Class Shares

    New settlement transitions are converted into source-class shares. These are
    the main reusable prior tables for later allocation methods. Zone-specific
    shares preserve local history where enough evidence exists; pooled shares serve
    as fallback priors for sparse or high-risk zones.
    """)
    return


@app.cell
def compute_source_share_tables(
    HIGH_SPATIAL_RISK_ZONE_NAMES,
    new_settlement_transitions,
    np,
):
    positive_new_settlement_transitions = new_settlement_transitions.loc[
        new_settlement_transitions["area_m2"] > 0
    ].copy()

    source_area_by_zone = (
        new_settlement_transitions.groupby(["zone", "start"], dropna=False)["area_m2"]
        .sum()
        .reset_index()
        .rename(columns={"start": "source_class", "area_m2": "new_settlement_area_m2"})
    )
    source_area_by_zone["zone_total_new_settlement_m2"] = source_area_by_zone.groupby(
        "zone"
    )["new_settlement_area_m2"].transform("sum")
    source_area_by_zone = source_area_by_zone.assign(
        source_share=np.where(
            source_area_by_zone["zone_total_new_settlement_m2"] > 0,
            source_area_by_zone["new_settlement_area_m2"]
            / source_area_by_zone["zone_total_new_settlement_m2"],
            0.0,
        ),
        high_spatial_risk=lambda _df: _df["zone"].isin(HIGH_SPATIAL_RISK_ZONE_NAMES),
    )

    historical_source_share_by_zone = source_area_by_zone.sort_values(
        ["zone", "source_share"],
        ascending=[True, False],
    ).reset_index(drop=True)

    source_area_by_year = (
        new_settlement_transitions.groupby(["year", "start"], dropna=False)["area_m2"]
        .sum()
        .reset_index()
        .rename(columns={"start": "source_class", "area_m2": "new_settlement_area_m2"})
    )
    source_area_by_year["year_total_new_settlement_m2"] = source_area_by_year.groupby(
        "year"
    )["new_settlement_area_m2"].transform("sum")
    source_share_by_year = source_area_by_year.assign(
        source_share=np.where(
            source_area_by_year["year_total_new_settlement_m2"] > 0,
            source_area_by_year["new_settlement_area_m2"]
            / source_area_by_year["year_total_new_settlement_m2"],
            0.0,
        )
    )

    pooled_source_prior = (
        new_settlement_transitions.groupby("start", dropna=False)["area_m2"]
        .sum()
        .reset_index()
        .rename(columns={"start": "source_class", "area_m2": "new_settlement_area_m2"})
    )
    pooled_total_new_settlement_m2 = float(
        pooled_source_prior["new_settlement_area_m2"].sum()
    )
    pooled_source_prior = pooled_source_prior.assign(
        source_share=lambda _df: (
            _df["new_settlement_area_m2"] / pooled_total_new_settlement_m2
            if pooled_total_new_settlement_m2
            else 0.0
        ),
        prior_scope="all_zones",
    ).sort_values("source_share", ascending=False)

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
    pooled_source_prior_excluding_high_risk = (
        pooled_source_prior_excluding_high_risk.assign(
            source_share=lambda _df: (
                _df["new_settlement_area_m2"] / pooled_non_risk_total_new_settlement_m2
                if pooled_non_risk_total_new_settlement_m2
                else 0.0
            ),
            prior_scope="excluding_high_spatial_risk_zones",
        ).sort_values("source_share", ascending=False)
    )

    pooled_source_prior
    return (
        historical_source_share_by_zone,
        pooled_source_prior,
        pooled_source_prior_excluding_high_risk,
        source_share_by_year,
    )


@app.cell
def show_pooled_prior_excluding_high_risk(
    pooled_source_prior_excluding_high_risk,
):
    pooled_source_prior_excluding_high_risk
    return


@app.cell
def show_zone_source_shares(historical_source_share_by_zone):
    historical_source_share_by_zone
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Temporal Stability

    If source shares shift sharply across historical years, a single pooled prior is
    less stable. The table below summarizes annual source-share variation across
    the historical record.
    """)
    return


@app.cell
def compute_temporal_stability(np, source_share_by_year):
    temporal_source_stability = (
        source_share_by_year.groupby("source_class", dropna=False)
        .agg(
            mean_annual_share=("source_share", "mean"),
            std_annual_share=("source_share", "std"),
            min_annual_share=("source_share", "min"),
            max_annual_share=("source_share", "max"),
            active_year_count=(
                "new_settlement_area_m2",
                lambda _series: int((_series > 0).sum()),
            ),
            total_new_settlement_area_m2=("new_settlement_area_m2", "sum"),
        )
        .reset_index()
    )
    temporal_source_stability = temporal_source_stability.assign(
        coefficient_of_variation=np.where(
            temporal_source_stability["mean_annual_share"] > 0,
            temporal_source_stability["std_annual_share"].fillna(0)
            / temporal_source_stability["mean_annual_share"],
            np.nan,
        )
    ).sort_values("mean_annual_share", ascending=False)

    temporal_source_stability
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Source Availability In The Baseline Year

    Historical source priors are only useful for allocation if candidate source
    classes still have available area in the baseline year. This section checks
    2020 non-settlement source availability and flags historically common source
    classes that are scarce at baseline.
    """)
    return


@app.cell
def compute_source_availability(
    CHEN_PIXEL_AREA_M2,
    COMMON_SOURCE_SHARE_THRESHOLD,
    COMPARISON_YEAR,
    NON_SETTLEMENT_LABELS,
    historical_source_share_by_zone,
    loaded_area_tables,
    loaded_zone_names,
    normalize_area_table,
    pd,
):
    _availability_rows = []
    for _zone_name in loaded_zone_names:
        _area = normalize_area_table(loaded_area_tables[_zone_name])
        if COMPARISON_YEAR not in _area.index:
            continue
        _availability_rows.extend(
            {
                "zone": _zone_name,
                "source_class": _source_class,
                "baseline_source_area_m2": float(
                    _area.loc[COMPARISON_YEAR, _source_class]
                ),
                "baseline_source_area_ha": float(
                    _area.loc[COMPARISON_YEAR, _source_class]
                )
                / 10_000.0,
            }
            for _source_class in NON_SETTLEMENT_LABELS
        )

    source_availability_2020 = pd.DataFrame(_availability_rows)

    source_availability_with_shares = historical_source_share_by_zone.merge(
        source_availability_2020,
        on=["zone", "source_class"],
        how="left",
    )
    source_availability_with_shares = source_availability_with_shares.assign(
        common_historical_source=lambda _df: (
            _df["source_share"] >= COMMON_SOURCE_SHARE_THRESHOLD
        ),
        scarce_in_baseline=lambda _df: (
            _df["baseline_source_area_m2"].fillna(0) < CHEN_PIXEL_AREA_M2
        ),
    )

    scarce_common_sources = source_availability_with_shares.loc[
        source_availability_with_shares["common_historical_source"]
        & source_availability_with_shares["scarce_in_baseline"]
        & (source_availability_with_shares["new_settlement_area_m2"] > 0)
    ].sort_values(["zone", "source_share"], ascending=[True, False])

    source_availability_summary = pd.DataFrame(
        [
            {
                "metric": "zone/source availability rows",
                "value": len(source_availability_2020),
            },
            {
                "metric": "scarce common source rows",
                "value": len(scarce_common_sources),
            },
            {
                "metric": "zones with scarce common source",
                "value": scarce_common_sources["zone"].nunique(),
            },
        ]
    )

    source_availability_summary
    return (scarce_common_sources,)


@app.cell
def show_scarce_common_sources(scarce_common_sources):
    scarce_common_sources
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Candidate Priors For Allocation

    The per-zone quality table combines evidence volume, temporal coverage, source
    dominance, spatial-risk context, and source availability. It recommends whether
    future allocation should use zone-specific shares, pooled shares, or manual
    review for each zone.
    """)
    return


@app.cell
def define_prior_quality_helper(
    MAX_DOMINANT_SOURCE_SHARE,
    MIN_ACTIVE_YEARS_FOR_ZONE_PRIOR,
    MIN_ZONE_PRIOR_AREA_M2,
    pd,
):
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

    "prior quality helper defined"
    return (assign_prior_quality_status,)


@app.cell
def compute_zone_prior_quality(
    HIGH_SPATIAL_RISK_ZONE_NAMES,
    assign_prior_quality_status,
    historical_source_share_by_zone,
    new_settlement_transitions,
    scarce_common_sources,
):
    zone_prior_base = (
        new_settlement_transitions.groupby("zone", dropna=False)
        .agg(
            total_new_settlement_m2=("area_m2", "sum"),
            active_year_count=(
                "area_m2",
                lambda _series: int(
                    (_series > 0)
                    .groupby(new_settlement_transitions.loc[_series.index, "year"])
                    .any()
                    .sum()
                ),
            ),
            positive_source_year_rows=("has_new_settlement_transition", "sum"),
        )
        .reset_index()
    )

    dominant_source_by_zone = (
        historical_source_share_by_zone.sort_values(
            ["zone", "source_share"], ascending=[True, False]
        )
        .groupby("zone", as_index=False)
        .first()
        .rename(
            columns={
                "source_class": "dominant_source_class",
                "source_share": "dominant_source_share",
            }
        )
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
            scarce_common_source_count=lambda _df: (
                _df["scarce_common_source_count"].fillna(0).astype(int)
            ),
            high_spatial_risk=lambda _df: _df["zone"].isin(
                HIGH_SPATIAL_RISK_ZONE_NAMES
            ),
        )
    )
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

    zone_prior_quality
    return prior_quality_summary, zone_prior_quality


@app.cell
def show_prior_quality_summary(prior_quality_summary):
    prior_quality_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visual Diagnostics

    The plots below summarize the empirical priors. The first compares pooled
    source shares with and without high spatial-risk zones. The second shows
    year-to-year source-share stability. The third summarizes which zones are good
    candidates for zone-specific priors versus pooled fallback priors.
    """)
    return


@app.cell
def plot_pooled_source_priors(
    pd,
    plt,
    pooled_source_prior,
    pooled_source_prior_excluding_high_risk,
    sns,
):
    sns.set_theme(style="whitegrid")

    pooled_prior_comparison = pd.concat(
        [pooled_source_prior, pooled_source_prior_excluding_high_risk],
        ignore_index=True,
    )

    _prior_fig, _prior_ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=pooled_prior_comparison,
        x="source_share",
        y="source_class",
        hue="prior_scope",
        ax=_prior_ax,
    )
    _prior_ax.set_xlabel("Source share of new settlement transitions")
    _prior_ax.set_ylabel("Historical source class")
    _prior_ax.set_title("Pooled historical source priors")
    _prior_ax.legend(title="Prior scope", loc="lower right")
    _prior_fig.tight_layout()

    pooled_prior_plot = _prior_fig

    pooled_prior_plot
    return


@app.cell
def plot_temporal_source_shares(
    plt,
    pooled_source_prior,
    sns,
    source_share_by_year,
):
    _top_sources = pooled_source_prior.head(6)["source_class"].tolist()
    _temporal_plot_data = source_share_by_year.loc[
        source_share_by_year["source_class"].isin(_top_sources)
    ]

    _temporal_fig, _temporal_ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(
        data=_temporal_plot_data,
        x="year",
        y="source_share",
        hue="source_class",
        marker="o",
        ax=_temporal_ax,
    )
    _temporal_ax.set_xlabel("Transition start year")
    _temporal_ax.set_ylabel("Annual source share")
    _temporal_ax.set_title("Temporal stability of top source classes")
    _temporal_ax.legend(
        title="Source class", bbox_to_anchor=(1.02, 1), loc="upper left"
    )
    _temporal_fig.tight_layout()

    temporal_source_share_plot = _temporal_fig

    temporal_source_share_plot
    return


@app.cell
def plot_prior_quality_status(plt, prior_quality_summary, sns):
    _quality_fig, _quality_ax = plt.subplots(figsize=(9, 4))
    sns.barplot(
        data=prior_quality_summary,
        x="zone_count",
        y="prior_quality_status",
        color="#4c78a8",
        ax=_quality_ax,
    )
    _quality_ax.set_xlabel("Zone count")
    _quality_ax.set_ylabel("Prior quality status")
    _quality_ax.set_title("Recommended prior strategy by zone")
    _quality_fig.tight_layout()

    prior_quality_plot = _quality_fig

    prior_quality_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recommended Priors

    The recommendation below is intended for `06_pseudo_transition_allocation.py`.
    It identifies which empirical prior table is the safest default and which zones
    should avoid unreviewed zone-specific allocation.
    """)
    return


@app.cell
def build_prior_recommendation(
    mo,
    new_settlement_transitions,
    pooled_source_prior,
    settlement_reconciliation,
    zone_prior_quality,
):
    _zone_specific_count = int(
        zone_prior_quality["prior_quality_status"]
        .isin(
            [
                "zone_specific_candidate",
                "zone_specific_with_dominance_warning",
                "zone_specific_with_availability_warning",
            ]
        )
        .sum()
    )
    _pooled_or_review_count = len(zone_prior_quality) - _zone_specific_count
    _top_source = str(pooled_source_prior.iloc[0]["source_class"])
    _top_source_share = float(pooled_source_prior.iloc[0]["source_share"])
    _second_source = str(pooled_source_prior.iloc[1]["source_class"])
    _second_source_share = float(pooled_source_prior.iloc[1]["source_share"])
    _max_net_reconciliation_diff = float(
        settlement_reconciliation["net_change_abs_diff_m2"].max(skipna=True)
    )
    _high_risk_prior_zones = ", ".join(
        zone_prior_quality.loc[
            zone_prior_quality["prior_quality_status"]
            == "review_or_pooled_spatial_risk",
            "zone",
        ].head(10)
    )

    if _pooled_or_review_count:
        _prior_recommendation = (
            "Use zone-specific priors where quality is sufficient, and use the "
            "pooled excluding-high-risk prior for sparse or high spatial-risk zones."
        )
    else:
        _prior_recommendation = (
            "Use zone-specific priors as the default, with pooled priors as fallback."
        )

    prior_recommendation = mo.md(
        f"""
    ### Prior Readout

    - Total historical new settlement area: `{new_settlement_transitions["area_ha"].sum():,.1f} ha`
    - Largest net settlement reconciliation difference: `{_max_net_reconciliation_diff:.6g} m2`
    - Dominant pooled source: `{_top_source}` (`{_top_source_share:.1%}`)
    - Second pooled source: `{_second_source}` (`{_second_source_share:.1%}`)
    - Zones suitable for zone-specific priors, with warnings allowed: `{_zone_specific_count}`
    - Zones needing pooled fallback or review: `{_pooled_or_review_count}`
    - Spatial-risk prior-review zones: `{_high_risk_prior_zones or "none"}`

    Recommendation: **{_prior_recommendation}**

    For the first allocation implementation, keep both `historical_source_share_by_zone`
    and `pooled_source_prior_excluding_high_risk` available. The pooled prior is a
    defensible fallback when a zone has sparse history, spatial-risk concerns, or a
    dominant source share that looks too brittle for future allocation.
    """
    )

    prior_recommendation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Limitations And Next Step

    These priors summarize historical aggregate transitions. They do not encode
    future suitability, adjacency, urban demand location, or Chen-grid spatial
    constraints. They also do not allocate future settlement demand; they only
    describe historical source behavior.

    Next, `05_chen_future_deltas.py` should convert Chen trajectories into decadal
    settlement-growth demand. The allocation notebook can then combine that demand
    with the source priors produced here.
    """)
    return


if __name__ == "__main__":
    app.run()
