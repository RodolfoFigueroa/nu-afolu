import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")

with app.setup:
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import marimo as mo

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["axes.titleweight"] = "bold"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 07 Pseudo-Transition Interpretation
    """)
    return


@app.cell(hide_code=True)
def _():
    RUN_DIR = Path("outputs") / "chen_pseudo_tables" / "latest"
    REPORTS_DIR = RUN_DIR / "reports"
    MANIFEST_PATH = REPORTS_DIR / "provenance_manifest.json"
    INTERVAL_REPORT_PATH = REPORTS_DIR / "allocation_interval_report.parquet"
    SOURCE_REPORT_PATH = REPORTS_DIR / "allocation_source_report.parquet"

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing run manifest: {MANIFEST_PATH}")
    if not INTERVAL_REPORT_PATH.exists():
        raise FileNotFoundError(f"Missing interval report: {INTERVAL_REPORT_PATH}")
    if not SOURCE_REPORT_PATH.exists():
        raise FileNotFoundError(f"Missing source allocation report: {SOURCE_REPORT_PATH}")

    with MANIFEST_PATH.open(encoding="utf-8") as _manifest_file:
        result_manifest = json.load(_manifest_file)

    interval_report = pd.read_parquet(INTERVAL_REPORT_PATH)
    source_report = pd.read_parquet(SOURCE_REPORT_PATH)

    run_methods = tuple(result_manifest.get("methods", sorted(interval_report["method"].unique())))
    run_ssps = tuple(result_manifest.get("ssps", sorted(interval_report["ssp"].unique())))
    run_zone_count = int(result_manifest.get("zone_count", interval_report["zone"].nunique()))
    run_artifact_count = int(result_manifest.get("artifact_count", 0))
    run_generated_at_utc = result_manifest.get("generated_at_utc", "unknown")

    run_context_table = pd.DataFrame(
        [
            {"field": "run directory", "value": str(RUN_DIR)},
            {"field": "generated at UTC", "value": run_generated_at_utc},
            {"field": "zones", "value": run_zone_count},
            {"field": "SSPs", "value": ", ".join(run_ssps)},
            {"field": "methods", "value": ", ".join(run_methods)},
            {"field": "baseline choice", "value": result_manifest.get("baseline_choice", "unknown")},
            {"field": "calibration choice", "value": result_manifest.get("calibration_choice", "unknown")},
            {"field": "negative delta policy", "value": result_manifest.get("negative_delta_policy", "unknown")},
            {"field": "interval semantics", "value": result_manifest.get("interval_semantics", "unknown")},
            {"field": "scenario table sets", "value": run_artifact_count},
        ]
    )

    run_context_table
    return (
        interval_report,
        run_methods,
        run_ssps,
        run_zone_count,
        source_report,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Why Pseudo-Transitions Are Needed

    Chen SSP data provide projected urban or settlement extent by decade and
    scenario. They do not say which AFOLU class was converted when settlement area
    expanded. The carbon model, however, needs transition tables with a concrete
    `start` class and `end` class.

    The pseudo-transition step therefore answers a constrained allocation question:

    > Given Chen-derived settlement growth for a zone, SSP, and decade, how should
    > that new settlement area be assigned back to AFOLU source classes?

    The total settlement demand is fixed by Chen. The three methods differ only in
    how they assign source classes for that demand.

    Method | Plain-English Rule | Main assumption | Best read as | Main limitation
     :---   | :--- | :--- | :--- | :---
    Historical shares | Allocate future settlement growth using the observed historical source-class shares for transitions into settlements | Past settlement expansion patterns are informative for future expansion | Empirical default | Can preserve historical patterns even when future expansion shifts spatially or economically
    Availability-constrained | Start with historical source shares, then weight them by source area still available at the start of each interval | Historical patterns matter, but classes with more available area are more plausible future sources | Conservative feasibility variant | Availability is not the same as urban suitability or adjacency
    Priority ranking | Allocate demand sequentially through a fixed source priority order before touching lower-priority classes | The chosen priority order is a defensible sensitivity assumption | Bounding or stress-test case | Strongly assumption-driven and can concentrate allocation into a few classes
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Method Mechanics

    The three methods share the same state update and differ only in how they
    construct the source-class allocation vector for each zone, SSP, and decade.

    For a zone `z`, SSP `s`, interval `t0 -> t1`, and non-settlement AFOLU source
    class `i`, notebook `06` works with:

    - `D[z, s, t0, t1]`: Chen-derived settlement-growth demand for the interval.
    - `A_i[t0]`: source-class area available at the start of the interval.
    - `h_i`: historical prior share for source class `i`.
    - `x_i`: allocated area from source class `i` into `settlements`.

    The demand is the positive Chen urban-area delta:

    ```text
    raw_delta = chen_urban_area[t1] - chen_urban_area[t0]
    D = max(raw_delta, 0)
    clipped_negative_delta = max(-raw_delta, 0)
    ```

    Negative Chen deltas are recorded but not converted into
    `settlements -> non-settlements` transitions. This keeps the first diagnostic
    workflow focused on urban expansion, which is the part Chen can directly
    constrain.

    ### Shared Transition-State Update

    For every interval, the transition matrix starts as pure persistence:

    ```text
    M[i, i] = A_i[t0] for every AFOLU class i
    ```

    After an allocation vector `x` is chosen, the matrix is edited only for
    settlement expansion:

    ```text
    M[source, source]      -= x_source
    M[source, settlements] += x_source
    ```

    The next interval starts from the updated area state:

    ```text
    A_source[t1]      = A_source[t0] - x_source
    A_settlements[t1] = A_settlements[t0] + sum(x_source)
    ```

    No non-settlement-to-non-settlement transitions are invented. Classes that are
    not used as settlement sources persist through the interval.

    ### Shared Capped Proportional Allocator

    Both `historical_shares` and `availability_constrained` use the same iterative
    capped proportional allocator:

    ```text
    remaining_demand = D
    remaining_available_i = max(A_i[t0], 0)
    weights_i = method-specific non-negative weights

    while remaining_demand > epsilon and some source remains available:
        active_weights_i = weights_i for sources with remaining_available_i > 0
        if sum(active_weights) == 0:
            active_weights_i = remaining_available_i

        proposed_i = remaining_demand * active_weights_i / sum(active_weights)
        capped_i = min(proposed_i, remaining_available_i)

        x_i += capped_i
        remaining_available_i -= capped_i
        remaining_demand -= sum(capped_i)
    ```

    This is important because it prevents impossible source depletion. If one class
    cannot supply its proposed share, the leftover demand is redistributed across
    the remaining eligible sources.

    ### Historical Shares

    This method estimates historical priors from observed transitions into
    `settlements`, excluding settlement persistence:

    ```text
    h_i = sum_y transition_area[y, start=i, end=settlements]
          / sum_j sum_y transition_area[y, start=j, end=settlements]
    ```

    If a zone has enough usable history, the implementation uses that zone's
    source mix. Otherwise it falls back to a pooled prior constructed from the
    broader historical transition evidence.

    The method-specific weights are simply:

    ```text
    weights_i = h_i
    ```

    The allocator then caps by `A_i[t0]` and redistributes any remainder. This
    means the method is historically grounded, but still cannot allocate more of a
    source class than exists at the start of an interval.

    ### Availability-Constrained

    This method uses the same historical prior, but modulates it by the current
    area state:

    ```text
    weights_i = h_i * A_i[t0]
    ```

    If the weighted prior collapses to zero, the implementation falls back to pure
    availability:

    ```text
    weights_i = A_i[t0]
    ```

    Because `A_i[t0]` is updated after every decade, this method is dynamically
    availability-aware. A class that was historically common as a settlement source
    loses influence as it becomes scarce. Conversely, abundant classes can receive
    more allocation than they would under historical shares alone.

    ### Priority Ranking

    This method does not use historical transition shares. It applies a strict
    ordered depletion rule:

    ```text
    remaining_demand = D

    for source in priority_order:
        x_source = min(remaining_demand, A_source[t0])
        remaining_demand -= x_source
        stop if remaining_demand == 0
    ```

    The priority order is:

    ```text
    croplands -> pastures -> grasslands -> shrublands -> other ->
    forests_secondary -> forests_primary -> forests_mangroves -> wetlands -> flooded
    ```

    This makes `priority_ranking` qualitatively different from the other two
    methods. It is not a weighted proportional allocation; it is an assumption
    stress test. The first sufficiently available classes can absorb most or all of
    the Chen demand before lower-priority classes are touched.
    """)
    return


@app.cell(hide_code=True)
def _():
    METHOD_LABELS = {
        "historical_shares": "Historical shares",
        "availability_constrained": "Availability constrained",
        "priority_ranking": "Priority ranking",
    }

    SOURCE_CLASS_ORDER = (
        "croplands",
        "shrublands",
        "forests_secondary",
        "other",
        "grasslands",
        "pastures",
        "wetlands",
        "flooded",
        "forests_mangroves",
        "forests_primary",
    )

    SOURCE_COLORS = {
        "croplands": "#4C78A8",
        "shrublands": "#F58518",
        "forests_secondary": "#54A24B",
        "other": "#B279A2",
        "grasslands": "#72B7B2",
        "pastures": "#E45756",
        "wetlands": "#9D755D",
        "flooded": "#439894",
        "forests_mangroves": "#59A14F",
        "forests_primary": "#2E6F40",
    }


    def format_area_ha(area_ha: float) -> str:
        return f"{area_ha:,.0f} ha" if abs(area_ha) >= 100 else f"{area_ha:,.2f} ha"


    def format_percent(value: float) -> str:
        return f"{100.0 * value:,.1f}%"


    def method_label(method: str) -> str:
        return METHOD_LABELS.get(method, method)

    return (
        METHOD_LABELS,
        SOURCE_CLASS_ORDER,
        SOURCE_COLORS,
        format_area_ha,
        format_percent,
        method_label,
    )


@app.cell(hide_code=True)
def _(interval_report):
    demand_intervals = interval_report.drop_duplicates(
        subset=["zone", "ssp", "start_year", "end_year"]
    ).copy()
    demand_intervals = demand_intervals.assign(
        demand_ha=lambda _df: _df["demand_m2"] / 10_000.0,
        clipped_negative_delta_ha=lambda _df: _df["clipped_negative_delta_m2"] / 10_000.0,
        interval_label=lambda _df: _df["start_year"].astype(str) + "-" + _df["end_year"].astype(str),
    )

    demand_by_ssp = (
        demand_intervals.groupby("ssp", dropna=False)
        .agg(
            zones=("zone", "nunique"),
            intervals=("start_year", "count"),
            positive_intervals=("demand_m2", lambda _series: int((_series > 0).sum())),
            zero_intervals=("demand_m2", lambda _series: int((_series == 0).sum())),
            total_demand_ha=("demand_ha", "sum"),
            clipped_negative_delta_ha=("clipped_negative_delta_ha", "sum"),
        )
        .reset_index()
        .sort_values("ssp")
    )

    demand_by_interval = (
        demand_intervals.groupby(["ssp", "start_year", "end_year", "interval_label"], dropna=False)
        .agg(
            zones=("zone", "nunique"),
            total_demand_ha=("demand_ha", "sum"),
            positive_zones=("demand_m2", lambda _series: int((_series > 0).sum())),
        )
        .reset_index()
        .sort_values(["ssp", "start_year"])
    )

    total_demand_ha = float(demand_intervals["demand_ha"].sum())
    largest_demand_row = demand_by_ssp.loc[demand_by_ssp["total_demand_ha"].idxmax()]
    largest_demand_ssp = str(largest_demand_row["ssp"])
    largest_demand_ha = float(largest_demand_row["total_demand_ha"])
    negative_delta_ha = float(demand_by_ssp["clipped_negative_delta_ha"].sum())

    demand_by_ssp
    return (
        demand_by_interval,
        demand_by_ssp,
        largest_demand_ha,
        largest_demand_ssp,
        negative_delta_ha,
        total_demand_ha,
    )


@app.cell(hide_code=True)
def _(
    format_area_ha,
    largest_demand_ha,
    largest_demand_ssp,
    negative_delta_ha,
    run_zone_count,
    total_demand_ha,
):
    mo.md(f"""
    ## Settlement-Growth Demand

    Across `{run_zone_count}` zones, Chen implies **{format_area_ha(total_demand_ha)}**
    of settlement expansion over the full 2020-2100 horizon. The largest total
    demand is under **{largest_demand_ssp}** with **{format_area_ha(largest_demand_ha)}**.

    Negative Chen deltas are not interpreted as de-urbanization in these outputs.
    They are clipped to zero before allocation; this run records
    **{format_area_ha(negative_delta_ha)}** of clipped negative deltas.
    """)
    return


@app.cell(hide_code=True)
def _(demand_by_ssp, run_ssps):
    demand_by_ssp_plot_data = demand_by_ssp.assign(
        total_demand_thousand_ha=lambda _df: _df["total_demand_ha"] / 1_000.0
    )

    demand_by_ssp_plot, _demand_by_ssp_ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=demand_by_ssp_plot_data,
        x="ssp",
        y="total_demand_thousand_ha",
        order=list(run_ssps),
        color="#4C78A8",
        ax=_demand_by_ssp_ax,
    )
    _demand_by_ssp_ax.set_xlabel("SSP")
    _demand_by_ssp_ax.set_ylabel("Settlement-growth demand (thousand ha)")
    _demand_by_ssp_ax.set_title("Total Chen-Derived Settlement Demand By SSP")
    _demand_by_ssp_ax.bar_label(_demand_by_ssp_ax.containers[0], fmt="%.0f", padding=3)
    demand_by_ssp_plot.tight_layout()

    demand_by_ssp_plot
    return


@app.cell(hide_code=True)
def _(demand_by_interval):
    demand_by_interval_plot_data = demand_by_interval.assign(
        total_demand_thousand_ha=lambda _df: _df["total_demand_ha"] / 1_000.0
    )

    demand_by_interval_plot, _demand_by_interval_ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=demand_by_interval_plot_data,
        x="start_year",
        y="total_demand_thousand_ha",
        hue="ssp",
        style="ssp",
        markers=True,
        dashes=False,
        ax=_demand_by_interval_ax,
    )
    _demand_by_interval_ax.set_xlabel("Interval start year")
    _demand_by_interval_ax.set_ylabel("Settlement-growth demand (thousand ha)")
    _demand_by_interval_ax.set_title("Decadal Settlement Demand")
    _demand_by_interval_ax.legend(title="SSP", ncol=3)
    demand_by_interval_plot.tight_layout()

    demand_by_interval_plot
    return


@app.cell(hide_code=True)
def _(interval_report, method_label):
    allocation_by_method_ssp = (
        interval_report.groupby(["method", "ssp"], dropna=False)
        .agg(
            zones=("zone", "nunique"),
            intervals=("start_year", "count"),
            demand_ha=("demand_m2", lambda _series: float(_series.sum() / 10_000.0)),
            allocated_ha=("allocated_m2", lambda _series: float(_series.sum() / 10_000.0)),
            unresolved_ha=("unresolved_demand_m2", lambda _series: float(_series.sum() / 10_000.0)),
            clipped_negative_delta_ha=("clipped_negative_delta_m2", lambda _series: float(_series.sum() / 10_000.0)),
            manual_review_zones=("zone", lambda _series: int(interval_report.loc[_series.index].loc[interval_report.loc[_series.index, "needs_manual_review"], "zone"].nunique())),
        )
        .reset_index()
        .assign(method_label=lambda _df: _df["method"].map(method_label))
    )

    allocation_by_method_ssp
    return


@app.cell(hide_code=True)
def _(SOURCE_CLASS_ORDER, method_label, source_report):
    source_allocation_by_method = (
        source_report.groupby(["method", "source_class"], dropna=False)
        .agg(
            allocated_m2=("allocated_m2", "sum"),
            exhausted_intervals=("source_exhausted", "sum"),
        )
        .reset_index()
        .assign(
            allocated_ha=lambda _df: _df["allocated_m2"] / 10_000.0,
            method_label=lambda _df: _df["method"].map(method_label),
        )
    )
    source_allocation_by_method["source_share"] = source_allocation_by_method.groupby("method")[
        "allocated_m2"
    ].transform(lambda _series: _series / _series.sum() if float(_series.sum()) else 0.0)

    source_allocation_by_method = source_allocation_by_method.assign(
        source_class=pd.Categorical(
            source_allocation_by_method["source_class"],
            categories=list(SOURCE_CLASS_ORDER),
            ordered=True,
        ),
        source_share_percent=lambda _df: 100.0 * _df["source_share"],
    ).sort_values(["method", "source_class"])

    top_sources_by_method = (
        source_allocation_by_method.sort_values(["method", "allocated_m2"], ascending=[True, False])
        .groupby("method", as_index=False)
        .head(5)
        .loc[:, ["method_label", "source_class", "allocated_ha", "source_share_percent"]]
        .assign(
            allocated_ha=lambda _df: _df["allocated_ha"].round(1),
            source_share_percent=lambda _df: _df["source_share_percent"].round(1),
        )
        .rename(
            columns={
                "method_label": "method",
                "source_class": "source class",
                "allocated_ha": "allocated ha",
                "source_share_percent": "share of method allocation (%)",
            }
        )
    )

    top_sources_by_method
    return (source_allocation_by_method,)


@app.cell(hide_code=True)
def _(format_percent, interval_report, source_allocation_by_method):
    historical_cropland_share = float(
        source_allocation_by_method.loc[
            (source_allocation_by_method["method"] == "historical_shares")
            & (source_allocation_by_method["source_class"].astype(str) == "croplands"),
            "source_share",
        ].sum()
    )
    historical_shrubland_share = float(
        source_allocation_by_method.loc[
            (source_allocation_by_method["method"] == "historical_shares")
            & (source_allocation_by_method["source_class"].astype(str) == "shrublands"),
            "source_share",
        ].sum()
    )
    availability_cropland_share = float(
        source_allocation_by_method.loc[
            (source_allocation_by_method["method"] == "availability_constrained")
            & (source_allocation_by_method["source_class"].astype(str) == "croplands"),
            "source_share",
        ].sum()
    )
    priority_cropland_share = float(
        source_allocation_by_method.loc[
            (source_allocation_by_method["method"] == "priority_ranking")
            & (source_allocation_by_method["source_class"].astype(str) == "croplands"),
            "source_share",
        ].sum()
    )

    manual_review_zone_count = int(
        interval_report.loc[interval_report["needs_manual_review"], "zone"].nunique()
    )

    mo.md(
        f"""
    ## Allocation Results

    All three methods allocate the same Chen-derived settlement demand by design.
    Their substantive difference is the **source-class mix**.

    In this run, `historical_shares` allocates most expansion to **croplands**
    ({format_percent(historical_cropland_share)}) and **shrublands**
    ({format_percent(historical_shrubland_share)}). The
    `availability_constrained` method shifts further toward croplands
    ({format_percent(availability_cropland_share)}), because available 2020 source
    area changes the effective weights. The `priority_ranking` sensitivity case
    puts **{format_percent(priority_cropland_share)}** of expansion into croplands
    before using lower-ranked classes.

    The compatibility diagnostics carried forward from earlier notebooks mark
    `{manual_review_zone_count}` zones for manual review. That is not an artifact
    failure; it is a caution that the Chen 2020 urban baseline and GLC 2020
    settlements baseline are not perfectly aligned in those zones.
    """
    )
    return


@app.cell(hide_code=True)
def _(
    SOURCE_CLASS_ORDER,
    SOURCE_COLORS,
    method_label,
    run_methods,
    source_allocation_by_method,
):
    source_share_pivot = (
        source_allocation_by_method.pivot_table(
            index="method_label",
            columns="source_class",
            values="source_share_percent",
            fill_value=0.0,
            observed=False,
        )
        .reindex([method_label(_method) for _method in run_methods])
        .reindex(columns=list(SOURCE_CLASS_ORDER), fill_value=0.0)
    )

    source_share_by_method_plot, _source_share_ax = plt.subplots(figsize=(10.5, 5.5))
    _source_share_bottom = np.zeros(len(source_share_pivot))
    for _source_class in SOURCE_CLASS_ORDER:
        _values = source_share_pivot[_source_class].to_numpy(dtype=float)
        _source_share_ax.bar(
            source_share_pivot.index,
            _values,
            bottom=_source_share_bottom,
            label=_source_class,
            color=SOURCE_COLORS.get(_source_class),
        )
        _source_share_bottom = _source_share_bottom + _values

    _source_share_ax.set_xlabel("Allocation method")
    _source_share_ax.set_ylabel("Share of allocated settlement expansion (%)")
    _source_share_ax.set_title("Source-Class Mix By Allocation Method")
    _source_share_ax.set_ylim(0, 100)
    _source_share_ax.legend(
        title="Source class",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )
    source_share_by_method_plot.tight_layout()

    source_share_by_method_plot
    return


@app.cell(hide_code=True)
def _(
    METHOD_LABELS,
    SOURCE_CLASS_ORDER,
    run_methods,
    source_allocation_by_method,
):
    source_share_wide = (
        source_allocation_by_method.pivot_table(
            index="source_class",
            columns="method",
            values="source_share",
            fill_value=0.0,
            observed=False,
        )
        .reindex(list(SOURCE_CLASS_ORDER), fill_value=0.0)
        .reindex(columns=list(run_methods), fill_value=0.0)
    )

    source_share_difference_from_historical = (
        source_share_wide.subtract(source_share_wide["historical_shares"], axis=0) * 100.0
    )
    source_share_difference_plot_data = source_share_difference_from_historical.loc[
        :, ["availability_constrained", "priority_ranking"]
    ].rename(columns=METHOD_LABELS)

    source_share_difference_plot, _source_difference_ax = plt.subplots(figsize=(8.5, 5.5))
    sns.heatmap(
        source_share_difference_plot_data,
        annot=True,
        fmt=".1f",
        center=0.0,
        cmap="vlag",
        linewidths=0.5,
        cbar_kws={"label": "Percentage-point difference"},
        ax=_source_difference_ax,
    )
    _source_difference_ax.set_xlabel("Method compared with historical shares")
    _source_difference_ax.set_ylabel("Source class")
    _source_difference_ax.set_title("How Source Shares Move Relative To Historical Shares")
    source_share_difference_plot.tight_layout()

    source_share_difference_plot
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reading The Differences

    The heatmap above compares each method's source-class share with
    `historical_shares`. Positive values mean the method assigns a larger share of
    settlement expansion to that class; negative values mean it assigns less.

    This is the main scientific sensitivity in the pseudo-transition workflow. The
    Chen demand signal fixes how much settlement area appears under each SSP, but
    the source allocation method decides whether that expansion is treated as
    coming mostly from croplands, shrublands, forests, or other classes.
    """)
    return


@app.cell(hide_code=True)
def _(method_label, source_report):
    source_allocation_by_method_ssp = (
        source_report.groupby(["method", "ssp", "source_class"], dropna=False)
        .agg(allocated_m2=("allocated_m2", "sum"))
        .reset_index()
        .assign(
            allocated_ha=lambda _df: _df["allocated_m2"] / 10_000.0,
            method_label=lambda _df: _df["method"].map(method_label),
        )
    )
    source_allocation_by_method_ssp["source_share"] = source_allocation_by_method_ssp.groupby(
        ["method", "ssp"]
    )["allocated_m2"].transform(lambda _series: _series / _series.sum() if float(_series.sum()) else 0.0)

    dominant_source_by_method_ssp = (
        source_allocation_by_method_ssp.sort_values(
            ["method", "ssp", "allocated_m2"],
            ascending=[True, True, False],
        )
        .groupby(["method", "ssp"], as_index=False)
        .first()
        .assign(
            allocated_ha=lambda _df: _df["allocated_ha"].round(1),
            source_share_percent=lambda _df: (100.0 * _df["source_share"]).round(1),
        )
        .loc[:, ["method_label", "ssp", "source_class", "allocated_ha", "source_share_percent"]]
        .rename(
            columns={
                "method_label": "method",
                "source_class": "dominant source class",
                "allocated_ha": "dominant source allocated ha",
                "source_share_percent": "dominant source share (%)",
            }
        )
    )

    dominant_source_by_method_ssp
    return (dominant_source_by_method_ssp,)


@app.cell(hide_code=True)
def _(dominant_source_by_method_ssp, run_ssps):
    dominant_source_plot_data = dominant_source_by_method_ssp.copy()

    dominant_source_plot, _dominant_source_ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=dominant_source_plot_data,
        x="ssp",
        y="dominant source share (%)",
        hue="method",
        order=list(run_ssps),
        ax=_dominant_source_ax,
    )
    _dominant_source_ax.set_xlabel("SSP")
    _dominant_source_ax.set_ylabel("Share of allocation in dominant source (%)")
    _dominant_source_ax.set_title("Concentration In The Dominant Source Class")
    _dominant_source_ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    dominant_source_plot.tight_layout()

    dominant_source_plot
    return


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ## Main Interpretation

    The three scenarios should not be interpreted as three different Chen futures.
    They use the same Chen settlement-growth demand. They are three different
    answers to the missing source-class problem.

    For a first diagnostic version, the strongest default is still
    `historical_shares`, because it is grounded in observed transitions from the
    same artifact family. `availability_constrained` is useful because it shows how
    the result changes when current source availability is allowed to pull against
    historical shares. `priority_ranking` is most useful as a sensitivity case: it
    shows what happens under a strong, transparent rule rather than an empirical
    source mix.

    The current 69-zone result is therefore best read as a method-sensitivity
    package. The demand totals tell us how much settlement expansion Chen implies;
    the allocation plots tell us how much the implied source classes depend on the
    chosen assumption.
    """)
    return


if __name__ == "__main__":
    app.run()
