# Marimo Notebook Roadmap

This document defines the planned notebooks for the Chen SSP feasibility and
scenario-analysis workflow. Each notebook should be readable on its own, but the
sequence is designed to build a defensible chain from input validation to carbon
model outputs.

All notebooks should follow the style guidance in
[`marimo_style_guide.md`](marimo_style_guide.md). In particular, every notebook
should use markdown cells generously to explain purpose, assumptions, section
transitions, outputs, and limitations.

Use plots and tables as explanatory artifacts, not as decoration. A notebook
should show tabular or visual output when it helps the reader compare zones,
inspect distributions, diagnose failures, or understand scenario results. It
should not attach a plot or table to every code cell by habit.

## `00_artifact_inventory.py`

### Purpose

Confirm which upstream artifacts and constants are available before any
feasibility analysis begins.

This notebook answers: "Is the workspace ready for Chen SSP analysis?"

### Inputs

- `OUT_PATH`
- available zone partitions or user-selected zones
- `src/nu_afolu/constants.py`
- historical artifact directories:
  - `bbox/ee`
  - `area_raster`
  - `transition_raster`
  - `area_table`
  - `transition_table`

### Recommended Markdown Sections

1. Title and goal
2. Required inputs
3. Project constants
4. Artifact discovery
5. Zone availability summary
6. Chen source availability
7. Readiness conclusion
8. Limitations and next notebook

### Core Analyses

- Print or table the configured `OUT_PATH`.
- List available zones for each required artifact type.
- Identify zones with complete artifact sets.
- Display `LABEL_LIST`, `CHEN_YEARS`, `SSP_NAMES`, and Chen collection id.
- Confirm whether Earth Engine initialization succeeds if required for later
  notebooks.

### Expected Outputs

- A table of zones and artifact availability.
- A list of zones eligible for analysis.
- A clear readiness statement.

### Acceptance Criteria

- The notebook identifies at least one zone with complete artifacts, or clearly
  explains that analysis cannot proceed.
- The notebook exposes the exact Chen years and SSP names that later notebooks
  will use.
- The notebook does not perform substantive compatibility analysis.

## `01_historical_contract_checks.py`

### Purpose

Validate that historical Dagster outputs satisfy the expected table contracts
before they are used as a baseline for Chen feasibility work.

This notebook answers: "Can we trust the historical `area_table` and
`transition_table` artifacts as analysis inputs?"

### Inputs

- Complete zone list from `00_artifact_inventory.py`
- historical `area_table` files
- historical `transition_table` files
- `LABEL_LIST`

### Recommended Markdown Sections

1. Title and goal
2. Historical artifact contract recap
3. Loaded zones
4. Area-table schema checks
5. Transition-table schema checks
6. Area/transition consistency checks
7. Edge cases: missing classes and 2021 -> 2022
8. Summary of pass/fail status
9. Limitations and next notebook

### Core Analyses

- Confirm `area_table` index represents years.
- Confirm `area_table` columns match the AFOLU labels.
- Confirm `transition_table` dimensions are `year`, `start`, and `end`.
- Confirm transition coordinates match the AFOLU labels.
- Check that all numeric values are finite and non-negative.
- Compare transition row sums with start-year areas.
- Compare transition column sums with next-year areas where next-year areas are
  available.
- Explicitly report the historical edge where `transition_table` includes
  2021 -> 2022 but `area_table` may stop at 2021.

### Expected Outputs

- A per-zone contract-check table.
- Summary statistics for area/transition reconciliation differences.
- A list of zones that pass, warn, or fail.

### Acceptance Criteria

- Later notebooks can filter to zones that pass required contract checks.
- Any tolerance used for reconciliation is documented.
- The notebook clearly distinguishes hard failures from expected historical
  edge cases.

## `02_chen_2020_compatibility.py`

### Purpose

Compare Chen 2020 urban area with GLC-derived 2020 `settlements` area at the
zone level.

This notebook answers: "Is Chen 2020 close enough to the historical settlements
baseline to justify further scenario construction?"

### Inputs

- Passing zones from `01_historical_contract_checks.py`
- `load_chen_analysis_zones`
- historical 2020 `area_table["settlements"]`
- Chen 2020 urban area by SSP

### Recommended Markdown Sections

1. Title and goal
2. Why 2020 is the comparison year
3. Loaded analysis zones
4. GLC settlements baseline
5. Chen 2020 urban area by SSP
6. Absolute and ratio error metrics
7. Worst-zone diagnostics
8. Visual comparison
9. Interpretation and decision points
10. Limitations and next notebook

### Core Analyses

- Compute GLC 2020 settlements area by zone.
- Compute Chen 2020 urban area by zone and SSP.
- Calculate absolute difference in square meters and hectares.
- Calculate ratio error, with careful handling of near-zero denominators.
- Rank zones by absolute error and ratio error.
- Produce scatterplots of GLC settlements versus Chen urban area.
- Produce distributions of mismatch by SSP.

### Expected Outputs

- Compatibility table by zone and SSP.
- Worst-zone table.
- Plots comparing GLC 2020 settlements and Chen 2020 urban area.
- Recommendation about whether further analysis should proceed globally, by
  subset of zones, or with calibration.

### Acceptance Criteria

- The notebook quantifies mismatch rather than only describing it.
- Small-zone ratio instability is called out.
- The notebook does not claim Chen urban is semantically identical to GLC
  settlements.

## `03_spatial_resolution_diagnostics.py`

### Purpose

Diagnose how the 30 m historical data and approximately 1 km Chen data interact
spatially.

This notebook answers: "Is the zone-level compatibility result hiding important
resolution or boundary problems?"

### Inputs

- Passing zones from previous notebooks
- observed `area_raster`
- Chen scenario masks
- `ChenAnalysisZone.resample_observed_settlement_mask`

### Recommended Markdown Sections

1. Title and goal
2. Why resolution mismatch matters
3. Selected diagnostic zones
4. Resampling approach
5. Settlement fractions on the Chen grid
6. Boundary and small-zone sensitivity
7. Spatial examples
8. Interpretation
9. Limitations and next notebook

### Core Analyses

- Aggregate observed settlement masks to the Chen grid using area or fraction
  logic.
- Compare Chen urban pixels with observed settlement fractions.
- Identify zones where a small number of Chen pixels drive most area.
- Report boundary sensitivity for zones that intersect partial Chen pixels.
- Produce maps or tabular summaries for representative zones.

### Expected Outputs

- Spatial diagnostic tables.
- Optional maps for selected zones.
- A list of high-risk zones where resolution mismatch may dominate.

### Acceptance Criteria

- The notebook treats spatial comparison as diagnostic rather than validation.
- It identifies whether a zone-level-only workflow is defensible for most zones.
- It flags zones that should be excluded, downweighted, or reviewed manually.

## `04_historical_settlement_transition_priors.py`

### Purpose

Summarize historical transitions into `settlements` to inform future source-class
allocation methods.

This notebook answers: "Historically, which AFOLU classes tend to become
settlements?"

### Inputs

- Passing historical `transition_table` artifacts
- historical `area_table` artifacts
- `LABEL_LIST`

### Recommended Markdown Sections

1. Title and goal
2. Why transition priors are needed
3. Historical transition data loaded
4. Zone-level settlement-source shares
5. Aggregate settlement-source shares
6. Temporal stability of source shares
7. Source availability in baseline year
8. Candidate priors for allocation
9. Limitations and next notebook

### Core Analyses

- Select transitions where `end == "settlements"`.
- Separate persistence (`settlements -> settlements`) from new settlement
  transitions.
- Compute source-class shares for new settlement area.
- Summarize shares by zone, year, and aggregate region.
- Compare historical settlement gains from transitions with `area_table`
  settlement differences.
- Identify source classes that are common historically but scarce in baseline
  availability.

### Expected Outputs

- Source-share tables by zone and aggregate.
- Visualizations of historical settlement-source distributions.
- Recommended historical priors for the allocation notebook.

### Acceptance Criteria

- The notebook produces at least one reusable source-share table.
- It distinguishes new settlement transitions from settlement persistence.
- It documents whether priors are stable enough to use by zone or should be
  pooled across zones.

## `05_chen_future_deltas.py`

### Purpose

Convert Chen urban-area projections into decadal settlement-growth demand by
zone and SSP.

This notebook answers: "How much new urban area does Chen imply for each future
decade?"

### Inputs

- Chen urban area by zone and SSP
- compatibility decisions from `02_chen_2020_compatibility.py`
- baseline choice from `analysis_contracts.md`

### Recommended Markdown Sections

1. Title and goal
2. Projection years and interval semantics
3. Baseline choice
4. Chen urban trajectories
5. Decadal deltas
6. Negative and non-monotonic deltas
7. Calibration options
8. Demand tables for allocation
9. Limitations and next notebook

### Core Analyses

- Compute Chen urban area trajectories by zone and SSP.
- Compute decadal deltas for every adjacent Chen year pair.
- Report positive, zero, and negative deltas.
- Clip negative deltas to zero for the first diagnostic allocation method, while
  recording clipped area separately.
- Optionally compare uncalibrated Chen trajectories with GLC-baseline-adjusted
  trajectories.

### Expected Outputs

- Decadal settlement-demand table by zone, SSP, start year, and end year.
- Negative-delta diagnostics.
- Baseline/calibration choice summary.

### Acceptance Criteria

- Future settlement demand is explicit and reproducible.
- Negative deltas are detected and documented.
- The notebook does not allocate source classes.

## `06_pseudo_transition_allocation.py`

### Purpose

Create candidate pseudo-`area_table` and pseudo-`transition_table` artifacts
from Chen settlement-growth demand.

This notebook answers: "Can we construct carbon-model-compatible diagnostic
scenario tables under explicit allocation assumptions?"

### Inputs

- Settlement-demand table from `05_chen_future_deltas.py`
- historical source priors from `04_historical_settlement_transition_priors.py`
- baseline area table
- `LABEL_LIST`
- candidate methods from
  [`pseudo_transition_methods.md`](pseudo_transition_methods.md)

### Recommended Markdown Sections

1. Title and goal
2. Artifact contract being reproduced
3. Allocation methods included
4. Baseline area state
5. Method 1: historical shares
6. Method 2: availability-constrained allocation
7. Method 3: priority/ranking allocation
8. Optional diagnostic baseline
9. Mass-balance checks
10. Output artifacts and provenance
11. Limitations and next notebook

### Core Analyses

- Apply each selected allocation method to each zone, SSP, and Chen interval.
- Construct pseudo-`area_table` objects.
- Construct pseudo-`transition_table` objects.
- Fill missing transition combinations with zero.
- Run required contract and mass-balance checks.
- Record allocation exhaustion or unresolved demand.
- Save diagnostic artifacts only after validation passes.

### Expected Outputs

- Pseudo-`area_table` artifacts.
- Pseudo-`transition_table` artifacts.
- Validation report by method and SSP.
- Provenance summary for each artifact set.

### Acceptance Criteria

- Every emitted pseudo-table matches the historical table contracts.
- Any unresolved demand or clipped negative delta is recorded.
- Invalid methods or scenarios are not silently passed to the carbon model.

## `07_carbon_model_scenario_runs.py`

### Purpose

Run the carbon model on validated pseudo-tables and collect emissions and
ancillary outputs.

This notebook answers: "What carbon outputs does each diagnostic scenario
produce?"

### Inputs

- Validated pseudo-`area_table` artifacts
- Validated pseudo-`transition_table` artifacts
- carbon model entry point
- scenario provenance metadata

### Recommended Markdown Sections

1. Title and goal
2. Carbon model input contract
3. Scenario artifact selection
4. Model run configuration
5. Execution
6. Output schema checks
7. Run summaries
8. Saved outputs
9. Limitations and next notebook

### Core Analyses

- Load validated pseudo-tables by scenario and method.
- Feed tables into the carbon model.
- Collect emissions in equivalent tons of CO2.
- Collect biomass decomposition and carbonization ancillary outputs.
- Check output schema and units.
- Save model outputs with scenario provenance.

### Expected Outputs

- Carbon model output tables by SSP, method, zone, and transition class.
- Ancillary output tables.
- Run-status report.

### Acceptance Criteria

- The notebook only runs scenarios with validated pseudo-inputs.
- Carbon outputs are labeled as diagnostic scenario results.
- Output files can be read by the final comparison notebook.

## `08_scenario_comparison_readout.py`

### Purpose

Summarize and interpret scenario carbon outputs across SSPs and allocation
methods.

This notebook answers: "What did the Chen SSP experiment show, and how sensitive
are results to assumptions?"

### Inputs

- Carbon model outputs from `07_carbon_model_scenario_runs.py`
- validation reports from `06_pseudo_transition_allocation.py`
- compatibility diagnostics from earlier notebooks

### Recommended Markdown Sections

1. Title and goal
2. Scenario sets included
3. Compatibility and validation recap
4. Emissions by SSP
5. Emissions by allocation method
6. Top contributing transitions
7. Zone-level drivers
8. Sensitivity and uncertainty
9. Decision summary
10. Recommended next steps

### Core Analyses

- Compare total emissions across SSPs.
- Compare total emissions across allocation methods.
- Identify transitions that contribute most to emissions or capture.
- Identify zones that drive scenario totals.
- Summarize sensitivity to baseline and allocation assumptions.
- State whether the Chen workflow is acceptable, needs more work, or should be
  rejected for carbon-model use.

### Expected Outputs

- Final summary tables.
- Plots for SSP and allocation-method comparisons.
- A written interpretation suitable for project review.

### Acceptance Criteria

- The notebook reads saved outputs rather than recomputing heavy Earth Engine
  reductions.
- It separates scientific findings from implementation limitations.
- It makes a clear recommendation about the status of Chen-derived scenario
  artifacts.
