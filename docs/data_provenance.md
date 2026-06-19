# AFOLU Chen Analysis Data Provenance

This document records the data lineage, processing steps, notebook handoffs, and expected outputs for the Chen SSP settlement-expansion analysis.

## Purpose

The Chen notebooks answer whether Chen et al. SSP urban-land projections are adequate enough, by zone and scenario, to justify manual review or further calibration for an AFOLU carbon-model workflow.

The notebooks do not currently produce approved carbon-model inputs. Chen-derived transition tables are diagnostic unless a later, explicit approval step promotes a reviewed subset.

## Primary Data Sources

### Historical Observed Land Use

- Source: GLC_FCS30D annual 30m land-cover data from the GEE Community Catalog.
- Documentation: https://gee-community-catalog.org/projects/glc_fcs/
- Earth Engine asset used by the Dagster graph: `projects/sat-io/open-datasets/GLC-FCS30D/annual`
- Analytical role: observed historical land use, including the observed `settlements` class.
- Expected historical decision window for this analysis: 2000 through 2020.

The Chen notebooks use 2020 as the source/baseline year. Historical growth diagnostics use 2000-2010, 2010-2020, and 2000-2020. If upstream materialization includes extra years, those years are not part of the Chen readiness decision unless a notebook section explicitly introduces them.

### Chen SSP Urban Projections

- Source: Chen et al. future urban land projection dataset.
- Earth Engine asset used by `nu_afolu.chen`: `projects/sat-io/open-datasets/FUTURE-URBAN-LAND/CHEN_2020_2100`
- Documentation: https://gee-community-catalog.org/projects/urban_projection/
- Analytical role: future urban expansion signal by SSP and decade.
- Years used: 2020, 2030, 2040, ..., 2100.
- Scenarios used: `SSP1` through `SSP5`.
- Urban value in this Earth Engine asset: `2`.

Chen is treated as a coarse 1km urban/non-urban signal. It is not treated as a complete future land-use map and does not directly identify exact 30m future transitions.

### External Built-Up Validation

- Source: GHSL built-up surface grid, release P2023A.
- Earth Engine asset used by `04_external_settlement_validation.py`: `JRC/GHSL/P2023A/GHS_BUILT_S`
- Documentation: https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_BUILT_S
- Analytical role: independent built-up evidence for Chen/GLC baseline agreement and near-term 2020-2030 growth plausibility.
- Years used in the external-validation notebook: 2020 and 2030.
- Band used: `built_surface`, expressed as built-up square metres per 100m grid cell.

GHSL is not a replacement observed baseline and is not a long-range future scenario input. It is an external validation anchor. GHSL 2030 is treated as near-term plausibility evidence rather than observed truth because the product is spatially-temporally interpolated or extrapolated through 2030.

## Upstream Processing

The upstream Dagster assets produce the observed historical artifacts consumed by the notebooks.

- `src/nu_afolu/defs/assets/graph.py`
  - Loads `projects/sat-io/open-datasets/GLC-FCS30D/annual`.
  - Builds class masks for the AFOLU classes in `src/nu_afolu/constants.py`.
  - Applies project-specific splits for classes such as primary/secondary forests and grasslands/pastures.
  - Produces per-zone `area_raster` and `transition_raster` Earth Engine artifacts.

- `src/nu_afolu/defs/assets/tables.py`
  - Reduces `area_raster` by zone and year.
  - Writes the per-zone `area_table` parquet artifact.

- `src/nu_afolu/chen.py`
  - Defines Chen collection constants, `ChenAnalysisZone`, and `ChenAnalysisZoneCollection`.
  - `load_chen_analysis_zones` loads per-zone `bbox`, `area_raster`, `transition_raster`, and `area_table` from `OUT_PATH`.
  - `observed_settlement_fraction_image` aggregates the 30m observed settlement mask onto Chen's 1km projection for calibration and exploration.

## Notebook Contracts

### `notebooks/01_calibration.py`

Question:

Are Chen 2020 urban extents adequate against the observed 2020 GLC-FCS30D-derived settlement layer?

Inputs:

- Per-zone `area_raster`, `transition_raster`, and `area_table` loaded from `OUT_PATH`.
- Chen SSP image collection.
- `SOURCE_YEAR = 2020`.

Processing:

- Compares Chen 2020 urban pixels with the observed 2020 `settlements` class.
- Aggregates the 30m observed settlement mask to Chen's 1km grid as fractional observed settlement.
- Computes area agreement, precision, recall, IoU, correction factors, and reliability labels.
- Runs scale-sensitivity checks at observed-settlement thresholds of 10%, 25%, and 50%.

Expected outputs:

- `OUT_PATH/chen/calibration.parquet`
- `OUT_PATH/chen/scale_sensitivity.parquet`

Interpretation:

These outputs are calibration and adequacy diagnostics. They are handoff artifacts for transition closure, not carbon-model inputs.

### `notebooks/02_transition_closure.py`

Question:

Given the calibration diagnostics, which Chen-derived future settlement-expansion cases are reasonable enough for manual review?

Inputs:

- Calibration artifact from `01_calibration.py`.
- Per-zone GLC-FCS30D-derived historical artifacts loaded from `OUT_PATH`.
- Chen 2020-2100 SSP urban projections.

Processing:

- Computes first Chen expansion year after 2020 for each SSP.
- Overlays Chen future expansion on the observed 2020 GLC-FCS30D-derived land-use raster.
- Builds diagnostic `source class -> settlements` transition rows.
- Applies raw and calibrated variants using the 2020 correction factor.
- Compares Chen decadal expansion against observed historical settlement growth from `manager.area_df`.
- Screens sensitive source classes: `forests_primary`, `forests_mangroves`, and `wetlands`; tracks `forests_secondary` as a watch class.
- Checks transition feasibility against observed 2020 source-class stock.
- Produces readiness and manual-review-priority labels.

Expected outputs:

- `OUT_PATH/chen/chen_expansion.parquet`
- `OUT_PATH/chen/chen_transitions.parquet`
- `OUT_PATH/chen/transition_feasibility.parquet`
- `OUT_PATH/chen/land_estimation_assessment.parquet`
- `OUT_PATH/chen/review_candidates.parquet`

Interpretation:

These outputs remain diagnostic. `df_chen_transitions` is shaped like a partial transition table, but it is not production-ready. A model-ready input requires a separate approved subset after manual review or explicit acceptance rules.

### `notebooks/03_method_exploration.py`

Question:

What method changes might improve or stress-test the Chen adequacy assessment?

Inputs:

- Canonical calibration and scale-sensitivity artifacts from `01_calibration.py`.
- The same GLC-FCS30D-derived observed baseline artifacts loaded from `OUT_PATH`.

Processing:

- Compares the canonical fractional method with thresholded observed-settlement methods.
- Adds buffered agreement methods to test spatial tolerance.
- Builds disagreement typologies that identify cases where high buffered agreement may mask area mismatch.

Expected outputs:

- `OUT_PATH/chen/exploration/method_comparison.parquet`
- `OUT_PATH/chen/exploration/method_summary.parquet`
- `OUT_PATH/chen/exploration/method_recommendation_candidates.parquet`
- `OUT_PATH/chen/exploration/disagreement_typology.parquet`
- `OUT_PATH/chen/exploration/disagreement_summary.parquet`

Interpretation:

These outputs are exploratory. Promoting a method requires updating `01_calibration.py` and rerunning the closure and validation workflow.

### `notebooks/04_external_settlement_validation.py`

Question:

Does independent built-up evidence support or undermine Chen's 2020 baseline and first-decade expansion signal?

Inputs:

- Validated closure artifacts from `02_transition_closure.py`.
- Per-zone GLC-FCS30D-derived historical artifacts loaded from `OUT_PATH`.
- GHSL built-up surface images for 2020 and 2030.

Processing:

- Sums GHSL 2020 built-up surface onto Chen's 1km grid.
- Compares GHSL 2020 against the GLC-FCS30D-derived 2020 settlement baseline and Chen 2020 urban baseline.
- Compares GHSL 2020-2030 built-up change against raw and calibrated Chen 2020-2030 expansion.
- Joins external support/conflict labels to the existing transition-closure readiness context without modifying readiness labels.

Expected outputs:

- `OUT_PATH/chen/external_validation/external_baseline_agreement.parquet`
- `OUT_PATH/chen/external_validation/external_growth_alignment.parquet`
- `OUT_PATH/chen/external_validation/external_review_flags.parquet`
- `OUT_PATH/chen/external_validation/external_validation_summary.parquet`

Interpretation:

These outputs are advisory validation products. They can guide manual review and method discussion, but they do not approve Chen-derived transitions for carbon-model input.

## Validation Code

The test suite and validation helpers check software and artifact contracts. They do not by themselves prove that Chen is acceptable as model input.

- `src/nu_afolu/artifact_validation.py`
  - Required columns and row counts.
  - Key coverage by zone, scenario, year, method, and calibration type.
  - Metric ranges and internal consistency.
  - Readiness-label and transition-feasibility consistency.

- `src/nu_afolu/transition_feasibility.py`
  - Checks cumulative future transition demand against observed 2020 source-class stock.

- `src/nu_afolu/metrics.py`
  - Shared calibration and agreement metric calculations.

- `src/nu_afolu/external_validation.py`
  - Shared external-validation label logic for GHSL support/conflict and advisory labels.

Recommended local checks:

```powershell
uv run python -m unittest discover
uv run pytest tests/test_artifact_validation.py tests/test_external_validation.py
uv run marimo check notebooks/01_calibration.py notebooks/02_transition_closure.py notebooks/03_method_exploration.py notebooks/04_external_settlement_validation.py
```

## Review Rules

Use the following rules when modifying the analysis:

1. Do not replace the GLC-FCS30D-derived observed baseline without documenting the new source and updating all affected notebooks.
2. Do not treat external datasets as the observed baseline unless the project explicitly changes that decision. External datasets can be validation evidence, not silent replacements.
3. Do not treat `df_chen_transitions` or `chen_transitions.parquet` as production-ready carbon-model input.
4. Preserve the distinction among diagnostics, manual-review candidates, and approved model inputs.
5. If a new method is promoted from exploration, update `01_calibration.py`, rerun `02_transition_closure.py`, and rerun artifact validation.
6. Keep `SOURCE_YEAR = 2020` explicit wherever Chen expansion is allocated onto observed land classes.
7. If using live marimo sessions, edit notebooks through marimo code-mode rather than directly editing the notebook `.py` files.
