# Primary Data Sources

## Historical Observed Land Use

- Source: GLC_FCS30D annual 30m land-cover data from the GEE Community Catalog.
- Documentation: https://gee-community-catalog.org/projects/glc_fcs/
- Earth Engine asset used by the Dagster graph: `projects/sat-io/open-datasets/GLC-FCS30D/annual`
- Analytical role: observed historical land use, including the observed `settlements` class.
- Expected historical decision window for this analysis: 2000 through 2020.

The Chen notebooks use 2020 as the source/baseline year. Historical growth diagnostics use 2000-2010, 2010-2020, and 2000-2020. If upstream materialization includes extra years, those years are not part of the Chen readiness decision unless a notebook section explicitly introduces them.

## Chen SSP Urban Projections

- Source: Chen et al. future urban land projection dataset.
- Earth Engine asset used by `nu_afolu.chen`: `projects/sat-io/open-datasets/FUTURE-URBAN-LAND/CHEN_2020_2100`
- Documentation: https://gee-community-catalog.org/projects/urban_projection/
- Analytical role: future urban expansion signal by SSP and decade.
- Years used: 2020, 2030, 2040, ..., 2100.
- Scenarios used: `SSP1` through `SSP5`.
- Urban value in this Earth Engine asset: `2`.

Chen is treated as a coarse 1km urban/non-urban signal. It is not treated as a complete future land-use map and does not directly identify exact 30m future transitions.

# External Built-Up Validation

- Source: GHSL built-up surface grid, release P2023A.
- Earth Engine asset used by `04_external_settlement_validation.py`: `JRC/GHSL/P2023A/GHS_BUILT_S`
- Documentation: https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_BUILT_S
- Analytical role: independent built-up evidence for Chen/GLC baseline agreement and near-term 2020-2030 growth plausibility.
- Years used in the external-validation notebook: 2020 and 2030.
- Band used: `built_surface`, expressed as built-up square metres per 100m grid cell.

GHSL is not a replacement observed baseline and is not a long-range future scenario input. It is an external validation anchor. GHSL 2030 is treated as near-term plausibility evidence rather than observed truth because the product is spatially-temporally interpolated or extrapolated through 2030.

# Upstream Processing

The upstream Dagster assets produce the observed historical artifacts consumed by the notebooks. They are located in `src/nu_afolu/defs/assets/graph.py` and `src/nu_afolu/defs/assets/tables.py`. The artifacts are structured as follows:

* `OUT_PATH/area_table/{ZONE}.parquet`: Parquet table detailing the observed area of each land-use class (rows) for each year (columns) for the specified zone, using the GLC_FCS30D land-cover classes. The units of the area values are square meters. 
* `OUT_PATH/transition_table/{ZONE}.nc`: NetCDF table detailing the observed transitions between land-use classes for each pair of consecutive years. The table has three dimensions: 
  * `year`: Starting year of the transition. This means that the transition from 2000 to 2001 is recorded in the `year=2000` slice of the table.
  * `start`: The starting land-use class of the transition.
  * `end`: The ending land-use class of the transition.
The values in the table represent the area of land that transitioned from the `start` class to the `end` class during the year specified by the `year` dimension, in square meters

# Validation Code

The test suite and validation helpers check software and artifact contracts. They do not by themselves prove that Chen is acceptable as model input.

- `src/nu_afolu/artifact_validation.py`
  - Required columns and row counts.
  - Key coverage by zone, scenario, year, method, and calibration type.
  - Metric ranges and internal consistency.
  - Readiness-label and transition-feasibility consistency.

- `src/nu_afolu/transition_feasibility.py`
  - Checks cumulative future transition demand against observed 2020 source-class stock.

- `src/nu_afolu/growth_plausibility.py`
  - Builds historical settlement-growth diagnostics and Chen-vs-history plausibility labels.

- `src/nu_afolu/metrics.py`
  - Shared calibration and agreement metric calculations.

- `src/nu_afolu/external_validation.py`
  - Shared external-validation label logic for GHSL support/conflict and advisory labels.