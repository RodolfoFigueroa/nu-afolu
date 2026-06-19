# nu-afolu

AFOLU / land-use analysis project using Earth Engine, Dagster assets, pandas/xarray, and marimo notebooks.

## Current Chen SSP Analysis

The active Chen workflow evaluates whether Chen et al. SSP urban-land projections are adequate enough to justify manual review or further calibration for a future AFOLU carbon-model workflow.

The notebooks are intentionally split:

- `notebooks/01_calibration.py` compares Chen 2020 urban extent against the observed 2020 settlement layer.
- `notebooks/02_transition_closure.py` builds diagnostic future `source class -> settlements` transition tables and readiness screens.
- `notebooks/03_method_exploration.py` stress-tests calibration methods and spatial-tolerance diagnostics.

Chen-derived transition tables are diagnostic only unless a later approval step explicitly promotes a reviewed subset.

## Data Provenance

Observed historical land use comes from the upstream GLC-FCS30D-derived artifacts:

- `area_raster`
- `transition_raster`
- `area_table`

The GLC_FCS30D source is documented at https://gee-community-catalog.org/projects/glc_fcs/ and loaded in the Dagster graph from `projects/sat-io/open-datasets/GLC-FCS30D/annual`.

Chen SSP urban projections are loaded from `projects/sat-io/open-datasets/FUTURE-URBAN-LAND/CHEN_2020_2100`.

For the full data lineage, processing contract, expected notebook outputs, and review rules, see `docs/data_provenance.md`.

## Validation

Run the lightweight software and notebook checks with:

```powershell
uv run python -m unittest discover
uv run marimo check notebooks/01_calibration.py notebooks/02_transition_closure.py notebooks/03_method_exploration.py
```

The validation suite checks artifact contracts and feasibility logic. It does not by itself approve Chen as model input.
