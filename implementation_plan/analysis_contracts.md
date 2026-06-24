# Shared Analysis Contracts

This document defines the shared contracts and assumptions that every notebook
in the Chen SSP workflow should follow. It exists so that humans and AI agents
do not have to rediscover table semantics or make implicit choices notebook by
notebook.

## Historical Artifact Contracts

Historical artifacts come from the upstream Dagster pipeline documented in
[`docs/upstream.md`](../docs/upstream.md).

### `area_table`

- File pattern: `OUT_PATH/area_table/{ZONE}.parquet`
- Object type when loaded: `pd.DataFrame`
- Index: `year`
- Columns: AFOLU labels from `nu_afolu.constants.LABEL_LIST`
- Values: area in square meters
- Important class for Chen comparison: `settlements`

The table should be interpreted as zone-level land-use area by calendar year.
It is an aggregate table, not a raster. Missing class/year combinations may
appear as missing values if the Earth Engine reducer did not return that group.

### `transition_table`

- File pattern: `OUT_PATH/transition_table/{ZONE}.nc`
- Object type when loaded: `xr.DataArray`
- Dimensions:
  - `year`: transition start year
  - `start`: AFOLU class at the start of the transition
  - `end`: AFOLU class at the end of the transition
- Values: area in square meters

The coordinate `year=Y` means the transition from `Y` to `Y + 1` in the
historical artifacts. Missing transition combinations are explicitly filled with
zero.

## Chen Projection Contracts

Chen SSP source details are documented in
[`docs/data_provenance.md`](../docs/data_provenance.md).

Use the project constants as the local code contract:

- Earth Engine collection:
  `projects/sat-io/open-datasets/FUTURE-URBAN-LAND/CHEN_2020_2100`
- Projection years: `2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100`
- Scenarios: `SSP1`, `SSP2`, `SSP3`, `SSP4`, `SSP5`
- Urban pixel value: `2`
- Non-urban pixel value: `1`
- Approximate resolution: 1 km

Chen is binary urban/non-urban. It does not provide full AFOLU classes and does
not identify which class is converted when urban area expands.

## Baseline And Stitching Defaults

Use 2020 as the primary compatibility year because the Earth Engine Chen
collection used by this project starts at 2020.

The notebooks should make baseline stitching explicit:

- Compare Chen 2020 urban area with GLC 2020 `settlements`.
- Report how much observed data exists after 2020 in the historical artifacts.
- Do not silently combine the historical 2021 area table or 2021 -> 2022
  transition with Chen 2020 -> 2030 deltas.
- When future pseudo-artifacts are created, record the baseline choice in the
  output metadata or directory naming.

Recommended default for the first implementation:

- Use GLC 2020 `settlements` as the baseline settlement area for calibrated
  pseudo-`area_table` construction.
- Use Chen decadal deltas as growth signals rather than replacing historical
  2020 areas wholesale.
- Preserve observed historical tables separately from future diagnostic tables.

This default can be revisited after the 2020 compatibility notebook quantifies
the GLC/Chen mismatch.

## Scenario Artifact Naming

Any future pseudo-artifact should be identifiable by:

- SSP scenario, such as `SSP2`;
- allocation method, such as `historical_shares`;
- baseline choice, such as `glc_2020`;
- calibration choice, such as `none` or `ratio_adjusted`;
- generation timestamp or code version when practical.

The initial implementation can encode this in directory names before a richer
metadata format exists.

Example:

```text
outputs/chen_pseudo_tables/
  historical_shares/
    baseline_glc_2020/
      calibration_none/
        SSP2/
          area_table/
          transition_table/
```

The exact output path can change during implementation, but the same pieces of
provenance must remain recoverable.

## Mass-Balance Expectations

Every pseudo-`area_table` and pseudo-`transition_table` must pass explicit
mass-balance checks before being used by the carbon model.

Required checks:

- All values are finite and non-negative.
- Pseudo-`area_table` uses the full AFOLU label set.
- Pseudo-`transition_table` uses dimensions `year`, `start`, and `end`.
- Missing transition combinations are represented as explicit zeros.
- For a transition start year `Y`, summing transitions over `end` should match
  the starting area vector for `Y`, within documented tolerance.
- Summing transitions over `start` should match the ending area vector for the
  next represented time step, within documented tolerance.
- Settlement gains assigned in the transition table should reconcile with the
  scenario settlement-area delta.
- Source classes cannot be allocated below zero.

Because Chen years are decadal, pseudo-transition years may represent decadal
intervals rather than annual `Y -> Y + 1` transitions. If that happens, the
notebook must clearly state the interval semantics and confirm that downstream
carbon-model code can handle them before running the model.

## Negative Or Non-Monotonic Chen Deltas

Chen urban area may be flat, noisy, or lower in a later decade for a given zone
and SSP. The notebooks should not assume monotonic growth without measuring it.

Default handling for the first implementation:

- Detect and report all negative decadal deltas.
- Do not convert negative deltas into `settlements -> non-settlements`
  transitions unless an explicit de-urbanization method is added.
- For diagnostic pseudo-transition construction, clip negative settlement
  deltas to zero and record the clipped area as unresolved scenario mismatch.

This keeps the first pseudo-transition methods focused on urban expansion, which
is the part Chen can most directly constrain.

## Resolution And Boundary Expectations

Chen is approximately 1 km, while the historical GLC-derived artifacts are based
on 30 m data. The analysis should avoid pixel-level equivalence claims.

Required expectations:

- Prefer zone-level area comparisons for compatibility checks.
- Use area-weighted reductions when comparing or clipping by zone geometry.
- Treat small zones and boundary-heavy zones as higher-risk.
- Use resampled observed settlement fractions on the Chen grid only as a
  diagnostic, not as proof that the two products are semantically identical.

## Carbon Model Boundary

The carbon model consumes table artifacts. It should not need to know whether a
table came from observed GLC reductions or Chen-derived pseudo-transitions.

For this project phase:

- Do not modify carbon-model internals unless required by interval semantics.
- Keep the carbon-model run notebook focused on feeding it compatible tables and
  collecting outputs.
- Clearly label all Chen-derived model outputs as scenario diagnostics.

