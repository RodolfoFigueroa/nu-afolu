# Data Provenance And Chen SSP Feasibility

This document is the shared baseline for human and AI agent developers working
on Chen SSP urban-projection analysis in this repository. It explains how the
Chen projection source relates to the historical Dagster artifacts documented in
[`docs/upstream.md`](upstream.md), and what must be true before Chen-derived
pseudo-transitions can be considered for the carbon model.

## Model Boundary

The carbon model is intentionally downstream of the land-use pipeline. It does
not need rasters or Earth Engine objects directly. It needs only:

- `area_table`: area by year and AFOLU class.
- `transition_table`: area by transition start year, start class, and end class.

Historical tables come from observed GLC_FCS30D-derived rasters reduced by the
Dagster pipeline. Chen SSP data do not provide those tables directly. Chen can
only be used after a separate feasibility and allocation step constructs
scenario-compatible pseudo-`area_table` and pseudo-`transition_table` artifacts.

Until a later review explicitly promotes a method, Chen-derived transition
tables should be treated as diagnostic scenario artifacts, not approved carbon
model input.

## Observed Historical Baseline

Observed historical AFOLU artifacts are documented in
[`docs/upstream.md`](upstream.md). The most relevant compatibility target is the
`settlements` column/class:

- GLC_FCS30D source collection:
  `projects/sat-io/open-datasets/GLC-FCS30D/annual`
- Project AFOLU class for urban-like historical land cover: `settlements`
- Historical table unit: square meters
- Historical raster resolution: 30 m
- Historical table interface:
  - `area_table.loc[year, "settlements"]`
  - `transition_table.sel(year=Y, end="settlements")`

For Chen feasibility, use 2020 as the baseline comparison year because the GEE
Chen collection starts at 2020. The upstream historical artifacts may contain
additional observed years depending on the generated graph; use
[`docs/upstream.md`](upstream.md) as the source of truth for the observed
artifact contract.

## Chen SSP Projection Source

The project uses the Google Earth Engine community-catalog version of Chen et
al.'s SSP urban land projections:

```text
projects/sat-io/open-datasets/FUTURE-URBAN-LAND/CHEN_2020_2100
```

In this repository, the contract is mirrored by constants in
`src/nu_afolu/constants.py`:

| Field | Project value |
| --- | --- |
| Earth Engine type | `ee.ImageCollection` |
| Years | `2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100` |
| Scenario bands | `SSP1`, `SSP2`, `SSP3`, `SSP4`, `SSP5` |
| Urban pixel value | `2` |
| Non-urban pixel value | `1` |
| Approximate resolution | 1 km |

Each image corresponds to one projection year. Each image has one band per SSP
scenario. The project interprets pixels equal to `2` as urban and masks or sums
those pixels when computing Chen urban area by scenario.

Important source nuance: the Chen paper and PANGAEA record describe a broader
product with a 2015 starting point. The Earth Engine collection used by this
repository is explicitly the `CHEN_2020_2100` collection, so local code and
analysis should treat 2020 as the first available Chen year unless a different
source asset is intentionally introduced.

## Compatibility Concerns

Chen 2020 urban area should be compared against GLC 2020 `settlements` area, but
this is a compatibility check rather than a strict validation. The two sources
are not identical products.

Key concerns:

- Semantic mismatch: Chen's historical urban data are based on GHSL concepts
  such as artificial cover and paved surfaces, while this project's
  `settlements` class is derived from GLC_FCS30D through `id_map.toml`.
- Resolution mismatch: GLC_FCS30D is 30 m, while Chen is approximately 1 km.
  Pixel-to-pixel comparison at 30 m is not meaningful.
- Temporal mismatch: Chen is decadal. It cannot directly provide annual
  transitions.
- Class mismatch: Chen is binary urban/non-urban. It does not provide full AFOLU
  class labels and cannot directly identify source classes for land converted to
  urban.
- Scenario uncertainty: Chen projections are model outputs under SSP
  assumptions. They should be documented as scenarios, not observations.

Preferred comparison pattern:

1. Compare area totals by analysis zone for Chen 2020 urban pixels and GLC 2020
   `settlements`.
2. Aggregate GLC settlements to the Chen grid, or compare both sources at the
   zone level using area-weighted reductions.
3. Record both absolute error and ratio error, because small zones can have
   unstable ratios.
4. Treat systematic mismatch as a reason to consider calibration or rejection
   before constructing pseudo-transitions.

## Future Pseudo-Transition Implications

Chen can constrain future settlement growth, but it cannot by itself specify a
complete AFOLU transition matrix. A future pseudo-transition method must make
the allocation rule explicit.

Safe default interpretation:

- New Chen urban area between decades can be interpreted as candidate land
  transitioning into `settlements`.
- Source classes for those transitions must come from a separate allocation
  method, such as overlap with the latest observed AFOLU raster, suitability
  ranking, or historical transition shares.
- Non-settlement AFOLU-to-AFOLU changes are outside Chen's direct information
  content unless another data source or assumption is added.
- Decadal Chen changes must not be silently treated as annual transitions.

If future code creates pseudo-`transition_table` artifacts from Chen, it should
preserve the downstream table contract from [`docs/upstream.md`](upstream.md):

- dimension `year` means transition start year;
- dimensions `start` and `end` use the fixed project AFOLU labels;
- values are square meters;
- missing transition combinations are explicit zeros.

## Agent Quick Reference

Agents may assume:

- The carbon model consumes `area_table` and `transition_table`, not Chen rasters
  directly.
- Chen data in this repo come from
  `projects/sat-io/open-datasets/FUTURE-URBAN-LAND/CHEN_2020_2100`.
- Chen urban pixels are value `2`.
- Chen scenarios are `SSP1` through `SSP5`.
- Chen years are decadal from 2020 through 2100.
- GLC `settlements` is the historical class to compare with Chen urban pixels.

Agents must not assume:

- Chen urban pixels are semantically identical to GLC `settlements`.
- Chen provides full AFOLU land-cover classes.
- Chen provides source classes for urban expansion.
- Chen provides annual transitions between decadal years.
- A 1 km Chen pixel should be downscaled to specific 30 m GLC pixels without an
  explicit allocation method.
- Chen-derived pseudo-transitions are approved carbon model inputs without a
  documented review decision.

## Sources

- GEE Community Catalog, "Global urban projections under SSPs (2020-2100)":
  <https://gee-community-catalog.org/projects/urban_projection/>
- Chen, G., Li, X., Liu, X. et al. "Global projections of future urban land
  expansion under shared socioeconomic pathways." Nature Communications 11, 537
  (2020): <https://www.nature.com/articles/s41467-020-14386-x>
- Chen et al. PANGAEA dataset record, "A global urban land expansion product at
  1-km resolution for 2015 to 2100 based on the SSP scenarios":
  <https://doi.pangaea.de/10.1594/PANGAEA.905890>
