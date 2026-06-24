# Upstream Dagster Artifacts

This document explains the observed historical artifacts that the Dagster pipeline
writes under `OUT_PATH`. These artifacts are the project baseline for downstream
notebooks and Chen SSP analysis.

The four main outputs are:

| Asset | File pattern | Python object when loaded | Purpose |
| --- | --- | --- | --- |
| `area_raster` | `OUT_PATH/area_raster/{ZONE}.json` | `ee.Image` | Per-pixel AFOLU class id for each observed year. |
| `transition_raster` | `OUT_PATH/transition_raster/{ZONE}.json` | `ee.Image` | Per-pixel transition code for each consecutive-year pair. |
| `area_table` | `OUT_PATH/area_table/{ZONE}.parquet` | `pd.DataFrame` | Zone-level area totals by year and AFOLU class. |
| `transition_table` | `OUT_PATH/transition_table/{ZONE}.nc` | `xr.DataArray` | Zone-level transition area totals by start year, start class, and end class. |

`{ZONE}` is one of the static Dagster `zone_partitions` from
`dagster_components.partitions`. Each asset is materialized once per zone
partition.

## Code Map

The relevant code lives in:

- `src/nu_afolu/definitions.py`: wires `OUT_PATH` into file IO managers.
- `src/nu_afolu/defs/assets/bbox.py`: builds zone geometries.
- `src/nu_afolu/defs/assets/graph.py`: builds `area_raster` and
  `transition_raster`.
- `src/nu_afolu/defs/assets/tables.py`: reduces rasters into `area_table` and
  `transition_table`.
- `src/nu_afolu/constants.py`: defines the AFOLU label order and loads
  `transition_dict.json`.
- `id_map.toml`: maps source GLC_FCS30D class ids into project AFOLU classes.
- `transition_dict.json`: maps numeric transition codes to
  `[start_label, end_label]` pairs.

The file IO managers use the asset key and partition key to construct paths
below `OUT_PATH`. For example, the `area_table` asset for partition `09.1.01`
is written to `OUT_PATH/area_table/09.1.01.parquet`.

The two raster artifacts are not downloaded raster files. They are serialized
Earth Engine computation graphs written as JSON by `EarthEngineManager`. Loading
them with the same manager, or with `ee.deserializer.decode`, reconstructs an
`ee.Image`.

## Dependency Flow

The normal upstream flow is:

1. `zone_bbox_shapely` reads the analysis-zone geometry from PostGIS tables.
2. `zone_bbox_ee` converts that simplified and buffered geometry to an
   Earth Engine polygon.
3. `build_glc_fcs_zone_rasters` loads the GLC_FCS30D annual image that intersects
   the zone, derives AFOLU class masks, and materializes:
   - `area_raster`
   - `transition_raster`
4. `area_table` reduces `area_raster` over the zone geometry.
5. `transition_table` reduces `transition_raster` over the zone geometry.

The raster assets must be materialized before the table assets because the
tables are reductions of the raster outputs.

## Source Data

The historical observed land-cover source is the Earth Engine image collection:

```text
projects/sat-io/open-datasets/GLC-FCS30D/annual
```

`load_glc30` filters this collection to the zone bounding geometry and expects
exactly one image. The source image has annual bands, and the pipeline uses
`year_to_band_name` to select the source band for each calendar year.

The current graph iterates over years `2000` through `2021` inclusive. For
`transition_raster`, a band named `Y` compares source year `Y` with source year
`Y + 1`, so band `2021` represents the 2021 -> 2022 transition.

## AFOLU Labels

The pipeline projects source classes into this fixed label order from
`LABEL_LIST`:

| Raster id | Label |
| --- | --- |
| 1 | `croplands` |
| 2 | `flooded` |
| 3 | `forests_mangroves` |
| 4 | `forests_primary` |
| 5 | `forests_secondary` |
| 6 | `grasslands` |
| 7 | `other` |
| 8 | `pastures` |
| 9 | `settlements` |
| 10 | `shrublands` |
| 11 | `wetlands` |

Most labels are direct masks over GLC_FCS30D class ranges from `id_map.toml`.
There are two important derived cases:

- `grasslands_to_pastures`, the source class range `130`, is split into
  `pastures` and `grasslands` using `ee.Image.random(42)` on the native GLC grid.
  Pixels where the random image is `<= 0.4` become `pastures`; the remaining
  pixels are merged into `grasslands`.
- `forests` is split into `forests_primary` and `forests_secondary` using
  `NASA/ORNL/global_forest_classification_2020/V1`. Pixels equal to `1` in that
  layer become `forests_primary`; the rest of the GLC forest mask becomes
  `forests_secondary`.

These splits are deterministic for a given source image and geometry because the
pasture discriminator uses a fixed random seed.

## `area_raster`

`area_raster` is produced by the `build_glc_fcs_zone_rasters` graph asset in
`src/nu_afolu/defs/assets/graph.py`.

Generation steps:

1. Build one Boolean mask image per AFOLU class. Each mask has the same annual
   band structure as the GLC_FCS30D source image.
2. For each year from `2000` through `2021`, `generate_area_raster` selects that
   year's band from each class mask.
3. It starts from a single-band image filled with `0`.
4. It loops over `LABEL_LIST` and writes `i + 1` wherever the class mask is true.
5. The yearly single-band rasters are stacked into one multiband `ee.Image` with
   bands named `2000`, `2001`, ..., `2021`.
6. The stacked image is reprojected to the native GLC_FCS30D projection.

Important contract details:

- File path: `OUT_PATH/area_raster/{ZONE}.json`.
- File format: serialized Earth Engine JSON, not GeoTIFF pixels.
- Bands: calendar years `2000` through `2021`.
- Pixel values: AFOLU raster ids `1` through `11` from the table above.
- Pixel value `0`: no AFOLU class was assigned by the current class mapping.
- Downstream code usually clips the deserialized image to
  `OUT_PATH/bbox/ee/{ZONE}.json` before analysis.

## `transition_raster`

`transition_raster` is produced by the same `build_glc_fcs_zone_rasters` graph
asset as `area_raster`.

Generation steps:

1. Reuse the same AFOLU class mask images generated for `area_raster`.
2. For each start year from `2000` through `2021`, `generate_transition_raster`
   selects the start-year band and the next-year band from every class mask.
3. For every `(start_label, end_label)` pair, it finds pixels where the start
   mask is true in year `Y` and the end mask is true in year `Y + 1`.
4. It writes the numeric transition code for that label pair. Codes come from
   `transition_dict.json`, loaded as `TRANSITION_LABEL_MAP`.
5. The yearly transition rasters are stacked into one multiband `ee.Image` with
   bands named by the transition start year: `2000`, `2001`, ..., `2021`.
6. The stacked image is reprojected to the native GLC_FCS30D projection.

Important contract details:

- File path: `OUT_PATH/transition_raster/{ZONE}.json`.
- File format: serialized Earth Engine JSON, not GeoTIFF pixels.
- Bands: transition start years `2000` through `2021`.
- Band meaning: band `Y` represents the transition from `Y` to `Y + 1`.
- Pixel values: numeric transition codes from `transition_dict.json`.
- Be careful with code `0`: in `transition_dict.json`, `0` means
  `croplands -> croplands`. It is not a generic no-data value for transition
  tables.

## `area_table`

`area_table` is produced by `src/nu_afolu/defs/assets/tables.py` from
`area_raster` and `bbox/ee`.

Generation steps:

1. Read the partition's `area_raster` `ee.Image`.
2. Read the partition's Earth Engine zone geometry from `bbox/ee`.
3. For each year band, combine `ee.Image.pixelArea()` with that band.
4. Run `reduceRegion` with `ee.Reducer.sum().group(...)` over the zone geometry:
   - `geometry=bbox`
   - `scale=30`
   - `maxPixels=1e10`
5. Convert grouped numeric class ids to AFOLU labels using
   `dict(enumerate(LABEL_LIST, start=1))`.
6. Build a pandas DataFrame and pivot it to one row per year and one column per
   label.

Important contract details:

- File path: `OUT_PATH/area_table/{ZONE}.parquet`.
- File format: Parquet DataFrame.
- Index: `year`.
- Columns: AFOLU labels, such as `croplands`, `settlements`, and `wetlands`.
- Values: area in square meters.
- Missing year/class combinations can appear as missing values if the reducer
  does not return that group for the zone.

Conceptually, a loaded table looks like:

```text
label       croplands  flooded  ...  wetlands
year
2000        ...        ...          ...
2001        ...        ...          ...
...
2021        ...        ...          ...
```

## `transition_table`

`transition_table` is produced by `src/nu_afolu/defs/assets/tables.py` from
`transition_raster` and `bbox/ee`.

Generation steps:

1. Read the partition's `transition_raster` `ee.Image`.
2. Read the partition's Earth Engine zone geometry from `bbox/ee`.
3. Iterate over each transition-start-year band.
4. For each band, combine `ee.Image.pixelArea()` with the transition-code band.
5. Run the same grouped `reduceRegion` pattern used by `area_table`.
6. Convert each numeric transition code to a `start-end` label using
   `transition_dict.json`.
7. Split `start-end` into separate `start` and `end` coordinates.
8. Convert the data to an `xarray.DataArray`, reindex both `start` and `end` to
   the full `LABEL_LIST`, and fill missing transition combinations with `0`.

Important contract details:

- File path: `OUT_PATH/transition_table/{ZONE}.nc`.
- File format: NetCDF DataArray.
- Dimensions:
  - `year`: transition start year.
  - `start`: AFOLU label at the start of the transition.
  - `end`: AFOLU label at the end of the transition.
- Values: area in square meters.
- Band/year meaning: `year=2000` is the 2000 -> 2001 transition,
  `year=2021` is the 2021 -> 2022 transition.
- Missing transition combinations are explicitly filled with `0`.

Conceptually, a loaded array can be read as:

```python
arr.sel(year=2020, start="croplands", end="settlements")
```

That value is the square meters of land in the zone that changed from
`croplands` in 2020 to `settlements` in 2021.

## Relationship Between Tables And Rasters

`area_table` and `transition_table` are reductions of the raster artifacts over
the zone geometry. The tables are usually the safer interface for aggregate
analysis because they contain explicit units and labeled dimensions.

Useful consistency checks:

- For a given year `Y`, `area_table.loc[Y].sum()` should be close to the mapped
  zone area represented by `area_raster` for that year.
- For a given transition year `Y`, `transition_table.sel(year=Y).sum()` should
  be close to the mapped zone area represented by `transition_raster` for that
  transition band.
- For a given transition year `Y`, summing `transition_table.sel(year=Y)` over
  `end` gives area by starting class. That should broadly align with
  `area_table.loc[Y]`.
- Summing `transition_table.sel(year=Y)` over `start` gives area by ending
  class. That should broadly align with `area_table.loc[Y + 1]` when that area
  year is available. As currently wired, `area_table` stops at 2021, while
  `transition_table` includes a 2021 -> 2022 transition.

Small differences can occur because these are Earth Engine reductions over
classified masks, not independent vector-area calculations.

## Loading The Artifacts

The canonical helper for downstream Chen work is `load_chen_analysis_zones` in
`src/nu_afolu/chen.py`. It loads:

- `OUT_PATH/bbox/ee/{ZONE}.json`
- `OUT_PATH/area_raster/{ZONE}.json`
- `OUT_PATH/transition_raster/{ZONE}.json`
- `OUT_PATH/area_table/{ZONE}.parquet`
- `OUT_PATH/transition_table/{ZONE}.nc`

Minimal manual loading looks like:

```python
import json
from pathlib import Path

import ee
import pandas as pd
import xarray as xr

out_path = Path("...")
zone = "09.1.01"

with (out_path / "area_raster" / f"{zone}.json").open() as f:
    area_raster = ee.Image(ee.deserializer.decode(json.load(f)))

with (out_path / "transition_raster" / f"{zone}.json").open() as f:
    transition_raster = ee.Image(ee.deserializer.decode(json.load(f)))

area_table = pd.read_parquet(out_path / "area_table" / f"{zone}.parquet")
transition_table = xr.open_dataarray(out_path / "transition_table" / f"{zone}.nc")
```

Remember that Earth Engine JSON files reconstruct lazy Earth Engine objects. Any
operation that calls `.getInfo()`, exports data, or reduces pixels will execute
work against Earth Engine.
