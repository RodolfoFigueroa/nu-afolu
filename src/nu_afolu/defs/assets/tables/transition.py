import ee
import pandas as pd
import xarray as xr
from dagster_components.partitions import zone_partitions

import dagster as dg
from nu_afolu.constants import LABEL_LIST, TRANSITION_LABEL_MAP, TRANSITION_NODATA
from nu_afolu.defs.assets.tables.common import (
    generate_band_iterator,
    get_mapping_key_safe,
    reduce_year_band,
)

TRANSITION_LABEL_MAP_PRETTY = {
    k: f"{v[0]}-{v[1]}" for k, v in TRANSITION_LABEL_MAP.items()
}


@dg.op(pool="earth_engine_pool")
def reduce_year_band_transition(
    context: dg.OpExecutionContext, img: ee.Image, bbox: ee.Geometry
) -> list[dict]:
    reduced = reduce_year_band(img, bbox, scale=30)
    mapping_key = get_mapping_key_safe(context)

    return [
        {
            "year": mapping_key,
            "label": TRANSITION_LABEL_MAP_PRETTY[label],
            "area_m2": area,
        }
        for label, area in reduced.items()
        if label != TRANSITION_NODATA
    ]


@dg.op(out=dg.Out(io_manager_key="dataarray_manager"))
def aggregate_year_bands_transition(reduced_list: list[list[dict]]) -> xr.DataArray:
    reduced_list_flat = [item for sublist in reduced_list for item in sublist]

    return (
        pd.DataFrame(reduced_list_flat)
        .assign(
            components=lambda df: df["label"].str.split("-"),
            start=lambda df: df["components"].str[0],
            end=lambda df: df["components"].str[1],
        )
        .drop(columns=["components", "label"])
        .set_index(["year", "start", "end"])["area_m2"]
        .to_xarray()
        .reindex({"start": LABEL_LIST, "end": LABEL_LIST})
        .fillna(0)
    )


@dg.graph_asset(
    key="transition_table",
    ins={
        "transition_raster": dg.AssetIn(key="transition_raster"),
        "bbox": dg.AssetIn(key=["bbox", "ee"]),
    },
    partitions_def=zone_partitions,
    group_name="tables",
)
def transition_table(transition_raster: ee.Image, bbox: ee.Geometry) -> xr.DataArray:
    band_imgs = generate_band_iterator(transition_raster)
    mapped = band_imgs.map(lambda img: reduce_year_band_transition(img, bbox))
    return aggregate_year_bands_transition(mapped.collect())
