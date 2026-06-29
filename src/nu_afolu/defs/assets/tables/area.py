import ee
import pandas as pd
from dagster_components.partitions import zone_partitions

import dagster as dg
from nu_afolu.constants import LABEL_LIST
from nu_afolu.defs.assets.tables.common import (
    generate_band_iterator,
    get_mapping_key_safe,
    reduce_year_band,
)

LABEL_MAP = dict(enumerate(LABEL_LIST, start=1))


@dg.op(pool="earth_engine_pool")
def reduce_year_band_area(
    context: dg.OpExecutionContext, img: ee.Image, bbox: ee.Geometry
) -> list[dict]:
    reduced = reduce_year_band(img, bbox, scale=30)
    mapping_key = get_mapping_key_safe(context)

    return [
        {
            "year": mapping_key,
            "label": LABEL_MAP[label],
            "area_m2": area,
        }
        for label, area in reduced.items()
    ]


@dg.op(out=dg.Out(io_manager_key="dataframe_manager"))
def aggregate_year_bands_area(reduced_list: list[list[dict]]) -> pd.DataFrame:
    reduced_list_flat = [item for sublist in reduced_list for item in sublist]
    return (
        pd.DataFrame(reduced_list_flat)
        .pivot_table(index="year", columns="label", values="area_m2")
        .reindex(columns=LABEL_LIST)
        .fillna(0)
    )


@dg.graph_asset(
    key="area_table",
    ins={
        "area_raster": dg.AssetIn(key="area_raster"),
        "bbox": dg.AssetIn(key=["bbox", "ee"]),
    },
    partitions_def=zone_partitions,
)
def area_table(area_raster: ee.Image, bbox: ee.Geometry) -> pd.DataFrame:
    band_imgs = generate_band_iterator(area_raster)
    mapped = band_imgs.map(lambda img: reduce_year_band_area(img, bbox))
    return aggregate_year_bands_area(mapped.collect())
