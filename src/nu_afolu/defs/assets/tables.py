import ee
import pandas as pd
from dagster_components.partitions import zone_partitions

import dagster as dg
from nu_afolu.constants import LABEL_LIST

LABEL_MAP = dict(enumerate(LABEL_LIST, start=1))


@dg.asset(
    key="area_table",
    ins={
        "area_raster": dg.AssetIn(key="area_raster"),
        "bbox": dg.AssetIn(key=["bbox", "ee"]),
    },
    partitions_def=zone_partitions,
    io_manager_key="dataframe_manager",
)
def area_table(area_raster: ee.Image, bbox: ee.Geometry) -> pd.DataFrame:
    band_names = area_raster.bandNames().getInfo()

    if not isinstance(band_names, list):
        err = f"Expected band names to be a list of strings, got {type(band_names)}"
        raise TypeError(err)

    rows = []
    for band in band_names:
        year_band = area_raster.select(band)
        result = (
            ee.Image.pixelArea()
            .addBands(year_band)
            .reduceRegion(
                reducer=ee.Reducer.sum().group(groupField=1, groupName="label"),
                geometry=bbox,
                scale=30,
                maxPixels=int(1e10),
            )
            .getInfo()
        )

        if result is None:
            err = f"Failed to compute area for band {band}."
            raise ValueError(err)

        rows.extend(
            [
                {
                    "year": int(band),
                    "label": LABEL_MAP[group["label"]],
                    "area_m2": group["sum"],
                }
                for group in result["groups"]
            ]
        )

    return pd.DataFrame(rows).pivot_table(
        index="year", columns="label", values="area_m2"
    )
