from collections.abc import Iterator

import ee
import pandas as pd
import xarray as xr
from dagster_components.partitions import zone_partitions

import dagster as dg
from nu_afolu.constants import LABEL_LIST, TRANSITION_LABEL_MAP

LABEL_MAP = dict(enumerate(LABEL_LIST, start=1))


TRANSITION_LABEL_MAP_PRETTY = {
    k: f"{v[0]}-{v[1]}" for k, v in TRANSITION_LABEL_MAP.items()
}


def get_band_names(img: ee.Image) -> list[str]:
    band_names = img.bandNames().getInfo()

    if not isinstance(band_names, list):
        err = f"Expected band names to be a list of strings, got {type(band_names)}"
        raise TypeError(err)

    return band_names


def reduce_year_band(year_band: ee.Image, bbox: ee.Geometry, *, scale: int) -> dict:
    result = (
        ee.Image.pixelArea()
        .addBands(year_band)
        .reduceRegion(
            reducer=ee.Reducer.sum().group(groupField=1, groupName="label"),
            geometry=bbox,
            scale=scale,
            maxPixels=int(1e10),
        )
        .getInfo()
    )

    if result is None:
        err = "Failed to compute area."
        raise ValueError(err)

    out = {}
    for group in result["groups"]:
        out[group["label"]] = group["sum"]

    return out


def reduce_all_year_bands(
    img: ee.Image, bbox: ee.Geometry, *, scale: int, label_map: dict[int, str]
) -> list[dict]:
    band_names = get_band_names(img)

    rows = []
    for band in band_names:
        year_band = img.select(band)
        reduced = reduce_year_band(year_band, bbox, scale=scale)

        rows.extend(
            [
                {
                    "year": int(band),
                    "label": label_map[label],
                    "area_m2": area,
                }
                for label, area in reduced.items()
            ]
        )

    return rows


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
    rows = reduce_all_year_bands(
        area_raster,
        bbox,
        scale=30,
        label_map=LABEL_MAP,
    )
    return pd.DataFrame(rows).pivot_table(
        index="year", columns="label", values="area_m2"
    )


@dg.op(out=dg.DynamicOut())
def generate_band_iterator(img: ee.Image) -> Iterator[dg.DynamicOutput[ee.Image]]:
    for band in get_band_names(img):
        yield dg.DynamicOutput(img.select(band), mapping_key=band)


@dg.op
def reduce_year_band_op(
    context: dg.OpExecutionContext, img: ee.Image, bbox: ee.Geometry
) -> list[dict]:
    reduced = reduce_year_band(img, bbox, scale=30)

    mapping_key = context.get_mapping_key()
    if mapping_key is None:
        err = (
            "Mapping key is None. This op should be called within a dynamic "
            "mapping context."
        )
        raise ValueError(err)

    return [
        {
            "year": int(mapping_key),
            "label": TRANSITION_LABEL_MAP_PRETTY[label],
            "area_m2": area,
        }
        for label, area in reduced.items()
    ]


@dg.op(out=dg.Out(io_manager_key="dataarray_manager"))
def aggregate_year_bands_op(reduced_list: list[list[dict]]) -> xr.DataArray:
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
)
def transition_table(transition_raster: ee.Image, bbox: ee.Geometry) -> xr.DataArray:
    band_imgs = generate_band_iterator(transition_raster)
    mapped = band_imgs.map(lambda img: reduce_year_band_op(img, bbox))
    return aggregate_year_bands_op(mapped.collect())
