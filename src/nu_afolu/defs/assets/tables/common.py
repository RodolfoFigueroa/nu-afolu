from collections.abc import Iterator

import ee

import dagster as dg


def get_band_names(img: ee.Image) -> list[str]:
    band_names = img.bandNames().getInfo()

    if not isinstance(band_names, list):
        err = f"Expected band names to be a list of strings, got {type(band_names)}"
        raise TypeError(err)

    return band_names


@dg.op(out=dg.DynamicOut())
def generate_band_iterator(img: ee.Image) -> Iterator[dg.DynamicOutput[ee.Image]]:
    for band in get_band_names(img):
        yield dg.DynamicOutput(img.select(band), mapping_key=band)


def get_mapping_key_safe(context: dg.OpExecutionContext) -> int:
    mapping_key = context.get_mapping_key()
    if mapping_key is None:
        err = (
            "Mapping key is None. This op should be called within a dynamic "
            "mapping context."
        )
        raise ValueError(err)
    return int(mapping_key)


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
