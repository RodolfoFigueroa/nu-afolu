from collections.abc import Iterator

import ee
import numpy as np
import pandas as pd
from dagster_components.partitions import zone_partitions

import dagster as dg
from nu_afolu.constants import LABEL_LIST
from nu_afolu.defs.resources import AFOLUClassMapResource, LabelResource
from nu_afolu.utils import year_to_band_name


def get_single_image_from_collection(col: ee.ImageCollection) -> ee.Image:
    size = col.size().getInfo()

    if not isinstance(size, int):
        err = f"Expected int, got {type(size)}"
        raise TypeError(err)
    if size == 0:
        err = "No images found in the specified bounding box."
        raise ValueError(err)
    if size > 1:
        err = f"Expected 1 image, found {size} images in the specified bounding box."
        raise ValueError(err)
    return col.first()


@dg.op
def load_glc30(bbox: ee.geometry.Geometry) -> ee.image.Image:
    filtered = ee.imagecollection.ImageCollection(
        "projects/sat-io/open-datasets/GLC-FCS30D/annual",
    ).filterBounds(bbox)
    return get_single_image_from_collection(filtered)


@dg.op
def get_native_projection(glc30: ee.image.Image) -> ee.projection.Projection:
    return glc30.projection()


@dg.op
def generate_transition_label_map() -> dict[str, list[str]]:
    multiplier = 10 ** np.ceil(np.log10(len(LABEL_LIST)))

    out = {}
    for i, start_label in enumerate(LABEL_LIST):
        for j, end_label in enumerate(LABEL_LIST):
            key = int(i * multiplier + j)
            if key in out:
                err = f"Key {key} already exists in the dictionary."
                raise ValueError(err)
            out[str(key)] = [start_label, end_label]

    return out


@dg.op
def generate_transition_raster(
    start_year: int,
    bbox: ee.geometry.Geometry,
    transition_label_map: dict[str, list[str]],
    croplands_img: ee.Image,
    flooded_img: ee.Image,
    forests_mangroves_img: ee.Image,
    forests_primary_img: ee.Image,
    forests_secondary_img: ee.Image,
    grasslands_img: ee.Image,
    other_img: ee.Image,
    pastures_img: ee.Image,
    settlements_img: ee.Image,
    shrublands_img: ee.Image,
    wetlands_img: ee.Image,
) -> ee.Image:
    end_year = start_year + 1
    start_band = year_to_band_name(start_year)
    end_band = year_to_band_name(end_year)

    masked = (
        ee.Image.constant(0).rename("class").uint8()
        # .clip(bbox)
    )

    img_map = {
        "croplands": croplands_img,
        "flooded": flooded_img,
        "forests_mangroves": forests_mangroves_img,
        "forests_primary": forests_primary_img,
        "forests_secondary": forests_secondary_img,
        "grasslands": grasslands_img,
        "other": other_img,
        "pastures": pastures_img,
        "settlements": settlements_img,
        "shrublands": shrublands_img,
        "wetlands": wetlands_img,
    }

    transition_label_map_inv = {
        tuple(value): int(key) for key, value in transition_label_map.items()
    }

    for start_label, start_img in img_map.items():
        for end_label, end_img in img_map.items():
            a: ee.Image = start_img.select(start_band).rename("class")
            b: ee.Image = end_img.select(end_band).rename("class")

            masked = masked.where(
                a.And(b),
                ee.Number(transition_label_map_inv[(start_label, end_label)]),
            )

    return masked


def class_mask_op_factory(
    class_name: str,
) -> dg.OpDefinition:
    @dg.op(
        name=f"generate_{class_name}_mask",
    )
    def _op(
        class_map_resource: AFOLUClassMapResource,
        glc30: ee.Image,
        bbox: ee.Geometry,
    ) -> ee.Image:
        label_spec: LabelResource = getattr(class_map_resource, class_name)

        mask = (
            ee.Image.constant([0] * 23).rename([f"b{i}" for i in range(1, 24)])
            # .clip(bbox)
        )

        for label_range in label_spec.ranges:
            if label_range[0] == label_range[1]:
                temp_mask = glc30.eq(ee.Number(label_range[0]))
            else:
                temp_mask = glc30.gte(ee.Number(label_range[0])).And(
                    glc30.lte(ee.Number(label_range[1]))
                )
            mask = mask.Or(temp_mask)

        return mask

    return _op


@dg.op
def generate_random_grasslands_to_pastures_discriminator(
    bbox: ee.Geometry,
    glc30: ee.Image,
) -> ee.Image:
    proj = glc30.projection().getInfo()

    if not isinstance(proj, dict):
        err = f"Expected dict, got {type(proj)}"
        raise TypeError(err)

    return (
        ee.Image.random(42)
        .reproject(crs=proj["crs"], crsTransform=proj["transform"])
        .lte(ee.Number(0.4))
        # .clip(bbox)
    )


@dg.op
def merge_grasslands(
    grasslands_to_pastures_img: ee.Image,
    grasslands_img: ee.Image,
    pastures_random_mask: ee.Image,
) -> ee.Image:
    return grasslands_to_pastures_img.And(
        pastures_random_mask.Not(),
    ).Or(grasslands_img)


@dg.op
def get_forest_discriminator(bbox: ee.Geometry) -> ee.Image:
    filtered = ee.ImageCollection(
        "NASA/ORNL/global_forest_classification_2020/V1",
    ).filterBounds(bbox)
    img = get_single_image_from_collection(filtered)
    return img.unmask(ee.Number(0)).eq(ee.Number(1))
    #     .mode()
    #     .clip(bbox)
    #     .unmask(ee.Number(0))
    #     .eq(ee.Number(1))
    # )


@dg.op
def get_forests_primary_mask(
    forests_img: ee.Image,
    forests_discriminator: ee.Image,
) -> ee.Image:
    return forests_img.And(forests_discriminator)


@dg.op
def get_forests_secondary_mask(
    forests_img: ee.Image,
    forests_discriminator: ee.Image,
) -> ee.Image:
    return forests_img.And(forests_discriminator.Not())


@dg.op
def get_pastures_mask(
    grasslands_to_pastures_img: ee.Image,
    pastures_random_discriminator: ee.Image,
) -> ee.Image:
    return grasslands_to_pastures_img.And(pastures_random_discriminator)


@dg.op(out=dg.DynamicOut(dagster_type=int))
def generate_glc_fcs_years() -> Iterator[dg.DynamicOutput[int]]:
    for year in range(2000, 2022):
        yield dg.DynamicOutput(year, mapping_key=str(year))


@dg.op
def convert_raster_to_table(
    raster: ee.Image,
    bbox: ee.Geometry,
    transition_label_map: dict[str, list[str]],
) -> pd.DataFrame:
    transition_img: ee.Image = raster.addBands(
        ee.Image.pixelArea(),
    ).select(
        ["area", "class"],
    )

    response = transition_img.reduceRegion(
        reducer=(ee.Reducer.sum().group(groupField=1, groupName="transition")),
        scale=30,
        geometry=bbox,
        maxPixels=int(1e10),
    ).getInfo()

    if response is None:
        err = "No data returned from reduceRegion."
        raise ValueError(err)

    rows = [
        {
            "label": transition_label_map[str(elem["transition"])],
            "area": float(elem["sum"]),
        }
        for elem in response["groups"]
    ]

    out = (
        pd.DataFrame(rows)
        .assign(
            start=lambda df: df["label"].str[0],
            end=lambda df: df["label"].str[1],
        )
        .drop(columns=["label"])
        .pivot_table(index="start", columns="end", values="area")
        .fillna(0)
    )

    for label in LABEL_LIST:
        if label not in out.index:
            out.loc[label] = 0

        if label not in out.columns:
            out[label] = 0

    return out.sort_index(axis=0).sort_index(axis=1)


@dg.op
def generate_area_raster(
    year: int,
    croplands_img: ee.image.Image,
    flooded_img: ee.image.Image,
    forests_mangroves_img: ee.image.Image,
    forests_primary_img: ee.image.Image,
    forests_secondary_img: ee.image.Image,
    grasslands_img: ee.image.Image,
    other_img: ee.image.Image,
    pastures_img: ee.image.Image,
    settlements_img: ee.image.Image,
    shrublands_img: ee.image.Image,
    wetlands_img: ee.image.Image,
) -> ee.image.Image:
    band = year_to_band_name(year)

    img_map = {
        "croplands": croplands_img,
        "flooded": flooded_img,
        "forests_mangroves": forests_mangroves_img,
        "forests_primary": forests_primary_img,
        "forests_secondary": forests_secondary_img,
        "grasslands": grasslands_img,
        "other": other_img,
        "pastures": pastures_img,
        "settlements": settlements_img,
        "shrublands": shrublands_img,
        "wetlands": wetlands_img,
    }

    out_img = ee.image.Image.constant(0).rename("class").uint8()
    for i, label in enumerate(LABEL_LIST):
        out_img = out_img.where(
            img_map[label].select(band).rename("class"),
            ee.Number(i + 1),
        )

    return out_img


@dg.op
def reduce_area_raster_to_table(
    img: ee.image.Image,
    bbox: ee.Geometry,
) -> pd.DataFrame:
    transition_img: ee.Image = img.addBands(ee.Image.pixelArea()).select(
        ["area", "class"],
    )

    response = transition_img.reduceRegion(
        reducer=(ee.Reducer.sum().group(groupField=1, groupName="transition")),
        scale=30,
        geometry=bbox,
        maxPixels=int(1e10),
    ).getInfo()

    if response is None:
        err = "No data returned from reduceRegion."
        raise ValueError(err)

    rows = [
        {
            "label": LABEL_LIST[elem["transition"] - 1],
            "area": float(elem["sum"]),
        }
        for elem in response["groups"]
        if elem["transition"] != 0
    ]

    return pd.DataFrame(rows).set_index("label")


@dg.op(
    out=dg.Out(io_manager_key="dataframe_manager"),
)
def merge_transition_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    table_frac_map = {}
    for key, df in enumerate(tables):
        start_year = key + 2000
        end_year = start_year + 1
        table_frac_map[f"{start_year}_{end_year}"] = df

    time_periods = [
        f"{start_year}_{end_year}"
        for start_year, end_year in zip(
            range(2000, 2022),
            range(2001, 2023),
            strict=True,
        )
    ]
    time_period_map = {key: i for i, key in enumerate(time_periods)}

    rows = []
    for period in time_periods:
        table = table_frac_map[period]
        for start_label in sorted(LABEL_LIST):
            for end_label in sorted(LABEL_LIST):
                rows.append(  # noqa: PERF401
                    {
                        "transition": f"pij_lndu_{start_label}_to_{end_label}",
                        "time_period": time_period_map[period],
                        "value": table.loc[start_label, end_label],
                    },
                )

    return pd.DataFrame(rows).pivot_table(
        index="time_period",
        columns="transition",
        values="value",
    )


@dg.op(
    out=dg.Out(io_manager_key="dataframe_manager"),
)
def merge_area_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    out = []
    for key, df in enumerate(tables):
        year = key + 2000
        temp = df.assign(year=int(year) - 2000)
        out.append(temp)

    out = (
        pd.concat(out)
        .pivot_table(index="label", columns="year", values="area")
        .divide(10_000)
    )

    for label in LABEL_LIST:
        if label not in out.index:
            out.loc[label] = 0.1

    return out.sort_index().sort_index(axis=1)


@dg.op(out=dg.Out(io_manager_key="earthengine_manager"))
def stack_rasters_into_bands(
    rasters: list[ee.Image],
    projection: ee.projection.Projection,
) -> ee.Image:
    print(projection.getInfo())
    return (
        ee.ImageCollection(rasters)
        .toBands()
        .rename([str(i + 2000) for i in range(len(rasters))])
        .reproject(projection)
    )


@dg.graph_multi_asset(
    ins={
        "bbox": dg.AssetIn(["bbox", "ee"]),
    },
    outs={"area_raster": dg.AssetOut(), "transition_raster": dg.AssetOut()},
    partitions_def=zone_partitions,
    group_name="class_mask_graph",
)
def build_glc_fcs_zone_rasters(bbox: ee.Geometry) -> dict[str, pd.DataFrame]:
    glc30 = load_glc30(bbox)
    native_proj = get_native_projection(glc30)
    transition_label_map = generate_transition_label_map()

    grasslands_base = class_mask_op_factory("grasslands")(glc30, bbox)
    grasslands_to_pastures_mask = class_mask_op_factory("grasslands_to_pastures")(
        glc30,
        bbox,
    )
    random_grasslands_to_pastures_discriminator = (
        generate_random_grasslands_to_pastures_discriminator(
            bbox,
            glc30,
        )
    )
    grasslands = merge_grasslands(
        grasslands_to_pastures_mask,
        grasslands_base,
        random_grasslands_to_pastures_discriminator,
    )
    pastures = get_pastures_mask(
        grasslands_to_pastures_mask,
        random_grasslands_to_pastures_discriminator,
    )

    forests_base = class_mask_op_factory("forests")(glc30, bbox)
    forests_discriminator = get_forest_discriminator(bbox)
    forests_primary = get_forests_primary_mask(forests_base, forests_discriminator)
    forests_secondary = get_forests_secondary_mask(forests_base, forests_discriminator)

    croplands = class_mask_op_factory("croplands")(glc30, bbox)
    flooded = class_mask_op_factory("flooded")(glc30, bbox)
    forests_mangroves = class_mask_op_factory("forests_mangroves")(glc30, bbox)
    other = class_mask_op_factory("other")(glc30, bbox)
    settlements = class_mask_op_factory("settlements")(glc30, bbox)
    shrublands = class_mask_op_factory("shrublands")(glc30, bbox)
    wetlands = class_mask_op_factory("wetlands")(glc30, bbox)

    wanted_years = generate_glc_fcs_years()

    transition_rasters = (
        wanted_years.map(
            lambda year: generate_transition_raster(
                year,
                bbox=bbox,
                transition_label_map=transition_label_map,
                croplands_img=croplands,
                flooded_img=flooded,
                forests_mangroves_img=forests_mangroves,
                forests_primary_img=forests_primary,
                forests_secondary_img=forests_secondary,
                grasslands_img=grasslands,
                other_img=other,
                pastures_img=pastures,
                settlements_img=settlements,
                shrublands_img=shrublands,
                wetlands_img=wetlands,
            )
        )
        # .map(
        #     lambda raster: convert_raster_to_table(
        #         raster, bbox=bbox, transition_label_map=transition_label_map
        #     )
        # )
    )

    area_rasters = (
        wanted_years.map(
            lambda year: generate_area_raster(
                year=year,
                croplands_img=croplands,
                flooded_img=flooded,
                forests_mangroves_img=forests_mangroves,
                forests_primary_img=forests_primary,
                forests_secondary_img=forests_secondary,
                grasslands_img=grasslands,
                other_img=other,
                pastures_img=pastures,
                settlements_img=settlements,
                shrublands_img=shrublands,
                wetlands_img=wetlands,
            )
        )
        # .map(lambda raster: reduce_area_raster_to_table(raster, bbox=bbox))
    )

    return {
        "area_raster": stack_rasters_into_bands(area_rasters.collect(), native_proj),
        "transition_raster": stack_rasters_into_bands(
            transition_rasters.collect(), native_proj
        ),
    }
