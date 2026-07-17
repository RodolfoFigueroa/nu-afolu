from collections.abc import Callable
from dataclasses import dataclass

import ee
from cfc_dagster_utils.partitions import zone_partitions

import dagster as dg
from nu_afolu.constants import (
    LABEL_LIST,
    TRANSITION_LABEL_MAP,
    TRANSITION_NODATA,
)
from nu_afolu.defs.resources import AFOLUClassMapResource, LabelResource
from nu_afolu.utils import year_to_band_name


@dataclass
class MaskManager:
    croplands: ee.Image
    flooded: ee.Image
    forests_mangroves: ee.Image
    forests_primary: ee.Image
    forests_secondary: ee.Image
    grasslands: ee.Image
    other: ee.Image
    pastures: ee.Image
    settlements: ee.Image
    shrublands: ee.Image
    wetlands: ee.Image

    @property
    def image_map(self) -> dict[str, ee.Image]:
        return {name: getattr(self, name) for name in LABEL_LIST}

    def generate_single_year_area_raster(
        self,
        year: int,
    ) -> ee.Image:
        band = year_to_band_name(year)

        out_img = ee.image.Image.constant(0).rename("class").uint8()
        for i, label in enumerate(LABEL_LIST):
            out_img = out_img.where(
                self.image_map[label].select(band).rename("class"),
                ee.Number(i + 1),
            )

        source_mask = next(iter(self.image_map.values())).select(band).mask()
        return out_img.updateMask(source_mask)

    def generate_single_year_transition_raster(
        self,
        start_year: int,
    ) -> ee.Image:
        end_year = start_year + 1
        start_band = year_to_band_name(start_year)
        end_band = year_to_band_name(end_year)

        masked = ee.Image.constant(TRANSITION_NODATA).rename("class").int16()

        transition_label_map_inv = {
            tuple(value): int(key) for key, value in TRANSITION_LABEL_MAP.items()
        }

        for start_label, start_img in self.image_map.items():
            for end_label, end_img in self.image_map.items():
                a: ee.Image = start_img.select(start_band).rename("class")
                b: ee.Image = end_img.select(end_band).rename("class")

                masked = masked.where(
                    a.And(b),
                    ee.Number(transition_label_map_inv[(start_label, end_label)]),
                )

        return masked

    def generate_area_raster(self, native_proj: ee.Projection) -> ee.Image:
        return stack_rasters_into_bands(
            [self.generate_single_year_area_raster(year) for year in range(2000, 2022)],
            native_proj,
        )

    def generate_transition_raster(self, native_proj: ee.Projection) -> ee.Image:
        stacked = stack_rasters_into_bands(
            [
                self.generate_single_year_transition_raster(year)
                for year in range(2000, 2021)
            ],
            native_proj,
        )
        nodata_mask = stacked.neq(ee.Number(TRANSITION_NODATA))
        return stacked.updateMask(nodata_mask)


def get_single_image_from_collection(col: ee.ImageCollection) -> ee.Image:
    size = col.size().getInfo()

    if not isinstance(size, int):
        err = f"Expected int, got {type(size)}"
        raise TypeError(err)
    if size == 0:
        err = "No images found in the specified bounding box."
        raise ValueError(err)

    if size == 1:
        out = col.first()
    else:
        native_proj = ee.Image(col.first()).projection()
        out = col.sort("system:index").mosaic().setDefaultProjection(native_proj)

    return out


def load_glc30(bbox: ee.Geometry | ee.ComputedObject) -> ee.Image:
    filtered = ee.ImageCollection(
        "projects/sat-io/open-datasets/GLC-FCS30D/annual",
    ).filterBounds(bbox)  # ty:ignore[invalid-argument-type]

    glc30 = get_single_image_from_collection(filtered)
    valid = glc30.neq(ee.Number(0)).And(glc30.neq(ee.Number(250)))

    return glc30.updateMask(valid)


def get_native_projection(glc30: ee.Image) -> ee.projection.Projection:
    return glc30.projection()


def class_mask_op_factory(
    class_name: str,
) -> Callable[[AFOLUClassMapResource, ee.Image], ee.Image]:
    def _f(
        class_map_resource: AFOLUClassMapResource,
        glc30: ee.Image,
    ) -> ee.Image:
        label_spec: LabelResource = getattr(class_map_resource, class_name)

        mask = ee.Image.constant([0] * 23).rename([f"b{i}" for i in range(1, 24)])

        for label_range in label_spec.ranges:
            if label_range[0] == label_range[1]:
                temp_mask = glc30.eq(ee.Number(label_range[0]))
            else:
                temp_mask = glc30.gte(ee.Number(label_range[0])).And(
                    glc30.lte(ee.Number(label_range[1]))
                )
            mask = mask.Or(temp_mask)

        return mask

    return _f


CLASS_MASK_FUNC_MAP = {
    name: class_mask_op_factory(name)
    for name in [
        "croplands",
        "flooded",
        "forests_mangroves",
        "other",
        "settlements",
        "shrublands",
        "wetlands",
    ]
}


def generate_random_grasslands_to_pastures_discriminator(
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
    )


def merge_grasslands(
    grasslands_to_pastures_img: ee.Image,
    grasslands_img: ee.Image,
    pastures_random_mask: ee.Image,
) -> ee.Image:
    return grasslands_to_pastures_img.And(
        pastures_random_mask.Not(),
    ).Or(grasslands_img)


def get_forest_discriminator(bbox: ee.Geometry) -> ee.Image:
    filtered = ee.ImageCollection(
        "NASA/ORNL/global_forest_classification_2020/V1",
    ).filterBounds(bbox)
    img = get_single_image_from_collection(filtered)
    return img.unmask(ee.Number(0)).eq(ee.Number(1))


def get_forests_primary_mask(
    forests_img: ee.Image,
    forests_discriminator: ee.Image,
) -> ee.Image:
    return forests_img.And(forests_discriminator)


def get_forests_secondary_mask(
    forests_img: ee.Image,
    forests_discriminator: ee.Image,
) -> ee.Image:
    return forests_img.And(forests_discriminator.Not())


def get_pastures_mask(
    grasslands_to_pastures_img: ee.Image,
    pastures_random_discriminator: ee.Image,
) -> ee.Image:
    return grasslands_to_pastures_img.And(pastures_random_discriminator)


def stack_rasters_into_bands(
    rasters: list[ee.Image],
    projection: ee.projection.Projection,
) -> ee.Image:
    return (
        ee.ImageCollection(rasters)
        .toBands()
        .rename([str(i + 2000) for i in range(len(rasters))])
        .reproject(projection)
    )


def build_forests(
    class_map_resource: AFOLUClassMapResource, glc30: ee.Image, bbox: ee.Geometry
) -> tuple[ee.Image, ee.Image]:
    forests_base = class_mask_op_factory("forests")(class_map_resource, glc30)
    forests_discriminator = get_forest_discriminator(bbox)
    forests_primary = get_forests_primary_mask(forests_base, forests_discriminator)
    forests_secondary = get_forests_secondary_mask(forests_base, forests_discriminator)

    return forests_primary, forests_secondary


def build_grasslands_and_pastures(
    class_map_resource: AFOLUClassMapResource, glc30: ee.Image
) -> tuple[ee.Image, ee.Image]:
    grasslands_base = class_mask_op_factory("grasslands")(class_map_resource, glc30)
    grasslands_to_pastures_mask = class_mask_op_factory("grasslands_to_pastures")(
        class_map_resource,
        glc30,
    )

    random_grasslands_to_pastures_discriminator = (
        generate_random_grasslands_to_pastures_discriminator(glc30)
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

    return grasslands, pastures


def get_all_class_masks(
    class_map_resource: AFOLUClassMapResource,
    glc30: ee.Image,
    bbox: ee.Geometry | ee.ComputedObject,
) -> MaskManager:
    grasslands, pastures = build_grasslands_and_pastures(class_map_resource, glc30)
    forests_primary, forests_secondary = build_forests(class_map_resource, glc30, bbox)  # ty:ignore[invalid-argument-type]

    return MaskManager(
        **{
            "grasslands": grasslands,
            "pastures": pastures,
            "forests_primary": forests_primary,
            "forests_secondary": forests_secondary,
            **{
                name: CLASS_MASK_FUNC_MAP[name](class_map_resource, glc30)
                for name in CLASS_MASK_FUNC_MAP
            },
        }
    )


@dg.multi_asset(
    ins={
        "bbox": dg.AssetIn(["bbox", "ee"]),
    },
    outs={
        "area_raster": dg.AssetOut(
            io_manager_key="earthengine_manager", group_name="area"
        ),
        "transition_raster": dg.AssetOut(
            io_manager_key="earthengine_manager", group_name="transition"
        ),
    },
    partitions_def=zone_partitions,
)
def rasters(
    class_map_resource: AFOLUClassMapResource, bbox: ee.Geometry
) -> tuple[ee.Image, ee.Image]:
    glc30 = load_glc30(bbox)
    native_proj = get_native_projection(glc30)
    mask_manager = get_all_class_masks(class_map_resource, glc30, bbox)

    return mask_manager.generate_area_raster(
        native_proj
    ), mask_manager.generate_transition_raster(
        native_proj,
    )


@dg.multi_asset(
    ins={
        "bbox": dg.AssetIn(["large", "bbox", "ee"]),
    },
    outs={
        "area_raster": dg.AssetOut(
            key=["large", "area_raster"],
            io_manager_key="earthengine_manager",
            group_name="area_large",
        ),
        "transition_raster": dg.AssetOut(
            key=["large", "transition_raster"],
            io_manager_key="earthengine_manager",
            group_name="transition_large",
        ),
    },
)
def rasters_large(
    class_map_resource: AFOLUClassMapResource, bbox: ee.ComputedObject
) -> tuple[ee.Image, ee.Image]:
    glc30 = load_glc30(bbox)
    native_proj = get_native_projection(glc30)
    mask_manager = get_all_class_masks(class_map_resource, glc30, bbox)

    return mask_manager.generate_area_raster(
        native_proj
    ), mask_manager.generate_transition_raster(
        native_proj,
    )
