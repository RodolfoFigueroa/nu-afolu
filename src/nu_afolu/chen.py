from __future__ import annotations

import json
from functools import cached_property
from typing import TYPE_CHECKING

import ee
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

CHEN_COLLECTION_ID = "projects/sat-io/open-datasets/FUTURE-URBAN-LAND/CHEN_2020_2100"
CHEN_URBAN_VALUE = 2
CHEN_YEARS = tuple(range(2020, 2101, 10))
SSP_NAMES = tuple(f"SSP{suffix}" for suffix in range(1, 6))


def reduce_ssp_col(
    col: ee.ImageCollection,
    *,
    geometry: ee.Geometry,
    scale: float,
    urban_value: int = CHEN_URBAN_VALUE,
    ssp_names: Sequence[str] = SSP_NAMES,
    years: Sequence[int] = CHEN_YEARS,
) -> pd.DataFrame:
    reduced: ee.FeatureCollection = col.map(
        lambda img: ee.Feature(
            geometry,
            img.eq(ee.Number(urban_value))
            .multiply(ee.Image.pixelArea())
            .reduceRegion(ee.Reducer.sum(), geometry=geometry, scale=scale),
        )
    )

    collected: dict[str, list[float]] = {}
    for property_name in ssp_names:
        arr = reduced.aggregate_array(property_name).getInfo()
        if not isinstance(arr, list):
            err = f"Expected list for property {property_name}, got {type(arr)}"
            raise TypeError(err)
        collected[property_name] = arr

    return pd.DataFrame(collected, index=list(years))


def _decode_ee_json(path: Path) -> ee.ComputedObject:
    with path.open() as file:
        return ee.deserializer.decode(json.load(file))


class Zone:
    def __init__(
        self,
        bbox: ee.Geometry,
        area_raster: ee.Image,
        transition_raster: ee.Image,
        area_df: pd.DataFrame,
        *,
        chen_collection: ee.ImageCollection,
        settlement_idx: int,
        chen_urban_value: int = CHEN_URBAN_VALUE,
        chen_years: Sequence[int] = CHEN_YEARS,
        ssp_names: Sequence[str] = SSP_NAMES,
    ) -> None:
        self._bbox: ee.Geometry = bbox
        self.area_raster: ee.Image = area_raster
        self.transition_raster: ee.Image = transition_raster
        self.area_df: pd.DataFrame = area_df
        self.chen_collection: ee.ImageCollection = chen_collection
        self.settlement_idx = settlement_idx
        self.chen_urban_value = chen_urban_value
        self.chen_years = tuple(chen_years)
        self.ssp_names = tuple(ssp_names)
        self._fields_to_invalidate = ("ssp_images", "area_chen")

    @property
    def bbox(self) -> ee.Geometry:
        return self._bbox

    @bbox.setter
    def bbox(self, value: ee.Geometry) -> None:
        self._bbox = value
        self._invalidate_bbox_dependent_fields()

    @bbox.deleter
    def bbox(self) -> None:
        del self._bbox
        self._invalidate_bbox_dependent_fields()

    def _invalidate_bbox_dependent_fields(self) -> None:
        for field in self._fields_to_invalidate:
            self.__dict__.pop(field, None)

    @cached_property
    def ssp_images(self) -> dict[str, ee.Image]:
        ssp_images: dict[str, ee.Image] = {}
        for name in self.ssp_names:
            ssp_images[name] = (
                self.chen_collection.select(name)
                .toBands()
                .rename([str(year) for year in self.chen_years])
                .eq(ee.Number(self.chen_urban_value))
                .clip(self.bbox)
                .selfMask()
            )
        return ssp_images

    @cached_property
    def area_chen(self) -> pd.DataFrame:
        return reduce_ssp_col(
            self.chen_collection,
            geometry=self.bbox,
            scale=1000,
            urban_value=self.chen_urban_value,
            ssp_names=self.ssp_names,
            years=self.chen_years,
        )

    @cached_property
    def settlement_mask(self) -> ee.Image:
        return self.area_raster.eq(ee.Number(self.settlement_idx))

    def resample_settlement_mask(self, ssp: str) -> ee.Image:
        return self.settlement_mask.reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=2048,
        ).reproject(self.ssp_images[ssp].projection())


class _ObservableDict(dict[str, Zone]):
    def __init__(self, on_change: Callable[[], None]) -> None:
        self._on_change = on_change
        super().__init__()

    def __setitem__(self, key: str, item: Zone) -> None:
        super().__setitem__(key, item)
        self._on_change()

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        self._on_change()


class GeoManager:
    def __init__(self) -> None:
        self._zones = _ObservableDict(self._invalidate)
        self._area_df: pd.DataFrame | None = None
        self._area_arr: xr.DataArray | None = None

    def _invalidate(self) -> None:
        self._area_df = None
        self._area_arr = None

    def __getitem__(self, key: str) -> Zone:
        return self._zones[key]

    def __setitem__(self, key: str, value: Zone) -> None:
        self._zones[key] = value

    def __delitem__(self, key: str) -> None:
        del self._zones[key]

    def __iter__(self) -> Iterator[tuple[str, Zone]]:
        return iter(self._zones.items())

    @property
    def zones(self) -> dict[str, Zone]:
        return self._zones

    @property
    def area_df(self) -> pd.DataFrame:
        if self._area_df is None:
            self._area_df = (
                pd.concat(
                    [
                        zone.area_df.assign(zone=key)
                        for key, zone in self._zones.items()
                    ]
                )
                .reset_index(names="year")
                .assign(year=lambda df: df["year"].astype(int))
                .set_index(["zone", "year"])
            )
        return self._area_df

    @property
    def area_arr(self) -> xr.DataArray:
        if self._area_arr is None:
            out = self.area_df.rename_axis(columns="category").stack().to_xarray()
            out.name = "area"
            if not isinstance(out, xr.DataArray):
                err = f"Expected area_arr to be an xarray.DataArray, got {type(out)}"
                raise TypeError(err)
            self._area_arr = out
        return self._area_arr


def chen_urban_mask(zone: Zone, scenario: str, year: int) -> ee.Image:
    return zone.ssp_images[scenario].select(str(year)).unmask(0).rename("chen_urban")


def observed_settlement_fraction_image(
    zone: Zone,
    scenario: str,
    source_year: int,
) -> ee.Image:
    chen_projection = zone.ssp_images[scenario].select(str(source_year)).projection()
    return (
        zone.settlement_mask.select(str(source_year))
        .unmask(0)
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=2048)
        .reproject(chen_projection)
        .rename("observed_settlement_fraction")
    )


def load_chen_manager(
    out_path: Path,
    zone_names: Sequence[str],
    chen_collection: ee.ImageCollection,
    settlement_idx: int,
) -> tuple[GeoManager, pd.DataFrame]:
    manager = GeoManager()
    missing_rows: list[dict[str, str]] = []

    for zone_name in zone_names:
        try:
            bbox = _decode_ee_json(out_path / "bbox" / "ee" / f"{zone_name}.json")
            area_raster = _decode_ee_json(
                out_path / "area_raster" / f"{zone_name}.json",
            )
            transition_raster = _decode_ee_json(
                out_path / "transition_raster" / f"{zone_name}.json",
            )
            area_df = pd.read_parquet(out_path / "area_table" / f"{zone_name}.parquet")
        except FileNotFoundError as exc:
            missing_rows.append({"zone": zone_name, "missing_path": str(exc.filename)})
            continue

        bbox = ee.Geometry(bbox)
        area_raster = ee.Image(area_raster).clip(bbox)
        area_raster = area_raster.updateMask(area_raster.neq(ee.Number(0)))
        transition_raster = ee.Image(transition_raster).clip(bbox)
        transition_raster = transition_raster.updateMask(
            transition_raster.neq(ee.Number(0)),
        )

        manager[zone_name] = Zone(
            bbox=bbox,
            area_raster=area_raster,
            transition_raster=transition_raster,
            area_df=area_df,
            chen_collection=chen_collection,
            settlement_idx=settlement_idx,
        )

    if not manager.zones:
        err = "No zones were loaded. Check OUT_PATH and upstream raster artifacts."
        raise ValueError(err)

    return manager, pd.DataFrame(missing_rows)
