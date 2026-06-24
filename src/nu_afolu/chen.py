import json
from collections.abc import Callable, Iterator, Sequence
from functools import cached_property
from pathlib import Path

import ee
import pandas as pd
import xarray as xr

from nu_afolu.constants import (
    CHEN_COLLECTION_ID,
    CHEN_URBAN_VALUE,
    CHEN_YEARS,
    LABEL_LIST,
    SSP_NAMES,
    TRANSITION_NODATA,
)

LABEL_MAP = dict(enumerate(LABEL_LIST, start=1))
LABEL_ID_BY_NAME = {label: idx for idx, label in LABEL_MAP.items()}
SETTLEMENT_IDX = LABEL_ID_BY_NAME["settlements"]


CHEN_COLLECTION = ee.ImageCollection(CHEN_COLLECTION_ID)


def reduce_chen_urban_area_by_scenario(
    col: ee.ImageCollection,
    *,
    geometry: ee.Geometry,
    scale: float,
) -> pd.DataFrame:
    """Reduce Chen urban pixels to area totals by scenario and year.

    Args:
        col: Chen Earth Engine image collection with one band per SSP scenario.
        geometry: Analysis geometry used to clip the reduction.
        scale: Earth Engine reduction scale in meters.

    Returns:
        DataFrame indexed by ``CHEN_YEARS`` with one column per SSP scenario.
        Values are urban area totals in square meters.

    Raises:
        TypeError: If Earth Engine does not return a list for a scenario.
    """
    reduced: ee.FeatureCollection = col.map(
        lambda img: ee.Feature(
            geometry,
            img.eq(ee.Number(CHEN_URBAN_VALUE))
            .multiply(ee.Image.pixelArea())
            .reduceRegion(ee.Reducer.sum(), geometry=geometry, scale=scale),
        )
    )

    collected: dict[str, list[float]] = {}
    for property_name in SSP_NAMES:
        arr = reduced.aggregate_array(property_name).getInfo()
        if not isinstance(arr, list):
            err = f"Expected list for property {property_name}, got {type(arr)}"
            raise TypeError(err)
        collected[property_name] = arr

    return pd.DataFrame(collected, index=list(CHEN_YEARS))


class ChenAnalysisZone:
    """Zone-level wrapper around observed artifacts and Chen projection helpers."""

    def __init__(
        self,
        bbox: ee.Geometry,
        area_raster: ee.Image,
        transition_raster: ee.Image,
        area_df: pd.DataFrame,
        transition_arr: xr.DataArray,
    ) -> None:
        """Initialize an analysis zone from generated observed artifacts.

        Args:
            bbox: Earth Engine geometry delimiting the analysis zone.
            area_raster: Observed GLC_FCS30D land-use raster clipped to the zone.
            transition_raster: Observed consecutive-year transition raster.
            area_df: Area table loaded from ``OUT_PATH/area_table/{ZONE}.parquet``.
                Values are square meters by land-use class and year.
            transition_arr: Transition table loaded from
                ``OUT_PATH/transition_table/{ZONE}.nc``. Values are square meters
                by transition start year, start class, and end class.
        """
        self._bbox: ee.Geometry = bbox
        self.area_raster: ee.Image = area_raster
        self.transition_raster: ee.Image = transition_raster
        self.area_df: pd.DataFrame = area_df
        self.transition_arr: xr.DataArray = transition_arr
        self._fields_to_invalidate = (
            "chen_urban_masks_by_scenario",
            "chen_urban_area_by_scenario_m2",
        )

    @property
    def bbox(self) -> ee.Geometry:
        """Return the Earth Engine geometry for this zone.

        Returns:
            The current zone bounding geometry.
        """
        return self._bbox

    @bbox.setter
    def bbox(self, value: ee.Geometry) -> None:
        """Set the zone geometry and clear bbox-dependent cached fields.

        Args:
            value: New Earth Engine geometry for this zone.
        """
        self._bbox = value
        self._invalidate_bbox_dependent_fields()

    @bbox.deleter
    def bbox(self) -> None:
        """Delete the zone geometry and clear bbox-dependent cached fields."""
        del self._bbox
        self._invalidate_bbox_dependent_fields()

    def _invalidate_bbox_dependent_fields(self) -> None:
        """Clear cached Chen fields derived from the current bounding box."""
        for field in self._fields_to_invalidate:
            self.__dict__.pop(field, None)

    @cached_property
    def chen_urban_masks_by_scenario(self) -> dict[str, ee.Image]:
        """Build clipped Chen urban masks for each SSP scenario.

        Returns:
            Mapping from SSP scenario name to a multi-band Earth Engine image.
            Bands are named by ``CHEN_YEARS`` and masked to Chen urban pixels.
        """
        masks_by_scenario: dict[str, ee.Image] = {}
        for name in SSP_NAMES:
            masks_by_scenario[name] = (
                CHEN_COLLECTION.select(name)
                .toBands()
                .rename([str(year) for year in CHEN_YEARS])
                .eq(ee.Number(CHEN_URBAN_VALUE))
                .clip(self.bbox)
                .selfMask()
            )
        return masks_by_scenario

    @cached_property
    def chen_urban_area_by_scenario_m2(self) -> pd.DataFrame:
        """Return Chen urban area totals for this zone in square meters.

        Returns:
            DataFrame indexed by Chen projection year with one column per SSP
            scenario.
        """
        return reduce_chen_urban_area_by_scenario(
            CHEN_COLLECTION,
            geometry=self.bbox,
            scale=1000,
        )

    @cached_property
    def observed_settlement_mask(self) -> ee.Image:
        """Build a mask of observed settlement pixels from the area raster.

        Returns:
            Earth Engine image where observed GLC_FCS30D settlement pixels are 1.
        """
        return self.area_raster.eq(ee.Number(SETTLEMENT_IDX))

    def resample_observed_settlement_mask(self, ssp: str) -> ee.Image:
        """Aggregate observed settlements to the Chen projection for an SSP.

        Args:
            ssp: SSP scenario name used to select the target Chen projection.

        Returns:
            Earth Engine image with observed settlement fractions reprojected to
            the scenario's Chen grid.

        Raises:
            KeyError: If ``ssp`` is not present in the Chen scenario masks.
        """
        return self.observed_settlement_mask.reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=2048,
        ).reproject(self.chen_urban_masks_by_scenario[ssp].projection())


class _ObservableDict(dict[str, ChenAnalysisZone]):
    """Dictionary that notifies a callback when zone membership changes."""

    def __init__(self, on_change: Callable[[], None]) -> None:
        """Initialize the observable dictionary.

        Args:
            on_change: Callback invoked after zone insertions and deletions.
        """
        self._on_change = on_change
        super().__init__()

    def __setitem__(self, key: str, item: ChenAnalysisZone) -> None:
        """Store a zone and mark aggregate caches as stale.

        Args:
            key: Zone name.
            item: Zone object to store.
        """
        super().__setitem__(key, item)
        self._on_change()

    def __delitem__(self, key: str) -> None:
        """Remove a zone and mark aggregate caches as stale.

        Args:
            key: Zone name to remove.

        Raises:
            KeyError: If ``key`` is not present.
        """
        super().__delitem__(key)
        self._on_change()


class ChenAnalysisZoneCollection:
    """Mutable collection of Chen analysis zones with cached aggregate views."""

    def __init__(self) -> None:
        """Initialize an empty zone collection."""
        self._zones = _ObservableDict(self._invalidate)
        self._area_df: pd.DataFrame | None = None
        self._area_arr: xr.DataArray | None = None
        self._transition_arr: xr.DataArray | None = None

    def _invalidate(self) -> None:
        """Clear cached aggregate views after zone membership changes."""
        self._area_df = None
        self._area_arr = None
        self._transition_arr = None

    def __getitem__(self, key: str) -> ChenAnalysisZone:
        """Return a zone by name.

        Args:
            key: Zone name.

        Returns:
            The matching analysis zone.

        Raises:
            KeyError: If ``key`` is not present.
        """
        return self._zones[key]

    def __setitem__(self, key: str, value: ChenAnalysisZone) -> None:
        """Add or replace a named zone.

        Args:
            key: Zone name.
            value: Analysis zone to store.
        """
        self._zones[key] = value

    def __delitem__(self, key: str) -> None:
        """Remove a named zone.

        Args:
            key: Zone name to remove.

        Raises:
            KeyError: If ``key`` is not present.
        """
        del self._zones[key]

    def __iter__(self) -> Iterator[tuple[str, ChenAnalysisZone]]:
        """Iterate over zone-name and zone pairs.

        Returns:
            Iterator over ``(zone_name, zone)`` tuples.
        """
        return iter(self._zones.items())

    @property
    def zones(self) -> dict[str, ChenAnalysisZone]:
        """Return the mutable mapping of loaded zones.

        Returns:
            Dictionary-like mapping from zone name to analysis zone. Direct
            membership changes invalidate cached aggregate views.
        """
        return self._zones

    @property
    def area_df(self) -> pd.DataFrame:
        """Return observed area artifacts combined across zones.

        Returns:
            DataFrame indexed by ``zone`` and ``year`` with one column per
            land-use category. Values are square meters from the generated
            ``OUT_PATH/area_table/{ZONE}.parquet`` artifacts.
        """
        if self._area_df is None:
            self._area_df = (
                pd.concat(
                    [zone.area_df.assign(zone=key) for key, zone in self._zones.items()]
                )
                .reset_index(names="year")
                .assign(year=lambda df: df["year"].astype(int))
                .set_index(["zone", "year"])
            )
        return self._area_df

    @property
    def area_arr(self) -> xr.DataArray:
        """Return observed area artifacts as an xarray data array.

        Returns:
            DataArray named ``area`` with ``zone``, ``year``, and ``category``
            coordinates. Values are square meters.

        Raises:
            TypeError: If the pandas-to-xarray conversion does not produce a
                DataArray.
        """
        if self._area_arr is None:
            out = self.area_df.rename_axis(columns="category").stack().to_xarray()
            out.name = "area"
            if not isinstance(out, xr.DataArray):
                err = f"Expected area_arr to be an xarray.DataArray, got {type(out)}"
                raise TypeError(err)
            self._area_arr = out
        return self._area_arr

    @property
    def transition_arr(self) -> xr.DataArray:
        """Return observed transition artifacts combined across zones.

        Returns:
            DataArray with an added ``zone`` dimension and the original
            transition artifact dimensions, typically ``year``, ``start``, and
            ``end``. Values are square meters from the generated
            ``OUT_PATH/transition_table/{ZONE}.nc`` artifacts.
        """
        if self._transition_arr is None:
            self._transition_arr = xr.concat(
                [zone.transition_arr for zone in self._zones.values()],
                dim=pd.Index(self._zones.keys(), name="zone"),
            )
        return self._transition_arr


def _decode_ee_json(path: Path) -> ee.ComputedObject:
    """Load a serialized Earth Engine JSON artifact.

    Args:
        path: Path to a JSON file containing an Earth Engine serialized object.

    Returns:
        Decoded Earth Engine computed object.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If ``path`` does not contain valid JSON.
    """
    with path.open() as file:
        return ee.deserializer.decode(json.load(file))


def load_chen_analysis_zones(
    out_path: Path,
    zone_names: Sequence[str],
) -> ChenAnalysisZoneCollection:
    """Load generated artifacts into a Chen analysis zone collection.

    Args:
        out_path: Root artifact directory containing ``bbox/ee``,
            ``area_raster``, ``transition_raster``, ``area_table``, and
            ``transition_table`` subdirectories.
        zone_names: Zone names to load. Zones with missing artifacts are skipped.

    Returns:
        Collection populated with successfully loaded analysis zones.

    Raises:
        ValueError: If none of the requested zones can be loaded.
    """
    manager = ChenAnalysisZoneCollection()

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
            transition_arr = xr.open_dataarray(
                out_path / "transition_table" / f"{zone_name}.nc",
            )
        except FileNotFoundError:
            continue

        bbox = ee.Geometry(bbox)
        area_raster = ee.Image(area_raster).clip(bbox)
        area_raster = area_raster.updateMask(area_raster.neq(ee.Number(0)))
        transition_raster = ee.Image(transition_raster).clip(bbox)
        transition_raster = transition_raster.updateMask(
            transition_raster.neq(ee.Number(TRANSITION_NODATA)),
        )

        manager[zone_name] = ChenAnalysisZone(
            bbox=bbox,
            area_raster=area_raster,
            transition_raster=transition_raster,
            area_df=area_df,
            transition_arr=transition_arr,
        )

    if not manager.zones:
        err = "No zones were loaded. Check OUT_PATH and upstream artifacts."
        raise ValueError(err)

    return manager
