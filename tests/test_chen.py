# ruff: noqa: S101

from __future__ import annotations

import math
import unittest

import pandas as pd

from nu_afolu.chen import GeoManager, Zone
from nu_afolu.utils import safe_ratio

EXPECTED_RATIO = 2.0
SOURCE_YEAR = 2020
ZONE_A_SETTLEMENTS = 10
ZONE_B_SETTLEMENTS = 20


def _make_zone(area_df: pd.DataFrame, bbox: object = "bbox") -> Zone:
    return Zone(
        bbox=bbox,
        area_raster=object(),
        transition_raster=object(),
        area_df=area_df,
        chen_collection=object(),
        settlement_idx=9,
    )


class SafeRatioTest(unittest.TestCase):
    def test_divides_nonzero_denominator(self) -> None:
        assert safe_ratio(6, 3) == EXPECTED_RATIO

    def test_zero_denominator_returns_nan(self) -> None:
        assert math.isnan(safe_ratio(6, 0))


class ZoneTest(unittest.TestCase):
    def test_bbox_assignment_invalidates_bbox_dependent_cache(self) -> None:
        zone = _make_zone(pd.DataFrame({"settlements": [1]}, index=[2020]))
        zone.__dict__["ssp_images"] = {"SSP1": object()}
        zone.__dict__["area_chen"] = pd.DataFrame()
        zone.__dict__["settlement_mask"] = object()

        zone.bbox = "new-bbox"

        assert "ssp_images" not in zone.__dict__
        assert "area_chen" not in zone.__dict__
        assert "settlement_mask" in zone.__dict__
        assert zone.bbox == "new-bbox"


class GeoManagerTest(unittest.TestCase):
    def test_area_df_combines_zone_tables_and_invalidates_on_change(self) -> None:
        manager = GeoManager()
        manager["zone-a"] = _make_zone(
            pd.DataFrame(
                {"settlements": [ZONE_A_SETTLEMENTS], "forests_primary": [90]},
                index=[SOURCE_YEAR],
            )
        )

        first = manager.area_df
        assert first.loc[("zone-a", SOURCE_YEAR), "settlements"] == ZONE_A_SETTLEMENTS

        manager["zone-b"] = _make_zone(
            pd.DataFrame(
                {"settlements": [ZONE_B_SETTLEMENTS], "forests_primary": [80]},
                index=[SOURCE_YEAR],
            )
        )

        second = manager.area_df
        assert first is not second
        assert second.loc[("zone-b", SOURCE_YEAR), "settlements"] == ZONE_B_SETTLEMENTS


if __name__ == "__main__":
    unittest.main()
