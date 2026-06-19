from __future__ import annotations

import math
import unittest
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from nu_afolu import chen
from nu_afolu.chen import (
    ChenAnalysisZone,
    ChenAnalysisZoneCollection,
    load_chen_analysis_zones,
)
from nu_afolu.utils import safe_ratio

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_RATIO = 2.0
SOURCE_YEAR = 2020
ZONE_A_SETTLEMENTS = 10
ZONE_B_SETTLEMENTS = 20
SETTLEMENT_IDX = 9


def _make_zone(area_df: pd.DataFrame, bbox: object = "bbox") -> ChenAnalysisZone:
    return ChenAnalysisZone(
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


class ChenAnalysisZoneTest(unittest.TestCase):
    def test_bbox_assignment_invalidates_bbox_dependent_cache(self) -> None:
        zone = _make_zone(pd.DataFrame({"settlements": [1]}, index=[2020]))
        zone.__dict__["chen_urban_masks_by_scenario"] = {"SSP1": object()}
        zone.__dict__["chen_urban_area_by_scenario_m2"] = pd.DataFrame()
        zone.__dict__["observed_settlement_mask"] = object()

        zone.bbox = "new-bbox"

        assert "chen_urban_masks_by_scenario" not in zone.__dict__
        assert "chen_urban_area_by_scenario_m2" not in zone.__dict__
        assert "observed_settlement_mask" in zone.__dict__
        assert zone.bbox == "new-bbox"


class ChenAnalysisZoneCollectionTest(unittest.TestCase):
    def test_area_df_combines_zone_tables_and_invalidates_on_change(self) -> None:
        manager = ChenAnalysisZoneCollection()
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


class _FakeImage:
    def clip(self, _bbox: object) -> _FakeImage:
        return self

    def updateMask(self, _mask: object) -> _FakeImage:  # noqa: N802
        return self

    def neq(self, _other: object) -> _FakeImage:
        return self


def test_load_chen_analysis_zones_loads_available_zones_and_reports_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    area_table_dir = tmp_path / "area_table"
    area_table_dir.mkdir()
    pd.DataFrame({"settlements": [ZONE_A_SETTLEMENTS]}, index=[SOURCE_YEAR]).to_parquet(
        area_table_dir / "zone-a.parquet",
    )

    fake_image = _FakeImage()
    monkeypatch.setattr(chen, "_decode_ee_json", lambda _path: fake_image)
    monkeypatch.setattr(chen.ee, "Geometry", lambda value: value)
    monkeypatch.setattr(chen.ee, "Image", lambda value: value)
    monkeypatch.setattr(chen.ee, "Number", lambda value: value)

    manager, missing = load_chen_analysis_zones(
        tmp_path,
        ["zone-a", "zone-b"],
        object(),
        SETTLEMENT_IDX,
    )

    assert list(manager.zones) == ["zone-a"]
    assert (
        manager["zone-a"].area_df.loc[SOURCE_YEAR, "settlements"]
        == ZONE_A_SETTLEMENTS
    )
    assert missing.to_dict("records") == [
        {"zone": "zone-b", "missing_path": str(area_table_dir / "zone-b.parquet")}
    ]


def test_load_chen_analysis_zones_raises_when_no_zones_load(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No zones were loaded"):
        load_chen_analysis_zones(tmp_path, ["zone-a"], object(), SETTLEMENT_IDX)


if __name__ == "__main__":
    unittest.main()
