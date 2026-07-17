import os
import time
from pathlib import Path
from typing import cast

import ee
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from xhistogram.xarray import histogram

import dagster as dg
from nu_afolu.constants import LABEL_MAP, TRANSITION_LABEL_MAP
from nu_afolu.defs.resources import PathResource

R = 6371000.0  # Earth's radius in meters
CHUNK_SIZE = 2048


def spawn_task(
    img: ee.Image,
    bbox: ee.Geometry,
    description: str,
    *,
    logger: dg.DagsterLogManager | None = None,
) -> bool:
    running_ops = [
        op
        for op in ee.data.listOperations()
        if op["metadata"]["state"] in ["RUNNING", "STARTING", "STARTED", "PENDING"]
    ]
    for op in running_ops:
        found_description = op["metadata"]["description"]
        if found_description.startswith(description):
            if logger is not None:
                msg = f"Raster task [{description}] found. Skipping."
                logger.info(msg)

            msg = f"Raster task [{description}] found."
            raise RuntimeError(msg)

    task = ee.batch.Export.image.toDrive(
        image=img,
        description=description,
        folder="nu_afolu_rasters",
        region=ee.Geometry(bbox),
        crs="EPSG:4326",
        scale=30,
        maxPixels=int(1e10),
    )
    task.start()

    if logger is not None:
        msg = f"Raster task [{description}] started."
        logger.info(msg)

    total_time = 0
    while task.active():
        time.sleep(10)
        total_time += 10
        if logger is not None:
            msg = (
                f"Raster task [{description}] is still running after "
                "{total_time} seconds."
            )
            logger.info(msg)

    return task.status()["state"] == "COMPLETED"


def materialized_raster_factory(raster_prefix: str) -> dg.AssetsDefinition:
    @dg.asset(
        key=[f"{raster_prefix}_raster_materialized"],
        ins={"raster": dg.AssetIn(key=[f"{raster_prefix}_raster", "large"])},
        group_name=f"{raster_prefix}_raster",
    )
    def _asset(
        context: dg.AssetExecutionContext, raster: ee.Image
    ) -> dg.MaterializeResult:
        result = spawn_task(
            raster,
            raster.geometry(),
            f"{raster_prefix}_raster_materialized",
            logger=context.log,
        )
        if result:
            return dg.MaterializeResult()

        msg = "Failed to materialize raster."
        raise RuntimeError(msg)

    return _asset


def reduce_tile(fpath: os.PathLike, *, vmin: int, vmax: int) -> xr.DataArray:
    ds = rxr.open_rasterio(
        fpath,
        chunks={"band": 1, "x": CHUNK_SIZE, "y": CHUNK_SIZE},
    )
    ds = cast("xr.DataArray", ds)
    ds = ds.fillna(0)
    ds.name = "raster"

    d_lat = np.deg2rad(float(abs(ds.y.diff(dim="y").to_numpy()[0])))
    d_lon = np.deg2rad(float(abs(ds.x.diff(dim="x").to_numpy()[0])))

    pixel_area = (R**2) * np.cos(np.deg2rad(ds.y)) * d_lon * d_lat
    pixel_area = cast("xr.DataArray", pixel_area)
    pixel_area = pixel_area.chunk({"y": CHUNK_SIZE})

    ds, pixel_area = xr.align(ds, pixel_area, join="exact")

    area_hist = histogram(
        ds,
        bins=[np.arange(vmin - 0.5, vmax + 0.5, 1.0)],
        dim=["y", "x"],
        weights=pixel_area,
    )
    area_hist = cast("xr.DataArray", area_hist)
    area_hist = area_hist.rename({"raster_bin": "label", "band": "year"})
    area_hist.name = "histogram"

    return area_hist.compute()


def reduce_all_tiles(dir_path: os.PathLike, *, vmin: int, vmax: int) -> xr.DataArray:
    dir_path = Path(dir_path)

    computed: list[xr.DataArray] = [
        reduce_tile(fpath, vmin=vmin, vmax=vmax) for fpath in dir_path.glob("*.tif")
    ]
    total = sum(computed)

    if not isinstance(total, xr.DataArray):
        err = f"Expected xr.DataArray, got {type(total)}"
        raise TypeError(err)

    return total


@dg.asset(
    key=["area_table", "large"],
    deps=["area_raster_materialized"],
    io_manager_key="dataframe_manager",
    group_name="area_table",
)
def area_table_large(path_resource: PathResource) -> pd.DataFrame:
    materialized_raster_dir = Path(path_resource.out_path) / "area_raster_materialized"

    return (
        reduce_all_tiles(materialized_raster_dir, vmin=1, vmax=12)
        .to_dataframe()
        .reset_index()
        .assign(
            label=lambda df: df["label"].map(LABEL_MAP),
            year=lambda df: df["year"] + 1999,
        )
        .pivot_table(index="year", columns="label", values="histogram")
    )


@dg.asset(
    key=["transition_table", "large"],
    deps=["transition_raster_materialized"],
    io_manager_key="dataarray_manager",
    group_name="transition_table",
)
def transition_table_large(path_resource: PathResource) -> xr.DataArray:
    materialized_raster_dir = (
        Path(path_resource.out_path) / "transition_raster_materialized"
    )

    temp = (
        reduce_all_tiles(
            materialized_raster_dir,
            vmin=min(TRANSITION_LABEL_MAP),
            vmax=max(TRANSITION_LABEL_MAP),
        )
        .to_dataframe()
        .reset_index()
        .assign(
            label=lambda df: df["label"].astype(int).map(TRANSITION_LABEL_MAP),
        )
        .dropna(subset=["label"])
        .assign(
            year=lambda df: df["year"] + 1999,
            start=lambda df: df["label"].str[0],
            end=lambda df: df["label"].str[1],
        )
        .drop(columns=["label"])
        .set_index(["year", "start", "end"])["histogram"]
        .rename("area_m2")
    )

    return xr.DataArray.from_series(temp)


dassets = [
    materialized_raster_factory("area"),
    materialized_raster_factory("transition"),
]
