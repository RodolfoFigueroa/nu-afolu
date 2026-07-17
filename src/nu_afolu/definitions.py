import os
from pathlib import Path

import ee
import toml
from cfc_dagster_utils.managers.dataframe import (
    DataFrameFileManager,
)
from cfc_dagster_utils.managers.earthengine import EarthEngineManager
from cfc_dagster_utils.managers.geodataframe import GeoDataFrameFileManager
from cfc_dagster_utils.managers.xarray import DataArrayFileManager
from cfc_dagster_utils.resources import PostgresResource

import dagster as dg
from nu_afolu.defs.resources import AFOLUClassMapResource, LabelResource, PathResource

ee.Initialize(project=os.environ["EARTHENGINE_PROJECT"])


def generate_class_map(id_map_path: os.PathLike) -> AFOLUClassMapResource:
    id_map_path = Path(id_map_path)

    with Path(id_map_path).open(encoding="utf8") as f:
        config = toml.load(f)

    spec_map = {}
    all_resources_map = {}
    for key, spec in config.items():
        resource = LabelResource(
            ranges=spec.get("ranges"),
        )

        spec_map[key] = resource
        all_resources_map[f"{key}_map"] = resource

    return AFOLUClassMapResource(**spec_map)


@dg.definitions
def defs() -> dg.Definitions:
    main_defs = dg.load_from_defs_folder(project_root=Path(__file__).parent.parent)

    class_map_path = Path(__file__).parents[2] / "config" / "id_map.toml"
    path_resource = PathResource(
        out_path=str(Path(__file__).parents[2] / "data" / "output")
    )
    extra_defs = dg.Definitions(
        resources={
            "class_map_resource": generate_class_map(class_map_path),
            "dataframe_manager": DataFrameFileManager(
                path_resource=path_resource, extension=".parquet"
            ),
            "geodataframe_manager": GeoDataFrameFileManager(
                path_resource=path_resource, extension=".geoparquet"
            ),
            "path_resource": path_resource,
            "postgres_resource": PostgresResource(
                host=dg.EnvVar("POSTGRES_HOST"),
                port=dg.EnvVar("POSTGRES_PORT"),
                user=dg.EnvVar("POSTGRES_USER"),
                password=dg.EnvVar("POSTGRES_PASSWORD"),
                db=dg.EnvVar("POSTGRES_DB"),
            ),
            "earthengine_manager": EarthEngineManager(
                path_resource=path_resource, extension=".json"
            ),
            "dataarray_manager": DataArrayFileManager(
                path_resource=path_resource, extension=".nc"
            ),
        },
    )
    return dg.Definitions.merge(main_defs, extra_defs)
