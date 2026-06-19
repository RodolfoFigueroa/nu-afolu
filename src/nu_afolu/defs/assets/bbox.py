import ee
import geopandas as gpd
import shapely
from dagster_components.partitions import zone_partitions
from dagster_components.resources import PostgresResource

import dagster as dg
from nu_afolu.utils import get_largest_geometry


@dg.asset(
    key=["bbox", "shapely"],
    partitions_def=zone_partitions,
    io_manager_key="geodataframe_manager",
    group_name="bbox",
    tags={"partitions": "zone"},
)
def zone_bbox_shapely(
    context: dg.AssetExecutionContext, postgres_resource: PostgresResource
) -> gpd.GeoDataFrame:
    with postgres_resource.connect() as conn:
        gdf = gpd.read_postgis(
            """
            SELECT census_2020_ageb.geometry from census_2020_ageb
            INNER JOIN census_2020_mun
                ON census_2020_ageb.cve_mun = census_2020_mun.cvegeo
            WHERE census_2020_mun.cve_met = %(zone)s
            """,
            conn,
            params={"zone": context.partition_key},
            geom_col="geometry",
        )

    geom = gdf["geometry"].union_all()

    if isinstance(geom, shapely.MultiPolygon):
        geom = get_largest_geometry(geom)

    return (
        gpd.GeoDataFrame(geometry=[geom], crs=gdf.crs)
        .assign(geometry=lambda df: df["geometry"].simplify(1000).buffer(10_000))
        .to_crs("EPSG:4326")
    )


@dg.asset(
    key=["bbox", "ee"],
    ins={"df_bbox": dg.AssetIn(["bbox", "shapely"])},
    partitions_def=zone_partitions,
    io_manager_key="earthengine_manager",
    group_name="bbox",
)
def zone_bbox_ee(df_bbox: gpd.GeoDataFrame) -> ee.geometry.Geometry:
    bbox_shapely = df_bbox["geometry"].item()

    if not isinstance(bbox_shapely, shapely.Polygon):
        err = f"Expected Polygon, got {type(bbox_shapely)}"
        raise TypeError(err)

    return ee.geometry.Geometry.Polygon(
        list(zip(*bbox_shapely.exterior.coords.xy, strict=False)),
    )
