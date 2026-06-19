import math

import shapely


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return math.nan
    return float(numerator / denominator)


def year_to_band_name(year: int | str) -> str:
    if isinstance(year, str):
        year = int(year)
    return f"b{year - 1999}"


def get_largest_geometry(
    multipoly: shapely.geometry.MultiPolygon,
) -> shapely.geometry.Polygon:
    max_area = 0
    largest_geom = None
    for geom in multipoly.geoms:
        if geom.area > max_area:
            max_area = geom.area
            largest_geom = geom

    if largest_geom is None:
        err = "No valid geometry found"
        raise ValueError(err)
    return largest_geom
