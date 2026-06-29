import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def import_dependencies():
    import os  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import ee  # noqa: PLC0415
    import marimo as mo  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    from dagster_components.partitions import zone_partitions  # noqa: PLC0415

    from nu_afolu.constants import (  # noqa: PLC0415
        CHEN_COLLECTION_ID,
        CHEN_URBAN_VALUE,
        CHEN_YEARS,
        LABEL_LIST,
        SSP_NAMES,
    )

    return (
        CHEN_COLLECTION_ID,
        CHEN_URBAN_VALUE,
        CHEN_YEARS,
        LABEL_LIST,
        Path,
        SSP_NAMES,
        ee,
        mo,
        os,
        pd,
        zone_partitions,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 00 Artifact Inventory

    This notebook checks whether the local workspace has the upstream artifacts and
    project constants needed before Chen SSP feasibility analysis begins. It is a
    readiness inventory only: it does not compare Chen urban area with observed GLC
    `settlements`, and it does not create pseudo-transition artifacts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal, Inputs, And Assumptions

    The first stage answers a simple question: is the workspace ready for the Chen
    SSP notebook sequence?

    The required historical artifact families are:

    - `bbox/ee/{ZONE}.json`
    - `area_raster/{ZONE}.json`
    - `transition_raster/{ZONE}.json`
    - `area_table/{ZONE}.parquet`
    - `transition_table/{ZONE}.nc`

    The inventory uses the canonical Dagster `zone_partitions` list when available
    and also reports any zones discovered directly from files under `OUT_PATH`.
    Missing artifacts are treated as readiness findings, not errors, so partial
    pipeline runs remain inspectable.
    """)
    return


@app.cell
def resolve_out_path(Path, os, pd):
    PROJECT_ROOT = Path.cwd()
    OUT_PATH_KEY = "OUT_PATH"

    def _read_dotenv_value(path: Path, key: str) -> str | None:
        if not path.exists():
            return None

        for _line in path.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _name, _value = _line.split("=", 1)
            if _name.strip() == key:
                return _value.strip().strip('"').strip("'")

        return None

    _out_path_raw = os.environ.get(OUT_PATH_KEY)
    OUT_PATH_SOURCE = "environment variable"
    if not _out_path_raw:
        _out_path_raw = _read_dotenv_value(PROJECT_ROOT / ".env", OUT_PATH_KEY)
        OUT_PATH_SOURCE = ".env file" if _out_path_raw else "missing"

    OUT_PATH = Path(_out_path_raw).expanduser() if _out_path_raw else None
    OUT_PATH_EXISTS = bool(OUT_PATH and OUT_PATH.exists())

    out_path_summary = pd.DataFrame(
        [
            {
                "setting": OUT_PATH_KEY,
                "source": OUT_PATH_SOURCE,
                "value": str(OUT_PATH) if OUT_PATH else "not configured",
                "exists": OUT_PATH_EXISTS,
            }
        ]
    )

    out_path_summary
    return OUT_PATH, OUT_PATH_EXISTS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Project Constants

    These constants define the AFOLU class labels and Chen projection contract used
    by later notebooks. This stage exposes them so a reader can verify that the
    notebook sequence is using the same local code contract as the Dagster assets.
    """)
    return


@app.cell
def summarize_project_constants(
    CHEN_COLLECTION_ID,
    CHEN_URBAN_VALUE,
    CHEN_YEARS,
    LABEL_LIST,
    SSP_NAMES,
    pd,
):
    project_constants_summary = pd.DataFrame(
        [
            {"name": "CHEN_COLLECTION_ID", "value": CHEN_COLLECTION_ID},
            {"name": "CHEN_URBAN_VALUE", "value": CHEN_URBAN_VALUE},
            {
                "name": "CHEN_YEARS",
                "value": ", ".join(str(year) for year in CHEN_YEARS),
            },
            {"name": "SSP_NAMES", "value": ", ".join(SSP_NAMES)},
            {"name": "AFOLU label count", "value": len(LABEL_LIST)},
        ]
    )

    project_constants_summary
    return


@app.cell
def show_label_list(LABEL_LIST, pd):
    label_table = pd.DataFrame(
        {
            "raster_id": range(1, len(LABEL_LIST) + 1),
            "label": LABEL_LIST,
        }
    )

    label_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Artifact Discovery

    The table below checks the expected artifact directories under `OUT_PATH`. A
    missing directory is not fatal for this notebook, but it means later stages will
    have no artifacts of that type to load unless the upstream Dagster assets are
    materialized first.
    """)
    return


@app.cell
def summarize_artifact_directories(OUT_PATH, Path, pd):
    ARTIFACT_SPECS = {
        "bbox_ee": {"relative_dir": Path("bbox") / "ee", "extension": ".json"},
        "area_raster": {"relative_dir": Path("area_raster"), "extension": ".json"},
        "transition_raster": {
            "relative_dir": Path("transition_raster"),
            "extension": ".json",
        },
        "area_table": {"relative_dir": Path("area_table"), "extension": ".parquet"},
        "transition_table": {
            "relative_dir": Path("transition_table"),
            "extension": ".nc",
        },
    }

    _artifact_directory_rows = []
    for _artifact_name, _spec in ARTIFACT_SPECS.items():
        _artifact_dir = OUT_PATH / _spec["relative_dir"] if OUT_PATH else None
        _file_count = (
            len(list(_artifact_dir.glob(f"*{_spec['extension']}")))
            if _artifact_dir and _artifact_dir.exists()
            else 0
        )
        _artifact_directory_rows.append(
            {
                "artifact": _artifact_name,
                "relative_dir": str(_spec["relative_dir"]),
                "extension": _spec["extension"],
                "directory_exists": bool(_artifact_dir and _artifact_dir.exists()),
                "file_count": _file_count,
            }
        )

    artifact_directory_summary = pd.DataFrame(_artifact_directory_rows)

    artifact_directory_summary
    return (ARTIFACT_SPECS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zone Availability Summary

    A zone is eligible for the next notebook only when all required artifact families
    are present. This is stricter than the later loading helper because an inventory
    should show partial failures instead of silently skipping them.
    """)
    return


@app.cell
def build_zone_inventory(ARTIFACT_SPECS, OUT_PATH, pd, zone_partitions):
    partition_zone_names = tuple(zone_partitions.get_partition_keys())

    artifact_zone_sets: dict[str, set[str]] = {}
    for _artifact_name, _spec in ARTIFACT_SPECS.items():
        _artifact_dir = OUT_PATH / _spec["relative_dir"] if OUT_PATH else None
        if _artifact_dir and _artifact_dir.exists():
            artifact_zone_sets[_artifact_name] = {
                _path.stem for _path in _artifact_dir.glob(f"*{_spec['extension']}")
            }
        else:
            artifact_zone_sets[_artifact_name] = set()

    _discovered_zone_set: set[str] = set()
    for _zone_set in artifact_zone_sets.values():
        _discovered_zone_set.update(_zone_set)

    discovered_zone_names = tuple(sorted(_discovered_zone_set))
    inventory_zone_names = tuple(
        sorted(set(partition_zone_names).union(discovered_zone_names))
    )

    _zone_inventory_rows = []
    for _zone_name in inventory_zone_names:
        _row = {
            "zone": _zone_name,
            "in_partition_def": _zone_name in partition_zone_names,
        }
        for _artifact_name in ARTIFACT_SPECS:
            _row[_artifact_name] = _zone_name in artifact_zone_sets[_artifact_name]
        _row["complete_artifact_set"] = all(
            _row[_artifact_name] for _artifact_name in ARTIFACT_SPECS
        )
        _zone_inventory_rows.append(_row)

    artifact_availability = pd.DataFrame(_zone_inventory_rows)
    if not artifact_availability.empty:
        artifact_availability = artifact_availability.sort_values("zone").reset_index(
            drop=True
        )

    eligible_zone_names = tuple(
        artifact_availability.loc[
            artifact_availability["complete_artifact_set"],
            "zone",
        ]
    )

    zone_availability_summary = pd.DataFrame(
        [
            {"metric": "canonical partition zones", "value": len(partition_zone_names)},
            {
                "metric": "zones discovered from artifacts",
                "value": len(discovered_zone_names),
            },
            {"metric": "zones in inventory table", "value": len(inventory_zone_names)},
            {
                "metric": "zones with complete artifact sets",
                "value": len(eligible_zone_names),
            },
        ]
    )

    zone_availability_summary
    return artifact_availability, eligible_zone_names


@app.cell
def show_artifact_availability(artifact_availability):
    artifact_availability
    return


@app.cell
def show_eligible_zones(eligible_zone_names, pd):
    eligible_zones_table = pd.DataFrame({"zone": eligible_zone_names})

    eligible_zones_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chen Source And Earth Engine Readiness

    The Chen collection is an Earth Engine source, so later notebooks need Earth
    Engine credentials and metadata access. This check is intentionally light: it
    initializes Earth Engine and reads collection metadata, but it does not reduce
    pixels over any zone.
    """)
    return


@app.cell
def check_earth_engine_readiness(CHEN_COLLECTION_ID, SSP_NAMES, ee, pd):
    def _short_error(exc: Exception) -> str:
        message = str(exc).replace("\n", " ")
        return message[:500] + ("..." if len(message) > 500 else "")

    def _check_earth_engine_readiness() -> dict[str, object]:
        try:
            ee.Initialize()
        except Exception as exc:  # noqa: BLE001
            return {
                "earth_engine_initialized": False,
                "chen_collection_metadata_readable": False,
                "chen_image_count": None,
                "first_image_bands": None,
                "expected_ssp_bands_present": False,
                "message": _short_error(exc),
            }

        try:
            _collection = ee.ImageCollection(CHEN_COLLECTION_ID)
            _image_count = _collection.size().getInfo()
            _first_image_bands = _collection.first().bandNames().getInfo()
            return {
                "earth_engine_initialized": True,
                "chen_collection_metadata_readable": True,
                "chen_image_count": _image_count,
                "first_image_bands": ", ".join(_first_image_bands),
                "expected_ssp_bands_present": set(SSP_NAMES).issubset(
                    _first_image_bands
                ),
                "message": "Earth Engine initialized and Chen collection metadata is readable.",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "earth_engine_initialized": True,
                "chen_collection_metadata_readable": False,
                "chen_image_count": None,
                "first_image_bands": None,
                "expected_ssp_bands_present": False,
                "message": _short_error(exc),
            }

    earth_engine_readiness = pd.DataFrame([_check_earth_engine_readiness()])

    earth_engine_readiness
    return (earth_engine_readiness,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Readiness Conclusion

    This final statement combines the path, artifact inventory, and Earth Engine
    metadata checks into a practical go/no-go signal for the next notebook. A ready
    workspace means at least one zone has every required historical artifact and the
    Chen source contract is visible.
    """)
    return


@app.cell
def build_readiness_conclusion(
    OUT_PATH,
    OUT_PATH_EXISTS,
    earth_engine_readiness,
    eligible_zone_names,
    mo,
):
    _ee_row = earth_engine_readiness.iloc[0].to_dict()
    if OUT_PATH is None:
        _readiness_status = "Not ready"
        _readiness_reason = "`OUT_PATH` is not configured."
    elif not OUT_PATH_EXISTS:
        _readiness_status = "Not ready"
        _readiness_reason = "`OUT_PATH` is configured but the directory does not exist."
    elif not eligible_zone_names:
        _readiness_status = "Not ready"
        _readiness_reason = (
            "No zone has a complete set of required historical artifacts."
        )
    elif not _ee_row["earth_engine_initialized"]:
        _readiness_status = "Partially ready"
        _readiness_reason = (
            "Historical artifacts are present, but Earth Engine did not initialize."
        )
    elif not _ee_row["chen_collection_metadata_readable"]:
        _readiness_status = "Partially ready"
        _readiness_reason = "Historical artifacts are present, but Chen collection metadata was not readable."
    elif not _ee_row["expected_ssp_bands_present"]:
        _readiness_status = "Partially ready"
        _readiness_reason = "Chen metadata was readable, but the expected SSP bands were not all present."
    else:
        _readiness_status = "Ready"
        _readiness_reason = "At least one complete historical zone is available and the Chen collection metadata is readable."

    readiness_conclusion = mo.md(
        f"""
    ### {_readiness_status}

    {_readiness_reason}

    - Complete zones available for `01_historical_contract_checks.py`: `{len(eligible_zone_names)}`
    - First complete zones: `{", ".join(eligible_zone_names[:10]) if eligible_zone_names else "none"}`
    - Historical artifact contracts were inventoried only; they were not validated here.
    - Chen 2020 compatibility was not evaluated here and remains the responsibility of `02_chen_2020_compatibility.py`.
    """
    )

    readiness_conclusion
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Limitations And Next Step

    This notebook does not validate schemas, reconcile area and transition totals,
    or compare Chen urban area against observed settlements. It also does not write
    any durable analysis artifact beyond the notebook itself.

    Next, `01_historical_contract_checks.py` should load only the complete zones
    listed above and validate the `area_table` and `transition_table` contracts in
    detail.
    """)
    return


if __name__ == "__main__":
    app.run()
