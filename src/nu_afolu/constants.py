import json
from pathlib import Path

LABEL_LIST = (
    "croplands",
    "flooded",
    "forests_mangroves",
    "forests_primary",
    "forests_secondary",
    "grasslands",
    "other",
    "pastures",
    "settlements",
    "shrublands",
    "wetlands",
)
LABEL_MAP = dict(enumerate(LABEL_LIST, start=1))

TRANSITION_NODATA = 9_999

with (Path(__file__).parents[2] / "config" / "transition_dict.json").open() as f:
    TRANSITION_LABEL_MAP = {int(k): v for k, v in json.load(f).items()}


CHEN_COLLECTION_ID = "projects/sat-io/open-datasets/FUTURE-URBAN-LAND/CHEN_2020_2100"
CHEN_URBAN_VALUE = 2
CHEN_YEARS = (2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100)

SSP_NAMES = ("SSP1", "SSP2", "SSP3", "SSP4", "SSP5")
