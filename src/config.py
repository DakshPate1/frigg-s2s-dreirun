"""Pipeline configuration: date ranges, zones, locations, paths."""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

# ── Directories ───────────────────────────────────────────────────────────────
DATA_RAW      = ROOT / "data" / "raw"
DATA_CLEAN    = ROOT / "data" / "clean"
DATA_ALIGNED  = ROOT / "data" / "aligned"
DATA_PROCESSED = ROOT / "data" / "processed"

for d in [DATA_RAW / "energycharts", DATA_RAW / "openmeteo", DATA_RAW / "fuel",
          DATA_RAW / "entsoe",
          DATA_CLEAN, DATA_ALIGNED, DATA_PROCESSED]:
    d.mkdir(parents=True, exist_ok=True)

# ── Zones ─────────────────────────────────────────────────────────────────────
ZONES = ["DE-LU", "ES"]

# Energy-Charts bidding zone codes (fallback source)
EC_BZN     = {"DE-LU": "DE-LU", "ES": "ES"}
EC_COUNTRY = {"DE-LU": "de",    "ES": "es"}

# ── ENTSOE Transparency Platform ──────────────────────────────────────────────
ENTSOE_TOKEN = os.environ.get("ENTSOE_TOKEN", "")
if not ENTSOE_TOKEN:
    raise EnvironmentError("Set ENTSOE_TOKEN env var (or add to .env). Get a free key at transparency.entsoe.eu.")

# Bidding-zone EIC codes
ENTSOE_ZONES = {
    "DE-LU": "10Y1001A1001A82H",
    "ES":    "10YES-REE------0",
}

# Neighboring zone EICs used for cross-border flow aggregation
ENTSOE_NEIGHBORS = {
    "DE-LU": [
        "10YFR-RTE------C",   # France
        "10YNL----------L",   # Netherlands
        "10YAT-APG------L",   # Austria
        "10YCH-SWISSGRIDZ",   # Switzerland
        "10YCZ-CEPS-----N",   # Czech Republic
        "10YPL-AREA-----S",   # Poland
        "10YDK-1--------W",   # Denmark West (DK1)
        "10YBE----------2",   # Belgium
    ],
    "ES": [
        "10YFR-RTE------C",   # France
        "10YPT-REN------W",   # Portugal
    ],
}

# ── Representative weather station locations ───────────────────────────────────
# DE-LU: Frankfurt (central Germany, industrial/population hub)
# ES:    Madrid (largest city, central, captures cooling-degree dynamics)
WEATHER_LOCATIONS = {
    "DE-LU": {"latitude": 50.11, "longitude": 8.68,  "label": "Frankfurt"},
    "ES":    {"latitude": 40.42, "longitude": -3.70,  "label": "Madrid"},
}

# ── Date range for training data ───────────────────────────────────────────────
TRAIN_START = "2021-01-01"
TRAIN_END   = "2026-05-06"   # inclusive; update before each run

# ── Evaluation window ──────────────────────────────────────────────────────────
# 2026-05-08 17:00 UTC → 2026-05-09 22:00 UTC  (30 hourly slots inclusive)
EVAL_START = "2026-05-08T17:00:00+00:00"
EVAL_END   = "2026-05-09T22:00:00+00:00"

# ── Generation type mapping (energy-charts → canonical names) ──────────────────
WIND_TYPES  = ["Wind offshore", "Wind onshore"]
SOLAR_TYPES = ["Solar"]
HYDRO_TYPES = ["Hydro Run-of-River", "Hydro water reservoir", "Hydro pumped storage"]
LOAD_TYPE   = "Load"

# ── ENTSOE generation PSR type → canonical names ─────────────────────────────
ENTSOE_WIND_TYPES  = ["Wind Offshore", "Wind Onshore"]
ENTSOE_SOLAR_TYPES = ["Solar"]
ENTSOE_HYDRO_TYPES = ["Hydro Pumped Storage", "Hydro Run-of-river and poundage", "Hydro Water Reservoir"]
