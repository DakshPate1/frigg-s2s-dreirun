"""Pipeline configuration: date ranges, zones, locations, paths."""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

# ── Directories ───────────────────────────────────────────────────────────────
DATA_RAW      = ROOT / "data" / "raw"
DATA_CLEAN    = ROOT / "data" / "clean"
DATA_ALIGNED  = ROOT / "data" / "aligned"
DATA_PROCESSED = ROOT / "data" / "processed"

for d in [DATA_RAW / "energycharts", DATA_RAW / "openmeteo", DATA_RAW / "fuel",
          DATA_CLEAN, DATA_ALIGNED, DATA_PROCESSED]:
    d.mkdir(parents=True, exist_ok=True)

# ── Zones ─────────────────────────────────────────────────────────────────────
ZONES = ["DE-LU", "ES"]

# Energy-Charts bidding zone codes
EC_BZN = {"DE-LU": "DE-LU", "ES": "ES"}
EC_COUNTRY = {"DE-LU": "de", "ES": "es"}

# ── Representative weather station locations ───────────────────────────────────
# DE-LU: Frankfurt (central Germany, industrial/population hub)
# ES:    Madrid (largest city, central, captures cooling-degree dynamics)
WEATHER_LOCATIONS = {
    "DE-LU": {"latitude": 50.11, "longitude": 8.68, "label": "Frankfurt"},
    "ES":    {"latitude": 40.42, "longitude": -3.70, "label": "Madrid"},
}

# ── Date range for training data ───────────────────────────────────────────────
TRAIN_START = "2021-01-01"
TRAIN_END   = "2026-05-05"   # today

# ── Evaluation window ──────────────────────────────────────────────────────────
# 2026-05-08 17:00 UTC → 2026-05-09 22:00 UTC  (30 hourly slots inclusive)
EVAL_START = "2026-05-08T17:00:00+00:00"
EVAL_END   = "2026-05-09T22:00:00+00:00"

# ── Generation type mapping (energy-charts → canonical names) ──────────────────
# Keys are substrings; order matters (first match wins per row)
WIND_TYPES  = ["Wind offshore", "Wind onshore"]
SOLAR_TYPES = ["Solar"]
HYDRO_TYPES = ["Hydro Run-of-River", "Hydro water reservoir", "Hydro pumped storage"]
LOAD_TYPE   = "Load"
