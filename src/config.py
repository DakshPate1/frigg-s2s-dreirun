"""Pipeline configuration: date ranges, zones, locations, paths."""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

# ── Directories ───────────────────────────────────────────────────────────────
DATA_RAW       = ROOT / "data" / "raw"
DATA_CLEAN     = ROOT / "data" / "clean"
DATA_ALIGNED   = ROOT / "data" / "aligned"
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

# Neighboring zone EICs
# DE-LU: FR, NL, CH, DK1 are the meaningful flow partners (see methodology)
# AT, PL, BE, CZ included for completeness but lower weight in model
# ES: FR is the only significant interconnection; PT included for Iberian flows
ENTSOE_NEIGHBORS = {
    "DE-LU": [
        "10YFR-RTE------C",   # France        — largest flow partner, nuclear signal
        "10YNL----------L",   # Netherlands   — gas hub, north wind correlation
        "10YCH-SWISSGRIDZ",   # Switzerland   — alpine hydro, arbitrage dampening
        "10YDK-1--------W",   # Denmark West  — Nordic wind export signal
        "10YAT-APG------L",   # Austria       — secondary, south Germany flows
        "10YPL-AREA-----S",   # Poland        — weak but included
        "10YBE----------2",   # Belgium       — weak but included
        "10YCZ-CEPS-----N",   # Czech Republic — weak but included
    ],
    "ES": [
        "10YFR-RTE------C",   # France   — only significant interconnection (~3 GW)
        "10YPT-REN------W",   # Portugal — Iberian internal flows
    ],
}

# ── Weather station locations ─────────────────────────────────────────────────
#
# Locations are chosen by physical role in price formation, NOT proximity to
# grid infrastructure nodes. Three categories per zone:
#
#   WIND   — capacity-weighted generation centers (where turbines actually are)
#   SOLAR  — capacity-weighted generation centers (highest irradiance / installed PV)
#   DEMAND — population-weighted demand centers  (where electricity is consumed)
#
# Additionally for DE-LU:
#   DK_WIND  — Danish North Sea offshore proxy (exports directly into north Germany)
#   CH_HYDRO — Swiss alpine precipitation proxy (snowmelt drives hydro availability)
#
# Weights reflect approximate installed capacity (wind/solar) or population (demand).
# Open-Meteo variables: wind_speed_100m, shortwave_radiation, temperature_2m, precipitation

WEATHER_LOCATIONS = {

    # ── DE-LU ────────────────────────────────────────────────────────────────
    # Wind: north Germany dominates (~70% of installed capacity)
    "DE_wind": [
        {"label": "Schleswig",  "latitude": 54.52, "longitude":  9.55, "weight": 0.35},
        {"label": "Emden",      "latitude": 53.37, "longitude":  7.21, "weight": 0.30},
        {"label": "Rostock",    "latitude": 54.09, "longitude": 12.10, "weight": 0.15},
        {"label": "Helgoland",  "latitude": 54.18, "longitude":  7.89, "weight": 0.12},  # North Sea offshore proxy
        {"label": "Ruegen",     "latitude": 54.43, "longitude": 13.38, "weight": 0.08},  # Baltic offshore proxy
    ],

    # Solar: south Germany dominates installed PV capacity
    "DE_solar": [
        {"label": "Munich",     "latitude": 48.14, "longitude": 11.58, "weight": 0.40},
        {"label": "Stuttgart",  "latitude": 48.78, "longitude":  9.18, "weight": 0.35},
        {"label": "Leipzig",    "latitude": 51.34, "longitude": 12.38, "weight": 0.25},
    ],

    # Demand: population-weighted major consumption centers
    "DE_demand": [
        {"label": "Berlin",     "latitude": 52.52, "longitude": 13.40, "weight": 0.40},
        {"label": "Frankfurt",  "latitude": 50.11, "longitude":  8.68, "weight": 0.35},
        {"label": "Cologne",    "latitude": 50.94, "longitude":  6.96, "weight": 0.25},
    ],

    # Danish wind: North Sea offshore exports directly into north Germany
    "DK_wind": [
        {"label": "Esbjerg",    "latitude": 55.47, "longitude":  8.45, "weight": 0.60},
        {"label": "Copenhagen", "latitude": 55.68, "longitude": 12.57, "weight": 0.40},
    ],

    # Swiss hydro proxy: alpine precipitation drives reservoir levels
    # May snowmelt is a key seasonal signal for CH→DE-LU power exports
    "CH_hydro": [
        {"label": "Lucerne",    "latitude": 47.05, "longitude":  8.31, "weight": 0.50},
        {"label": "Davos",      "latitude": 46.80, "longitude":  9.84, "weight": 0.50},
    ],

    # ── ES ───────────────────────────────────────────────────────────────────
    # Wind: main corridors — Castile, Ebro valley, Andalusia coast
    "ES_wind": [
        {"label": "Burgos",      "latitude": 42.34, "longitude": -3.70, "weight": 0.40},
        {"label": "Zaragoza",    "latitude": 41.65, "longitude": -0.89, "weight": 0.35},
        {"label": "Cadiz",       "latitude": 36.53, "longitude": -6.30, "weight": 0.25},
    ],

    # Solar: highest irradiance zones — Andalusia, Murcia, Castile-La Mancha
    "ES_solar": [
        {"label": "Seville",     "latitude": 37.39, "longitude": -5.99, "weight": 0.40},
        {"label": "Murcia",      "latitude": 37.98, "longitude": -1.13, "weight": 0.35},
        {"label": "CiudadReal",  "latitude": 38.99, "longitude": -3.92, "weight": 0.25},
    ],

    # Demand: population-weighted major cities
    "ES_demand": [
        {"label": "Madrid",      "latitude": 40.42, "longitude": -3.70, "weight": 0.40},
        {"label": "Barcelona",   "latitude": 41.39, "longitude":  2.15, "weight": 0.35},
        {"label": "Valencia",    "latitude": 39.47, "longitude": -0.38, "weight": 0.25},
    ],

    # Hydro proxy: Duero and Ebro basin precipitation → reservoir levels
    # ES hydro is a major supply-side swing factor unlike DE-LU
    "ES_hydro": [
        {"label": "Zamora",      "latitude": 41.50, "longitude": -5.74, "weight": 0.50},
        {"label": "Huesca",      "latitude": 42.14, "longitude": -0.41, "weight": 0.50},
    ],
}

# ── Open-Meteo variable mapping by location group ─────────────────────────────
OPENMETEO_VARIABLES = {
    "DE_wind":   ["wind_speed_100m"],
    "DE_solar":  ["shortwave_radiation"],
    "DE_demand": ["temperature_2m"],
    "DK_wind":   ["wind_speed_100m"],
    "CH_hydro":  ["precipitation", "temperature_2m"],
    "ES_wind":   ["wind_speed_100m"],
    "ES_solar":  ["shortwave_radiation"],
    "ES_demand": ["temperature_2m"],
    "ES_hydro":  ["precipitation"],
}

# ── Aggregation helper (use in feature engineering) ───────────────────────────
def weighted_average(dfs_and_weights):
    """
    Compute capacity/population-weighted average across location DataFrames.

    Usage:
        agg = weighted_average([
            (df_schleswig, 0.35),
            (df_emden,     0.30),
            ...
        ])
    """
    total_weight = sum(w for _, w in dfs_and_weights)
    result = sum(df * w for df, w in dfs_and_weights)
    return result / total_weight

# ── Date range for training data ───────────────────────────────────────────────
# 3 years of clean post-crisis data. Avoids 2021-2022 energy crisis distortion.
# A binary crisis_period flag is added in feature engineering for any
# pre-2023 data if the range is extended for sensitivity analysis.
TRAIN_START = "2023-05-01"
TRAIN_END   = "2026-05-07"   # inclusive; update before each run

# Crisis period flag boundaries (used in feature engineering)
CRISIS_START = "2021-08-01"
CRISIS_END   = "2022-12-31"

# ── Evaluation window ──────────────────────────────────────────────────────────
# Monday 11 May 2026 02:00 CEST → Tuesday 12 May 2026 01:00 CEST (inclusive)
# CEST = UTC+2 in May
# = 24 hourly slots (00:00 UTC → 23:00 UTC on May 11)
EVAL_START = "2026-05-11T02:00:00+02:00"
EVAL_END   = "2026-05-12T01:00:00+02:00"
EVAL_HOURS = 24

# ── Generation type mapping (energy-charts → canonical names) ──────────────────
WIND_TYPES  = ["Wind offshore", "Wind onshore"]
SOLAR_TYPES = ["Solar"]
HYDRO_TYPES = ["Hydro Run-of-River", "Hydro water reservoir", "Hydro pumped storage"]
LOAD_TYPE   = "Load"

# ── ENTSOE generation PSR type → canonical names ──────────────────────────────
ENTSOE_WIND_TYPES  = ["Wind Offshore", "Wind Onshore"]
ENTSOE_SOLAR_TYPES = ["Solar"]
ENTSOE_HYDRO_TYPES = ["Hydro Pumped Storage", "Hydro Run-of-river and poundage", "Hydro Water Reservoir"]
ENTSOE_NUCLEAR_TYPES = ["Nuclear"]  # Critical for ES — ~7 GW active capacity

# ── Scoring ───────────────────────────────────────────────────────────────────
PINBALL_Q = 0.45   # Competition metric: overestimation penalised 1.22x more
                   # Train p50 model at q=0.45, NOT q=0.50
