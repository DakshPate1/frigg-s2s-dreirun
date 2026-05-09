"""Long-term pipeline configuration.

Single source of truth for:
- file paths
- zone codes
- ENTSOE EICs
- generation taxonomy (PSR types -> canonical tech buckets)
- engineering constants (heat rates, emissions factors)
- capacity / policy roadmap (DE-LU coal exit, ES PNIEC, etc.)
- regulatory parameters (Iberian gas cap)
- fuel tickers (yfinance)

The merit-order model consumes everything below.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Load .env from Frigg root ─────────────────────────────────────────────────
# The token lives in Frigg/.env (two levels up from src/).
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[2] / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # dotenv optional; user can export ENTSOE_API_TOKEN directly

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_CLEAN     = ROOT / "data" / "clean"
DATA_ALIGNED   = ROOT / "data" / "aligned"
DATA_PROCESSED = ROOT / "data" / "processed"

for d in [DATA_RAW / "entsoe", DATA_RAW / "fuel", DATA_RAW / "static",
          DATA_CLEAN, DATA_ALIGNED, DATA_PROCESSED]:
    d.mkdir(parents=True, exist_ok=True)

# ── Zones ─────────────────────────────────────────────────────────────────────
TARGET_ZONES = ["DE-LU", "ES"]

ENTSOE_ZONES = {
    "DE-LU": "10Y1001A1001A82H",
    "ES":    "10YES-REE------0",
}

ENTSOE_TOKEN = os.environ.get("ENTSOE_TOKEN") or os.environ.get("ENTSOE_API_TOKEN", "")
if not ENTSOE_TOKEN:
    raise EnvironmentError(
        "Set ENTSOE_API_TOKEN env var (or add to Frigg/.env). "
        "Get a free key at transparency.entsoe.eu."
    )

# ── Date ranges ───────────────────────────────────────────────────────────────
# History used for calibration + capacity-factor estimation.
HISTORY_START = "2018-01-01"
HISTORY_END   = "2026-05-01"   # cutoff for fully-observed months

# Forecast horizon: out to 2045 covers EU 2040 climate targets + 2038 coal exit + ES nuclear phase-out.
FORECAST_END_YEAR = 2045

# ── Generation taxonomy ───────────────────────────────────────────────────────
# Canonical technology buckets used by the merit-order model. Order = roughly
# ascending marginal cost (renewables/nuclear cheapest, oil peakers most expensive).
TECHS = [
    "wind_solar",   # zero marginal cost
    "hydro",        # near-zero MC; flexibility resource
    "nuclear",
    "lignite",
    "coal",
    "gas_ccgt",
    "oil_peaker",
    "biomass_other",
]

# ENTSOE PSR-type strings (entsoe-py canonical names) -> our buckets.
ENTSOE_PSR_TO_TECH = {
    "Wind Onshore":                       "wind_solar",
    "Wind Offshore":                      "wind_solar",
    "Solar":                              "wind_solar",
    "Hydro Run-of-river and poundage":    "hydro",
    "Hydro Water Reservoir":              "hydro",
    "Hydro Pumped Storage":               "hydro",
    "Nuclear":                            "nuclear",
    "Fossil Brown coal/Lignite":          "lignite",
    "Fossil Hard coal":                   "coal",
    "Fossil Gas":                         "gas_ccgt",
    "Fossil Oil":                         "oil_peaker",
    "Fossil Oil shale":                   "oil_peaker",
    "Biomass":                            "biomass_other",
    "Waste":                              "biomass_other",
    "Geothermal":                         "biomass_other",
    "Other renewable":                    "biomass_other",
    "Other":                              "biomass_other",
    "Marine":                             "biomass_other",
    "Fossil Peat":                        "biomass_other",
    "Fossil Coal-derived gas":            "gas_ccgt",
    "Energy storage":                     "hydro",   # batteries + pumped, zero-MC flexible
}

# ── Engineering constants per tech ────────────────────────────────────────────
# heat_rate_mwh_per_mwh: thermal energy input per MWh of electricity (= 1 / efficiency).
#   - CCGT modern: ~50% -> 2.0
#   - Hard coal:   ~40% -> 2.5
#   - Lignite:     ~35% -> 2.85
#   - Oil peaker:  ~38% -> 2.6
# emissions_t_per_mwh_fuel: tonnes CO2 per MWh of *fuel*. Multiply by heat rate
#   to get tCO2 per MWh of electricity.
#   Sources: IPCC defaults / EEA. These are stylized; document in methodology.
TECH_PARAMS = {
    "wind_solar":     {"heat_rate": 0.0,  "co2_per_mwh_fuel": 0.0,  "fuel": None},
    "hydro":          {"heat_rate": 0.0,  "co2_per_mwh_fuel": 0.0,  "fuel": None},
    "nuclear":        {"heat_rate": 0.0,  "co2_per_mwh_fuel": 0.0,  "fuel": "uranium_flat"},  # ~5 EUR/MWh flat
    "lignite":        {"heat_rate": 2.85, "co2_per_mwh_fuel": 0.36, "fuel": "lignite_flat"},  # ~3 EUR/MWh thermal flat (mine-mouth)
    "coal":           {"heat_rate": 2.50, "co2_per_mwh_fuel": 0.34, "fuel": "coal_api2"},
    "gas_ccgt":       {"heat_rate": 2.00, "co2_per_mwh_fuel": 0.20, "fuel": "gas_ttf"},
    "oil_peaker":     {"heat_rate": 2.60, "co2_per_mwh_fuel": 0.27, "fuel": "oil_brent"},
    "biomass_other":  {"heat_rate": 0.0,  "co2_per_mwh_fuel": 0.0,  "fuel": None},  # treated as price-taker baseload
}

NUCLEAR_FLAT_MC_EUR_PER_MWH  = 5.0
LIGNITE_FUEL_FLAT_EUR_PER_MWH_TH = 3.0   # mine-mouth, near-flat historically

# ── Fuel tickers (yfinance) ───────────────────────────────────────────────────
# Note: yfinance returns daily close. Forward-filled to monthly means in cleaning.
FUEL_TICKERS = {
    "gas_ttf":   "TTF=F",      # Dutch TTF natural gas futures (EUR/MWh)
    "carbon":    "KRBN",       # KraneShares carbon ETF — EUA proxy (USD; converted in features)
    "coal_api2": "MTF=F",      # API2 Rotterdam coal futures (USD/t)
    "oil_brent": "BZ=F",       # Brent crude futures (USD/bbl)
}

# Conversions (used in features.py)
COAL_USD_PER_T_TO_EUR_PER_MWH_TH = 1.0 / 8.14   # ~8.14 MWh thermal per tonne hard coal; USD≈EUR (rough)
OIL_USD_PER_BBL_TO_EUR_PER_MWH_TH = 1.0 / 1.70  # ~1.7 MWh thermal per barrel
CARBON_KRBN_TO_EUR_PER_T = 1.0                  # KRBN price ≈ EUA EUR/t (rough proxy; documented as a known limitation)

# ── Iberian gas cap mechanism (ES-only structural feature) ────────────────────
# Active 2022-06 → 2023-12. Modeled as a soft cap on gas marginal cost in ES.
# After 2023-12, no cap (free market). For long-run scenarios, a "cap returns"
# scenario can re-enable this.
IBERIAN_CAP_PERIODS = [
    {"start": "2022-06-15", "end": "2022-12-31", "cap_eur_per_mwh_gas": 40.0},
    {"start": "2023-01-01", "end": "2023-05-31", "cap_eur_per_mwh_gas": 55.0},
    {"start": "2023-06-01", "end": "2023-12-31", "cap_eur_per_mwh_gas": 65.0},
]

# ── Capacity / policy roadmap (stylized; for long-run scenarios) ──────────────
# These are *defaults* for the baseline scenario. Alternative scenarios will
# perturb these in the modeling notebook.
#
# Sources (to cite in methodology):
#   - DE Kohleausstiegsgesetz (2020): coal exit by 2038 (target 2030).
#   - DE EEG / WindSeeG: 215 GW solar + 145 GW wind by 2030.
#   - ES PNIEC 2024 update: 76 GW solar + 62 GW wind + 22 GW storage by 2030.
#   - ES nuclear phase-out plan: Almaraz 2027–28, others by 2035.
#
# Format: {zone: {tech: {year: capacity_GW}}}. Linear-interpolated between years.
CAPACITY_ROADMAP = {
    # Knots refit to ENTSOE-observed 2025 values + realistic growth pace
    # (~14 GW/yr DE wind+solar, ~6 GW/yr ES; gradual coal exit through 2038).
    # Long-run knots still anchored to EU policy targets but reached by 2035-2040.
    "DE-LU": {
        "wind_solar":   {2024: 146, 2025: 160, 2026: 174, 2028: 200,
                         2030: 230, 2035: 320, 2040: 410, 2045: 480},
        "hydro":        {2024:  15, 2045:  15},
        "nuclear":      {2024:   0, 2045:   0},
        "lignite":      {2024:  18, 2025: 15, 2028: 12, 2030: 9,
                         2035:  3, 2038:   0, 2045:   0},
        "coal":         {2024:  18, 2025: 16, 2028: 12, 2030: 8,
                         2035:  3, 2038:   0, 2045:   0},
        "gas_ccgt":     {2024:  38, 2025: 39, 2028: 40, 2030: 40,
                         2035: 38, 2040: 32, 2045: 25},
        "oil_peaker":   {2024:   4, 2025:  7, 2030:  6, 2045:  3},
        "biomass_other":{2024:  13, 2030: 13, 2045:  13},
    },
    "ES": {
        "wind_solar":   {2024:  54, 2025:  60, 2026:  68, 2028:  82,
                         2030:  98, 2035: 138, 2040: 178, 2045: 220},
        "hydro":        {2024:  20, 2045:  20},
        "nuclear":      {2024:   7, 2026:  7, 2028:   6, 2030:   6,
                         2032:  3, 2035:   0, 2045:   0},
        "lignite":      {2024:   0, 2045:   0},
        "coal":         {2024:   3, 2025:  2, 2027:   1, 2030:   0, 2045:   0},
        "gas_ccgt":     {2024:  30, 2030: 28, 2040: 22, 2045:  18},
        "oil_peaker":   {2024:   1, 2045:   1},
        "biomass_other":{2024:   2, 2045:   3},
    },
}

# ── Demand projection (annual TWh, baseline scenario) ─────────────────────────
# DE-LU: ~500 TWh today, electrification → ~700 TWh by 2045.
# ES: ~240 TWh today, electrification → ~370 TWh by 2045.
DEMAND_ROADMAP_TWH = {
    # Anchored at ENTSOE-observed 2024-2025; modest electrification through 2045.
    "DE-LU": {2024: 470, 2025: 471, 2026: 478, 2030: 520, 2040: 620, 2045: 680},
    "ES":    {2024: 232, 2025: 238, 2026: 245, 2030: 270, 2040: 320, 2045: 360},
}

# ── Monte Carlo distributions (long-run scenario uncertainty) ────────────────
# These define the prior over structural drivers used for the fan chart in the
# modelling notebook. Each per-shock stdev is scaled by sqrt(years-ahead) at
# sampling time, so 2026 has a tight cone and 2045 has a wide cone.
#
# Distributions are defined as (mean, base_stdev). Lognormal is used for
# multiplicative parameters (fuel/carbon/RES); normal for demand.
MC_DISTRIBUTIONS = {
    # Fossil-fuel shocks: a single shared factor drives gas + coal + oil with
    # tech-specific noise on top.  This keeps "Russia 2.0" and "gas glut"
    # scenarios as natural draws instead of hand-coded.
    "fossil_shock":     {"dist": "lognormal", "mean": 1.0, "sigma_base": 0.30},
    "gas_idiosync":     {"dist": "lognormal", "mean": 1.0, "sigma_base": 0.20},
    "coal_idiosync":    {"dist": "lognormal", "mean": 1.0, "sigma_base": 0.15},
    "oil_idiosync":     {"dist": "lognormal", "mean": 1.0, "sigma_base": 0.20},
    # Carbon: separate factor (more policy-driven, less correlated with fossils)
    "carbon":           {"dist": "lognormal", "mean": 1.0, "sigma_base": 0.20},
    # Capacity buildout pace
    "res_buildout":     {"dist": "lognormal", "mean": 1.0, "sigma_base": 0.10},
    # Demand growth (electrification path)
    "demand":           {"dist": "lognormal", "mean": 1.0, "sigma_base": 0.05},
    # ES Iberian cap reactivation: Bernoulli per year
    "iberian_cap_prob": 0.05,   # 5% prob per year that the cap is re-enabled
    "iberian_cap_level":{"mean": 70.0, "sigma": 15.0},  # cap level when active
}

MC_N_DRAWS    = 200
MC_RANDOM_SEED = 42

# ── Forward fuel curve (mean-reversion model) ────────────────────────────────
# At anchor month t, expected fuel price h months ahead =
#     theta + (S_t - theta) * rho^h
# where theta is the long-run mean, rho is monthly mean-reversion speed.
# Source: Schwartz (1997) one-factor commodity model.
# Parameters chosen to match observed TTF half-life ~6 months in 2018-2024.
FORWARD_CURVE = {
    "gas_ttf":   {"theta_eur_per_mwh": 30.0, "rho_monthly": 0.88},
    "coal_api2": {"theta_eur_per_mwh": 12.0, "rho_monthly": 0.92},   # in EUR/MWh thermal
    "oil_brent": {"theta_eur_per_mwh": 35.0, "rho_monthly": 0.95},   # ditto
    "carbon":    {"theta_eur_per_t":   60.0, "rho_monthly": 0.95},   # EU ETS slow mean-reversion
}

# ── GPR (Geopolitical Risk index, Iacoviello) regime detector ────────────────
# Source: https://www.matteoiacoviello.com/gpr.htm
# Index is normalized to 1985-2019 mean = 100. We use the historical 80th
# percentile as the "crisis" threshold; above it, the MC bands widen.
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
GPR_CRISIS_PERCENTILE = 0.80
GPR_CRISIS_SIGMA_MULTIPLIER = 1.8   # bands multiplied by this when GPR above threshold

# ── Capacity factors (annual average; used for energy-balance checks) ─────────
# Stylized values — refined later from ENTSOE actual generation / installed capacity.
CAPACITY_FACTORS = {
    "wind_solar":    {"DE-LU": 0.20, "ES": 0.27},   # ES has more solar share -> higher CF
    "hydro":         {"DE-LU": 0.30, "ES": 0.25},
    "nuclear":       {"DE-LU": 0.85, "ES": 0.85},
    "lignite":       {"DE-LU": 0.55, "ES": 0.50},
    "coal":          {"DE-LU": 0.40, "ES": 0.30},
    "gas_ccgt":      {"DE-LU": 0.35, "ES": 0.30},
    "oil_peaker":    {"DE-LU": 0.05, "ES": 0.05},
    "biomass_other": {"DE-LU": 0.55, "ES": 0.55},
}
