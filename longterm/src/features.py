"""
Stage 4 — Feature engineering.

Produces data/processed/final_dataset.parquet, which is what the merit-order
model and the methodology notebook will consume.

Key derived features:
  - Per-tech monthly marginal cost (EUR/MWh), built from fuel + carbon prices
    and engineering constants (heat rate, emissions factor) in config.TECH_PARAMS.
  - Iberian gas-cap flag (ES only) — historical periods from config.IBERIAN_CAP_PERIODS.
  - Annual structural panel forward-extended to FORECAST_END_YEAR using the
    config.CAPACITY_ROADMAP / DEMAND_ROADMAP_TWH baselines (linear interpolation
    between knot years; constant-extrapolated outside knot range).

Output schemas:
  data/processed/marginal_costs_monthly.parquet
      index:   (month, zone, tech)
      columns: mc_eur_per_mwh, fuel_component, carbon_component, capped
  data/processed/structural_extended.parquet
      index:   (year, zone)
      columns: capacity_<tech>_gw (forecast-extended), demand_twh
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from config import (
    DATA_ALIGNED, DATA_PROCESSED,
    TARGET_ZONES, TECHS,
    TECH_PARAMS,
    NUCLEAR_FLAT_MC_EUR_PER_MWH, LIGNITE_FUEL_FLAT_EUR_PER_MWH_TH,
    COAL_USD_PER_T_TO_EUR_PER_MWH_TH, OIL_USD_PER_BBL_TO_EUR_PER_MWH_TH,
    CARBON_KRBN_TO_EUR_PER_T,
    IBERIAN_CAP_PERIODS,
    CAPACITY_ROADMAP, DEMAND_ROADMAP_TWH,
    FORECAST_END_YEAR,
    HISTORY_END,
    FORWARD_CURVE,
)


# ── Forward fuel curve (mean-reversion model) ────────────────────────────────

def forward_fuel_price(spot: float, h_months: int, theta: float, rho: float) -> float:
    """Schwartz one-factor forward expectation:  E[F_{t+h}] = theta + (S_t - theta) * rho^h.

    spot:    observed price at anchor (in same units as theta)
    h_months: months ahead
    theta:   long-run mean (config.FORWARD_CURVE[fuel]['theta_*'])
    rho:     monthly mean-reversion (config.FORWARD_CURVE[fuel]['rho_monthly'])
    """
    if pd.isna(spot):
        return np.nan
    return theta + (spot - theta) * (rho ** max(0, int(h_months)))

log = logging.getLogger(__name__)


# ── Fuel-price normalisation (raw market units -> EUR/MWh thermal) ───────────

def _fuel_eur_per_mwh_thermal(fuel: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw fuel series (gas EUR/MWh, coal USD/t, oil USD/bbl) to a
    common unit: EUR per MWh of *thermal* energy.

    Carbon (KRBN) is left in EUR/tCO2 — applied separately via emissions factor.
    """
    df = fuel.copy()
    out = pd.DataFrame(index=df.index)

    # TTF gas already EUR/MWh thermal
    if "gas_ttf" in df.columns:
        out["gas_ttf_eur_per_mwh_th"] = df["gas_ttf"]

    # Coal: USD/t -> EUR/MWh thermal (rough USD≈EUR)
    if "coal_api2" in df.columns:
        out["coal_eur_per_mwh_th"] = df["coal_api2"] * COAL_USD_PER_T_TO_EUR_PER_MWH_TH

    # Oil: USD/bbl -> EUR/MWh thermal
    if "oil_brent" in df.columns:
        out["oil_eur_per_mwh_th"] = df["oil_brent"] * OIL_USD_PER_BBL_TO_EUR_PER_MWH_TH

    # Carbon: KRBN price as proxy for EUR/tCO2 (documented limitation)
    if "carbon" in df.columns:
        out["carbon_eur_per_t"] = df["carbon"] * CARBON_KRBN_TO_EUR_PER_T

    # Lignite: near-flat mine-mouth thermal cost (no traded ticker)
    out["lignite_eur_per_mwh_th"] = LIGNITE_FUEL_FLAT_EUR_PER_MWH_TH

    return out


def _fuel_key_for_tech(tech: str) -> str | None:
    """Map config.TECH_PARAMS[tech]['fuel'] to the column name produced above."""
    f = TECH_PARAMS[tech]["fuel"]
    return {
        "gas_ttf":      "gas_ttf_eur_per_mwh_th",
        "coal_api2":    "coal_eur_per_mwh_th",
        "oil_brent":    "oil_eur_per_mwh_th",
        "lignite_flat": "lignite_eur_per_mwh_th",
        "uranium_flat": None,   # handled via NUCLEAR_FLAT_MC_EUR_PER_MWH
        None:           None,
    }.get(f, None)


# ── Marginal cost ────────────────────────────────────────────────────────────

def _apply_iberian_cap(month: pd.Timestamp,
                       fuel_th_eur_per_mwh: float) -> tuple[float, bool]:
    """
    If `month` falls inside an Iberian gas-cap period and the fuel cost
    exceeds the cap, return (cap, True). Otherwise return (input, False).
    Cap applies to the gas-fuel component of marginal cost; carbon adder
    is computed by the caller after this function.
    """
    m_naive = pd.Timestamp(month).tz_localize(None) if pd.Timestamp(month).tzinfo else pd.Timestamp(month)
    for p in IBERIAN_CAP_PERIODS:
        if pd.Timestamp(p["start"]) <= m_naive <= pd.Timestamp(p["end"]):
            cap = p["cap_eur_per_mwh_gas"]
            if fuel_th_eur_per_mwh > cap:
                return cap, True
    return fuel_th_eur_per_mwh, False


def build_marginal_costs() -> pd.DataFrame:
    """
    Monthly marginal cost in EUR/MWh per (month, zone, tech).
      MC = heat_rate * fuel_eur_per_mwh_th + heat_rate * co2_per_mwh_fuel * carbon_eur_per_t
    Iberian cap (ES only, historical) is applied to the gas fuel component.
    """
    fuel_raw = pd.read_parquet(DATA_ALIGNED / "fuel_monthly.parquet")
    # Strip DA-price columns, keep only fuel/carbon
    fuel_cols = [c for c in fuel_raw.columns if not c.startswith("da_price_")]
    fuel = fuel_raw[fuel_cols].copy()
    fuel.index = pd.to_datetime(fuel.index)
    fuel = fuel.sort_index()
    # alignment.py outer-joins fuel + DA prices; if DA prices have a more recent
    # month than fuels (yfinance lag), trailing rows have NaN fuels. Re-ffill here.
    fuel = fuel.ffill()

    norm = _fuel_eur_per_mwh_thermal(fuel)

    rows = []
    for month in norm.index:
        carbon = float(norm.loc[month, "carbon_eur_per_t"]) if "carbon_eur_per_t" in norm.columns and pd.notna(norm.loc[month, "carbon_eur_per_t"]) else 0.0
        for zone in TARGET_ZONES:
            for tech in TECHS:
                params = TECH_PARAMS[tech]
                heat_rate = params["heat_rate"]
                emis      = params["co2_per_mwh_fuel"]

                if tech == "nuclear":
                    mc, fuel_comp, carbon_comp, capped = NUCLEAR_FLAT_MC_EUR_PER_MWH, NUCLEAR_FLAT_MC_EUR_PER_MWH, 0.0, False
                elif heat_rate == 0.0:
                    # wind/solar/hydro/biomass treated as zero-MC
                    mc, fuel_comp, carbon_comp, capped = 0.0, 0.0, 0.0, False
                else:
                    fuel_key = _fuel_key_for_tech(tech)
                    fuel_th  = float(norm.loc[month, fuel_key]) if fuel_key and fuel_key in norm.columns and pd.notna(norm.loc[month, fuel_key]) else np.nan
                    if np.isnan(fuel_th):
                        mc, fuel_comp, carbon_comp, capped = np.nan, np.nan, np.nan, False
                    else:
                        # Iberian cap (ES + gas only)
                        if zone == "ES" and tech == "gas_ccgt":
                            fuel_th, capped = _apply_iberian_cap(month, fuel_th)
                        else:
                            capped = False
                        fuel_comp   = heat_rate * fuel_th
                        carbon_comp = heat_rate * emis * carbon
                        mc          = fuel_comp + carbon_comp

                rows.append({
                    "month":          month,
                    "zone":           zone,
                    "tech":           tech,
                    "mc_eur_per_mwh": mc,
                    "fuel_component": fuel_comp,
                    "carbon_component": carbon_comp,
                    "capped":         capped,
                })

    out = pd.DataFrame(rows)
    out_path = DATA_PROCESSED / "marginal_costs_monthly.parquet"
    out.to_parquet(out_path, index=False)
    log.info("Saved %s — %d rows", out_path.name, len(out))
    return out


# ── Forward-extended structural panel ────────────────────────────────────────

def _interpolate_roadmap(knots: dict, years: range) -> pd.Series:
    """Linearly interpolate a roadmap dict {year: value} across `years`.
    Constant-extrapolated outside the knot range."""
    if not knots:
        return pd.Series([np.nan] * len(years), index=years)
    s = pd.Series(knots).sort_index()
    full = pd.Series(index=list(years), dtype=float)
    full.loc[s.index.intersection(full.index)] = s.loc[s.index.intersection(full.index)]
    full = full.interpolate(method="linear")
    full = full.ffill().bfill()
    return full


def build_structural_extended() -> pd.DataFrame:
    """Combine observed structural panel (history) with roadmap (future).

    The current calendar year is treated as missing — ENTSOE returns YTD-only
    values that would otherwise be misread as full-year totals (e.g. 167 TWh
    YTD-April vs ~480 TWh full year). For partial years, fall through to the
    roadmap interpolation instead.
    """
    hist = pd.read_parquet(DATA_ALIGNED / "structural_annual.parquet").reset_index()

    # Drop the current calendar year and any future years from observed data
    # so the roadmap fills them in.
    current_year = pd.Timestamp(HISTORY_END).year
    hist = hist[hist["year"] < current_year]

    years = range(int(hist["year"].min()), FORECAST_END_YEAR + 1)
    rows = []
    for zone in TARGET_ZONES:
        zhist = hist[hist["zone"] == zone].set_index("year")
        for year in years:
            row = {"year": year, "zone": zone}
            for tech in TECHS:
                cap_col = f"capacity_{tech}_gw"
                obs = zhist.loc[year, cap_col] if (year in zhist.index and cap_col in zhist.columns) else np.nan
                if pd.notna(obs):
                    row[cap_col] = obs
                else:
                    knots = CAPACITY_ROADMAP.get(zone, {}).get(tech, {})
                    row[cap_col] = _interpolate_roadmap(knots, years).loc[year] if knots else np.nan
            # Demand
            dem_obs = zhist.loc[year, "demand_twh"] if (year in zhist.index and "demand_twh" in zhist.columns) else np.nan
            if pd.notna(dem_obs):
                row["demand_twh"] = dem_obs
            else:
                row["demand_twh"] = _interpolate_roadmap(DEMAND_ROADMAP_TWH.get(zone, {}), years).loc[year]
            rows.append(row)

    out = pd.DataFrame(rows).sort_values(["zone", "year"]).set_index(["year", "zone"])
    out_path = DATA_PROCESSED / "structural_extended.parquet"
    out.to_parquet(out_path)
    log.info("Saved %s — %s shape", out_path.name, out.shape)
    return out


def build_features():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("=== Features ===")
    build_marginal_costs()
    build_structural_extended()
    log.info("=== Features complete ===")


if __name__ == "__main__":
    build_features()
