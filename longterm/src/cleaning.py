"""
Stage 2 — Cleaning.

Inputs:  data/raw/entsoe/*.csv, data/raw/fuel/*.csv
Outputs: data/clean/*.parquet

Operations:
  - Map ENTSOE PSR types to canonical tech buckets (config.ENTSOE_PSR_TO_TECH).
  - Aggregate capacity / generation per (year, zone, tech).
  - Resample fuel prices: daily -> monthly mean; carry trading-day gaps via ffill.
  - Sanity-cast types, normalize units, drop empties.

No interpolation across years (long-term: gaps are real, not noise to smooth).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import (
    DATA_RAW, DATA_CLEAN,
    TARGET_ZONES, TECHS,
    ENTSOE_PSR_TO_TECH,
    HISTORY_START, HISTORY_END,
)

_YEAR_MIN = pd.Timestamp(HISTORY_START).year
_YEAR_MAX = pd.Timestamp(HISTORY_END).year

log = logging.getLogger(__name__)


# ── Capacity ─────────────────────────────────────────────────────────────────

def clean_installed_capacity() -> pd.DataFrame:
    """Aggregate ENTSOE installed capacity to (year, zone, tech) -> MW."""
    frames = []
    for zone in TARGET_ZONES:
        path = DATA_RAW / "entsoe" / f"installed_capacity_{zone}.csv"
        if not path.exists():
            log.warning("missing %s", path)
            continue
        df = pd.read_csv(path)
        df["tech"] = df["psr_type"].map(ENTSOE_PSR_TO_TECH)
        unmapped = df.loc[df["tech"].isna(), "psr_type"].unique()
        if len(unmapped) > 0:
            log.warning("unmapped PSR types in %s: %s", zone, list(unmapped))
            df["tech"] = df["tech"].fillna("biomass_other")
        frames.append(df)

    if not frames:
        raise RuntimeError("no installed-capacity CSVs found")
    full = pd.concat(frames, ignore_index=True)
    agg = (full.groupby(["year", "zone", "tech"], as_index=False)["capacity_mw"]
                .sum()
                .sort_values(["zone", "year", "tech"]))
    # MW -> GW for readability
    agg["capacity_gw"] = agg["capacity_mw"] / 1000.0
    out_path = DATA_CLEAN / "capacity_by_tech.parquet"
    agg.to_parquet(out_path, index=False)
    log.info("Saved %s — %d rows", out_path.name, len(agg))
    return agg


# ── Generation ───────────────────────────────────────────────────────────────

def clean_generation() -> pd.DataFrame:
    """Aggregate annual actual generation to (year, zone, tech) -> TWh."""
    frames = []
    for zone in TARGET_ZONES:
        path = DATA_RAW / "entsoe" / f"generation_annual_{zone}.csv"
        if not path.exists():
            log.warning("missing %s", path)
            continue
        df = pd.read_csv(path)
        df["tech"] = df["psr_type"].map(ENTSOE_PSR_TO_TECH).fillna("biomass_other")
        frames.append(df)

    if not frames:
        log.warning("no generation CSVs — skipping")
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)
    full = full[(full["year"] >= _YEAR_MIN) & (full["year"] <= _YEAR_MAX)]
    agg = (full.groupby(["year", "zone", "tech"], as_index=False)["generation_twh"]
                .sum()
                .sort_values(["zone", "year", "tech"]))
    out_path = DATA_CLEAN / "generation_by_tech.parquet"
    agg.to_parquet(out_path, index=False)
    log.info("Saved %s — %d rows", out_path.name, len(agg))
    return agg


# ── Demand ───────────────────────────────────────────────────────────────────

def clean_demand() -> pd.DataFrame:
    """Concatenate annual demand per zone."""
    frames = []
    for zone in TARGET_ZONES:
        path = DATA_RAW / "entsoe" / f"load_annual_{zone}.csv"
        if not path.exists():
            log.warning("missing %s", path)
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise RuntimeError("no load CSVs found")
    full = pd.concat(frames, ignore_index=True).sort_values(["zone", "year"])
    full = full[(full["year"] >= _YEAR_MIN) & (full["year"] <= _YEAR_MAX)]
    out_path = DATA_CLEAN / "demand_annual.parquet"
    full.to_parquet(out_path, index=False)
    log.info("Saved %s — %d rows", out_path.name, len(full))
    return full


# ── Day-ahead prices (calibration target) ────────────────────────────────────

def clean_da_prices() -> pd.DataFrame:
    """Concatenate monthly DA price means per zone."""
    frames = []
    for zone in TARGET_ZONES:
        path = DATA_RAW / "entsoe" / f"da_prices_monthly_{zone}.csv"
        if not path.exists():
            log.warning("missing %s", path)
            continue
        df = pd.read_csv(path, parse_dates=["month"])
        frames.append(df)
    if not frames:
        raise RuntimeError("no DA-price CSVs found")
    full = pd.concat(frames, ignore_index=True).sort_values(["zone", "month"])
    out_path = DATA_CLEAN / "da_prices_monthly.parquet"
    full.to_parquet(out_path, index=False)
    log.info("Saved %s — %d rows", out_path.name, len(full))
    return full


# ── GPR (monthly already; just standardize) ──────────────────────────────────

def clean_gpr() -> pd.DataFrame:
    path = DATA_RAW / "fuel" / "gpr.csv"
    if not path.exists():
        log.warning("missing %s — skipping", path)
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["month"]).sort_values("month")
    out_path = DATA_CLEAN / "gpr_monthly.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Saved %s — %d rows", out_path.name, len(df))
    return df


# ── Fuel prices ──────────────────────────────────────────────────────────────

def clean_fuel_prices() -> pd.DataFrame:
    """Resample daily fuel/carbon series to monthly means, join into one panel."""
    series = {}
    for path in sorted((DATA_RAW / "fuel").glob("*.csv")):
        name = path.stem
        if name == "gpr":   # GPR is monthly, handled separately
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.set_index("date").sort_index()
        # Drop any non-numeric leftover columns
        col = [c for c in df.columns if c.lower() not in ("ticker",)][0]
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        # Forward-fill weekends/holidays before resampling so monthly means
        # aren't biased by fewer trading days in some months.
        s = s.asfreq("D").ffill()
        monthly = s.resample("MS").mean()
        series[name] = monthly

    if not series:
        log.warning("no fuel CSVs — skipping")
        return pd.DataFrame()
    panel = pd.concat(series, axis=1)
    # yfinance lags real-time by ~3 weeks; the trailing month often has missing
    # data even though earlier months are complete. Forward-fill so the latest
    # month inherits the previous month's prices for marginal-cost calc.
    panel = panel.ffill()
    panel.index.name = "month"
    panel = panel.reset_index()
    out_path = DATA_CLEAN / "fuel_monthly.parquet"
    panel.to_parquet(out_path, index=False)
    log.info("Saved %s — %d rows × %d series", out_path.name, len(panel), len(series))
    return panel


# ── Orchestrator ─────────────────────────────────────────────────────────────

def clean_all():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("=== Cleaning ===")
    clean_installed_capacity()
    clean_generation()
    clean_demand()
    clean_da_prices()
    clean_fuel_prices()
    clean_gpr()
    log.info("=== Cleaning complete ===")


if __name__ == "__main__":
    clean_all()
