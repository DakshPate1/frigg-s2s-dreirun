"""
Stage 2 — Cleaning.

Per dataset:
  1. Parse timestamps → UTC datetime index
  2. Enforce hourly frequency (reindex to full hourly grid)
  3. Handle missing values:
     - gaps ≤ MAX_INTERP_GAP hours → linear interpolation
     - gaps > MAX_INTERP_GAP       → remain NaN (flagged, handled in alignment)
  4. Sort chronologically
  5. Save as parquet to data/clean/

ENTSOE is the primary source for prices, generation, load, and cross-border flows.
energy-charts.info cleaners are retained for fallback use.

Output files:
  data/clean/prices_{zone}.parquet
  data/clean/generation_{zone}.parquet
  data/clean/crossborder_{zone}.parquet
  data/clean/weather_{zone}.parquet
  data/clean/fuel_prices.parquet
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from config import (
    DATA_RAW, DATA_CLEAN, ZONES,
    WIND_TYPES, SOLAR_TYPES, HYDRO_TYPES, LOAD_TYPE,
    TRAIN_START, TRAIN_END,
)

log = logging.getLogger(__name__)

MAX_INTERP_GAP = 24  # hours: interpolate gaps up to this size (covers 21h chunk-boundary gaps)
MAX_LARGE_GAP  = 48  # hours: gaps larger than this will be logged as warnings


# ── Shared helpers ────────────────────────────────────────────────────────────

def _utc_index(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Parse timestamp column to UTC-aware datetime and set as index."""
    df  = df.copy()
    col = df[ts_col]
    if pd.api.types.is_integer_dtype(col):
        df.index = pd.to_datetime(col, unit="s", utc=True)
    else:
        df.index = pd.to_datetime(col, utc=True)
    df.index.name = "timestamp"
    return df.drop(columns=[ts_col])


def _enforce_hourly(df: pd.DataFrame) -> pd.DataFrame:
    start = df.index.min().floor("h")
    end   = df.index.max().ceil("h")
    full  = pd.date_range(start, end, freq="h", tz="UTC")
    return df.reindex(full)


def _interpolate(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        s = df[col]
        if not s.isna().any():
            continue
        df[col] = s.interpolate(method="linear", limit=MAX_INTERP_GAP)
        remaining = df[col].isna().sum()
        if remaining:
            log.warning("Column '%s': %d NaN remain after interpolation", col, remaining)
    return df


def _flag_large_gaps(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        s = df[col].isna()
        if not s.any():
            continue
        runs      = s.ne(s.shift()).cumsum()
        gap_sizes = s.groupby(runs).sum()
        big       = gap_sizes[gap_sizes > MAX_LARGE_GAP]
        if len(big):
            log.warning("Column '%s': %d gaps > %dh", col, len(big), MAX_LARGE_GAP)


def _save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    path = DATA_CLEAN / name
    df.to_parquet(path)
    log.info("Saved %s — %d rows", name, len(df))
    return df


# ── ENTSOE prices ─────────────────────────────────────────────────────────────

def clean_entsoe_prices(zone: str) -> pd.DataFrame:
    path = DATA_RAW / "entsoe" / f"prices_{zone}.csv"
    log.info("Cleaning ENTSOE prices %s from %s", zone, path)
    raw = pd.read_csv(path)
    df  = _utc_index(raw, "timestamp").sort_index()
    df  = _enforce_hourly(df)
    _flag_large_gaps(df, ["price"])
    df  = _interpolate(df, ["price"])
    return _save(df, f"prices_{zone}.parquet")


# ── ENTSOE generation + load ──────────────────────────────────────────────────

def clean_entsoe_generation(zone: str) -> pd.DataFrame:
    """
    ENTSOE generation CSV already has canonical column names from ingestion.
    Columns: timestamp, wind_generation, solar_generation, hydro_generation, load
    """
    path = DATA_RAW / "entsoe" / f"generation_{zone}.csv"
    log.info("Cleaning ENTSOE generation %s from %s", zone, path)
    raw  = pd.read_csv(path)
    df   = _utc_index(raw, "timestamp").sort_index()

    # Ensure all canonical columns exist
    for col in ["wind_generation", "solar_generation", "hydro_generation", "load"]:
        if col not in df.columns:
            df[col] = np.nan

    for col in ["wind_generation", "solar_generation", "hydro_generation"]:
        df[col] = df[col].clip(lower=0)

    df = _enforce_hourly(df)
    cols = ["wind_generation", "solar_generation", "hydro_generation", "load"]
    _flag_large_gaps(df, cols)
    df = _interpolate(df, cols)
    return _save(df, f"generation_{zone}.parquet")


# ── ENTSOE cross-border flows ─────────────────────────────────────────────────

def clean_entsoe_crossborder(zone: str) -> pd.DataFrame:
    path = DATA_RAW / "entsoe" / f"crossborder_{zone}.csv"
    log.info("Cleaning ENTSOE crossborder %s from %s", zone, path)
    raw = pd.read_csv(path)
    df  = _utc_index(raw, "timestamp").sort_index()
    df  = _enforce_hourly(df)
    _flag_large_gaps(df, ["net_imports"])
    # Cross-border flows can be negative (net exporter) — interpolate but don't clip
    df  = _interpolate(df, ["net_imports"])
    return _save(df, f"crossborder_{zone}.parquet")


# ── ENTSOE day-ahead forecasts ────────────────────────────────────────────────

def clean_entsoe_forecasts(zone: str) -> pd.DataFrame:
    """
    Clean ENTSOE day-ahead load + wind/solar generation forecasts.

    Returns empty DataFrame if the raw CSV doesn't exist (forecasts are optional;
    the model falls back to actuals if this file is absent).
    """
    path = DATA_RAW / "entsoe" / f"forecasts_{zone}.csv"
    if not path.exists():
        log.info("No forecast CSV for %s — skipping (run pipeline to generate)", zone)
        return pd.DataFrame()

    log.info("Cleaning ENTSOE forecasts %s from %s", zone, path)
    raw  = pd.read_csv(path)
    df   = _utc_index(raw, "timestamp").sort_index()
    df   = _enforce_hourly(df)

    fcst_cols = [c for c in df.columns if c in [
        "load_forecast", "wind_generation_forecast", "solar_generation_forecast"]]
    if not fcst_cols:
        return df

    for col in ["wind_generation_forecast", "solar_generation_forecast"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    _flag_large_gaps(df, fcst_cols)
    df = _interpolate(df, fcst_cols)
    return _save(df, f"forecasts_{zone}.parquet")


# ── Weather (Open-Meteo) — unchanged ─────────────────────────────────────────

def clean_weather(zone: str) -> pd.DataFrame:
    path = DATA_RAW / "openmeteo" / f"weather_{zone}.csv"
    log.info("Cleaning weather %s from %s", zone, path)
    raw = pd.read_csv(path)
    df  = _utc_index(raw, "time").sort_index()

    # Multi-city format has columns like DE_wind_wind_speed_100m, DE_demand_temperature_2m.
    # Pass all columns through and create legacy aliases so alignment CRITICAL_COLS pass.
    multi_city = any("_wind_speed_100m" in c or "_temperature_2m" in c for c in df.columns)
    if multi_city:
        temp_col  = next((c for c in df.columns if "_temperature_2m"       in c), None)
        wind_col  = next((c for c in df.columns if "_wind_speed_100m"      in c), None)
        solar_col = next((c for c in df.columns if "_shortwave_radiation"  in c), None)
        if temp_col:
            df["temperature"]     = df[temp_col]
        if wind_col:
            df["wind_speed"]      = df[wind_col]
        if solar_col:
            df["solar_radiation"] = df[solar_col].clip(lower=0)
        legacy_cols = [c for c in ["temperature", "wind_speed", "solar_radiation"] if c in df.columns]
    else:
        df   = df[["temperature", "wind_speed", "solar_radiation"]]
        df["solar_radiation"] = df["solar_radiation"].clip(lower=0)
        legacy_cols = ["temperature", "wind_speed", "solar_radiation"]

    df = _enforce_hourly(df)
    _flag_large_gaps(df, legacy_cols)
    df = _interpolate(df, legacy_cols)
    return _save(df, f"weather_{zone}.parquet")


# ── Fuel prices (yfinance) — unchanged ───────────────────────────────────────

def clean_fuel_prices() -> pd.DataFrame:
    path = DATA_RAW / "fuel" / "fuel_prices.csv"
    log.info("Cleaning fuel prices from %s", path)
    raw  = pd.read_csv(path, parse_dates=["date"])
    raw  = raw.sort_values("date").drop_duplicates("date").set_index("date")
    raw.index = pd.to_datetime(raw.index)

    daily = pd.date_range(raw.index.min(), raw.index.max(), freq="D")
    raw   = raw.reindex(daily)
    fuel_cols = [c for c in ["gas_price", "coal_price", "carbon_price"] if c in raw.columns]
    for col in fuel_cols:
        raw[col] = raw[col].ffill()

    hourly_idx = pd.date_range(
        raw.index.min().normalize(),
        raw.index.max().normalize() + pd.Timedelta(hours=23),
        freq="h", tz="UTC",
    )
    hourly = pd.DataFrame(index=hourly_idx)
    hourly.index.name = "timestamp"
    raw.index = raw.index.tz_localize("UTC")
    for col in fuel_cols:
        hourly[col] = raw[col].reindex(hourly_idx, method="ffill")
    return _save(hourly, "fuel_prices.parquet")


# ── Fallback: energy-charts cleaners ─────────────────────────────────────────

def clean_prices_ec(zone: str) -> pd.DataFrame:
    """Fallback cleaner for energy-charts prices CSV (unix_seconds column)."""
    path = DATA_RAW / "energycharts" / f"prices_{zone}.csv"
    raw  = pd.read_csv(path)
    df   = _utc_index(raw, "unix_seconds").sort_index()
    df   = _enforce_hourly(df)
    _flag_large_gaps(df, ["price"])
    df   = _interpolate(df, ["price"])
    return _save(df, f"prices_{zone}.parquet")


def clean_generation_ec(zone: str) -> pd.DataFrame:
    """Fallback cleaner for energy-charts generation CSV (collapses raw columns)."""
    from config import WIND_TYPES, SOLAR_TYPES, HYDRO_TYPES, LOAD_TYPE
    path = DATA_RAW / "energycharts" / f"generation_{zone}.csv"
    raw  = pd.read_csv(path)
    df   = _utc_index(raw, "unix_seconds").sort_index()
    available = df.columns.tolist()

    wind_cols  = [c for c in available if any(wt in c for wt in WIND_TYPES)]
    solar_cols = [c for c in available if any(st in c for st in SOLAR_TYPES)]
    hydro_cols = [c for c in available if any(ht in c for ht in HYDRO_TYPES)]

    out = pd.DataFrame(index=df.index)
    out["wind_generation"]  = df[wind_cols].sum(axis=1).clip(lower=0)
    out["solar_generation"] = df[solar_cols].sum(axis=1).clip(lower=0)
    out["hydro_generation"] = df[hydro_cols].sum(axis=1).clip(lower=0)
    out["load"]             = df[LOAD_TYPE] if LOAD_TYPE in df.columns else np.nan

    out = _enforce_hourly(out)
    cols = ["wind_generation", "solar_generation", "hydro_generation", "load"]
    _flag_large_gaps(out, cols)
    out = _interpolate(out, cols)
    return _save(out, f"generation_{zone}.parquet")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def clean_all() -> None:
    """Clean all data sources. Expects ENTSOE raw CSVs to exist."""
    for zone in ZONES:
        log.info("=== Cleaning zone: %s ===", zone)
        clean_entsoe_prices(zone)
        clean_entsoe_generation(zone)
        clean_entsoe_crossborder(zone)
        clean_entsoe_forecasts(zone)
        clean_weather(zone)
    clean_fuel_prices()
    log.info("=== Cleaning complete ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    clean_all()
