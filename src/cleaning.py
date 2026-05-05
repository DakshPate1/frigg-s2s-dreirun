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

Output files (one per source-zone or global):
  data/clean/prices_{zone}.parquet
  data/clean/generation_{zone}.parquet
  data/clean/weather_{zone}.parquet
  data/clean/fuel_prices.parquet
"""

from __future__ import annotations

import logging
from datetime import timezone

import numpy as np
import pandas as pd

from config import (
    DATA_RAW, DATA_CLEAN, ZONES,
    WIND_TYPES, SOLAR_TYPES, HYDRO_TYPES, LOAD_TYPE,
    TRAIN_START, TRAIN_END,
)

log = logging.getLogger(__name__)

MAX_INTERP_GAP = 3   # hours: interpolate gaps up to this size
MAX_LARGE_GAP  = 24  # hours: gaps larger than this will be logged as warnings


def _utc_index(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Parse timestamp column to UTC-aware datetime and set as index."""
    df = df.copy()
    col = df[ts_col]

    if pd.api.types.is_integer_dtype(col):
        df.index = pd.to_datetime(col, unit="s", utc=True)
    else:
        df.index = pd.to_datetime(col, utc=True)

    df.index.name = "timestamp"
    df = df.drop(columns=[ts_col])
    return df


def _enforce_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to full hourly grid, forward-filling index gaps with NaN values."""
    start = df.index.min().floor("h")
    end   = df.index.max().ceil("h")
    full  = pd.date_range(start, end, freq="h", tz="UTC")
    return df.reindex(full)


def _interpolate(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Linear interpolation for small gaps; leave large gaps as NaN."""
    for col in cols:
        s = df[col]
        # identify NaN runs
        mask = s.isna()
        if not mask.any():
            continue
        # interpolate, but only fill runs ≤ MAX_INTERP_GAP
        s_interp = s.interpolate(method="linear", limit=MAX_INTERP_GAP)
        df[col] = s_interp

        remaining = df[col].isna().sum()
        if remaining:
            log.warning("Column '%s': %d NaN remain after interpolation (large gaps)", col, remaining)
    return df


def _flag_large_gaps(df: pd.DataFrame, cols: list[str]) -> None:
    """Log warnings for NaN runs larger than MAX_LARGE_GAP."""
    for col in cols:
        s = df[col].isna()
        if not s.any():
            continue
        runs = s.ne(s.shift()).cumsum()
        gap_sizes = s.groupby(runs).sum()
        big = gap_sizes[gap_sizes > MAX_LARGE_GAP]
        if len(big):
            log.warning("Column '%s' has %d gaps > %dh", col, len(big), MAX_LARGE_GAP)


# ── Prices ────────────────────────────────────────────────────────────────────

def clean_prices(zone: str) -> pd.DataFrame:
    path = DATA_RAW / "energycharts" / f"prices_{zone}.csv"
    log.info("Cleaning prices for %s from %s", zone, path)
    raw = pd.read_csv(path)
    df = _utc_index(raw, "unix_seconds")
    df = df.sort_index()
    df = _enforce_hourly(df)
    _flag_large_gaps(df, ["price"])
    df = _interpolate(df, ["price"])
    out_path = DATA_CLEAN / f"prices_{zone}.parquet"
    df.to_parquet(out_path)
    log.info("Saved %s — %d rows, %d NaN", out_path.name, len(df), df["price"].isna().sum())
    return df


# ── Generation + Load ─────────────────────────────────────────────────────────

def clean_generation(zone: str) -> pd.DataFrame:
    path = DATA_RAW / "energycharts" / f"generation_{zone}.csv"
    log.info("Cleaning generation for %s from %s", zone, path)
    raw = pd.read_csv(path)
    df = _utc_index(raw, "unix_seconds")
    df = df.sort_index()

    # Collapse raw columns into canonical features
    available = df.columns.tolist()

    wind_cols  = [c for c in available if any(wt in c for wt in WIND_TYPES)]
    solar_cols = [c for c in available if any(st in c for st in SOLAR_TYPES)]
    hydro_cols = [c for c in available if any(ht in c for ht in HYDRO_TYPES)]

    out = pd.DataFrame(index=df.index)
    out["wind_generation"]  = df[wind_cols].sum(axis=1)
    out["solar_generation"] = df[solar_cols].sum(axis=1)
    out["hydro_generation"] = df[hydro_cols].sum(axis=1)
    out["load"]             = df[LOAD_TYPE] if LOAD_TYPE in df.columns else np.nan

    # Negative values: pumped storage consumption can make wind_generation negative
    # Clamp to zero — we only want actual generation, not consumption
    for col in ["wind_generation", "solar_generation", "hydro_generation"]:
        out[col] = out[col].clip(lower=0)

    out = _enforce_hourly(out)
    cols = ["wind_generation", "solar_generation", "hydro_generation", "load"]
    _flag_large_gaps(out, cols)
    out = _interpolate(out, cols)

    out_path = DATA_CLEAN / f"generation_{zone}.parquet"
    out.to_parquet(out_path)
    log.info("Saved %s — %d rows", out_path.name, len(out))
    return out


# ── Weather ───────────────────────────────────────────────────────────────────

def clean_weather(zone: str) -> pd.DataFrame:
    path = DATA_RAW / "openmeteo" / f"weather_{zone}.csv"
    log.info("Cleaning weather for %s from %s", zone, path)
    raw = pd.read_csv(path)

    # Open-Meteo time column: "2021-01-01T00:00" (local UTC strings)
    df = _utc_index(raw, "time")
    df = df.sort_index()
    df = df[["temperature", "wind_speed", "solar_radiation"]]

    df = _enforce_hourly(df)
    cols = ["temperature", "wind_speed", "solar_radiation"]
    _flag_large_gaps(df, cols)
    df = _interpolate(df, cols)

    # Solar radiation cannot be negative (sensor noise)
    df["solar_radiation"] = df["solar_radiation"].clip(lower=0)

    out_path = DATA_CLEAN / f"weather_{zone}.parquet"
    df.to_parquet(out_path)
    log.info("Saved %s — %d rows", out_path.name, len(df))
    return df


# ── Fuel prices ───────────────────────────────────────────────────────────────

def clean_fuel_prices() -> pd.DataFrame:
    path = DATA_RAW / "fuel" / "fuel_prices.csv"
    log.info("Cleaning fuel prices from %s", path)
    raw = pd.read_csv(path, parse_dates=["date"])

    raw = raw.sort_values("date").drop_duplicates("date").set_index("date")
    raw.index = pd.to_datetime(raw.index)

    # Build daily UTC-midnight index
    start = raw.index.min()
    end   = raw.index.max()
    daily = pd.date_range(start, end, freq="D")
    raw = raw.reindex(daily)

    # Forward-fill weekends and holidays (prices don't change on non-trading days)
    raw["gas_price"]    = raw["gas_price"].ffill()
    raw["carbon_price"] = raw["carbon_price"].ffill()

    # Expand to hourly by repeating daily value each hour
    hourly_idx = pd.date_range(
        start.normalize(),
        end.normalize() + pd.Timedelta(hours=23),
        freq="h",
        tz="UTC",
    )
    hourly = pd.DataFrame(index=hourly_idx)
    hourly.index.name = "timestamp"

    # Align on date
    raw.index = raw.index.tz_localize("UTC")
    hourly["gas_price"]    = raw["gas_price"].reindex(hourly_idx, method="ffill")
    hourly["carbon_price"] = raw["carbon_price"].reindex(hourly_idx, method="ffill")

    out_path = DATA_CLEAN / "fuel_prices.parquet"
    hourly.to_parquet(out_path)
    log.info("Saved %s — %d rows", out_path.name, len(hourly))
    return hourly


# ── Orchestrator ──────────────────────────────────────────────────────────────

def clean_all() -> None:
    for zone in ZONES:
        log.info("=== Cleaning zone: %s ===", zone)
        clean_prices(zone)
        clean_generation(zone)
        clean_weather(zone)
    clean_fuel_prices()
    log.info("=== Cleaning complete ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    clean_all()
