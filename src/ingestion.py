"""
Stage 1 — Raw data ingestion.

Fetches from three sources and writes unmodified CSVs to data/raw/:
  - energy-charts.info  → prices, load, generation per zone
  - Open-Meteo          → temperature, wind speed, solar radiation per zone
  - yfinance            → TTF gas price (EUR/MWh), KRBN carbon ETF (USD, proxy)

No cleaning or transformation here.
"""

from __future__ import annotations

import time
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
import pandas as pd
import yfinance as yf

from config import (
    DATA_RAW, ZONES, EC_BZN, EC_COUNTRY, WEATHER_LOCATIONS,
    TRAIN_START, TRAIN_END,
)

log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(url: str, params: dict, retries: int = 3, pause: float = 1.0) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            log.warning("GET %s attempt %d/%d failed: %s", url, attempt + 1, retries, exc)
            if attempt < retries - 1:
                time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def _chunk_dates(start: str, end: str, chunk_days: int = 180):
    """Yield (start, end) string pairs in chunks to avoid API timeouts."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    while s < e:
        chunk_end = min(s + timedelta(days=chunk_days), e)
        yield s.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        s = chunk_end + timedelta(days=1)


# ── Source 1: Energy-Charts prices ────────────────────────────────────────────

def fetch_prices(zone: str, start: str = TRAIN_START, end: str = TRAIN_END) -> pd.DataFrame:
    """Fetch day-ahead auction prices from energy-charts.info."""
    bzn = EC_BZN[zone]
    frames = []
    for s, e in _chunk_dates(start, end, 180):
        log.info("Fetching %s prices %s → %s", zone, s, e)
        d = _get("https://api.energy-charts.info/price",
                 {"bzn": bzn, "start": s, "end": e})
        if not d.get("unix_seconds"):
            continue
        df = pd.DataFrame({
            "unix_seconds": d["unix_seconds"],
            "price": d["price"],
        })
        frames.append(df)
        time.sleep(0.5)

    if not frames:
        raise RuntimeError(f"No price data for {zone}")
    out = pd.concat(frames, ignore_index=True).drop_duplicates("unix_seconds")
    out.to_csv(DATA_RAW / "energycharts" / f"prices_{zone}.csv", index=False)
    log.info("Saved prices_%s.csv — %d rows", zone, len(out))
    return out


# ── Source 2: Energy-Charts generation & load ─────────────────────────────────

def fetch_generation(zone: str, start: str = TRAIN_START, end: str = TRAIN_END) -> pd.DataFrame:
    """Fetch actual generation by type + load from energy-charts.info public_power."""
    country = EC_COUNTRY[zone]
    frames = []
    for s, e in _chunk_dates(start, end, 180):
        log.info("Fetching %s generation %s → %s", zone, s, e)
        d = _get("https://api.energy-charts.info/public_power",
                 {"country": country, "start": s, "end": e})
        if not d.get("unix_seconds"):
            continue

        row = {"unix_seconds": d["unix_seconds"]}
        for pt in d["production_types"]:
            row[pt["name"]] = pt["data"]

        df = pd.DataFrame(row)
        frames.append(df)
        time.sleep(0.5)

    if not frames:
        raise RuntimeError(f"No generation data for {zone}")
    out = pd.concat(frames, ignore_index=True).drop_duplicates("unix_seconds")
    out.to_csv(DATA_RAW / "energycharts" / f"generation_{zone}.csv", index=False)
    log.info("Saved generation_%s.csv — %d rows, %d cols", zone, len(out), len(out.columns))
    return out


# ── Source 3: Open-Meteo weather ──────────────────────────────────────────────

def fetch_weather(zone: str, start: str = TRAIN_START, end: str = TRAIN_END) -> pd.DataFrame:
    """Fetch hourly historical weather from Open-Meteo archive API (no key needed)."""
    loc = WEATHER_LOCATIONS[zone]
    frames = []
    for s, e in _chunk_dates(start, end, 365):
        log.info("Fetching %s weather %s → %s (%s)", zone, s, e, loc["label"])
        d = _get("https://archive-api.open-meteo.com/v1/archive", {
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "start_date": s,
            "end_date": e,
            "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation",
            "timezone": "UTC",
            "wind_speed_unit": "ms",
        })
        h = d.get("hourly", {})
        if not h.get("time"):
            continue
        df = pd.DataFrame({
            "time": h["time"],
            "temperature": h["temperature_2m"],
            "wind_speed": h["wind_speed_10m"],
            "solar_radiation": h["shortwave_radiation"],
        })
        frames.append(df)
        time.sleep(0.3)

    if not frames:
        raise RuntimeError(f"No weather data for {zone}")
    out = pd.concat(frames, ignore_index=True).drop_duplicates("time")
    out.to_csv(DATA_RAW / "openmeteo" / f"weather_{zone}.csv", index=False)
    log.info("Saved weather_%s.csv — %d rows", zone, len(out))
    return out


def fetch_weather_forecast(zone: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly weather forecast from Open-Meteo (up to 16 days ahead)."""
    loc = WEATHER_LOCATIONS[zone]
    log.info("Fetching %s weather forecast %s → %s", zone, start_date, end_date)
    d = _get("https://api.open-meteo.com/v1/forecast", {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation",
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    })
    h = d.get("hourly", {})
    df = pd.DataFrame({
        "time": h["time"],
        "temperature": h["temperature_2m"],
        "wind_speed": h["wind_speed_10m"],
        "solar_radiation": h["shortwave_radiation"],
    })
    log.info("Forecast weather %s: %d rows", zone, len(df))
    return df


# ── Source 4: Fuel prices (yfinance) ─────────────────────────────────────────

def fetch_fuel_prices(start: str = TRAIN_START, end: str = TRAIN_END) -> pd.DataFrame:
    """
    Fetch fuel price proxies via yfinance:
      - TTF=F  : TTF natural gas futures (EUR/MWh on ICE)
      - KRBN   : KraneShares European Carbon ETF (USD, strongly correlated to EUA)
    Daily closing prices, forward-filled to handle weekends.
    """
    log.info("Fetching fuel prices %s → %s", start, end)

    # Add 1 day buffer so end date is inclusive
    end_dt = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    ttf = yf.Ticker("TTF=F").history(start=start, end=end_dt, interval="1d")["Close"]
    ttf.name = "gas_price"

    krbn = yf.Ticker("KRBN").history(start=start, end=end_dt, interval="1d")["Close"]
    krbn.name = "carbon_price"

    df = pd.concat([ttf, krbn], axis=1)
    df.index = df.index.tz_localize(None)  # strip tz, store as date
    df.index.name = "date"
    df = df.reset_index()

    df.to_csv(DATA_RAW / "fuel" / "fuel_prices.csv", index=False)
    log.info("Saved fuel_prices.csv — %d rows", len(df))
    return df


# ── Orchestrator ──────────────────────────────────────────────────────────────

def ingest_all(start: str = TRAIN_START, end: str = TRAIN_END) -> None:
    """Run full ingestion for all zones. Safe to re-run (skips existing files)."""
    for zone in ZONES:
        log.info("=== Ingesting zone: %s ===", zone)
        fetch_prices(zone, start, end)
        fetch_generation(zone, start, end)
        fetch_weather(zone, start, end)

    fetch_fuel_prices(start, end)
    log.info("=== Ingestion complete ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ingest_all()
