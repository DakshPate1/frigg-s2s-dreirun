"""
Stage 1 — Raw data ingestion.

Primary source: ENTSOE Transparency Platform (prices, load, generation, cross-border flows)
Supplementary: Open-Meteo (weather), yfinance (fuel prices)
Fallback:      energy-charts.info (retained as functions; not called by default)

Cross-border flows → net_imports feature (MW, positive = net importer)
  DE-LU: aggregated over 8 European neighbors
  ES:    aggregated over FR + PT
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

import requests
import pandas as pd
import yfinance as yf

from config import (
    DATA_RAW, ZONES,
    EC_BZN, EC_COUNTRY,
    ENTSOE_TOKEN, ENTSOE_ZONES, ENTSOE_NEIGHBORS,
    ENTSOE_WIND_TYPES, ENTSOE_SOLAR_TYPES, ENTSOE_HYDRO_TYPES,
    WEATHER_LOCATIONS, WIND_TYPES, SOLAR_TYPES, HYDRO_TYPES, LOAD_TYPE,
    TRAIN_START, TRAIN_END,
)

log = logging.getLogger(__name__)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _chunk_dates(start: str, end: str, chunk_days: int = 180):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")
    while s < e:
        chunk_end = min(s + timedelta(days=chunk_days), e)
        yield s.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        s = chunk_end + timedelta(days=1)


def _ts(date_str: str) -> pd.Timestamp:
    """Convert YYYY-MM-DD to timezone-aware Timestamp for ENTSOE queries."""
    return pd.Timestamp(date_str, tz="Europe/Brussels")


def _resample_hourly(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Resample to hourly mean if data arrives at sub-hourly resolution."""
    if len(obj) < 2:
        return obj
    diffs = obj.index.to_series().diff().dropna()
    if diffs.min() < pd.Timedelta("1h"):
        if isinstance(obj, pd.Series):
            return obj.resample("h").mean()
        return obj.resample("h").mean()
    return obj


def _get(url: str, params: dict, retries: int = 3, pause: float = 1.0) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            log.warning("GET %s attempt %d/%d: %s", url, attempt + 1, retries, exc)
            if attempt < retries - 1:
                time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"Failed {url} after {retries} attempts")


@lru_cache(maxsize=1)
def _entsoe_client():
    from entsoe import EntsoePandasClient
    return EntsoePandasClient(api_key=ENTSOE_TOKEN)


# ── ENTSOE: Prices ────────────────────────────────────────────────────────────

def fetch_entsoe_prices(zone: str, start: str = TRAIN_START, end: str = TRAIN_END) -> pd.DataFrame:
    """Fetch day-ahead prices from ENTSOE Transparency Platform."""
    client = _entsoe_client()
    eic    = ENTSOE_ZONES[zone]
    frames = []

    for s, e in _chunk_dates(start, end, 180):
        log.info("ENTSOE prices %s  %s → %s", zone, s, e)
        try:
            series = client.query_day_ahead_prices(eic, start=_ts(s), end=_ts(e))
            series = _resample_hourly(series)
            series = series.tz_convert("UTC")
            df = series.reset_index()
            df.columns = ["timestamp", "price"]
            df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            frames.append(df)
            time.sleep(1)
        except Exception as exc:
            log.warning("ENTSOE prices %s %s→%s failed: %s", zone, s, e, exc)

    if not frames:
        raise RuntimeError(f"No ENTSOE price data for {zone}")
    out = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")
    out.to_csv(DATA_RAW / "entsoe" / f"prices_{zone}.csv", index=False)
    log.info("Saved entsoe/prices_%s.csv — %d rows", zone, len(out))
    return out


# ── ENTSOE: Generation + Load ─────────────────────────────────────────────────

def _parse_entsoe_generation(gen_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse ENTSOE MultiIndex generation columns into canonical features."""
    df = gen_df.copy()

    # Drop 'Actual Consumption' columns (pumped-storage load) — keep generation only
    if isinstance(df.columns, pd.MultiIndex):
        mask = df.columns.get_level_values(1) == "Actual Aggregated"
        df = df.loc[:, mask]
        df.columns = df.columns.get_level_values(0)

    wind_cols  = [c for c in df.columns if any(t in str(c) for t in ENTSOE_WIND_TYPES)]
    solar_cols = [c for c in df.columns if any(t in str(c) for t in ENTSOE_SOLAR_TYPES)]
    hydro_cols = [c for c in df.columns if any(t in str(c) for t in ENTSOE_HYDRO_TYPES)]

    out = pd.DataFrame(index=df.index)
    out["wind_generation"]  = df[wind_cols].sum(axis=1).clip(lower=0)  if wind_cols  else 0.0
    out["solar_generation"] = df[solar_cols].sum(axis=1).clip(lower=0) if solar_cols else 0.0
    out["hydro_generation"] = df[hydro_cols].sum(axis=1).clip(lower=0) if hydro_cols else 0.0
    return out


def _parse_entsoe_load(load_obj: Any) -> pd.Series:
    """Extract load Series from whatever query_load returns."""
    if isinstance(load_obj, pd.DataFrame):
        # Flatten MultiIndex or pick first column
        if isinstance(load_obj.columns, pd.MultiIndex):
            col = load_obj.columns[0]
            return load_obj[col].squeeze()
        return load_obj.iloc[:, 0].squeeze()
    return load_obj  # already a Series


def fetch_entsoe_generation(zone: str, start: str = TRAIN_START, end: str = TRAIN_END) -> pd.DataFrame:
    """Fetch actual generation per type + load from ENTSOE."""
    client = _entsoe_client()
    eic    = ENTSOE_ZONES[zone]
    frames = []

    for s, e in _chunk_dates(start, end, 180):
        log.info("ENTSOE generation %s  %s → %s", zone, s, e)
        try:
            gen_raw  = client.query_generation(eic, start=_ts(s), end=_ts(e))
            load_raw = client.query_load(eic, start=_ts(s), end=_ts(e))

            gen_raw  = _resample_hourly(gen_raw)
            load_raw = _resample_hourly(_parse_entsoe_load(load_raw))

            gen_raw  = gen_raw.tz_convert("UTC")
            load_raw = load_raw.tz_convert("UTC")

            gen_cols = _parse_entsoe_generation(gen_raw)
            gen_cols["load"] = load_raw.reindex(gen_cols.index)

            gen_cols = gen_cols.reset_index()
            gen_cols.rename(columns={gen_cols.columns[0]: "timestamp"}, inplace=True)
            gen_cols["timestamp"] = pd.to_datetime(gen_cols["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            frames.append(gen_cols)
            time.sleep(1.5)
        except Exception as exc:
            log.warning("ENTSOE generation %s %s→%s failed: %s", zone, s, e, exc)

    if not frames:
        raise RuntimeError(f"No ENTSOE generation data for {zone}")
    out = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")
    out.to_csv(DATA_RAW / "entsoe" / f"generation_{zone}.csv", index=False)
    log.info("Saved entsoe/generation_%s.csv — %d rows", zone, len(out))
    return out


# ── ENTSOE: Cross-border flows → net_imports ─────────────────────────────────

def fetch_entsoe_crossborder(zone: str, start: str = TRAIN_START, end: str = TRAIN_END) -> pd.DataFrame:
    """
    Fetch physical cross-border flows for all neighbor pairs and compute net imports.

    net_imports (MW) = sum over neighbors N of [flow(N→zone) − flow(zone→N)]
    Positive  → zone is a net importer (draws power from neighbors, upward price pressure)
    Negative  → zone is a net exporter (surplus production, downward price pressure)
    """
    client    = _entsoe_client()
    zone_eic  = ENTSOE_ZONES[zone]
    neighbors = ENTSOE_NEIGHBORS[zone]
    chunk_nets: list[pd.Series] = []

    for s, e in _chunk_dates(start, end, 180):
        ts_s, ts_e = _ts(s), _ts(e)
        net_chunk: pd.Series | None = None

        for nbr_eic in neighbors:
            try:
                exports = client.query_crossborder_flows(zone_eic, nbr_eic, start=ts_s, end=ts_e)
                imports = client.query_crossborder_flows(nbr_eic, zone_eic, start=ts_s, end=ts_e)

                exports = _resample_hourly(exports).tz_convert("UTC")
                imports = _resample_hourly(imports).tz_convert("UTC")

                idx = exports.index.union(imports.index)
                net = imports.reindex(idx, fill_value=0) - exports.reindex(idx, fill_value=0)

                net_chunk = net if net_chunk is None else net_chunk.add(net.reindex(net_chunk.index, fill_value=0), fill_value=0)
                log.info("  crossborder %s ↔ %s  %s→%s  ok", zone, nbr_eic[:8], s, e)
                time.sleep(0.5)
            except Exception as exc:
                log.warning("  crossborder %s ↔ %s  %s→%s  skipped: %s", zone, nbr_eic[:8], s, e, exc)

        if net_chunk is not None:
            chunk_nets.append(net_chunk)

    if not chunk_nets:
        raise RuntimeError(f"No cross-border flow data for {zone}")

    combined = pd.concat(chunk_nets).sort_index()
    combined = combined[~combined.index.duplicated()]
    df = combined.reset_index()
    df.columns = ["timestamp", "net_imports"]
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df.to_csv(DATA_RAW / "entsoe" / f"crossborder_{zone}.csv", index=False)
    log.info("Saved entsoe/crossborder_%s.csv — %d rows  mean=%.0f MW", zone, len(df), df["net_imports"].mean())
    return df


# ── Source 3: Open-Meteo weather (unchanged) ──────────────────────────────────

def fetch_weather(zone: str, start: str = TRAIN_START, end: str = TRAIN_END) -> pd.DataFrame:
    """Fetch hourly historical weather from Open-Meteo archive API (no key needed)."""
    loc    = WEATHER_LOCATIONS[zone]
    frames = []
    for s, e in _chunk_dates(start, end, 365):
        log.info("Fetching %s weather %s → %s (%s)", zone, s, e, loc["label"])
        d = _get("https://archive-api.open-meteo.com/v1/archive", {
            "latitude":        loc["latitude"],
            "longitude":       loc["longitude"],
            "start_date":      s,
            "end_date":        e,
            "hourly":          "temperature_2m,wind_speed_10m,shortwave_radiation",
            "timezone":        "UTC",
            "wind_speed_unit": "ms",
        })
        h = d.get("hourly", {})
        if not h.get("time"):
            continue
        frames.append(pd.DataFrame({
            "time":             h["time"],
            "temperature":      h["temperature_2m"],
            "wind_speed":       h["wind_speed_10m"],
            "solar_radiation":  h["shortwave_radiation"],
        }))
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
        "latitude":        loc["latitude"],
        "longitude":       loc["longitude"],
        "start_date":      start_date,
        "end_date":        end_date,
        "hourly":          "temperature_2m,wind_speed_10m,shortwave_radiation",
        "timezone":        "UTC",
        "wind_speed_unit": "ms",
    })
    h = d.get("hourly", {})
    return pd.DataFrame({
        "time":            h["time"],
        "temperature":     h["temperature_2m"],
        "wind_speed":      h["wind_speed_10m"],
        "solar_radiation": h["shortwave_radiation"],
    })


# ── Source 4: Fuel prices — yfinance (unchanged) ──────────────────────────────

def fetch_fuel_prices(start: str = TRAIN_START, end: str = TRAIN_END) -> pd.DataFrame:
    """Fetch TTF gas futures (EUR/MWh) and KRBN carbon ETF via yfinance."""
    from datetime import datetime, timedelta
    log.info("Fetching fuel prices %s → %s", start, end)
    end_dt = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    ttf  = yf.Ticker("TTF=F").history(start=start, end=end_dt, interval="1d")["Close"]
    ttf.name = "gas_price"
    krbn = yf.Ticker("KRBN").history(start=start, end=end_dt, interval="1d")["Close"]
    krbn.name = "carbon_price"

    df = pd.concat([ttf, krbn], axis=1)
    df.index = df.index.tz_localize(None)
    df.index.name = "date"
    df = df.reset_index()
    df.to_csv(DATA_RAW / "fuel" / "fuel_prices.csv", index=False)
    log.info("Saved fuel_prices.csv — %d rows", len(df))
    return df


# ── Fallback: energy-charts.info ─────────────────────────────────────────────
# Not called by default; retained for debugging / gap-filling.

def fetch_prices_ec(zone: str, start: str = TRAIN_START, end: str = TRAIN_END) -> pd.DataFrame:
    """Fallback: day-ahead prices from energy-charts.info (no API key, rate-limited)."""
    bzn    = EC_BZN[zone]
    frames = []
    for s, e in _chunk_dates(start, end, 180):
        log.info("EC prices %s %s → %s", zone, s, e)
        d = _get("https://api.energy-charts.info/price", {"bzn": bzn, "start": s, "end": e})
        if not d.get("unix_seconds"):
            continue
        frames.append(pd.DataFrame({"unix_seconds": d["unix_seconds"], "price": d["price"]}))
        time.sleep(0.5)
    if not frames:
        raise RuntimeError(f"No EC price data for {zone}")
    out = pd.concat(frames, ignore_index=True).drop_duplicates("unix_seconds")
    out.to_csv(DATA_RAW / "energycharts" / f"prices_{zone}.csv", index=False)
    return out


def fetch_generation_ec(zone: str, start: str = TRAIN_START, end: str = TRAIN_END) -> pd.DataFrame:
    """Fallback: generation + load from energy-charts.info."""
    country = EC_COUNTRY[zone]
    frames  = []
    for s, e in _chunk_dates(start, end, 180):
        log.info("EC generation %s %s → %s", zone, s, e)
        d = _get("https://api.energy-charts.info/public_power", {"country": country, "start": s, "end": e})
        if not d.get("unix_seconds"):
            continue
        row = {"unix_seconds": d["unix_seconds"]}
        for pt in d["production_types"]:
            row[pt["name"]] = pt["data"]
        frames.append(pd.DataFrame(row))
        time.sleep(0.5)
    if not frames:
        raise RuntimeError(f"No EC generation data for {zone}")
    out = pd.concat(frames, ignore_index=True).drop_duplicates("unix_seconds")
    out.to_csv(DATA_RAW / "energycharts" / f"generation_{zone}.csv", index=False)
    return out


# ── Orchestrator ──────────────────────────────────────────────────────────────

def ingest_all(start: str = TRAIN_START, end: str = TRAIN_END) -> None:
    """Full ingestion via ENTSOE (primary) + Open-Meteo + yfinance."""
    for zone in ZONES:
        log.info("━" * 56)
        log.info("ENTSOE ingestion: %s", zone)
        fetch_entsoe_prices(zone, start, end)
        fetch_entsoe_generation(zone, start, end)
        fetch_entsoe_crossborder(zone, start, end)
        fetch_weather(zone, start, end)

    fetch_fuel_prices(start, end)
    log.info("━" * 56)
    log.info("Ingestion complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ingest_all()
