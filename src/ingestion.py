"""
Stage 1 — Raw data ingestion.

Primary source: ENTSOE Transparency Platform (prices, load, generation, cross-border flows,
                neighbor prices, ES nuclear unavailability)
Supplementary: Open-Meteo (weather — multi-city, capacity-weighted, 100m wind)
               yfinance (fuel prices: TTF gas, coal, carbon)
Fallback:      energy-charts.info (retained as functions; not called by default)

Changes vs original:
  - Weather: 1 city per zone → 26 cities across 9 location groups (see config.py)
  - Wind height: wind_speed_10m → wind_speed_100m (turbine hub height)
  - Neighbor prices: new fetch_neighbor_prices() pulls FR, NL, CH, DK, PT day-ahead prices
  - Coal price: added API2 coal proxy via yfinance (MTF=F)
  - ES nuclear: new fetch_entsoe_nuclear() pulls unavailability notices → available_mw
  - ECMWF ensemble spread: new fetch_weather_ensemble() for uncertainty quantification
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
import numpy as np
import yfinance as yf

from config import (
    DATA_RAW, ZONES,
    EC_BZN, EC_COUNTRY,
    ENTSOE_TOKEN, ENTSOE_ZONES, ENTSOE_NEIGHBORS,
    ENTSOE_WIND_TYPES, ENTSOE_SOLAR_TYPES, ENTSOE_HYDRO_TYPES,
    ENTSOE_NUCLEAR_TYPES,
    WEATHER_LOCATIONS,
    WIND_TYPES, SOLAR_TYPES, HYDRO_TYPES, LOAD_TYPE,
    TRAIN_START, TRAIN_END,
    OPENMETEO_VARIABLES,
)

log = logging.getLogger(__name__)

# EIC codes for neighbor zones we want day-ahead prices for
NEIGHBOR_PRICE_EICS = {
    "FR":  "10YFR-RTE------C",
    "NL":  "10YNL----------L",
    "CH":  "10YCH-SWISSGRIDZ",
    "DK":  "10YDK-1--------W",
    "PT":  "10YPT-REN------W",
}


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


def _get(url: str, params: dict, retries: int = 3, pause: float = 1.0,
         api_key: str | None = None) -> dict:
    """Helper for robust GET requests with retries and optional API key."""
    query_params = params.copy()
    if api_key:
        query_params["apikey"] = api_key

    for attempt in range(retries):
        try:
            r = requests.get(url, params=query_params, timeout=60)
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

def fetch_entsoe_prices(zone: str, start: str = TRAIN_START,
                         end: str = TRAIN_END) -> pd.DataFrame:
    """Fetch day-ahead prices from ENTSOE Transparency Platform."""
    client = _entsoe_client()
    eic    = ENTSOE_ZONES[zone]
    frames = []

    for s, e in _chunk_dates(start, end, 180):
        log.info("ENTSOE prices %s  %s → %s", zone, s, e)
        try:
            series = client.query_day_ahead_prices(
                eic, start=_ts(s), end=_ts(e) + pd.Timedelta(days=1)
            )
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


# ── ENTSOE: Neighbor prices ───────────────────────────────────────────────────

def fetch_neighbor_prices(start: str = TRAIN_START,
                           end: str = TRAIN_END) -> pd.DataFrame:
    """
    Fetch day-ahead prices for key neighbor zones: FR, NL, CH, DK, PT.

    These are used as:
      - Cross-border price signal features (lag_24h)
      - Transmission spread proxies (zone_price - neighbor_price)
      - FR is used by both DE-LU and ES models
      - DK used by DE-LU only (Nordic wind export signal)
      - CH used by DE-LU only (alpine hydro arbitrage)
      - NL used by DE-LU only (gas hub, north wind correlation)
      - PT used by ES only (Iberian internal flows)
    """
    client = _entsoe_client()
    all_frames = {name: [] for name in NEIGHBOR_PRICE_EICS}

    for name, eic in NEIGHBOR_PRICE_EICS.items():
        for s, e in _chunk_dates(start, end, 180):
            log.info("ENTSOE neighbor prices %s  %s → %s", name, s, e)
            try:
                series = client.query_day_ahead_prices(
                    eic, start=_ts(s), end=_ts(e) + pd.Timedelta(days=1)
                )
                series = _resample_hourly(series).tz_convert("UTC")
                df = series.reset_index()
                df.columns = ["timestamp", f"{name}_price"]
                df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                all_frames[name].append(df)
                time.sleep(1)
            except Exception as exc:
                log.warning("ENTSOE neighbor %s %s→%s failed: %s", name, s, e, exc)

    # Merge all neighbors on timestamp
    merged = None
    for name, frames in all_frames.items():
        if not frames:
            log.warning("No data for neighbor %s — skipping", name)
            continue
        df = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")
        merged = df if merged is None else merged.merge(df, on="timestamp", how="outer")

    if merged is None:
        raise RuntimeError("No neighbor price data downloaded")

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged.to_csv(DATA_RAW / "entsoe" / "prices_neighbors.csv", index=False)
    log.info("Saved entsoe/prices_neighbors.csv — %d rows × %d cols",
             len(merged), len(merged.columns))
    return merged


# ── ENTSOE: Generation + Load ─────────────────────────────────────────────────

def _parse_entsoe_generation(gen_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse ENTSOE MultiIndex generation columns into canonical features."""
    df = gen_df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        mask = df.columns.get_level_values(1) == "Actual Aggregated"
        df = df.loc[:, mask]
        df.columns = df.columns.get_level_values(0)

    wind_cols    = [c for c in df.columns if any(t in str(c) for t in ENTSOE_WIND_TYPES)]
    solar_cols   = [c for c in df.columns if any(t in str(c) for t in ENTSOE_SOLAR_TYPES)]
    hydro_cols   = [c for c in df.columns if any(t in str(c) for t in ENTSOE_HYDRO_TYPES)]
    nuclear_cols = [c for c in df.columns if any(t in str(c) for t in ENTSOE_NUCLEAR_TYPES)]

    out = pd.DataFrame(index=df.index)
    out["wind_generation"]    = df[wind_cols].sum(axis=1).clip(lower=0)    if wind_cols    else 0.0
    out["solar_generation"]   = df[solar_cols].sum(axis=1).clip(lower=0)   if solar_cols   else 0.0
    out["hydro_generation"]   = df[hydro_cols].sum(axis=1).clip(lower=0)   if hydro_cols   else 0.0
    out["nuclear_generation"] = df[nuclear_cols].sum(axis=1).clip(lower=0) if nuclear_cols else 0.0
    return out


def _parse_entsoe_load(load_obj: Any) -> pd.Series:
    """Extract load Series from whatever query_load returns."""
    if isinstance(load_obj, pd.DataFrame):
        if isinstance(load_obj.columns, pd.MultiIndex):
            col = load_obj.columns[0]
            return load_obj[col].squeeze()
        return load_obj.iloc[:, 0].squeeze()
    return load_obj


def fetch_entsoe_generation(zone: str, start: str = TRAIN_START,
                             end: str = TRAIN_END) -> pd.DataFrame:
    """Fetch actual generation per type + load from ENTSOE."""
    client = _entsoe_client()
    eic    = ENTSOE_ZONES[zone]
    frames = []

    for s, e in _chunk_dates(start, end, 180):
        log.info("ENTSOE generation %s  %s → %s", zone, s, e)
        try:
            ts_end   = _ts(e) + pd.Timedelta(days=1)
            gen_raw  = client.query_generation(eic, start=_ts(s), end=ts_end)
            load_raw = client.query_load(eic, start=_ts(s), end=ts_end)

            gen_raw  = _resample_hourly(gen_raw)
            load_raw = _resample_hourly(_parse_entsoe_load(load_raw))
            gen_raw  = gen_raw.tz_convert("UTC")
            load_raw = load_raw.tz_convert("UTC")

            gen_cols = _parse_entsoe_generation(gen_raw)
            gen_cols["load"] = load_raw.reindex(gen_cols.index)

            gen_cols = gen_cols.reset_index()
            gen_cols.rename(columns={gen_cols.columns[0]: "timestamp"}, inplace=True)
            gen_cols["timestamp"] = pd.to_datetime(
                gen_cols["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
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


# ── ENTSOE: ES Nuclear Unavailability ─────────────────────────────────────────

def fetch_entsoe_nuclear(zone: str = "ES", start: str = TRAIN_START,
                          end: str = TRAIN_END) -> pd.DataFrame:
    """
    Fetch nuclear unavailability notices from ENTSOE REMIT for ES.

    Spain has ~7 GW of active nuclear capacity. Planned and unplanned outages
    shift the supply curve significantly. This function computes hourly
    available nuclear MW = installed_capacity - unavailable_MW.

    Only meaningful for ES (DE-LU nuclear decommissioned April 2023).
    """
    if zone != "ES":
        log.info("Nuclear unavailability only meaningful for ES — skipping %s", zone)
        return pd.DataFrame()

    client = _entsoe_client()
    eic    = ENTSOE_ZONES[zone]
    frames = []

    for s, e in _chunk_dates(start, end, 90):  # smaller chunks for REMIT data
        log.info("ENTSOE nuclear unavailability %s  %s → %s", zone, s, e)
        try:
            ts_end = _ts(e) + pd.Timedelta(days=1)
            unavail = client.query_unavailability_of_generation_units(
                eic, start=_ts(s), end=ts_end, docstatus=None
            )
            if unavail is not None and len(unavail) > 0:
                # Filter nuclear only
                if "businessType" in unavail.columns:
                    pass  # keep all, filter by plant type below
                if "productionType.name" in unavail.columns:
                    nuclear = unavail[
                        unavail["productionType.name"].str.contains(
                            "Nuclear", na=False, case=False
                        )
                    ]
                else:
                    nuclear = unavail  # fallback: use all if column missing

                frames.append(nuclear)
            time.sleep(2)
        except Exception as exc:
            log.warning("ENTSOE nuclear %s %s→%s failed: %s", zone, s, e, exc)

    if not frames:
        log.warning("No nuclear unavailability data for %s — using constant 7000 MW", zone)
        # Return constant 7GW available as fallback
        idx = pd.date_range(start=start, end=end, freq="h", tz="UTC")
        out = pd.DataFrame({"timestamp": idx.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "nuclear_available_mw": 7000.0})
        out.to_csv(DATA_RAW / "entsoe" / f"nuclear_{zone}.csv", index=False)
        return out

    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(DATA_RAW / "entsoe" / f"nuclear_raw_{zone}.csv", index=False)
    log.info("Saved nuclear_raw_%s.csv — %d rows", zone, len(raw))

    # Aggregate to hourly unavailable MW
    # ES installed nuclear capacity ~7,000 MW (as of 2024)
    ES_NUCLEAR_INSTALLED_MW = 7117.0

    try:
        # Build hourly series of unavailable MW
        idx = pd.date_range(
            start=pd.Timestamp(start, tz="UTC"),
            end=pd.Timestamp(end, tz="UTC"),
            freq="h"
        )
        unavail_series = pd.Series(0.0, index=idx)

        for _, row in raw.iterrows():
            try:
                t_start = pd.Timestamp(row.get("start", row.get("Start", None)), tz="UTC")
                t_end   = pd.Timestamp(row.get("end",   row.get("End",   None)), tz="UTC")
                mw      = float(row.get("unavailableMW", row.get("Unavailable Capacity", 0)))
                mask    = (idx >= t_start) & (idx < t_end)
                unavail_series[mask] += mw
            except Exception:
                continue

        available = (ES_NUCLEAR_INSTALLED_MW - unavail_series).clip(lower=0)
        out = pd.DataFrame({
            "timestamp":           available.index.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "nuclear_available_mw": available.values,
        })
    except Exception as exc:
        log.warning("Nuclear aggregation failed (%s) — using constant 7000 MW", exc)
        idx = pd.date_range(start=start, end=end, freq="h", tz="UTC")
        out = pd.DataFrame({"timestamp": idx.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "nuclear_available_mw": 7000.0})

    out.to_csv(DATA_RAW / "entsoe" / f"nuclear_{zone}.csv", index=False)
    log.info("Saved nuclear_%s.csv — %d rows", zone, len(out))
    return out


# ── ENTSOE: Cross-border flows → net_imports ──────────────────────────────────

def fetch_entsoe_crossborder(zone: str, start: str = TRAIN_START,
                              end: str = TRAIN_END) -> pd.DataFrame:
    """
    Fetch physical cross-border flows for all neighbor pairs and compute net imports.
    net_imports (MW) = sum over neighbors N of [flow(N→zone) − flow(zone→N)]
    """
    client    = _entsoe_client()
    eic       = ENTSOE_ZONES[zone]
    neighbors = ENTSOE_NEIGHBORS[zone]
    frames    = []

    for s, e in _chunk_dates(start, end, 180):
        log.info("ENTSOE cross-border %s  %s → %s", zone, s, e)
        net = None
        ts_s = _ts(s)
        ts_e = _ts(e) + pd.Timedelta(days=1)

        for nb_eic in neighbors:
            try:
                exp = client.query_crossborder_flows(eic, nb_eic, start=ts_s, end=ts_e)
                imp = client.query_crossborder_flows(nb_eic, eic, start=ts_s, end=ts_e)
                exp = _resample_hourly(exp).tz_convert("UTC")
                imp = _resample_hourly(imp).tz_convert("UTC")
                flow = imp.reindex(exp.index, fill_value=0) - exp
                net  = flow if net is None else net.add(flow, fill_value=0)
                time.sleep(0.5)
            except Exception as exc:
                log.warning("  Cross-border %s↔%s %s: %s", zone, nb_eic, s, exc)

        if net is not None:
            df = net.reset_index()
            df.columns = ["timestamp", "net_imports"]
            df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            frames.append(df)

    if not frames:
        log.warning("No cross-border data for %s", zone)
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")
    out.to_csv(DATA_RAW / "entsoe" / f"crossborder_{zone}.csv", index=False)
    log.info("Saved crossborder_%s.csv — %d rows", zone, len(out))
    return out


# ── ENTSOE: Day-ahead forecasts ───────────────────────────────────────────────

def fetch_entsoe_forecasts(zone: str, start: str = TRAIN_START,
                            end: str = TRAIN_END) -> pd.DataFrame:
    """Fetch day-ahead load + wind/solar generation forecasts from ENTSOE."""
    client = _entsoe_client()
    eic    = ENTSOE_ZONES[zone]
    frames = []

    for s, e in _chunk_dates(start, end, 180):
        log.info("ENTSOE forecasts %s  %s → %s", zone, s, e)
        ts_s = _ts(s)
        ts_e = _ts(e) + pd.Timedelta(days=1)
        chunk_data: dict[str, pd.Series] = {}

        try:
            lf = client.query_load_forecast(eic, start=ts_s, end=ts_e)
            if isinstance(lf, pd.DataFrame):
                lf = lf.iloc[:, 0]
            lf = _resample_hourly(lf).tz_convert("UTC")
            chunk_data["load_forecast"] = lf
        except Exception as exc:
            log.warning("  Load forecast %s %s: %s", zone, s, exc)

        try:
            ws = client.query_wind_and_solar_forecast(eic, start=ts_s, end=ts_e)
            ws = _resample_hourly(ws).tz_convert("UTC")
            if isinstance(ws.columns, pd.MultiIndex):
                ws.columns = ws.columns.get_level_values(0)
            wind_cols  = [c for c in ws.columns if any(t in str(c) for t in ENTSOE_WIND_TYPES)]
            solar_cols = [c for c in ws.columns if any(t in str(c) for t in ENTSOE_SOLAR_TYPES)]
            if wind_cols:
                chunk_data["wind_generation_forecast"]  = ws[wind_cols].sum(axis=1).clip(lower=0)
            if solar_cols:
                chunk_data["solar_generation_forecast"] = ws[solar_cols].sum(axis=1).clip(lower=0)
        except Exception as exc:
            log.warning("  Wind/solar forecast %s %s: %s", zone, s, exc)

        if chunk_data:
            chunk_df = pd.DataFrame(chunk_data)
            chunk_df = chunk_df.reset_index()
            chunk_df.rename(columns={chunk_df.columns[0]: "timestamp"}, inplace=True)
            chunk_df["timestamp"] = pd.to_datetime(
                chunk_df["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            frames.append(chunk_df)
        time.sleep(1.5)

    if not frames:
        log.warning("No ENTSOE forecast data for %s — skipping", zone)
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")
    out.to_csv(DATA_RAW / "entsoe" / f"forecasts_{zone}.csv", index=False)
    log.info("Saved forecasts_%s.csv — %d rows", zone, len(out))
    return out


# ── Open-Meteo: Multi-city weather ───────────────────────────────────────────

def _fetch_single_location_history(label: str, lat: float, lon: float,
                                    variables: list[str],
                                    start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical weather for a single coordinate from Open-Meteo archive API.
    Returns DataFrame with timestamp + requested variable columns.
    """
    frames = []
    for s, e in _chunk_dates(start, end, 365):
        log.info("  Open-Meteo history %s (%s→%s)", label, s, e)
        d = _get("https://archive-api.open-meteo.com/v1/archive", {
            "latitude":        lat,
            "longitude":       lon,
            "start_date":      s,
            "end_date":        e,
            "hourly":          ",".join(variables),
            "timezone":        "UTC",
            "wind_speed_unit": "ms",
        })
        h = d.get("hourly", {})
        if not h.get("time"):
            continue
        df = pd.DataFrame({"time": h["time"]})
        for v in variables:
            df[v] = h.get(v, np.nan)
        frames.append(df)
        time.sleep(0.5)

    if not frames:
        log.warning("No weather history for %s", label)
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("time")


def _fetch_single_location_forecast(label: str, lat: float, lon: float,
                                     variables: list[str],
                                     start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch weather forecast for a single coordinate from Open-Meteo forecast API.
    Used for May 11-12 prediction window.
    """
    log.info("  Open-Meteo forecast %s (%s→%s)", label, start_date, end_date)
    d = _get("https://api.open-meteo.com/v1/forecast", {
        "latitude":        lat,
        "longitude":       lon,
        "start_date":      start_date,
        "end_date":        end_date,
        "hourly":          ",".join(variables),
        "timezone":        "UTC",
        "wind_speed_unit": "ms",
    })
    h = d.get("hourly", {})
    if not h.get("time"):
        return pd.DataFrame()
    df = pd.DataFrame({"time": h["time"]})
    for v in variables:
        df[v] = h.get(v, np.nan)
    return df


def fetch_weather(zone: str, start: str = TRAIN_START,
                   end: str = TRAIN_END) -> pd.DataFrame:
    """
    Fetch multi-city historical weather and compute capacity/population-weighted
    aggregates per zone.

    Uses WEATHER_LOCATIONS from config.py which defines 9 location groups:
      DE_wind, DE_solar, DE_demand, DK_wind, CH_hydro   (for DE-LU model)
      ES_wind, ES_solar, ES_demand, ES_hydro             (for ES model)

    Key improvement over original:
      - 1 city per zone → up to 15 cities (DE-LU) / 11 cities (ES)
      - wind_speed_10m → wind_speed_100m (turbine hub height)
      - Capacity-weighted aggregation for generation sites
      - Population-weighted aggregation for demand sites

    Output columns (zone-prefixed, already aggregated):
      DE_wind_speed, DE_solar_radiation, DE_temperature
      DK_wind_speed, CH_precipitation, CH_temperature
      ES_wind_speed, ES_solar_radiation, ES_temperature, ES_hydro_precipitation
    """
    from config import WEATHER_LOCATIONS, OPENMETEO_VARIABLES

    # Determine which location groups belong to this zone's model
    if zone == "DE-LU":
        groups = ["DE_wind", "DE_solar", "DE_demand", "DK_wind", "CH_hydro"]
    else:  # ES
        groups = ["ES_wind", "ES_solar", "ES_demand", "ES_hydro"]

    aggregated = {}

    for group in groups:
        locations = WEATHER_LOCATIONS.get(group, [])
        variables = OPENMETEO_VARIABLES.get(group, [])
        if not locations or not variables:
            continue

        group_frames = []
        weights      = []

        for loc in locations:
            df = _fetch_single_location_history(
                label=loc["label"],
                lat=loc["latitude"],
                lon=loc["longitude"],
                variables=variables,
                start=start,
                end=end,
            )
            if df.empty:
                continue
            df = df.set_index("time")
            group_frames.append(df)
            weights.append(loc["weight"])

        if not group_frames:
            log.warning("No data for location group %s", group)
            continue

        # Normalize weights in case some locations failed
        total_w = sum(weights)
        norm_w  = [w / total_w for w in weights]

        # Weighted average across locations
        weighted = sum(df * w for df, w in zip(group_frames, norm_w))

        # Rename columns with group prefix
        for col in weighted.columns:
            col_name = f"{group}_{col}"
            aggregated[col_name] = weighted[col]

    if not aggregated:
        raise RuntimeError(f"No weather data for {zone}")

    out = pd.DataFrame(aggregated)
    out.index.name = "time"
    out = out.reset_index()
    out.to_csv(DATA_RAW / "openmeteo" / f"weather_{zone}.csv", index=False)
    log.info("Saved weather_%s.csv — %d rows × %d cols", zone, len(out), len(out.columns))
    return out


def fetch_weather_forecast(zone: str, start_date: str,
                            end_date: str) -> pd.DataFrame:
    """
    Fetch multi-city weather forecast for the evaluation window (May 11-12).
    Mirrors fetch_weather() but uses forecast API instead of archive API.
    """
    from config import WEATHER_LOCATIONS, OPENMETEO_VARIABLES

    if zone == "DE-LU":
        groups = ["DE_wind", "DE_solar", "DE_demand", "DK_wind", "CH_hydro"]
    else:
        groups = ["ES_wind", "ES_solar", "ES_demand", "ES_hydro"]

    aggregated = {}

    for group in groups:
        locations = WEATHER_LOCATIONS.get(group, [])
        variables = OPENMETEO_VARIABLES.get(group, [])
        if not locations or not variables:
            continue

        group_frames = []
        weights      = []

        for loc in locations:
            df = _fetch_single_location_forecast(
                label=loc["label"],
                lat=loc["latitude"],
                lon=loc["longitude"],
                variables=variables,
                start_date=start_date,
                end_date=end_date,
            )
            if df.empty:
                continue
            df = df.set_index("time")
            group_frames.append(df)
            weights.append(loc["weight"])

        if not group_frames:
            continue

        total_w = sum(weights)
        norm_w  = [w / total_w for w in weights]
        weighted = sum(df * w for df, w in zip(group_frames, norm_w))

        for col in weighted.columns:
            aggregated[f"{group}_{col}"] = weighted[col]

    out = pd.DataFrame(aggregated)
    out.index.name = "time"
    out = out.reset_index()
    return out


def fetch_weather_ensemble(zone: str, start_date: str,
                            end_date: str) -> pd.DataFrame:
    """
    Fetch ECMWF ensemble weather forecast spread for uncertainty quantification.

    The spread between ensemble members (std across 51 ECMWF runs) is used to
    dynamically widen/narrow p025-p975 prediction intervals:
      - Wide spread → high meteorological uncertainty → wider intervals
      - Narrow spread → confident forecast → tighter intervals

    Uses the primary location only (not all 26 cities) for efficiency.
    Returns ensemble_wind_std and ensemble_solar_std per hour.
    """
    if zone == "DE-LU":
        loc = {"latitude": 54.52, "longitude": 9.55, "label": "Schleswig"}
        variables = ["wind_speed_100m", "shortwave_radiation"]
    else:
        loc = {"latitude": 37.39, "longitude": -5.99, "label": "Seville"}
        variables = ["wind_speed_100m", "shortwave_radiation"]

    log.info("Open-Meteo ECMWF ensemble %s (%s→%s)", zone, start_date, end_date)

    try:
        d = _get("https://ensemble-api.open-meteo.com/v1/ensemble", {
            "latitude":        loc["latitude"],
            "longitude":       loc["longitude"],
            "start_date":      start_date,
            "end_date":        end_date,
            "hourly":          ",".join(variables),
            "models":          "ecmwf_ifs025",
            "timezone":        "UTC",
            "wind_speed_unit": "ms",
        })
    except Exception as exc:
        log.warning("ECMWF ensemble fetch failed: %s — returning empty", exc)
        return pd.DataFrame()

    h = d.get("hourly", {})
    if not h.get("time"):
        return pd.DataFrame()

    times = h["time"]
    result = pd.DataFrame({"time": times})

    # Each variable comes as multiple members: wind_speed_100m_member01, etc.
    for var in variables:
        member_cols = [k for k in h.keys() if k.startswith(var) and "member" in k]
        if member_cols:
            member_data = np.array([h[c] for c in member_cols])
            result[f"{var}_ensemble_mean"] = np.nanmean(member_data, axis=0)
            result[f"{var}_ensemble_std"]  = np.nanstd(member_data,  axis=0)

    result.to_csv(DATA_RAW / "openmeteo" / f"ensemble_{zone}.csv", index=False)
    log.info("Saved ensemble_%s.csv — %d rows", zone, len(result))
    return result


# ── Fuel prices ───────────────────────────────────────────────────────────────

def fetch_fuel_prices(start: str = TRAIN_START, end: str = TRAIN_END) -> pd.DataFrame:
    """
    Fetch fuel prices via yfinance:
      TTF=F   — Dutch TTF natural gas futures (EUR/MWh proxy)
      MTF=F   — Coal futures (API2 proxy) — NEW: relevant for DE-LU coal peakers
      KRBN    — KraneShares Global Carbon ETF (EU ETS proxy)

    All daily frequency — forward-filled to hourly in feature engineering.

    Note on TTF=F: yfinance returns USD/MMBtu. We keep raw units and let the
    model learn the scaling — adding a unit-conversion here would break if
    yfinance changes the contract.
    """
    end_dt = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    log.info("Fetching fuel prices %s → %s", start, end)

    results = {}

    # TTF gas
    try:
        ttf = yf.Ticker("TTF=F").history(start=start, end=end_dt, interval="1d")["Close"]
        ttf.name = "gas_price"
        results["gas_price"] = ttf
        log.info("TTF gas: %d rows", len(ttf))
    except Exception as exc:
        log.warning("TTF fetch failed: %s", exc)

    # Coal (API2 proxy via Rotterdam futures)
    try:
        coal = yf.Ticker("MTF=F").history(start=start, end=end_dt, interval="1d")["Close"]
        coal.name = "coal_price"
        results["coal_price"] = coal
        log.info("Coal: %d rows", len(coal))
    except Exception as exc:
        log.warning("Coal fetch failed (MTF=F): %s — trying XAD=F", exc)
        try:
            coal2 = yf.Ticker("XAD=F").history(start=start, end=end_dt, interval="1d")["Close"]
            coal2.name = "coal_price"
            results["coal_price"] = coal2
        except Exception as exc2:
            log.warning("Coal fallback also failed: %s — coal_price will be missing", exc2)

    # Carbon (EU ETS proxy via KRBN ETF)
    try:
        krbn = yf.Ticker("KRBN").history(start=start, end=end_dt, interval="1d")["Close"]
        krbn.name = "carbon_price"
        results["carbon_price"] = krbn
        log.info("KRBN carbon: %d rows", len(krbn))
    except Exception as exc:
        log.warning("KRBN fetch failed: %s", exc)

    if not results:
        raise RuntimeError("All fuel price fetches failed")

    df = pd.concat(results.values(), axis=1)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df = df.reset_index()
    df.to_csv(DATA_RAW / "fuel" / "fuel_prices.csv", index=False)
    log.info("Saved fuel_prices.csv — %d rows × %d cols", len(df), len(df.columns))
    return df


# ── Fallback: energy-charts.info ─────────────────────────────────────────────

def fetch_prices_ec(zone: str, start: str = TRAIN_START,
                     end: str = TRAIN_END) -> pd.DataFrame:
    """Fallback: day-ahead prices from energy-charts.info (no API key, rate-limited)."""
    bzn    = EC_BZN[zone]
    frames = []
    for s, e in _chunk_dates(start, end, 180):
        log.info("EC prices %s %s → %s", zone, s, e)
        d = _get("https://api.energy-charts.info/price", {"bzn": bzn, "start": s, "end": e})
        if not d.get("unix_seconds"):
            continue
        frames.append(pd.DataFrame({
            "unix_seconds": d["unix_seconds"],
            "price":        d["price"],
        }))
        time.sleep(0.5)
    if not frames:
        raise RuntimeError(f"No EC price data for {zone}")
    out = pd.concat(frames, ignore_index=True).drop_duplicates("unix_seconds")
    out.to_csv(DATA_RAW / "energycharts" / f"prices_{zone}.csv", index=False)
    return out


def fetch_generation_ec(zone: str, start: str = TRAIN_START,
                         end: str = TRAIN_END) -> pd.DataFrame:
    """Fallback: generation + load from energy-charts.info."""
    country = EC_COUNTRY[zone]
    frames  = []
    for s, e in _chunk_dates(start, end, 180):
        log.info("EC generation %s %s → %s", zone, s, e)
        d = _get("https://api.energy-charts.info/public_power",
                 {"country": country, "start": s, "end": e})
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
    """
    Full ingestion pipeline:
      1. ENTSOE prices (DE-LU, ES)
      2. ENTSOE generation + load (DE-LU, ES)
      3. ENTSOE cross-border flows (DE-LU, ES)
      4. ENTSOE day-ahead forecasts (DE-LU, ES)
      5. ENTSOE neighbor prices (FR, NL, CH, DK, PT)    ← NEW
      6. ENTSOE ES nuclear unavailability               ← NEW
      7. Open-Meteo multi-city weather (DE-LU, ES)      ← UPDATED (26 cities, 100m wind)
      8. Fuel prices: TTF gas + coal + carbon           ← UPDATED (added coal)
    """
    for zone in ZONES:
        log.info("━" * 60)
        log.info("ENTSOE ingestion: %s", zone)
        fetch_entsoe_prices(zone, start, end)
        fetch_entsoe_generation(zone, start, end)
        fetch_entsoe_crossborder(zone, start, end)
        fetch_entsoe_forecasts(zone, start, end)

        if zone == "ES":
            fetch_entsoe_nuclear(zone, start, end)

        log.info("Open-Meteo weather: %s (multi-city)", zone)
        fetch_weather(zone, start, end)

    log.info("━" * 60)
    log.info("Neighbor prices: FR, NL, CH, DK, PT")
    fetch_neighbor_prices(start, end)

    log.info("Fuel prices: TTF + coal + carbon")
    fetch_fuel_prices(start, end)

    log.info("━" * 60)
    log.info("Ingestion complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    ingest_all()
