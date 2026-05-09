"""
Stage 1 — Raw data ingestion (long-term pipeline).

Fetches only what the merit-order / LRMC model needs:
  - ENTSOE installed capacity per production type   (annual, per zone)
  - ENTSOE day-ahead prices                         (hourly → resampled monthly)
  - ENTSOE actual load                              (hourly → resampled annual)
  - ENTSOE actual generation per type               (hourly → resampled annual)
  - yfinance fuel prices: TTF gas, EUA proxy, coal API2, Brent oil

Writes raw CSVs to data/raw/{entsoe,fuel}/.

Design notes:
  - Hourly data is fetched at hourly granularity (so we can choose resampling
    later in cleaning), but stored only at the granularity we actually use:
      prices  -> monthly mean  (calibration target)
      load    -> annual sum    (demand projection)
      gen     -> annual sum per tech (capacity-factor estimation)
    This keeps raw/ small and the pipeline fast.
  - No weather, no cross-border flows, no graph-context zones.
"""

from __future__ import annotations

import time
import signal
import logging
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

import pandas as pd
import yfinance as yf

from config import (
    DATA_RAW,
    TARGET_ZONES, ENTSOE_ZONES, ENTSOE_TOKEN,
    HISTORY_START, HISTORY_END,
    FUEL_TICKERS, GPR_URL,
)

log = logging.getLogger(__name__)


# ── Helpers (adapted from team_repo/src/ingestion.py) ─────────────────────────

def _chunk_dates(start: str, end: str, chunk_days: int = 180):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")
    while s < e:
        chunk_end = min(s + timedelta(days=chunk_days), e)
        yield s.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        s = chunk_end + timedelta(days=1)


def _ts(date_str: str) -> pd.Timestamp:
    return pd.Timestamp(date_str, tz="Europe/Brussels")


@lru_cache(maxsize=1)
def _entsoe_client():
    from entsoe import EntsoePandasClient
    return EntsoePandasClient(api_key=ENTSOE_TOKEN)


class _TimeoutError(Exception):
    pass


def _with_timeout(fn, timeout_s: int, *args, **kwargs):
    def _handler(signum, frame):
        raise _TimeoutError(f"call exceeded {timeout_s}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_s)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _retry(fn, *args, retries: int = 4, base_sleep: float = 4.0, timeout_s: int = 90, **kwargs):
    last_exc = None
    for attempt in range(retries):
        try:
            return _with_timeout(fn, timeout_s, *args, **kwargs)
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            transient = ("503" in msg or "502" in msg or "504" in msg
                         or "timeout" in msg.lower() or "Service Unavailable" in msg
                         or isinstance(exc, _TimeoutError))
            if attempt < retries - 1 and transient:
                wait = base_sleep * (2 ** attempt)
                log.warning("  transient %s — sleeping %.1fs (attempt %d/%d)",
                            msg[:80], wait, attempt + 1, retries)
                time.sleep(wait)
                continue
            raise
    raise last_exc


# ── ENTSOE: Installed capacity per production type ───────────────────────────

def fetch_installed_capacity(zone: str,
                             start: str = HISTORY_START,
                             end: str = HISTORY_END) -> pd.DataFrame:
    """
    Annual installed capacity per PSR type.

    Returns long-format DataFrame with columns: [year, psr_type, capacity_mw, zone].
    ENTSOE returns one row per year-start; we fetch year-by-year and concat.
    """
    client = _entsoe_client()
    eic    = ENTSOE_ZONES[zone]

    s_year = pd.Timestamp(start).year
    e_year = pd.Timestamp(end).year
    rows = []
    for year in range(s_year, e_year + 1):
        log.info("ENTSOE installed capacity %s %d", zone, year)
        try:
            df = _retry(
                client.query_installed_generation_capacity,
                eic,
                start=pd.Timestamp(f"{year}-01-01", tz="Europe/Brussels"),
                end=pd.Timestamp(f"{year}-12-31", tz="Europe/Brussels"),
            )
            # entsoe-py returns DataFrame indexed by timestamp, columns = PSR types.
            if df is None or df.empty:
                log.warning("  empty capacity for %s %d", zone, year)
                continue
            # Take the first (year-start) snapshot.
            snap = df.iloc[0]
            for psr_type, mw in snap.items():
                if pd.notna(mw) and mw > 0:
                    rows.append({
                        "year": year,
                        "psr_type": psr_type,
                        "capacity_mw": float(mw),
                        "zone": zone,
                    })
            time.sleep(1)
        except Exception as exc:
            log.warning("  failed %s %d: %s", zone, year, exc)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"No installed-capacity data fetched for {zone}")
    out.to_csv(DATA_RAW / "entsoe" / f"installed_capacity_{zone}.csv", index=False)
    log.info("Saved installed_capacity_%s.csv — %d rows", zone, len(out))
    return out


# ── ENTSOE: Day-ahead prices (hourly → stored monthly) ───────────────────────

def fetch_da_prices(zone: str,
                    start: str = HISTORY_START,
                    end: str = HISTORY_END) -> pd.DataFrame:
    """Fetch hourly DA prices, resample to monthly mean. Used for calibration."""
    client = _entsoe_client()
    eic    = ENTSOE_ZONES[zone]
    frames = []

    for s, e in _chunk_dates(start, end, 180):
        log.info("ENTSOE prices %s  %s → %s", zone, s, e)
        try:
            series = _retry(
                client.query_day_ahead_prices,
                eic,
                start=_ts(s),
                end=_ts(e) + pd.Timedelta(days=1),
            )
            series = series.tz_convert("UTC")
            frames.append(series)
            time.sleep(1)
        except Exception as exc:
            msg = repr(exc) if not str(exc) else str(exc)
            log.warning("  no data %s %s→%s: %s", zone, s, e, msg)

    if not frames:
        raise RuntimeError(f"No DA-price data for {zone}")
    full = pd.concat(frames).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    # Resample to monthly mean
    monthly = full.resample("MS").mean().to_frame("price_eur_per_mwh")
    monthly["zone"] = zone
    monthly.index.name = "month"
    monthly = monthly.reset_index()
    monthly.to_csv(DATA_RAW / "entsoe" / f"da_prices_monthly_{zone}.csv", index=False)
    log.info("Saved da_prices_monthly_%s.csv — %d rows", zone, len(monthly))
    return monthly


# ── ENTSOE: Load (hourly → stored annual sum) ────────────────────────────────

def fetch_load_annual(zone: str,
                      start: str = HISTORY_START,
                      end: str = HISTORY_END) -> pd.DataFrame:
    """Fetch hourly load, sum to annual TWh."""
    client = _entsoe_client()
    eic    = ENTSOE_ZONES[zone]
    frames = []

    for s, e in _chunk_dates(start, end, 180):
        log.info("ENTSOE load %s  %s → %s", zone, s, e)
        try:
            obj = _retry(
                client.query_load,
                eic,
                start=_ts(s),
                end=_ts(e) + pd.Timedelta(days=1),
            )
            # entsoe-py returns DataFrame with one column 'Actual Load'
            if isinstance(obj, pd.DataFrame):
                series = obj.iloc[:, 0]
            else:
                series = obj
            series = series.tz_convert("UTC")
            frames.append(series)
            time.sleep(1)
        except Exception as exc:
            msg = repr(exc) if not str(exc) else str(exc)
            log.warning("  no data %s %s→%s: %s", zone, s, e, msg)

    if not frames:
        raise RuntimeError(f"No load data for {zone}")
    full = pd.concat(frames).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    # ENTSOE returns 15-min for some zones; resample to hourly mean first so
    # the sum is granularity-agnostic.  Hourly MW * 1h = MWh; sum per year, /1e6 -> TWh.
    full = full.resample("h").mean()
    annual = full.resample("YS").sum() / 1e6   # MWh -> TWh
    annual = annual.to_frame("demand_twh")
    annual["zone"] = zone
    annual["year"] = annual.index.year
    annual = annual.reset_index(drop=True)[["year", "zone", "demand_twh"]]
    # Drop spurious year-boundary rows: Brussels-tz start (e.g. 2018-01-01 00:00 CET
    # = 2017-12-31 23:00 UTC) leaks one hour into the prior year, creating a tiny
    # ghost row in the resample. Keep only years inside the requested range.
    s_year, e_year = pd.Timestamp(start).year, pd.Timestamp(end).year
    annual = annual[(annual["year"] >= s_year) & (annual["year"] <= e_year)]
    annual.to_csv(DATA_RAW / "entsoe" / f"load_annual_{zone}.csv", index=False)
    log.info("Saved load_annual_%s.csv — %d rows", zone, len(annual))
    return annual


# ── ENTSOE: Actual generation per type (hourly → annual sum) ─────────────────

def fetch_generation_annual(zone: str,
                            start: str = HISTORY_START,
                            end: str = HISTORY_END) -> pd.DataFrame:
    """Fetch hourly actual generation per PSR type, sum to annual TWh per type."""
    client = _entsoe_client()
    eic    = ENTSOE_ZONES[zone]
    frames = []

    for s, e in _chunk_dates(start, end, 180):
        log.info("ENTSOE generation %s  %s → %s", zone, s, e)
        try:
            df = _retry(
                client.query_generation,
                eic,
                start=_ts(s),
                end=_ts(e) + pd.Timedelta(days=1),
                psr_type=None,
            )
            if df is None or df.empty:
                continue
            # Drop pumped-storage 'Actual Consumption' if multiindex
            if isinstance(df.columns, pd.MultiIndex):
                mask = df.columns.get_level_values(1) == "Actual Aggregated"
                df = df.loc[:, mask]
                df.columns = df.columns.get_level_values(0)
            df = df.tz_convert("UTC")
            frames.append(df)
            time.sleep(1)
        except Exception as exc:
            msg = repr(exc) if not str(exc) else str(exc)
            log.warning("  no data %s %s→%s: %s", zone, s, e, msg)

    if not frames:
        raise RuntimeError(f"No generation data for {zone}")
    full = pd.concat(frames).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    # Resample 15-min -> hourly mean first (some zones use 15-min granularity).
    full = full.resample("h").mean()
    # Annual sum per PSR type. Hourly MW -> MWh -> TWh
    annual = full.resample("YS").sum() / 1e6
    annual = annual.reset_index().melt(
        id_vars=annual.index.name or "index",
        var_name="psr_type",
        value_name="generation_twh",
    )
    # First column post-reset will be the timestamp; rename robustly
    ts_col = annual.columns[0]
    annual = annual.rename(columns={ts_col: "year_start"})
    annual["year"] = pd.to_datetime(annual["year_start"]).dt.year
    annual["zone"] = zone
    annual = annual[["year", "zone", "psr_type", "generation_twh"]]
    annual = annual[annual["generation_twh"].fillna(0) > 0]
    # Drop spurious year-boundary rows (see fetch_load_annual).
    s_year, e_year = pd.Timestamp(start).year, pd.Timestamp(end).year
    annual = annual[(annual["year"] >= s_year) & (annual["year"] <= e_year)]
    annual.to_csv(DATA_RAW / "entsoe" / f"generation_annual_{zone}.csv", index=False)
    log.info("Saved generation_annual_%s.csv — %d rows", zone, len(annual))
    return annual


# ── GPR (geopolitical-risk index, Iacoviello) ────────────────────────────────

def fetch_gpr(start: str = HISTORY_START, end: str = HISTORY_END) -> pd.DataFrame:
    """Download Iacoviello's monthly Geopolitical Risk index, save monthly CSV."""
    import requests, io
    log.info("GPR %s", GPR_URL)
    r = requests.get(GPR_URL, timeout=60)
    r.raise_for_status()
    # .xls (legacy Excel) — needs xlrd
    df = pd.read_excel(io.BytesIO(r.content), sheet_name=0)
    # Keep only the headline columns we'll use
    keep = ["month", "GPR", "GPRT", "GPRA"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["month"])
    df = df[(df["month"] >= pd.Timestamp(start)) & (df["month"] <= pd.Timestamp(end))]
    df.to_csv(DATA_RAW / "fuel" / "gpr.csv", index=False)
    log.info("Saved gpr.csv — %d rows", len(df))
    return df


# ── yfinance: Fuel prices ────────────────────────────────────────────────────

def fetch_fuel_prices(start: str = HISTORY_START, end: str = HISTORY_END) -> dict:
    """Daily fuel/carbon prices from yfinance. Saves one CSV per series."""
    out = {}
    for name, ticker in FUEL_TICKERS.items():
        log.info("yfinance %s (%s)", name, ticker)
        try:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=False)
            if df is None or df.empty:
                log.warning("  empty %s", ticker)
                continue
            # Flatten any MultiIndex returned by yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            close = df["Close"].rename(name)
            close.index = close.index.tz_localize("UTC") if close.index.tz is None else close.index
            close.to_frame().reset_index().rename(columns={"Date": "date"}).to_csv(
                DATA_RAW / "fuel" / f"{name}.csv", index=False
            )
            out[name] = close
        except Exception as exc:
            log.warning("  failed %s: %s", ticker, exc)
    return out


# ── Orchestrator ─────────────────────────────────────────────────────────────

def ingest_all():
    """Run all ingestion steps for both target zones."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("=== Long-term ingestion: %s → %s ===", HISTORY_START, HISTORY_END)
    log.info("Zones: %s", TARGET_ZONES)

    for zone in TARGET_ZONES:
        log.info("--- %s ---", zone)
        fetch_installed_capacity(zone)
        fetch_da_prices(zone)
        fetch_load_annual(zone)
        fetch_generation_annual(zone)

    log.info("--- fuel prices ---")
    fetch_fuel_prices()

    log.info("--- GPR index ---")
    fetch_gpr()

    log.info("=== Ingestion complete ===")


if __name__ == "__main__":
    ingest_all()
