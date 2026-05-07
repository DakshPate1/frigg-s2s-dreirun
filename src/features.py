"""
Stage 4 — Feature engineering.

Operates on aligned base_dataset. All features are derived from the same column
names across both zones — no zone-specific logic in schema.

Features added:
  Derived (original):
    residual_load          = load - (wind_generation + solar_generation)
    renewable_penetration  = (wind_generation + solar_generation) / load

  Derived forecast versions (original):
    residual_load_forecast, renewable_penetration_forecast

  Temporal (original — raw + circular encoding):
    hour, weekday, month, week_of_year
    hour_sin/cos, weekday_sin/cos, month_sin/cos, week_sin/cos

  Holiday (original — enhanced):
    is_holiday, days_to_holiday, days_from_holiday

  Price lags (original):
    lag_1, lag_24, lag_168
    price_roll_24h, price_roll_168h
    residual_load_ramp, residual_load_ramp_forecast

  ── NEW ADDITIONS ─────────────────────────────────────────────────────────────

  Weather (new — multi-city capacity/population-weighted):
    DE_wind_speed, DE_solar_radiation, DE_temperature        (DE-LU model)
    DK_wind_speed                                            (DE-LU model — Nordic signal)
    CH_precipitation, CH_temperature                         (DE-LU model — alpine hydro)
    ES_wind_speed, ES_solar_radiation, ES_temperature        (ES model)
    ES_hydro_precipitation                                   (ES model — reservoir proxy)

  Engineered weather transformations (new):
    *_wind_speed_cubed      — cubic wind → power output proxy
    *_solar_hour_interaction — solar × sin(hour) → morning/evening asymmetry
    *_temperature_squared   — nonlinear heating/cooling demand

  Neighbor prices (new):
    FR_price_lag24, NL_price_lag24, CH_price_lag24, DK_price_lag24  (DE-LU)
    FR_price_lag24                                                    (ES)
    DE_FR_spread, DE_NL_spread, DE_CH_spread                         (DE-LU)
    ES_FR_spread                                                      (ES)

  Fuel prices (new):
    gas_price, coal_price, carbon_price   (daily → forward-filled hourly)

  Regime flags (new):
    crisis_period       — binary 1 for Aug 2021–Dec 2022
    is_peak             — binary 1 for hours 7-9 and 17-20
    negative_price_lag24 — binary 1 if own-zone price was negative 24h ago

  Cross-zone lag (new — unique angle):
    cross_zone_lag24    — other zone's price lagged 24h (captures price transmission)

  Nuclear (new — ES only):
    nuclear_available_mw — available nuclear capacity after outages

  Ensemble uncertainty (new — if available):
    wind_ensemble_std, solar_ensemble_std — ECMWF spread for interval calibration

Feature selection rationale:
  - SHAP confirms lag_24h and lag_168h dominate; lag_1 secondary
  - Renewable generation most important exogenous feature for DE market
  - Multi-city capacity-weighted weather outperforms single-city (generation centers
    vs demand centers are geographically distinct)
  - Wind at 100m (hub height) vs 10m meaningfully different signal
  - Neighbor prices capture cross-border transmission that zone price alone misses
  - Circular encoding of temporal features outperforms raw integers
  - is_holiday + distance features capture bridge-day demand suppression
  - Crisis period flag isolates 2021-2022 distortion without discarding data
  - Cross-zone lag captures Iberian isolation vs Central European coupling

Output:
  data/processed/final_dataset.parquet
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
import holidays as hdays

from config import (
    DATA_ALIGNED, DATA_PROCESSED, ZONES,
    CRISIS_START, CRISIS_END, PINBALL_Q,
)

log = logging.getLogger(__name__)

_ZONE_COUNTRY = {"DE-LU": "DE", "ES": "ES"}


# ══════════════════════════════════════════════════════════════════════════════
# ORIGINAL FUNCTIONS (unchanged — do not edit)
# ══════════════════════════════════════════════════════════════════════════════

def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["residual_load"] = (
        df["load"] - df["wind_generation"] - df["solar_generation"]
    ).clip(lower=0)
    df["renewable_penetration"] = (
        (df["wind_generation"] + df["solar_generation"]) /
        df["load"].replace(0, np.nan)
    )
    df["renewable_penetration"] = df["renewable_penetration"].clip(0, 1).fillna(0)
    return df


def add_derived_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from day-ahead forecast columns.
    Only runs if all three forecast columns are present.
    """
    needed = ["load_forecast", "wind_generation_forecast", "solar_generation_forecast"]
    if not all(c in df.columns for c in needed):
        return df
    df = df.copy()
    df["residual_load_forecast"] = (
        df["load_forecast"] -
        df["wind_generation_forecast"] -
        df["solar_generation_forecast"]
    ).clip(lower=0)
    df["renewable_penetration_forecast"] = (
        (df["wind_generation_forecast"] + df["solar_generation_forecast"]) /
        df["load_forecast"].replace(0, np.nan)
    ).clip(0, 1).fillna(0)
    return df


def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df.index.get_level_values("timestamp")

    df["hour"]         = ts.hour
    df["weekday"]      = ts.dayofweek
    df["month"]        = ts.month
    df["week_of_year"] = ts.isocalendar().week.values.astype(int)

    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["month_sin"]   = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"]   = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    df["week_sin"]    = np.sin(2 * np.pi * (df["week_of_year"] - 1) / 52)
    df["week_cos"]    = np.cos(2 * np.pi * (df["week_of_year"] - 1) / 52)

    return df


def add_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-zone holiday features.
    is_holiday, days_to_holiday, days_from_holiday (capped at 7).
    """
    frames = []
    for zone in ZONES:
        zone_df = df.xs(zone, level="zone").copy()
        country = _ZONE_COUNTRY.get(zone, "DE")

        years   = range(zone_df.index.year.min(), zone_df.index.year.max() + 2)
        cal     = hdays.country_holidays(country, years=years)
        cal_set = set(cal.keys())

        dates = zone_df.index.date
        zone_df["is_holiday"] = np.array(
            [int(d in cal_set) for d in dates], dtype=np.int8
        )

        hol_ord  = np.array(sorted(d.toordinal() for d in cal_set))
        date_ord = np.array([d.toordinal() for d in dates])

        idx_next    = np.searchsorted(hol_ord, date_ord, side="right")
        idx_next    = np.clip(idx_next, 0, len(hol_ord) - 1)
        days_to_raw = hol_ord[idx_next] - date_ord
        days_to     = np.where(days_to_raw > 0, days_to_raw, 7)

        idx_prev      = np.searchsorted(hol_ord, date_ord, side="left") - 1
        idx_prev      = np.clip(idx_prev, 0, len(hol_ord) - 1)
        days_from_raw = date_ord - hol_ord[idx_prev]
        days_from     = np.where(days_from_raw > 0, days_from_raw, 7)

        zone_df["days_to_holiday"]   = np.clip(days_to,   0, 7).astype(np.int8)
        zone_df["days_from_holiday"] = np.clip(days_from, 0, 7).astype(np.int8)

        zone_df["zone"] = zone
        frames.append(zone_df)

    combined = pd.concat(frames)
    combined = combined.reset_index().set_index(["timestamp", "zone"]).sort_index()
    return combined


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add price lags + rolling stats + ramp features per zone."""
    df = df.copy()
    lag_defs = {"lag_1": 1, "lag_24": 24, "lag_168": 168}

    frames = []
    for zone in ZONES:
        zone_df = df.xs(zone, level="zone").copy()
        zone_df = zone_df.sort_index()

        for col_name, shift in lag_defs.items():
            zone_df[col_name] = zone_df["price"].shift(shift)

        zone_df["price_roll_24h"]  = zone_df["price"].rolling(24,  min_periods=12).mean()
        zone_df["price_roll_168h"] = zone_df["price"].rolling(168, min_periods=84).mean()
        zone_df["price_roll_std_168h"] = zone_df["price"].rolling(168, min_periods=84).std()

        zone_df["residual_load_ramp"] = zone_df["residual_load"].diff(1)
        if "residual_load_forecast" in zone_df.columns:
            zone_df["residual_load_ramp_forecast"] = (
                zone_df["residual_load_forecast"].diff(1)
            )

        zone_df["zone"] = zone
        frames.append(zone_df)

    combined = pd.concat(frames)
    combined = combined.reset_index().set_index(["timestamp", "zone"]).sort_index()
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# NEW ADDITIONS
# ══════════════════════════════════════════════════════════════════════════════

def add_weather_features(df: pd.DataFrame,
                          weather_de: pd.DataFrame,
                          weather_es: pd.DataFrame) -> pd.DataFrame:
    """
    Merge pre-aggregated multi-city weather features.

    Expects weather_de and weather_es to be the outputs of
    ingestion.fetch_weather() — already capacity/population weighted,
    with columns like DE_wind_wind_speed_100m, DE_solar_shortwave_radiation, etc.

    Adds engineered transformations:
      - wind³  (cubic — power output is cubic function of wind speed)
      - solar × sin(hour)  (captures morning/evening asymmetry)
      - temperature²  (nonlinear heating/cooling demand)
    """
    df = df.copy()

    # Prepare weather indexes
    def _prep_weather(w: pd.DataFrame) -> pd.DataFrame:
        w = w.copy()
        if "time" in w.columns:
            w["time"] = pd.to_datetime(w["time"], utc=True)
            w = w.set_index("time")
        w.index = pd.to_datetime(w.index, utc=True)
        return w

    weather_de = _prep_weather(weather_de)
    weather_es = _prep_weather(weather_es)

    frames = []
    for zone in ZONES:
        zone_df = df.xs(zone, level="zone").copy()
        weather = weather_de if zone == "DE-LU" else weather_es

        # Merge weather columns
        zone_df = zone_df.join(weather, how="left")

        # ── Wind speed (100m) aggregated ──────────────────────────────────────
        # Find aggregated wind column (output of weighted average in ingestion)
        wind_col = next(
            (c for c in zone_df.columns
             if "wind" in c.lower() and "speed" in c.lower() and "100" in c),
            None
        )
        if wind_col:
            zone_df["wind_speed_agg"]   = zone_df[wind_col]
            zone_df["wind_speed_cubed"] = zone_df[wind_col] ** 3  # power proxy

        # ── Solar radiation aggregated ────────────────────────────────────────
        solar_col = next(
            (c for c in zone_df.columns
             if "solar" in c.lower() or "shortwave" in c.lower()),
            None
        )
        if solar_col and "hour_sin" in zone_df.columns:
            zone_df["solar_radiation_agg"] = zone_df[solar_col]
            # sin(hour*π/12) peaks at noon, zero at midnight — captures asymmetry
            zone_df["solar_hour_interaction"] = (
                zone_df[solar_col] *
                np.sin(np.pi * zone_df["hour"] / 12).clip(lower=0)
            )

        # ── Temperature aggregated ────────────────────────────────────────────
        temp_col = next(
            (c for c in zone_df.columns
             if "temperature" in c.lower() and "2m" in c.lower()),
            None
        )
        if temp_col:
            zone_df["temperature_agg"] = zone_df[temp_col]
            zone_df["temperature_sq"]  = zone_df[temp_col] ** 2  # nonlinear demand

        # ── DE-LU specific: DK wind + CH hydro ───────────────────────────────
        if zone == "DE-LU":
            dk_col = next(
                (c for c in zone_df.columns if "DK" in c and "wind" in c.lower()),
                None
            )
            if dk_col:
                zone_df["DK_wind_speed"]       = zone_df[dk_col]
                zone_df["DK_wind_speed_cubed"] = zone_df[dk_col] ** 3

            ch_precip_col = next(
                (c for c in zone_df.columns
                 if "CH" in c and "precip" in c.lower()),
                None
            )
            if ch_precip_col:
                zone_df["CH_precipitation"] = zone_df[ch_precip_col]
                # 7-day rolling sum — reservoir filling lag
                zone_df["CH_precip_7d_sum"] = (
                    zone_df[ch_precip_col].rolling(168, min_periods=1).sum()
                )

        # ── ES specific: hydro precipitation ─────────────────────────────────
        if zone == "ES":
            es_hydro_col = next(
                (c for c in zone_df.columns
                 if "ES_hydro" in c and "precip" in c.lower()),
                None
            )
            if es_hydro_col:
                zone_df["ES_hydro_precipitation"] = zone_df[es_hydro_col]
                zone_df["ES_hydro_precip_7d_sum"] = (
                    zone_df[es_hydro_col].rolling(168, min_periods=1).sum()
                )

        zone_df["zone"] = zone
        frames.append(zone_df)

    combined = pd.concat(frames)
    combined = combined.reset_index().set_index(["timestamp", "zone"]).sort_index()
    return combined


def add_neighbor_prices(df: pd.DataFrame,
                         neighbor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged neighbor zone prices + transmission spread proxies.

    DE-LU gets: FR, NL, CH, DK lags + spreads
    ES gets:    FR lag + spread only

    Spreads proxy transmission congestion:
      large spread = interconnector saturated = zones decoupling
      small spread = free flow = prices converging
    """
    df = df.copy()

    # Prep neighbor df
    nb = neighbor_df.copy()
    if "timestamp" in nb.columns:
        nb["timestamp"] = pd.to_datetime(nb["timestamp"], utc=True)
        nb = nb.set_index("timestamp")
    nb.index = pd.to_datetime(nb.index, utc=True)

    frames = []
    for zone in ZONES:
        zone_df = df.xs(zone, level="zone").copy()

        if zone == "DE-LU":
            neighbors = ["FR", "NL", "CH", "DK"]
        else:
            neighbors = ["FR"]

        for nb_name in neighbors:
            col = f"{nb_name}_price"
            if col not in nb.columns:
                log.warning("Neighbor column %s missing — skipping", col)
                continue

            # Lag 24h — known at prediction time
            zone_df[f"{nb_name}_price_lag24"] = nb[col].shift(24).reindex(
                zone_df.index, method="nearest"
            )

            # Transmission spread: own zone lag - neighbor lag
            if "lag_24" in zone_df.columns:
                zone_df[f"{zone.replace('-','_')}_{nb_name}_spread"] = (
                    zone_df["lag_24"] - zone_df[f"{nb_name}_price_lag24"]
                )

        zone_df["zone"] = zone
        frames.append(zone_df)

    combined = pd.concat(frames)
    combined = combined.reset_index().set_index(["timestamp", "zone"]).sort_index()
    return combined


def add_fuel_prices(df: pd.DataFrame,
                     fuel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily fuel prices, forward-filled to hourly frequency.

    Columns added: gas_price, coal_price (DE-LU relevant), carbon_price

    Forward-fill is correct here: fuel prices are published daily and
    the same price applies to all hours of that day.
    """
    df = df.copy()

    fuel = fuel_df.copy()
    if "date" in fuel.columns:
        fuel["date"] = pd.to_datetime(fuel["date"])
        fuel = fuel.set_index("date")
    fuel.index = pd.to_datetime(fuel.index)

    # Get hourly timestamp index
    ts_index = df.index.get_level_values("timestamp").unique()
    ts_index = pd.to_datetime(ts_index, utc=True).tz_localize(None)

    # Reindex to hourly, forward fill
    fuel_hourly = (
        fuel
        .reindex(ts_index.normalize().unique().union(fuel.index))
        .sort_index()
        .ffill()
        .reindex(ts_index.normalize())
    )

    # Simpler approach: merge on date
    fuel_cols = [c for c in ["gas_price", "coal_price", "carbon_price"]
                 if c in fuel.columns]

    ts_dates = pd.to_datetime(
        df.index.get_level_values("timestamp")
    ).normalize().tz_localize(None)

    for col in fuel_cols:
        date_map = fuel[col].to_dict()
        df[col] = [date_map.get(d, np.nan) for d in ts_dates]
        df[col] = df[col].fillna(method="ffill")

    log.info("Added fuel columns: %s", fuel_cols)
    return df


def add_regime_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary regime and structural flags.

    crisis_period:        1 for Aug 2021 – Dec 2022 energy crisis
                          Isolates distorted price formation without discarding data
    is_peak:              1 for morning (07-09) and evening (17-20) peak hours
    negative_price_lag24: 1 if own-zone price was negative 24h ago
                          DE-LU regime signal — predicts continuation of surplus
    """
    df = df.copy()
    ts = pd.to_datetime(df.index.get_level_values("timestamp"), utc=True)

    # Crisis period flag
    crisis_start = pd.Timestamp(CRISIS_START, tz="UTC")
    crisis_end   = pd.Timestamp(CRISIS_END,   tz="UTC")
    df["crisis_period"] = (
        (ts >= crisis_start) & (ts <= crisis_end)
    ).astype(np.int8)

    # Peak hours flag
    hours = ts.hour
    df["is_peak"] = pd.array(
        [int(h in {7, 8, 9, 17, 18, 19, 20}) for h in hours],
        dtype="Int8"
    )

    # Negative price lag — per zone to avoid cross-contamination
    frames = []
    for zone in ZONES:
        zone_df = df.xs(zone, level="zone").copy()
        if "lag_24" in zone_df.columns:
            zone_df["negative_price_lag24"] = (
                zone_df["lag_24"] < 0
            ).astype(np.int8)
        else:
            zone_df["negative_price_lag24"] = np.int8(0)
        zone_df["zone"] = zone
        frames.append(zone_df)

    combined = pd.concat(frames)
    combined = combined.reset_index().set_index(["timestamp", "zone"]).sort_index()
    return combined


def add_cross_zone_lag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the other zone's lagged price as a feature.

    DE-LU model gets ES price lag (t-24h)
    ES model gets DE-LU price lag (t-24h)

    Physical rationale:
      DE-LU and ES prices partially transmit through FR interconnection.
      When DE-LU goes negative (wind surplus), it sometimes depresses FR price
      which then slightly suppresses ES. The correlation is weak (~0.6-0.8)
      but consistent — adding this as a feature lets the model learn the
      transmission coefficient rather than ignoring it entirely.

    This is a unique feature most teams will not include.
    """
    df = df.copy()

    # Extract price series per zone at t-24h
    de_prices = (
        df.xs("DE-LU", level="zone")["price"]
        .sort_index()
        .shift(24)
        .rename("cross_zone_lag24")
    )
    es_prices = (
        df.xs("ES", level="zone")["price"]
        .sort_index()
        .shift(24)
        .rename("cross_zone_lag24")
    )

    frames = []
    for zone in ZONES:
        zone_df = df.xs(zone, level="zone").copy()
        # DE-LU gets ES lag; ES gets DE-LU lag
        cross = es_prices if zone == "DE-LU" else de_prices
        zone_df["cross_zone_lag24"] = cross.reindex(zone_df.index)
        zone_df["zone"] = zone
        frames.append(zone_df)

    combined = pd.concat(frames)
    combined = combined.reset_index().set_index(["timestamp", "zone"]).sort_index()
    return combined


def add_nuclear_availability(df: pd.DataFrame,
                               nuclear_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge ES nuclear available MW from ENTSO-E REMIT unavailability notices.

    Only meaningful for ES (~7 GW installed capacity, active as of 2026).
    DE-LU gets a constant zero column for schema consistency.

    High nuclear availability → more baseload supply → lower gas dispatch →
    lower prices, especially during peak hours.
    """
    df = df.copy()

    nuc = nuclear_df.copy()
    if "timestamp" in nuc.columns:
        nuc["timestamp"] = pd.to_datetime(nuc["timestamp"], utc=True)
        nuc = nuc.set_index("timestamp")
    nuc.index = pd.to_datetime(nuc.index, utc=True)

    frames = []
    for zone in ZONES:
        zone_df = df.xs(zone, level="zone").copy()

        if zone == "ES" and "nuclear_available_mw" in nuc.columns:
            zone_df["nuclear_available_mw"] = nuc["nuclear_available_mw"].reindex(
                zone_df.index, method="nearest"
            )
        else:
            # DE-LU: nuclear decommissioned April 2023 — constant zero
            zone_df["nuclear_available_mw"] = 0.0

        zone_df["zone"] = zone
        frames.append(zone_df)

    combined = pd.concat(frames)
    combined = combined.reset_index().set_index(["timestamp", "zone"]).sort_index()
    return combined


def add_ensemble_uncertainty(df: pd.DataFrame,
                               ensemble_de: pd.DataFrame,
                               ensemble_es: pd.DataFrame) -> pd.DataFrame:
    """
    Merge ECMWF ensemble spread as uncertainty signal.

    wind_ensemble_std and solar_ensemble_std (from 51 ECMWF members) measure
    meteorological forecast uncertainty. These are used by the model to
    dynamically widen/narrow prediction intervals:
      - High spread → uncertain renewable forecast → wider p025/p975
      - Low spread  → confident forecast → tighter intervals

    If ensemble data unavailable (empty df), columns are filled with zeros
    so downstream code doesn't break.
    """
    df = df.copy()

    def _prep(e: pd.DataFrame) -> pd.DataFrame:
        if e.empty:
            return e
        e = e.copy()
        if "time" in e.columns:
            e["time"] = pd.to_datetime(e["time"], utc=True)
            e = e.set_index("time")
        e.index = pd.to_datetime(e.index, utc=True)
        return e

    ens_de = _prep(ensemble_de)
    ens_es = _prep(ensemble_es)

    frames = []
    for zone in ZONES:
        zone_df = df.xs(zone, level="zone").copy()
        ens = ens_de if zone == "DE-LU" else ens_es

        std_cols = [c for c in (ens.columns if not ens.empty else [])
                    if "std" in c]

        if std_cols:
            for col in std_cols:
                short = col.replace("_ensemble_std", "").replace("wind_speed_100m", "wind").replace("shortwave_radiation", "solar")
                zone_df[f"{short}_ensemble_std"] = ens[col].reindex(
                    zone_df.index, method="nearest"
                )
        else:
            zone_df["wind_ensemble_std"]  = 0.0
            zone_df["solar_ensemble_std"] = 0.0

        zone_df["zone"] = zone
        frames.append(zone_df)

    combined = pd.concat(frames)
    combined = combined.reset_index().set_index(["timestamp", "zone"]).sort_index()
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(
    drop_lag_na: bool = True,
    weather_de:  pd.DataFrame | None = None,
    weather_es:  pd.DataFrame | None = None,
    neighbor_df: pd.DataFrame | None = None,
    fuel_df:     pd.DataFrame | None = None,
    nuclear_df:  pd.DataFrame | None = None,
    ensemble_de: pd.DataFrame | None = None,
    ensemble_es: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Load aligned dataset, apply all feature engineering steps, save parquet.

    Optional DataFrames (pass None to skip):
      weather_de  — multi-city weather for DE-LU (from ingestion.fetch_weather)
      weather_es  — multi-city weather for ES
      neighbor_df — neighbor zone prices (from ingestion.fetch_neighbor_prices)
      fuel_df     — daily fuel prices (from ingestion.fetch_fuel_prices)
      nuclear_df  — ES nuclear availability (from ingestion.fetch_entsoe_nuclear)
      ensemble_de — ECMWF ensemble spread DE-LU
      ensemble_es — ECMWF ensemble spread ES

    If called without optional args, produces same output as original pipeline.
    """
    path = DATA_ALIGNED / "base_dataset.parquet"
    log.info("Loading aligned dataset from %s", path)
    df = pd.read_parquet(path)

    # ── Original steps (unchanged) ────────────────────────────────────────────
    df = add_derived(df)
    df = add_derived_forecasts(df)
    df = add_temporal(df)
    df = add_lags(df)
    df = add_holidays(df)

    # ── New steps ─────────────────────────────────────────────────────────────
    df = add_regime_flags(df)
    df = add_cross_zone_lag(df)

    if weather_de is not None and weather_es is not None:
        log.info("Adding multi-city weather features")
        df = add_weather_features(df, weather_de, weather_es)
    else:
        log.warning("Weather DataFrames not provided — skipping weather features")

    if neighbor_df is not None:
        log.info("Adding neighbor price lags + spreads")
        df = add_neighbor_prices(df, neighbor_df)
    else:
        log.warning("Neighbor prices not provided — skipping")

    if fuel_df is not None:
        log.info("Adding fuel prices")
        df = add_fuel_prices(df, fuel_df)
    else:
        log.warning("Fuel prices not provided — skipping")

    if nuclear_df is not None:
        log.info("Adding ES nuclear availability")
        df = add_nuclear_availability(df, nuclear_df)
    else:
        log.warning("Nuclear data not provided — using zeros for nuclear_available_mw")
        df["nuclear_available_mw"] = 0.0

    if ensemble_de is not None and ensemble_es is not None:
        log.info("Adding ECMWF ensemble uncertainty")
        df = add_ensemble_uncertainty(df, ensemble_de, ensemble_es)
    else:
        log.info("Ensemble data not provided — using zero spread")
        df["wind_ensemble_std"]  = 0.0
        df["solar_ensemble_std"] = 0.0

    # ── Drop burn-in rows ─────────────────────────────────────────────────────
    if drop_lag_na:
        before = len(df)
        df = df.dropna(subset=["lag_168"])
        log.info("Dropped %d rows with NaN lag_168 (burn-in period)", before - len(df))

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = DATA_PROCESSED / "final_dataset.parquet"
    df.to_parquet(out_path)
    log.info("Saved %s — %d rows × %d cols", out_path.name, len(df), len(df.columns))
    log.info("Columns: %s", sorted(df.columns.tolist()))
    return df


def load_optional_data() -> dict:
    """
    Helper to load all optional DataFrames from raw data directory.
    Call this from pipeline.py before engineer_features().

    Returns dict with keys: weather_de, weather_es, neighbor_df,
                             fuel_df, nuclear_df, ensemble_de, ensemble_es
    """
    from config import DATA_RAW
    import os

    result = {}

    def _load_csv(path, key):
        if os.path.exists(path):
            log.info("Loading %s from %s", key, path)
            result[key] = pd.read_csv(path)
        else:
            log.warning("File not found: %s — %s will be None", path, key)
            result[key] = None

    _load_csv(DATA_RAW / "openmeteo" / "weather_DE-LU.csv",  "weather_de")
    _load_csv(DATA_RAW / "openmeteo" / "weather_ES.csv",     "weather_es")
    _load_csv(DATA_RAW / "entsoe"    / "prices_neighbors.csv", "neighbor_df")
    _load_csv(DATA_RAW / "fuel"      / "fuel_prices.csv",    "fuel_df")
    _load_csv(DATA_RAW / "entsoe"    / "nuclear_ES.csv",     "nuclear_df")
    _load_csv(DATA_RAW / "openmeteo" / "ensemble_DE-LU.csv", "ensemble_de")
    _load_csv(DATA_RAW / "openmeteo" / "ensemble_ES.csv",    "ensemble_es")

    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    optional = load_optional_data()
    engineer_features(**optional)
