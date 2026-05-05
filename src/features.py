"""
Stage 4 — Feature engineering.

Operates on aligned base_dataset. All features are derived from the same column
names across both zones — no zone-specific logic.

Features added:
  Derived:
    residual_load          = load - (wind_generation + solar_generation)
    renewable_penetration  = (wind_generation + solar_generation) / load

  Temporal:
    hour, weekday, month

  Price lags (strictly backward-looking, computed per zone independently):
    lag_1    = price shifted 1h
    lag_24   = price shifted 24h
    lag_168  = price shifted 168h (7 days)

  Rolling (per zone):
    price_roll_24h   = 24h rolling mean of price
    price_roll_168h  = 168h rolling mean of price

Output:
  data/processed/final_dataset.parquet
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np

from config import DATA_ALIGNED, DATA_PROCESSED, ZONES

log = logging.getLogger(__name__)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["residual_load"]         = (df["load"] - df["wind_generation"] - df["solar_generation"]).clip(lower=0)
    df["renewable_penetration"] = (df["wind_generation"] + df["solar_generation"]) / df["load"].replace(0, np.nan)
    df["renewable_penetration"] = df["renewable_penetration"].clip(0, 1).fillna(0)
    return df


def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # timestamp is the first level of the MultiIndex
    ts = df.index.get_level_values("timestamp")
    df["hour"]    = ts.hour
    df["weekday"] = ts.dayofweek   # 0 = Monday
    df["month"]   = ts.month
    return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add price lags per zone. Groups by zone to avoid cross-zone leakage."""
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

        zone_df["zone"] = zone
        frames.append(zone_df)

    combined = pd.concat(frames)
    combined = combined.reset_index().set_index(["timestamp", "zone"]).sort_index()
    return combined


def engineer_features(drop_lag_na: bool = True) -> pd.DataFrame:
    """Load aligned dataset, add all features, save final parquet."""
    path = DATA_ALIGNED / "base_dataset.parquet"
    log.info("Loading aligned dataset from %s", path)
    df = pd.read_parquet(path)

    df = add_derived(df)
    df = add_temporal(df)
    df = add_lags(df)

    if drop_lag_na:
        # Drop rows where lag_168 is NaN (first 168h per zone)
        before = len(df)
        df = df.dropna(subset=["lag_168"])
        log.info("Dropped %d rows with NaN lag_168 (burn-in period)", before - len(df))

    out_path = DATA_PROCESSED / "final_dataset.parquet"
    df.to_parquet(out_path)
    log.info("Saved %s — %d rows × %d cols", out_path.name, len(df), len(df.columns))
    log.info("Columns: %s", df.columns.tolist())
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engineer_features()
