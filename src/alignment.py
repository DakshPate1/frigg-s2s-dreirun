"""
Stage 3 — Alignment.

Joins all cleaned datasets on a common hourly UTC timestamp index per zone,
then stacks both zones into a single DataFrame with a (timestamp, zone) MultiIndex.

Output:
  data/aligned/base_dataset.parquet
"""

from __future__ import annotations

import logging
import pandas as pd

from config import DATA_CLEAN, DATA_ALIGNED, ZONES

log = logging.getLogger(__name__)

# Columns that must be non-null for a row to be kept
CRITICAL_COLS = ["price", "load", "wind_generation", "solar_generation", "temperature"]


def align_zone(zone: str) -> pd.DataFrame:
    """Join cleaned sources for a single zone on UTC hourly index."""
    prices  = pd.read_parquet(DATA_CLEAN / f"prices_{zone}.parquet")
    gen     = pd.read_parquet(DATA_CLEAN / f"generation_{zone}.parquet")
    weather = pd.read_parquet(DATA_CLEAN / f"weather_{zone}.parquet")
    fuel    = pd.read_parquet(DATA_CLEAN / "fuel_prices.parquet")

    # Cross-border flows — optional; present when ENTSOE pipeline has run
    xborder_path = DATA_CLEAN / f"crossborder_{zone}.parquet"
    xborder = pd.read_parquet(xborder_path) if xborder_path.exists() else None

    # Build common hourly index spanning required sources
    all_indices = [prices.index, gen.index, weather.index, fuel.index]
    start = max(idx.min() for idx in all_indices)
    end   = min(idx.max() for idx in all_indices)
    log.info("Zone %s common window: %s → %s", zone, start, end)

    common = pd.date_range(start, end, freq="h", tz="UTC", name="timestamp")

    df = pd.DataFrame(index=common)
    df = df.join(prices.reindex(common),  how="left")
    df = df.join(gen.reindex(common),     how="left")
    df = df.join(weather.reindex(common), how="left")
    df = df.join(fuel.reindex(common),    how="left")

    if xborder is not None:
        df = df.join(xborder.reindex(common), how="left")
        log.info("Zone %s: cross-border flows joined (%d non-null)", zone, df["net_imports"].notna().sum())
    else:
        log.info("Zone %s: no crossborder parquet — net_imports not included", zone)

    # Drop rows missing any critical column
    before = len(df)
    df = df.dropna(subset=CRITICAL_COLS)
    dropped = before - len(df)
    if dropped:
        log.warning("Zone %s: dropped %d rows with missing critical cols", zone, dropped)

    df["zone"] = zone
    log.info("Zone %s aligned: %d rows", zone, len(df))
    return df


def align_all() -> pd.DataFrame:
    """Align both zones, stack into MultiIndex DataFrame, save parquet."""
    frames = []
    for zone in ZONES:
        log.info("=== Aligning zone: %s ===", zone)
        frames.append(align_zone(zone))

    combined = pd.concat(frames, axis=0)
    combined = combined.reset_index().set_index(["timestamp", "zone"]).sort_index()

    out_path = DATA_ALIGNED / "base_dataset.parquet"
    combined.to_parquet(out_path)
    log.info("Saved %s — %d rows × %d cols", out_path.name, len(combined), len(combined.columns))
    log.info("Columns: %s", combined.columns.tolist())

    # Sanity: check no NaN in critical columns
    for col in CRITICAL_COLS:
        n_nan = combined[col].isna().sum()
        if n_nan:
            log.error("ALIGNMENT ERROR: %d NaN in column '%s'", n_nan, col)
        else:
            log.info("  ✓ %s: no NaN", col)

    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    align_all()
