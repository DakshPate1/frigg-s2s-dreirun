"""
Stage 5 — Validation.

Checks the final dataset for:
  1. No missing values in required columns
  2. Continuous hourly timestamps per zone (no gaps)
  3. Correct zone values
  4. Reasonable value ranges
  5. Price lag integrity (lag_1 matches actual shift)

Raises RuntimeError on hard failures; logs warnings for soft issues.
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np

from config import DATA_PROCESSED, ZONES

log = logging.getLogger(__name__)

REQUIRED_COLS = [
    "price", "load", "wind_generation", "solar_generation", "hydro_generation",
    "temperature", "wind_speed", "solar_radiation",
    "gas_price", "carbon_price",
    "residual_load", "renewable_penetration",
    "hour", "weekday", "month",
    "lag_1", "lag_24", "lag_168",
]

RANGE_CHECKS = {
    "price":               (-500, 3000),
    "load":                (0, 200_000),
    "wind_generation":     (0, 100_000),
    "solar_generation":    (0, 100_000),
    "hydro_generation":    (0, 100_000),
    "temperature":         (-40, 50),
    "wind_speed":          (0, 60),
    "solar_radiation":     (0, 1500),
    "gas_price":           (0, 400),
    "carbon_price":        (0, 200),
    "renewable_penetration": (0, 1.01),
}


def validate(path: str | None = None) -> bool:
    if path is None:
        path = DATA_PROCESSED / "final_dataset.parquet"

    log.info("Validating %s", path)
    df = pd.read_parquet(path)

    errors = 0
    warnings = 0

    # 1. Required columns present
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        log.error("FAIL: missing columns: %s", missing_cols)
        errors += 1
    else:
        log.info("  ✓ All required columns present")

    # 2. No NaN in required columns
    for col in REQUIRED_COLS:
        if col not in df.columns:
            continue
        n = df[col].isna().sum()
        if n:
            log.error("FAIL: %d NaN in column '%s'", n, col)
            errors += 1
        else:
            log.info("  ✓ %s: no NaN", col)

    # 3. Zones
    actual_zones = set(df.index.get_level_values("zone").unique())
    expected_zones = set(ZONES)
    if actual_zones != expected_zones:
        log.error("FAIL: expected zones %s, got %s", expected_zones, actual_zones)
        errors += 1
    else:
        log.info("  ✓ Zones: %s", actual_zones)

    # 4. Continuous hourly timestamps per zone
    # Warn for small gaps (e.g. real-world outages like the Apr 2025 ES blackout);
    # error only for large gaps that indicate a systematic ingestion problem.
    GAP_WARN_THRESHOLD  =  50   # warn if missing < 50 timestamps
    GAP_ERROR_THRESHOLD = 200   # error if missing >= 200 timestamps
    for zone in ZONES:
        ts = df.xs(zone, level="zone").index.sort_values()
        expected = pd.date_range(ts.min(), ts.max(), freq="h", tz="UTC")
        missing = expected.difference(ts)
        n = len(missing)
        if n == 0:
            log.info("  ✓ Zone %s: continuous hourly timestamps (%d rows)", zone, len(ts))
        elif n < GAP_WARN_THRESHOLD:
            log.warning("WARN: Zone %s: %d missing hourly timestamps (real-world gap — acceptable)", zone, n)
            warnings += 1
        elif n < GAP_ERROR_THRESHOLD:
            log.error("FAIL: Zone %s: %d missing hourly timestamps", zone, n)
            errors += 1
        else:
            log.error("FAIL: Zone %s: %d missing hourly timestamps (large gap — check ingestion)", zone, n)
            errors += 1

    # 5. Value range checks
    for col, (lo, hi) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        out = df[col].dropna()
        n_low  = (out < lo).sum()
        n_high = (out > hi).sum()
        if n_low or n_high:
            log.warning("WARN: '%s': %d below %.1f, %d above %.1f", col, n_low, lo, n_high, hi)
            warnings += 1
        else:
            log.info("  ✓ %s range [%.1f, %.1f] ok", col, out.min(), out.max())

    # 6. Lag integrity spot-check (lag_1 should match price shifted 1h)
    for zone in ZONES:
        z = df.xs(zone, level="zone").sort_index()
        check = (z["price"].shift(1) - z["lag_1"]).abs().max()
        if check > 1e-6:
            log.error("FAIL: Zone %s: lag_1 mismatch (max diff=%.6f)", zone, check)
            errors += 1
        else:
            log.info("  ✓ Zone %s: lag_1 integrity ok", zone)

    log.info("Validation done: %d errors, %d warnings", errors, warnings)

    if errors:
        raise RuntimeError(f"Dataset validation failed with {errors} error(s)")

    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    validate()
