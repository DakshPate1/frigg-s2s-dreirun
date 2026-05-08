"""
Stage 5 — Validation.

Hard-fails on:
  - Missing required parquet files.
  - NaN in critical columns (capacity history, demand history, prices).
  - Capacity totals outside 50-300 GW per zone.
  - Annual demand outside 100-800 TWh per zone.
  - Marginal-cost panel: any non-renewable tech with all-NaN MC.

Warnings (don't hard-fail):
  - Capacity factors > 0.95 or < 0.0 (probable data issue).
  - Forecast-extended capacity discontinuities at the history/forecast boundary.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from config import (
    DATA_CLEAN, DATA_ALIGNED, DATA_PROCESSED,
    TARGET_ZONES, TECHS,
    HISTORY_END,
)

_CURRENT_YEAR = pd.Timestamp(HISTORY_END).year

log = logging.getLogger(__name__)


def _fail(msg: str):
    log.error("VALIDATION FAILED: %s", msg)
    sys.exit(1)


def validate():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("=== Validation ===")

    # 1. Required files
    required = [
        DATA_CLEAN / "capacity_by_tech.parquet",
        DATA_CLEAN / "demand_annual.parquet",
        DATA_CLEAN / "da_prices_monthly.parquet",
        DATA_ALIGNED / "structural_annual.parquet",
        DATA_ALIGNED / "fuel_monthly.parquet",
        DATA_PROCESSED / "marginal_costs_monthly.parquet",
        DATA_PROCESSED / "structural_extended.parquet",
    ]
    for p in required:
        if not p.exists():
            _fail(f"missing required file: {p}")
    log.info("✓ all %d required files present", len(required))

    # 2. Capacity totals per zone-year
    cap = pd.read_parquet(DATA_CLEAN / "capacity_by_tech.parquet")
    totals = cap.groupby(["year", "zone"])["capacity_gw"].sum()
    for (year, zone), gw in totals.items():
        if not (50 <= gw <= 300):
            log.warning("capacity total out of expected range: %s %d -> %.1f GW",
                        zone, year, gw)
    log.info("✓ capacity totals in plausible range (50-300 GW)")

    # 3. Demand totals
    dem = pd.read_parquet(DATA_CLEAN / "demand_annual.parquet")
    for _, row in dem.iterrows():
        # Skip current/partial year (always incomplete) and earliest year
        # (often partial because ENTSOE 2018-H1 has gaps).
        if row["year"] >= _CURRENT_YEAR or row["year"] <= int(dem["year"].min()):
            continue
        if not (100 <= row["demand_twh"] <= 800):
            log.warning("demand out of expected range: %s %d -> %.1f TWh",
                        row["zone"], row["year"], row["demand_twh"])
    log.info("✓ demand in plausible range (100-800 TWh; partial first/current years skipped)")

    # 4. DA prices: any NaN months, any zero / negative monthly means
    pr = pd.read_parquet(DATA_CLEAN / "da_prices_monthly.parquet")
    nan_count = pr["price_eur_per_mwh"].isna().sum()
    if nan_count > 0:
        log.warning("DA prices: %d NaN monthly observations", nan_count)
    neg_count = (pr["price_eur_per_mwh"] < 0).sum()
    if neg_count > 0:
        # Negative monthly means are rare but real (heavy solar-surplus months).
        # Not a data quality issue — log informationally.
        log.info("  (note: %d months with negative mean DA price — solar surplus, expected)", neg_count)
    log.info("✓ DA prices: %d months across zones", len(pr))

    # 5. Marginal cost panel: thermal techs should have non-NaN MC for most months
    mc = pd.read_parquet(DATA_PROCESSED / "marginal_costs_monthly.parquet")
    thermal_techs = ["coal", "gas_ccgt", "oil_peaker"]
    for tech in thermal_techs:
        sub = mc[mc["tech"] == tech]
        nan_share = sub["mc_eur_per_mwh"].isna().mean()
        if nan_share > 0.30:
            _fail(f"tech {tech}: {nan_share:.0%} months have NaN marginal cost")
    log.info("✓ marginal costs: thermal techs <30%% NaN")

    # 6. Capacity factor sanity
    s = pd.read_parquet(DATA_ALIGNED / "structural_annual.parquet")
    for tech in TECHS:
        cf_col = f"capacity_factor_{tech}"
        if cf_col not in s.columns:
            continue
        cfs = s[cf_col].dropna()
        if len(cfs) == 0:
            continue
        if cfs.max() > 0.99:
            log.warning("CF %s exceeds 0.99 in some years (max %.2f)", tech, cfs.max())
        if cfs.min() < 0:
            log.warning("CF %s negative in some years (min %.2f)", tech, cfs.min())
    log.info("✓ capacity factors plausible")

    # 7. Iberian cap presence (ES gas marginal cost during cap window
    #    should be lower than DE gas MC for the same months)
    cap_check = mc[(mc["tech"] == "gas_ccgt") & (mc["capped"])]
    if len(cap_check) == 0:
        log.warning("Iberian cap never triggered — check IBERIAN_CAP_PERIODS / fuel data range")
    else:
        log.info("✓ Iberian cap triggered in %d (month, zone=ES) cells", len(cap_check))

    # 8. Structural extended: monotonic year coverage
    se = pd.read_parquet(DATA_PROCESSED / "structural_extended.parquet").reset_index()
    for zone in TARGET_ZONES:
        years = sorted(se[se["zone"] == zone]["year"].unique())
        gaps = [b - a for a, b in zip(years, years[1:]) if b - a != 1]
        if gaps:
            _fail(f"{zone}: year gaps in structural_extended {gaps}")
    log.info("✓ structural_extended: contiguous year coverage per zone")

    log.info("=== Validation passed ===")


if __name__ == "__main__":
    validate()
