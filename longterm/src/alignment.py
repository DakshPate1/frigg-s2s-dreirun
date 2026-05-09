"""
Stage 3 — Alignment.

Builds two aligned panels from the cleaned parquet files:

  1. structural_annual.parquet
       Index:   (year, zone)
       Columns: capacity_<tech>_gw  for each tech
                generation_<tech>_twh
                demand_twh
                capacity_factor_<tech>      (= generation / (capacity * 8760))
                price_eur_per_mwh_avg       (annual mean, calibration target)

  2. fuel_monthly_aligned.parquet
       Index:   month (UTC month start)
       Columns: gas_ttf, carbon, coal_api2, oil_brent, plus DA price per zone.

These are the inputs the merit-order model and the calibration step will read.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_CLEAN, DATA_ALIGNED, TARGET_ZONES, TECHS

log = logging.getLogger(__name__)

HOURS_PER_YEAR = 8760


# ── Structural annual panel ──────────────────────────────────────────────────

def build_structural_annual() -> pd.DataFrame:
    cap  = pd.read_parquet(DATA_CLEAN / "capacity_by_tech.parquet")
    gen  = pd.read_parquet(DATA_CLEAN / "generation_by_tech.parquet") \
                if (DATA_CLEAN / "generation_by_tech.parquet").exists() \
                else pd.DataFrame(columns=["year", "zone", "tech", "generation_twh"])
    dem  = pd.read_parquet(DATA_CLEAN / "demand_annual.parquet")
    pr_m = pd.read_parquet(DATA_CLEAN / "da_prices_monthly.parquet")

    # Pivot capacity wide
    cap_w = cap.pivot_table(index=["year", "zone"], columns="tech",
                            values="capacity_gw", aggfunc="sum",
                            fill_value=0.0)
    cap_w.columns = [f"capacity_{t}_gw" for t in cap_w.columns]

    # Pivot generation wide (may be empty -> handled)
    if not gen.empty:
        gen_w = gen.pivot_table(index=["year", "zone"], columns="tech",
                                values="generation_twh", aggfunc="sum",
                                fill_value=0.0)
        gen_w.columns = [f"generation_{t}_twh" for t in gen_w.columns]
    else:
        gen_w = pd.DataFrame(index=cap_w.index)

    # Demand
    dem_w = dem.set_index(["year", "zone"])[["demand_twh"]]

    # Annual mean price from monthly
    pr_m["year"] = pd.to_datetime(pr_m["month"]).dt.year
    pr_a = pr_m.groupby(["year", "zone"])["price_eur_per_mwh"].mean() \
               .to_frame("price_eur_per_mwh_avg")

    panel = cap_w.join(gen_w, how="left") \
                 .join(dem_w, how="left") \
                 .join(pr_a, how="left") \
                 .sort_index()

    # Capacity factors (only when both capacity and generation are present)
    for tech in TECHS:
        cap_col = f"capacity_{tech}_gw"
        gen_col = f"generation_{tech}_twh"
        cf_col  = f"capacity_factor_{tech}"
        if cap_col in panel.columns and gen_col in panel.columns:
            denom = panel[cap_col] * HOURS_PER_YEAR / 1000.0   # GW * h -> GWh; / 1000 -> TWh
            with np.errstate(divide="ignore", invalid="ignore"):
                panel[cf_col] = (panel[gen_col] / denom).where(denom > 0)
            panel[cf_col] = panel[cf_col].clip(lower=0, upper=1)

    out_path = DATA_ALIGNED / "structural_annual.parquet"
    panel.to_parquet(out_path)
    log.info("Saved %s — %s shape", out_path.name, panel.shape)
    return panel


# ── Fuel + DA price monthly panel ────────────────────────────────────────────

def build_fuel_monthly() -> pd.DataFrame:
    fuel = pd.read_parquet(DATA_CLEAN / "fuel_monthly.parquet")
    pr_m = pd.read_parquet(DATA_CLEAN / "da_prices_monthly.parquet")

    fuel = fuel.set_index("month").sort_index()

    # Pivot DA prices wide
    pr_w = pr_m.pivot_table(index="month", columns="zone",
                            values="price_eur_per_mwh", aggfunc="mean")
    pr_w.columns = [f"da_price_{z}_eur_per_mwh" for z in pr_w.columns]

    panel = fuel.join(pr_w, how="outer").sort_index()

    # Join GPR if available
    gpr_path = DATA_CLEAN / "gpr_monthly.parquet"
    if gpr_path.exists():
        gpr = pd.read_parquet(gpr_path).set_index("month").sort_index()
        # Match fuel panel's index timezone
        if panel.index.tz is not None and gpr.index.tz is None:
            gpr.index = gpr.index.tz_localize("UTC")
        elif panel.index.tz is None and gpr.index.tz is not None:
            gpr.index = gpr.index.tz_localize(None)
        panel = panel.join(gpr, how="left")

    out_path = DATA_ALIGNED / "fuel_monthly.parquet"
    panel.to_parquet(out_path)
    log.info("Saved %s — %s shape", out_path.name, panel.shape)
    return panel


def align_all():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("=== Alignment ===")
    build_structural_annual()
    build_fuel_monthly()
    log.info("=== Alignment complete ===")


if __name__ == "__main__":
    align_all()
