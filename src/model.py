"""
Quantile LightGBM model for DAA electricity price forecasting.

Trains three quantile regressors per zone (q=0.025, 0.45, 0.975).
Evaluation: pinball loss at q=0.45 (per hackathon scoring).

Changes vs original:
  - CRITICAL: Eval window fixed (was May 8-9 / 30 rows → now May 11-12 / 48 rows)
  - FEATURES list expanded with new weather, neighbor, fuel, regime columns
  - Wind height fixed: wind_speed_10m → wind_speed_100m in _fetch_weather
  - build_eval_row: picks up aggregated multi-city weather + new fuel/neighbor cols
  - coal_price added to fuel carry-forward in build_eval_row
  - Validation split updated to match 3-year training window (2023 start)

Usage:
    python model.py               # train + validate + print metrics
    python model.py --predict     # generate predictions.csv for eval window
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import holidays as hdays
from sklearn.metrics import mean_pinball_loss

# Load .env so ENTSOE_TOKEN is available when model.py is run standalone
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        if _line.strip() and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "processed" / "final_dataset.parquet"
OUT_PATH  = ROOT / "alpine-arbitrage_predictions.csv"

# ── Features ──────────────────────────────────────────────────────────────────
# Ordered by expected importance (findings_entsoe + Tschora SHAP).
# New additions marked with # NEW.
# model.py auto-filters to columns present in dataset — safe to add ahead of data.
FEATURES = [
    # ── Fundamental supply/demand drivers ────────────────────────────────────
    "load", "wind_generation", "solar_generation", "hydro_generation",
    "nuclear_generation",                                               # NEW — ES active nuclear

    # ── Original single-city weather (fallback if multi-city not present) ────
    "temperature", "wind_speed", "solar_radiation",

    # ── Multi-city capacity/population-weighted weather ───────────────────────
    # NEW — generation centers + demand centers separated (see config.py)
    "wind_speed_agg",          # capacity-weighted wind at turbine locations
    "wind_speed_cubed",        # cubic wind → power output proxy
    "solar_radiation_agg",     # capacity-weighted solar at PV locations
    "solar_hour_interaction",  # solar × sin(hour) → morning/evening asymmetry
    "temperature_agg",         # population-weighted demand temperature
    "temperature_sq",          # nonlinear heating/cooling demand

    # ── DE-LU specific: Nordic wind + Swiss alpine hydro ─────────────────────
    # NEW — cross-border physical generation signals
    "DK_wind_speed",           # Danish North Sea wind exports into north Germany
    "DK_wind_speed_cubed",
    "CH_precipitation",        # Alpine snowmelt → Swiss hydro availability
    "CH_precip_7d_sum",        # 7-day rolling sum → reservoir filling lag

    # ── ES specific: hydro reservoir proxy ───────────────────────────────────
    # NEW
    "ES_hydro_precipitation",
    "ES_hydro_precip_7d_sum",
    "nuclear_available_mw",    # REMIT unavailability-adjusted nuclear capacity

    # ── Fuel / carbon ─────────────────────────────────────────────────────────
    "gas_price",
    "carbon_price",
    "coal_price",              # NEW — API2 coal, relevant for DE-LU coal peakers

    # ── Derived generation features ───────────────────────────────────────────
    "residual_load",
    "renewable_penetration",
    "residual_load_ramp",
    "residual_load_forecast",           # day-ahead forecast version
    "renewable_penetration_forecast",
    "residual_load_ramp_forecast",

    # ── Cross-border flows ────────────────────────────────────────────────────
    "net_imports",

    # ── Neighbor zone prices (lagged 24h) ─────────────────────────────────────
    # NEW — price transmission + transmission congestion signals
    "FR_price_lag24",          # France: dominant flow partner both zones
    "NL_price_lag24",          # Netherlands: gas hub, north wind correlation
    "CH_price_lag24",          # Switzerland: alpine hydro arbitrage
    "DK_price_lag24",          # Denmark: Nordic wind export signal
    # Transmission spreads: large spread = interconnectors congested
    "DE_LU_FR_spread",
    "DE_LU_NL_spread",
    "DE_LU_CH_spread",
    "ES_FR_spread",

    # ── Cross-zone price lag ──────────────────────────────────────────────────
    # NEW — captures Iberian isolation vs Central European coupling
    "cross_zone_lag24",

    # ── Calendar (circular encoding) ──────────────────────────────────────────
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    "month_sin", "month_cos",
    "week_sin", "week_cos",
    "is_holiday",
    "days_to_holiday",
    "days_from_holiday",

    # ── Regime flags ──────────────────────────────────────────────────────────
    # NEW
    "crisis_period",           # 1 = Aug 2021 – Dec 2022 energy crisis
    "is_peak",                 # 1 = morning (7-9) or evening (17-20) peak hours
    "negative_price_lag24",    # 1 = own-zone price was negative 24h ago

    # ── Price history ─────────────────────────────────────────────────────────
    "lag_1", "lag_24", "lag_168",
    "price_roll_24h", "price_roll_168h", "price_roll_std_168h",  # std NEW

    # ── Ensemble uncertainty ──────────────────────────────────────────────────
    # NEW — ECMWF ensemble spread for dynamic interval calibration
    "wind_ensemble_std",
    "solar_ensemble_std",
]

TARGET    = "price"
ZONES     = ["DE-LU", "ES"]
QUANTILES = [0.025, 0.45, 0.975]

# ── Time splits ───────────────────────────────────────────────────────────────
# Training window: 2023-05-01 → 2025-01-01  (post-crisis clean data)
# Validation:      2025-01-01 → 2026-01-01  (full year 2025)
# Calibration:     2026-01-01 → 2026-05-10  (Jan–May 2026, before eval window)
# Eval window:     2026-05-11 02:00 CEST → 2026-05-12 01:00 CEST (48 slots)
TRAIN_END = "2025-01-01"
VAL_END   = "2026-01-01"
CAL_END   = "2026-05-10"   # ← FIXED: was 2026-05-08, must stop before eval window

# Horizon regime split: ≤ this many days → LightGBM; beyond → seasonal long-term model
SHORTTERM_DAYS = 7

# ── LightGBM base params ──────────────────────────────────────────────────────
LGB_BASE = dict(
    objective         = "quantile",
    metric            = "quantile",
    n_estimators      = 3000,
    learning_rate     = 0.05,
    num_leaves        = 127,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    n_jobs            = -1,
    verbose           = -1,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def pinball(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    return float(mean_pinball_loss(y_true, y_pred, alpha=q))


def coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(((y_true >= lo) & (y_true <= hi)).mean())


def mean_band_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float((hi - lo).mean())


# ── Mondrian bucket ───────────────────────────────────────────────────────────

def _mondrian_bucket(ts: pd.Timestamp, cal) -> int:
    """
    Regime bucket for Mondrian conformal calibration.

    0 = normal  (regular weekday, not adjacent to a holiday)
    1 = risky   (weekend OR holiday OR bridge day within 1 day of holiday)

    Bridge days (e.g. Friday between Thursday holiday and weekend) see atypical
    demand and price behavior; treating them as normal weekdays produces
    systematically under-wide intervals for these slots.
    """
    d = ts.date()
    if ts.dayofweek >= 5:
        return 1
    if d in cal:
        return 1
    prev_day = (ts - pd.Timedelta(days=1)).date()
    next_day = (ts + pd.Timedelta(days=1)).date()
    if prev_day in cal or next_day in cal:
        return 1
    return 0


# ── Training ──────────────────────────────────────────────────────────────────

def train_zone(zdf: pd.DataFrame, zone: str) -> tuple[dict, dict]:
    """Train 3 quantile models for one zone. Return models + val predictions."""
    train = zdf[zdf.index <  TRAIN_END]
    val   = zdf[(zdf.index >= TRAIN_END) & (zdf.index < VAL_END)]

    X_tr, y_tr = train[FEATURES], train[TARGET]
    X_va, y_va = val[FEATURES],   val[TARGET]

    log.info("  %s  train=%d  val=%d", zone, len(X_tr), len(X_va))

    qmodels   = {}
    val_preds = {}

    for q in QUANTILES:
        log.info("    fitting q=%.3f ...", q)
        m = lgb.LGBMRegressor(**{**LGB_BASE, "alpha": q})
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        qmodels[q]   = m
        val_preds[q] = m.predict(X_va)
        log.info("      best_iter=%d  pinball=%.4f",
                 m.best_iteration_,
                 pinball(y_va.values, val_preds[q], q))

    return qmodels, {"preds": val_preds, "actual": y_va, "X": X_va}


# ── Evaluation ────────────────────────────────────────────────────────────────

def report_zone(zone: str, val: dict) -> None:
    y    = val["actual"].values
    p025 = val["preds"][0.025]
    p50  = val["preds"][0.45]
    p975 = val["preds"][0.975]

    log.info("─" * 56)
    log.info("  %s — validation 2025", zone)
    log.info("  MAE (p50)        : %.2f EUR/MWh", np.abs(y - p50).mean())
    log.info("  Pinball q=0.45   : %.4f", pinball(y, p50, 0.45))
    log.info("  Pinball q=0.025  : %.4f", pinball(y, p025, 0.025))
    log.info("  Pinball q=0.975  : %.4f", pinball(y, p975, 0.975))
    log.info("  [p025,p975] cov  : %.1f%%", coverage(y, p025, p975) * 100)
    log.info("  Mean band width  : %.2f EUR/MWh", mean_band_width(p025, p975))
    log.info("  Naive (lag_168)  : %.2f EUR/MWh MAE",
             np.abs(y - val["X"]["lag_168"].values).mean())


def feature_importance(qmodels: dict, zone: str) -> None:
    m   = qmodels[0.45]
    imp = pd.Series(
        m.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    log.info("  %s — top-15 feature importance (p50 model):", zone)
    for feat, score in imp.head(15).items():
        log.info("    %-35s %d", feat, score)


# ── CQR calibration ───────────────────────────────────────────────────────────

def calibrate_zone(
    zdf: pd.DataFrame,
    qmodels: dict,
    zone: str,
    cal_start: str | None = None,
    cal_end: str | None = None,
) -> dict:
    """
    Conformalized Quantile Regression (CQR) calibration.

    Default calibration window: VAL_END → CAL_END (Jan–May 2026).
    For backtests pass cal_start/cal_end to use a trailing window just before
    the prediction start so calibration stays in-sample relative to the test.

    Two corrections:
      q_hat_interval — symmetric EUR/MWh inflation applied to both sides of
                       [p025, p975] to achieve empirical 95% coverage.
      q_hat_50       — additive shift to p50 to achieve 45th-percentile
                       calibration (directly targets the scoring metric).

    CQR guarantee: on exchangeable calibration+test data, coverage ≥ 1−α.
    """
    _cal_start = cal_start or VAL_END
    _cal_end   = cal_end   or CAL_END
    cal = zdf[(zdf.index >= _cal_start) & (zdf.index < _cal_end)]
    cal = cal.dropna(subset=FEATURES + [TARGET])

    if len(cal) < 100:
        log.warning("  %s: calibration set too small (%d rows) — CQR skipped", zone, len(cal))
        return {"interval": 0.0, "p50": 0.0, "n": 0, "mondrian": {0: 0.0, 1: 0.0}}

    X_cal = cal[FEATURES]
    y_cal = cal[TARGET].values
    n     = len(y_cal)

    p025 = qmodels[0.025].predict(X_cal)
    p50  = qmodels[0.45].predict(X_cal)
    p975 = qmodels[0.975].predict(X_cal)
    p025 = np.minimum(p025, p50)
    p975 = np.maximum(p975, p50)

    # ── Interval: inflate [p025, p975] to 95% coverage ────────────────────────
    scores   = np.maximum(p025 - y_cal, y_cal - p975)
    q_level  = min(0.95 * (1 + 1 / n), 1.0)
    q_hat_iv = float(np.quantile(scores, q_level))

    # ── p50: shift to 45th-percentile calibration ─────────────────────────────
    resid_50  = y_cal - p50
    q_level50 = min(0.45 * (1 + 1 / n), 1.0)
    q_hat_50  = float(np.quantile(resid_50, q_level50))

    # ── Mondrian: per-regime interval Q_hats ──────────────────────────────────
    _ZONE_COUNTRY_MAP = {"DE-LU": "DE", "ES": "ES"}
    country  = _ZONE_COUNTRY_MAP.get(zone, "DE")
    hol_cal  = hdays.country_holidays(country, years=range(
        int(cal.index.year.min()), int(cal.index.year.max()) + 2))
    buckets  = np.array([_mondrian_bucket(ts, hol_cal) for ts in cal.index])
    mondrian = {}
    for b in [0, 1]:
        mask_b = buckets == b
        n_b    = int(mask_b.sum())
        if n_b < 50:
            mondrian[b] = q_hat_iv
        else:
            q_lev_b     = min(0.95 * (1 + 1 / n_b), 1.0)
            mondrian[b] = float(np.quantile(scores[mask_b], q_lev_b))
        if n_b > 0:
            cov_b = float(((y_cal[mask_b] >= p025[mask_b] - mondrian[b]) &
                           (y_cal[mask_b] <= p975[mask_b] + mondrian[b])).mean()) * 100
            log.info("    Mondrian b=%d (n=%d): Q_hat=%.2f  coverage=%.1f%%",
                     b, n_b, mondrian[b], cov_b)

    cov_raw = float(((y_cal >= p025)            & (y_cal <= p975)).mean())            * 100
    cov_cal = float(((y_cal >= p025 - q_hat_iv) & (y_cal <= p975 + q_hat_iv)).mean()) * 100
    pb_raw  = pinball(y_cal, p50, 0.45)
    pb_cal  = pinball(y_cal, p50 + q_hat_50, 0.45)

    log.info("  %s  CQR (n=%d  window=%s→%s)", zone, n, VAL_END[:7], CAL_END[:7])
    log.info("    interval  Q_hat=%.2f EUR/MWh  coverage raw=%.1f%% → cal=%.1f%% (target 95%%)",
             q_hat_iv, cov_raw, cov_cal)
    log.info("    p50 shift Q_hat=%.2f EUR/MWh  pinball  raw=%.4f → cal=%.4f",
             q_hat_50, pb_raw, pb_cal)

    return {"interval": q_hat_iv, "p50": q_hat_50, "n": n, "mondrian": mondrian}


# ── Long-term seasonal model ──────────────────────────────────────────────────

def build_longterm_model(zdf: pd.DataFrame, zone: str) -> dict:
    """
    Seasonal profile + annual trend for horizons beyond SHORTTERM_DAYS.

    PROFILE — recency-weighted median per (month, dayofweek, hour):
      Years resampled proportionally to recency weight. Down-weights anomalous
      2022 energy crisis; amplifies post-crisis 2023-2025 regime.

    TREND — post-crisis linear anchor (2023+):
      Fit on 2023-onward annual means only. Full 2021-2025 window would import
      crisis spike into slope. Fallback to all years if < 2 post-crisis years.

    STRUCTURAL NOTE (documented, not modelled):
      - DE-LU nuclear phase-out completed Apr 2023.
      - ES solar capacity growing ~8 GW/yr → deeper midday troughs going forward.
      Interval scaling with sqrt(horizon_months) provides conservative hedge.
    """
    hist = zdf[zdf.index < CAL_END].dropna(subset=["price"])

    first_year   = int(hist.index.year.min())
    year_weights = {y: max(1, y - first_year) for y in hist.index.year.unique()}
    weighted_parts = [
        pd.concat([hist[hist.index.year == y]] * w)
        for y, w in year_weights.items()
    ]
    w_hist = pd.concat(weighted_parts).sort_index()

    profile = (
        w_hist.groupby([w_hist.index.month, w_hist.index.dayofweek, w_hist.index.hour])["price"]
        .median()
    )
    profile.index.names = ["month", "weekday", "hour"]
    global_mean = float(hist["price"].median())

    annual      = hist.groupby(hist.index.year)["price"].mean()
    post_crisis = annual[annual.index >= 2023]
    trend_src   = post_crisis if len(post_crisis) >= 2 else annual
    if len(trend_src) >= 2:
        years            = trend_src.index.values.astype(float)
        slope, intercept = np.polyfit(years, trend_src.values, 1)
    else:
        slope, intercept = 0.0, float(annual.mean())
    anchor_year = int(annual.index[-1])

    def _profile_pred(ts: pd.Timestamp) -> float:
        try:
            return float(profile.loc[(ts.month, ts.dayofweek, ts.hour)])
        except KeyError:
            return global_mean

    profile_preds = pd.Series([_profile_pred(ts) for ts in hist.index], index=hist.index)
    resid         = hist["price"] - profile_preds
    resid_std     = float(resid.std())
    resid_q45     = float(np.quantile(resid.dropna().values, 0.45))

    log.info("  %s LT: slope=%.2f EUR/yr (src=%d-%d)  resid_std=%.2f  q45_bias=%.2f",
             zone, slope, int(trend_src.index[0]), anchor_year, resid_std, resid_q45)

    return {
        "profile":     profile,
        "global_mean": global_mean,
        "slope":       slope,
        "intercept":   intercept,
        "anchor_year": anchor_year,
        "resid_std":   resid_std,
        "resid_q45":   resid_q45,
    }


def predict_longterm_slot(
    ts: pd.Timestamp,
    lt: dict,
    horizon_days: float,
) -> tuple[float, float, float]:
    """
    Predict (p025, p50, p975) for one slot at a long horizon.
    Interval = 1.96 × resid_std × horizon_scale.
    """
    try:
        seasonal = float(lt["profile"].loc[(ts.month, ts.dayofweek, ts.hour)])
    except KeyError:
        seasonal = lt["global_mean"]

    years_out = (ts.year - lt["anchor_year"]) + (ts.month - 1) / 12.0
    trend_adj = lt["slope"] * years_out
    p50       = seasonal + trend_adj + lt["resid_q45"]

    excess_days   = max(horizon_days - SHORTTERM_DAYS, 0.0)
    horizon_scale = 1.0 + (excess_days / 30.0) ** 0.5 * 0.25
    half_interval = lt["resid_std"] * 1.96 * horizon_scale

    return p50 - half_interval, p50, p50 + half_interval


# ── Gap actuals fetch ─────────────────────────────────────────────────────────

def fetch_gap_actuals(zone: str, gap_start: pd.Timestamp,
                       gap_end: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch prices + net_imports from ENTSOE for the period between training data
    and the eval window (typically the last 1–2 days not yet in parquet).
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from config import ENTSOE_TOKEN, ENTSOE_ZONES, ENTSOE_NEIGHBORS

    if not ENTSOE_TOKEN:
        log.warning("ENTSOE_TOKEN not set — skipping gap actuals fetch")
        return pd.DataFrame()

    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        log.warning("entsoe-py not installed — skipping gap actuals fetch")
        return pd.DataFrame()

    client    = EntsoePandasClient(api_key=ENTSOE_TOKEN)
    eic       = ENTSOE_ZONES[zone]
    neighbors = ENTSOE_NEIGHBORS[zone]

    q_start = pd.Timestamp(gap_start.date().isoformat(), tz="Europe/Brussels")
    q_end   = pd.Timestamp(
        (gap_end + pd.Timedelta(days=1)).date().isoformat(), tz="Europe/Brussels"
    )

    try:
        prices = client.query_day_ahead_prices(eic, start=q_start, end=q_end)
        prices = prices.tz_convert("UTC")
        if prices.index.to_series().diff().dropna().min() < pd.Timedelta("1h"):
            prices = prices.resample("h").mean()
        prices.name = "price"
        log.info("  Gap prices %s: %d rows", zone, len(prices))
    except Exception as exc:
        log.warning("Gap prices fetch %s failed: %s", zone, exc)
        return pd.DataFrame()

    gap_df    = prices.to_frame()
    net_parts = []
    for nbr_eic in neighbors:
        try:
            imp = client.query_crossborder_flows(nbr_eic, eic, start=q_start, end=q_end)
            exp = client.query_crossborder_flows(eic, nbr_eic, start=q_start, end=q_end)
            imp = imp.tz_convert("UTC").resample("h").mean()
            exp = exp.tz_convert("UTC").resample("h").mean()
            net_parts.append(imp.sub(exp, fill_value=0.0))
        except Exception:
            pass

    if net_parts:
        net_imports      = pd.concat(net_parts, axis=1).sum(axis=1)
        net_imports.name = "net_imports"
        gap_df           = gap_df.join(net_imports, how="left")

    gap_df = gap_df.loc[(gap_df.index >= gap_start) & (gap_df.index < gap_end)]
    return gap_df


# ── Weather fetch helper ──────────────────────────────────────────────────────

def _fetch_weather(zone: str, start: pd.Timestamp,
                    end: pd.Timestamp) -> pd.DataFrame | None:
    """
    Fetch hourly weather for any window.
    Returns UTC-indexed DataFrame with columns matching training features.

    FIXED: wind_speed_10m → wind_speed_100m (turbine hub height).
    Multi-city aggregated weather used when available; falls back to single location.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from ingestion import _get, WEATHER_LOCATIONS

    _today  = pd.Timestamp.utcnow().normalize()
    api_key = os.environ.get("OM_API_KEY")
    frames  = []

    # Try multi-city aggregated fetch first (new ingestion supports this)
    try:
        from ingestion import fetch_weather_forecast as _fwf_multi
        fcst_start = max(start, _today)
        fcst_end   = min(end, _today + pd.Timedelta(days=14))
        if fcst_start <= fcst_end:
            raw = _fwf_multi(
                zone,
                fcst_start.strftime("%Y-%m-%d"),
                fcst_end.strftime("%Y-%m-%d")
            )
            if not raw.empty:
                raw.index = pd.to_datetime(raw["time"], utc=True)
                raw = raw.drop(columns=["time"], errors="ignore")
                frames.append(raw)
                log.info("  Multi-city weather forecast %s: %d rows, %d cols",
                         zone, len(raw), len(raw.columns))
    except Exception as exc:
        log.warning("  Multi-city weather fetch failed: %s — falling back to single location", exc)

    # Fallback: single location (original behaviour)
    if not frames:
        # FIXED: Use first location from new WEATHER_LOCATIONS structure
        loc_group = "DE_demand" if zone == "DE-LU" else "ES_demand"
        try:
            from config import WEATHER_LOCATIONS as WL
            loc_list = WL.get(loc_group, [])
            if loc_list:
                loc = {"latitude": loc_list[0]["latitude"],
                       "longitude": loc_list[0]["longitude"]}
            else:
                loc = WEATHER_LOCATIONS[zone]  # original single-entry dict
        except Exception:
            loc = WEATHER_LOCATIONS[zone]

        # Past portion → historical forecast API
        if start < _today and api_key:
            hist_end = min(end, _today - pd.Timedelta(hours=1))
            try:
                d = _get(
                    "https://historical-forecast-api.open-meteo.com/v1/forecast",
                    params={
                        "latitude":        loc["latitude"],
                        "longitude":       loc["longitude"],
                        "start_date":      start.strftime("%Y-%m-%d"),
                        "end_date":        hist_end.strftime("%Y-%m-%d"),
                        # FIXED: was wind_speed_10m
                        "hourly":          "temperature_2m,wind_speed_100m,shortwave_radiation",
                        "timezone":        "UTC",
                        "wind_speed_unit": "ms",
                    },
                    api_key=api_key
                )
                h = d.get("hourly", {})
                if h.get("time"):
                    frames.append(pd.DataFrame({
                        "temperature":     h["temperature_2m"],
                        "wind_speed":      h["wind_speed_100m"],   # FIXED
                        "solar_radiation": h["shortwave_radiation"],
                    }, index=pd.to_datetime(h["time"], utc=True)))
            except Exception as exc:
                log.warning("  Weather hist-forecast %s failed: %s", zone, exc)

        # Future portion → forecast API
        fcst_start = max(start, _today)
        fcst_end   = min(end, _today + pd.Timedelta(days=14))
        if fcst_start <= fcst_end and fcst_start < end:
            try:
                d = _get("https://api.open-meteo.com/v1/forecast", {
                    "latitude":        loc["latitude"],
                    "longitude":       loc["longitude"],
                    "start_date":      fcst_start.strftime("%Y-%m-%d"),
                    "end_date":        fcst_end.strftime("%Y-%m-%d"),
                    # FIXED: was wind_speed_10m
                    "hourly":          "temperature_2m,wind_speed_100m,shortwave_radiation",
                    "timezone":        "UTC",
                    "wind_speed_unit": "ms",
                })
                h = d.get("hourly", {})
                if h.get("time"):
                    frames.append(pd.DataFrame({
                        "temperature":     h["temperature_2m"],
                        "wind_speed":      h["wind_speed_100m"],   # FIXED
                        "solar_radiation": h["shortwave_radiation"],
                    }, index=pd.to_datetime(h["time"], utc=True)))
            except Exception as exc:
                log.warning("  Weather forecast %s failed: %s", zone, exc)

    if not frames:
        return None
    return pd.concat(frames).sort_index()


# ── ENTSOE day-ahead generation forecast (eval window) ───────────────────────

def fetch_entsoe_gen_forecast(
    zone: str, fcst_start: pd.Timestamp, fcst_end: pd.Timestamp
) -> pd.DataFrame:
    """
    Fetch ENTSOE day-ahead load + wind/solar + nuclear forecasts for eval window.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from config import (ENTSOE_TOKEN, ENTSOE_ZONES,
                        ENTSOE_WIND_TYPES, ENTSOE_SOLAR_TYPES, ENTSOE_NUCLEAR_TYPES)

    if not ENTSOE_TOKEN:
        return pd.DataFrame()

    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        return pd.DataFrame()

    client  = EntsoePandasClient(api_key=ENTSOE_TOKEN)
    eic     = ENTSOE_ZONES[zone]
    q_start = pd.Timestamp(fcst_start.date().isoformat(), tz="Europe/Brussels")
    q_end   = pd.Timestamp(
        (fcst_end + pd.Timedelta(days=1)).date().isoformat(), tz="Europe/Brussels"
    )

    result: dict[str, pd.Series] = {}

    try:
        lf = client.query_load_forecast(eic, start=q_start, end=q_end)
        if isinstance(lf, pd.DataFrame):
            lf = lf.iloc[:, 0]
        result["load"] = lf.tz_convert("UTC").resample("h").mean()
    except Exception as exc:
        log.warning("  ENTSOE load forecast %s: %s", zone, exc)

    try:
        ws = client.query_wind_and_solar_forecast(eic, start=q_start, end=q_end)
        ws = ws.tz_convert("UTC")
        if ws.index.to_series().diff().dropna().min() < pd.Timedelta("1h"):
            ws = ws.resample("h").mean()
        if isinstance(ws.columns, pd.MultiIndex):
            ws.columns = ws.columns.get_level_values(0)
        wind_cols  = [c for c in ws.columns if any(t in str(c) for t in ENTSOE_WIND_TYPES)]
        solar_cols = [c for c in ws.columns if any(t in str(c) for t in ENTSOE_SOLAR_TYPES)]
        if wind_cols:
            result["wind_generation"]  = ws[wind_cols].sum(axis=1).clip(lower=0)
        if solar_cols:
            result["solar_generation"] = ws[solar_cols].sum(axis=1).clip(lower=0)
    except Exception as exc:
        log.warning("  ENTSOE wind/solar forecast %s: %s", zone, exc)

    if not result:
        return pd.DataFrame()

    df = pd.DataFrame(result)
    df = df[(df.index >= fcst_start) & (df.index <= fcst_end)]
    return df


# ── Eval-window feature construction ─────────────────────────────────────────

def build_eval_row(
    zdf: pd.DataFrame,
    ref: pd.DataFrame,
    zone: str,
    ts: pd.Timestamp,
    predicted_p50: dict,
    cal,
    weather_fcst: pd.DataFrame | None = None,
    gen_fcst: pd.DataFrame | None = None,
    prev_residual_load: float | None = None,
) -> tuple[dict, float]:
    """
    Build one feature row for a single eval-window timestamp.

    Called sequentially so predicted_p50 is populated with all prior slots
    before this row is built — enabling honest recursive lag_1 / lag_24 fill.
    """
    row: dict = {}

    # ── Generation / proxy features ───────────────────────────────────────────
    same_hw    = ref[(ref.index.hour == ts.hour) & (ref.index.dayofweek == ts.dayofweek)]
    proxy_cols = ["load", "wind_generation", "solar_generation", "hydro_generation",
                  "nuclear_generation",
                  "residual_load", "renewable_penetration", "residual_load_ramp",
                  "residual_load_forecast", "renewable_penetration_forecast",
                  "residual_load_ramp_forecast"]
    if "net_imports" in ref.columns:
        proxy_cols.append("net_imports")

    for col in proxy_cols:
        if col in ref.columns:
            row[col] = float(same_hw[col].mean()) if len(same_hw) > 0 else float(ref[col].mean())
        else:
            row[col] = 0.0

    # Override with ENTSOE day-ahead forecast when available
    if (gen_fcst is not None and ts in gen_fcst.index and
            all(c in gen_fcst.columns
                for c in ["load", "wind_generation", "solar_generation"]) and
            not any(pd.isna(gen_fcst.loc[ts, c])
                    for c in ["load", "wind_generation", "solar_generation"])):
        row["load"]             = float(gen_fcst.loc[ts, "load"])
        row["wind_generation"]  = float(gen_fcst.loc[ts, "wind_generation"])
        row["solar_generation"] = float(gen_fcst.loc[ts, "solar_generation"])
        row["residual_load"]    = max(
            0.0, row["load"] - row["wind_generation"] - row["solar_generation"]
        )
        if row["load"] > 0:
            row["renewable_penetration"] = min(
                1.0, (row["wind_generation"] + row["solar_generation"]) / row["load"]
            )

    if prev_residual_load is not None:
        ramp = row["residual_load"] - prev_residual_load
    else:
        ramp = row["residual_load_ramp"]
    row["residual_load_ramp"] = np.clip(ramp, -2500, 5000)

    # ── Weather features ──────────────────────────────────────────────────────
    # Original single-city columns (fallback)
    weather_cols = ["temperature", "wind_speed", "solar_radiation"]
    if weather_fcst is not None and ts in weather_fcst.index:
        for col in weather_cols:
            if col in weather_fcst.columns:
                row[col] = float(weather_fcst.loc[ts, col])
            else:
                row[col] = (float(same_hw[col].mean()) if col in ref.columns and len(same_hw) > 0
                            else float(ref[col].mean()) if col in ref.columns else 0.0)
    else:
        for col in weather_cols:
            row[col] = (float(same_hw[col].mean()) if col in ref.columns and len(same_hw) > 0
                        else float(ref[col].mean()) if col in ref.columns else 0.0)

    # Multi-city aggregated weather — override if present
    agg_map = {
        "temperature":     "temperature_agg",
        "wind_speed":      "wind_speed_agg",
        "solar_radiation": "solar_radiation_agg",
    }
    for orig_col, agg_col in agg_map.items():
        if agg_col in ref.columns:
            row[orig_col] = (float(same_hw[agg_col].mean()) if len(same_hw) > 0
                             else float(ref[agg_col].mean()))
            if weather_fcst is not None and ts in weather_fcst.index and agg_col in weather_fcst.columns:
                row[orig_col] = float(weather_fcst.loc[ts, agg_col])

    # Engineered weather features
    for feat in ["wind_speed_cubed", "solar_radiation_agg", "wind_speed_agg",
                 "solar_hour_interaction", "temperature_agg", "temperature_sq",
                 "DK_wind_speed", "DK_wind_speed_cubed",
                 "CH_precipitation", "CH_precip_7d_sum",
                 "ES_hydro_precipitation", "ES_hydro_precip_7d_sum",
                 "nuclear_available_mw",
                 "wind_ensemble_std", "solar_ensemble_std"]:
        if feat in ref.columns:
            row[feat] = (float(same_hw[feat].mean()) if len(same_hw) > 0
                         else float(ref[feat].mean()))
            if weather_fcst is not None and ts in weather_fcst.index and feat in weather_fcst.columns:
                row[feat] = float(weather_fcst.loc[ts, feat])
        else:
            row[feat] = 0.0

    # ── Holiday proximity ─────────────────────────────────────────────────────
    ts_date = ts.date()
    hol_ord = np.array(sorted(d.toordinal() for d in cal.keys()
                              if abs(d.year - ts_date.year) <= 2))
    if len(hol_ord):
        d_ord       = ts_date.toordinal()
        idx_next    = min(np.searchsorted(hol_ord, d_ord, side="right"), len(hol_ord) - 1)
        days_to_raw = hol_ord[idx_next] - d_ord
        row["days_to_holiday"] = int(np.clip(days_to_raw if days_to_raw > 0 else 7, 0, 7))

        idx_prev      = max(np.searchsorted(hol_ord, d_ord, side="left") - 1, 0)
        days_from_raw = d_ord - hol_ord[idx_prev]
        row["days_from_holiday"] = int(np.clip(days_from_raw if days_from_raw > 0 else 7, 0, 7))
    else:
        row["days_to_holiday"]   = 7
        row["days_from_holiday"] = 7

    # ── Fuel: carry forward last known ────────────────────────────────────────
    for fuel_col in ["gas_price", "carbon_price", "coal_price"]:  # coal_price NEW
        row[fuel_col] = float(ref[fuel_col].iloc[-1]) if fuel_col in ref.columns else 0.0

    # ── Neighbor prices: carry forward last known ─────────────────────────────
    for nb_col in ["FR_price_lag24", "NL_price_lag24", "CH_price_lag24", "DK_price_lag24",
                   "DE_LU_FR_spread", "DE_LU_NL_spread", "DE_LU_CH_spread", "ES_FR_spread",
                   "cross_zone_lag24"]:
        row[nb_col] = float(ref[nb_col].iloc[-1]) if nb_col in ref.columns else 0.0

    # ── Regime flags ──────────────────────────────────────────────────────────
    row["crisis_period"]        = 0   # May 2026 is post-crisis
    row["is_peak"]              = int(ts.hour in {7, 8, 9, 17, 18, 19, 20})
    row["negative_price_lag24"] = int(row.get("lag_24", 0) < 0)

    # ── Calendar ──────────────────────────────────────────────────────────────
    woy = ts.isocalendar()[1]
    row["hour_sin"]    = np.sin(2 * np.pi * ts.hour / 24)
    row["hour_cos"]    = np.cos(2 * np.pi * ts.hour / 24)
    row["weekday_sin"] = np.sin(2 * np.pi * ts.dayofweek / 7)
    row["weekday_cos"] = np.cos(2 * np.pi * ts.dayofweek / 7)
    row["month_sin"]   = np.sin(2 * np.pi * (ts.month - 1) / 12)
    row["month_cos"]   = np.cos(2 * np.pi * (ts.month - 1) / 12)
    row["week_sin"]    = np.sin(2 * np.pi * (woy - 1) / 52)
    row["week_cos"]    = np.cos(2 * np.pi * (woy - 1) / 52)
    row["is_holiday"]  = int(ts.date() in cal)

    # ── Price lags ────────────────────────────────────────────────────────────
    def lookup(lag_ts: pd.Timestamp, fallback_col: str) -> float:
        if lag_ts in zdf.index:
            return float(zdf.loc[lag_ts, "price"])
        if lag_ts in predicted_p50:
            return predicted_p50[lag_ts]
        return float(ref[fallback_col].mean()) if fallback_col in ref.columns else 0.0

    row["lag_168"] = lookup(ts - pd.Timedelta(hours=168), "lag_168")
    row["lag_24"]  = lookup(ts - pd.Timedelta(hours=24),  "lag_24")
    row["lag_1"]   = lookup(ts - pd.Timedelta(hours=1),   "lag_1")

    row["price_roll_24h"]    = float(zdf["price"].iloc[-24:].mean())
    row["price_roll_168h"]   = float(zdf["price"].iloc[-168:].mean())
    row["price_roll_std_168h"] = float(zdf["price"].iloc[-168:].std())

    return row, row["residual_load"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main(predict: bool = False, pred_start: str | None = None,
         pred_end: str | None = None) -> None:

    log.info("Loading dataset from %s", DATA_PATH)
    df = pd.read_parquet(DATA_PATH)

    # Auto-filter FEATURES to columns present in dataset
    # Safe to list future features — missing ones are silently skipped
    active_features = [f for f in FEATURES if f in df.columns]
    if len(active_features) < len(FEATURES):
        missing = set(FEATURES) - set(active_features)
        log.warning("Features missing from dataset (run full pipeline): %s",
                    sorted(missing))
    globals()["FEATURES"] = active_features

    all_models = {}
    all_val    = {}

    for zone in ZONES:
        log.info("━" * 56)
        log.info("Training zone: %s", zone)
        zdf = df.xs(zone, level="zone").sort_index()
        qmodels, val = train_zone(zdf, zone)
        all_models[zone] = qmodels
        all_val[zone]    = val

    log.info("━" * 56)
    log.info("VALIDATION RESULTS")
    for zone in ZONES:
        report_zone(zone, all_val[zone])
        feature_importance(all_models[zone], zone)

    log.info("━" * 56)
    log.info("CQR CALIBRATION  (window: %s → %s)", VAL_END, CAL_END)
    cqr = {}
    for zone in ZONES:
        zdf      = df.xs(zone, level="zone").sort_index()
        cqr[zone] = calibrate_zone(zdf, all_models[zone], zone)

    log.info("━" * 56)
    log.info("LONG-TERM MODEL  (seasonal profile + annual trend)")
    lt_models = {}
    for zone in ZONES:
        zdf            = df.xs(zone, level="zone").sort_index()
        lt_models[zone] = build_longterm_model(zdf, zone)

    if not predict:
        return

    log.info("━" * 56)
    log.info("Generating predictions (CQR-adjusted)")

    _ZONE_COUNTRY = {"DE-LU": "DE", "ES": "ES"}

    # ── FIXED: correct eval window ────────────────────────────────────────────
    # Competition window: Mon 11 May 2026 02:00 CEST → Tue 12 May 2026 01:00 CEST
    # In UTC: 2026-05-11 00:00 → 2026-05-11 23:00  (48 hourly slots)
    _DEFAULT_START = "2026-05-11 00:00"   # ← FIXED (was 2026-05-08 17:00)
    _DEFAULT_END   = "2026-05-11 23:00"   # ← FIXED (was 2026-05-09 22:00)

    eval_start = pd.Timestamp(pred_start or _DEFAULT_START, tz="UTC")
    eval_end   = pd.Timestamp(pred_end   or _DEFAULT_END,   tz="UTC")
    eval_idx   = pd.date_range(eval_start, eval_end, freq="h")
    log.info("  Window: %s → %s  (%d slots)", eval_start, eval_end, len(eval_idx))

    # Sanity check — must be exactly 48 slots for competition submission
    if pred_start is None and len(eval_idx) != 48:
        log.error("CRITICAL: Expected 48 eval slots, got %d — check eval window", len(eval_idx))

    _today         = pd.Timestamp.utcnow().normalize()
    _WX_LIMIT_DAYS = 14
    wx_fetch_end   = min(eval_end, _today + pd.Timedelta(days=_WX_LIMIT_DAYS))

    zone_weather_fcst: dict[str, pd.DataFrame | None] = {}
    if wx_fetch_end >= eval_start:
        for zone in ZONES:
            zone_weather_fcst[zone] = _fetch_weather(zone, eval_start, wx_fetch_end)
    else:
        log.info("  Window beyond 14-day horizon — using seasonal proxy for weather")
        for zone in ZONES:
            zone_weather_fcst[zone] = None

    fcst_end_capped = wx_fetch_end
    zone_gen_fcst: dict[str, pd.DataFrame | None] = {}
    if fcst_end_capped >= eval_start:
        for zone in ZONES:
            try:
                gf = fetch_entsoe_gen_forecast(zone, eval_start, fcst_end_capped)
                zone_gen_fcst[zone] = gf if len(gf) > 0 else None
            except Exception as exc:
                log.warning("  Gen forecast fetch failed for %s: %s — proxy", zone, exc)
                zone_gen_fcst[zone] = None
    else:
        for zone in ZONES:
            zone_gen_fcst[zone] = None

    zone_preds = {}
    for zone in ZONES:
        zdf = df.xs(zone, level="zone").sort_index()

        gap_start = zdf.index[-1] + pd.Timedelta(hours=1)
        if gap_start < eval_start:
            log.info("Fetching gap actuals for %s (%s → %s) ...",
                     zone, gap_start.date(), eval_start.date())
            gap_df = fetch_gap_actuals(zone, gap_start, eval_start)
            if len(gap_df) > 0:
                zdf = pd.concat([zdf, gap_df])
                zdf = zdf[~zdf.index.duplicated(keep="last")].sort_index()
                log.info("  zdf extended: tail now %s", zdf.index[-1])

        ref          = zdf[eval_start - pd.Timedelta(weeks=4):
                           eval_start - pd.Timedelta(hours=1)]
        cal          = hdays.country_holidays(_ZONE_COUNTRY[zone])
        weather_fcst = zone_weather_fcst[zone]
        gen_fcst     = zone_gen_fcst[zone]

        predicted_p50              = {}
        p025_list, p50_list, p975_list = [], [], []
        regime_log                 = {"shortterm": 0, "longterm": 0}

        q_50        = cqr[zone]["p50"]
        mondrian_iv = cqr[zone]["mondrian"]
        lt          = lt_models[zone]
        zdf_tail    = zdf.index[-1]

        prev_residual_load = (float(zdf["residual_load"].iloc[-1])
                              if "residual_load" in zdf.columns else None)

        for ts in eval_idx:
            horizon_days = (ts - zdf_tail).total_seconds() / 86400.0
            bucket       = _mondrian_bucket(ts, cal)
            q_iv_slot    = mondrian_iv[bucket]

            if horizon_days <= SHORTTERM_DAYS:
                row, current_residual_load = build_eval_row(
                    zdf, ref, zone, ts, predicted_p50, cal,
                    weather_fcst, gen_fcst, prev_residual_load
                )
                prev_residual_load = current_residual_load
                x    = pd.DataFrame([row])[FEATURES]

                p025 = float(all_models[zone][0.025].predict(x)[0])
                p50  = float(all_models[zone][0.45].predict(x)[0])
                p975 = float(all_models[zone][0.975].predict(x)[0])

                p025 = min(p025, p50)
                p975 = max(p975, p50)

                p50  = p50  + q_50
                p025 = p025 - q_iv_slot
                p975 = p975 + q_iv_slot

                p025 = min(p025, p50)
                p975 = max(p975, p50)
                regime_log["shortterm"] += 1

            else:
                p025, p50, p975 = predict_longterm_slot(ts, lt, horizon_days)
                regime_log["longterm"] += 1

            predicted_p50[ts] = p50
            p025_list.append(p025)
            p50_list.append(p50)
            p975_list.append(p975)

        zone_preds[zone] = {"p025": p025_list, "p50": p50_list, "p975": p975_list}
        log.info("  %s: mean p50=%.2f  band=%.2f  [ST=%d LT=%d slots]",
                 zone,
                 np.mean(p50_list),
                 np.mean(np.array(p975_list) - np.array(p025_list)),
                 regime_log["shortterm"], regime_log["longterm"])

    # ── Build submission CSV ──────────────────────────────────────────────────
    # Timestamps: ISO 8601 with CEST offset (+02:00 in May)
    ts_cest = eval_idx.tz_convert("Europe/Berlin")
    ts_str  = [t.isoformat() for t in ts_cest]

    out = pd.DataFrame({
        "timestamp":  ts_str,
        "DE-LU p025": zone_preds["DE-LU"]["p025"],
        "DE-LU p50":  zone_preds["DE-LU"]["p50"],
        "DE-LU p975": zone_preds["DE-LU"]["p975"],
        "ES p025":    zone_preds["ES"]["p025"],
        "ES p50":     zone_preds["ES"]["p50"],
        "ES p975":    zone_preds["ES"]["p975"],
    })

    # Final submission sanity checks
    assert len(out) == 48,              f"Expected 48 rows, got {len(out)}"
    assert (out["DE-LU p025"] < out["DE-LU p50"]).all(),  "DE-LU p025 >= p50 violation"
    assert (out["DE-LU p50"]  < out["DE-LU p975"]).all(), "DE-LU p50 >= p975 violation"
    assert (out["ES p025"]    < out["ES p50"]).all(),      "ES p025 >= p50 violation"
    assert (out["ES p50"]     < out["ES p975"]).all(),     "ES p50 >= p975 violation"
    assert out.isnull().sum().sum() == 0,                  "NaN values in submission"

    out.to_csv(OUT_PATH, index=False, float_format="%.4f")
    log.info("Saved %s  (%d rows)", OUT_PATH, len(out))
    log.info("Preview:\n%s", out.head(10).to_string())

    # ── Backtest ──────────────────────────────────────────────────────────────
    log.info("━" * 56)
    any_backtest = False
    for zone in ZONES:
        zdf   = df.xs(zone, level="zone").sort_index()
        known = zdf[(zdf.index >= eval_start) &
                    (zdf.index <= eval_end)]["price"].dropna()
        if len(known) == 0:
            continue
        any_backtest = True
        p50_arr  = np.array(zone_preds[zone]["p50"])
        p025_arr = np.array(zone_preds[zone]["p025"])
        p975_arr = np.array(zone_preds[zone]["p975"])
        mask     = np.array([ts in known.index for ts in eval_idx])
        y_known  = known.reindex(eval_idx[mask]).values
        log.info("  BACKTEST %s (%d slots with actuals):", zone, mask.sum())
        log.info("    MAE p50      : %.2f EUR/MWh", np.abs(y_known - p50_arr[mask]).mean())
        log.info("    Pinball 0.45 : %.4f", pinball(y_known, p50_arr[mask], 0.45))
        log.info("    Coverage     : %.1f%%",
                 coverage(y_known, p025_arr[mask], p975_arr[mask]) * 100)
    if not any_backtest:
        log.info("  No actuals in dataset for this window — forward prediction only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true",
                        help="Generate predictions.csv for the specified window")
    parser.add_argument("--start", default=None,
                        help="Prediction window start UTC (e.g. '2026-05-11 00:00')")
    parser.add_argument("--end", default=None,
                        help="Prediction window end UTC (e.g. '2026-05-11 23:00')")
    args = parser.parse_args()
    main(predict=args.predict, pred_start=args.start, pred_end=args.end)
