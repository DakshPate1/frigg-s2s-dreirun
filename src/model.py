"""
Quantile LightGBM model for DAA electricity price forecasting.

Trains three quantile regressors per zone (q=0.025, 0.45, 0.975).
Evaluation: pinball loss at q=0.45 (per hackathon scoring).

Usage:
    python model.py               # train + validate + print metrics
    python model.py --predict     # also generate predictions.csv for eval window
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
ROOT       = Path(__file__).parent.parent
DATA_PATH  = ROOT / "data" / "processed" / "final_dataset.parquet"
OUT_PATH   = ROOT / "alpine-arbitrage_predictions.csv"

# ── Features ──────────────────────────────────────────────────────────────────
# Ordered by expected importance (findings_entsoe + Tschora SHAP):
# load/generation > lag_24 > lag_168 > weather > gas > calendar > lag_1 (recursive)
FEATURES = [
    # Fundamental drivers
    "load", "wind_generation", "solar_generation", "hydro_generation",
    "temperature", "wind_speed", "solar_radiation",
    # Fuel / carbon
    "gas_price", "carbon_price",
    # Derived
    "residual_load", "renewable_penetration",
    # Cross-border flows (net imports MW; ENTSOE source)
    "net_imports",
    # Calendar — circular encoding
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    "month_sin", "month_cos",
    "week_sin", "week_cos",
    "is_holiday",
    # Price history
    "lag_1", "lag_24", "lag_168",
    "price_roll_24h", "price_roll_168h",
]

TARGET    = "price"
ZONES     = ["DE-LU", "ES"]
QUANTILES = [0.025, 0.45, 0.975]

# Train on 2021–2024; validate on 2025; calibrate on Jan–May 2026; eval window is May 2026
TRAIN_END = "2025-01-01"
VAL_END   = "2026-01-01"
CAL_END   = "2026-05-08"   # stop before eval window

# Horizon regime split: ≤ this many days → LightGBM; beyond → seasonal long-term model
SHORTTERM_DAYS = 7

# ── LightGBM base params ──────────────────────────────────────────────────────
LGB_BASE = dict(
    objective        = "quantile",
    metric           = "quantile",
    n_estimators     = 3000,
    learning_rate    = 0.05,
    num_leaves       = 127,
    min_child_samples= 20,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 0.1,
    n_jobs           = -1,
    verbose          = -1,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def pinball(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    return float(mean_pinball_loss(y_true, y_pred, alpha=q))


def coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(((y_true >= lo) & (y_true <= hi)).mean())


def mean_band_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float((hi - lo).mean())


# ── Training ──────────────────────────────────────────────────────────────────

def train_zone(zdf: pd.DataFrame, zone: str) -> tuple[dict, dict]:
    """Train 3 quantile models for one zone. Return models + val predictions."""
    train = zdf[zdf.index <  TRAIN_END]
    val   = zdf[(zdf.index >= TRAIN_END) & (zdf.index < VAL_END)]

    X_tr, y_tr = train[FEATURES], train[TARGET]
    X_va, y_va = val[FEATURES],   val[TARGET]

    log.info("  %s  train=%d  val=%d", zone, len(X_tr), len(X_va))

    qmodels = {}
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
        qmodels[q]    = m
        val_preds[q]  = m.predict(X_va)
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
    m = qmodels[0.45]
    imp = pd.Series(
        m.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    log.info("  %s — top-10 feature importance (p50 model):", zone)
    for feat, score in imp.head(10).items():
        log.info("    %-30s %d", feat, score)


# ── CQR calibration ───────────────────────────────────────────────────────────

def calibrate_zone(zdf: pd.DataFrame, qmodels: dict, zone: str) -> dict:
    """
    Conformalized Quantile Regression (CQR) calibration.

    Calibration window: VAL_END → CAL_END (Jan–May 2026).
    Held out from both training (ends 2025-01-01) and reported validation (2025).

    Two corrections:
      q_hat_interval — symmetric EUR/MWh inflation applied to both sides of
                       [p025, p975] to achieve empirical 95% coverage.
      q_hat_50       — additive shift to p50 to achieve 45th-percentile
                       calibration (directly targets the scoring metric).

    CQR guarantee: on exchangeable calibration+test data, coverage ≥ 1−α.
    """
    cal = zdf[(zdf.index >= VAL_END) & (zdf.index < CAL_END)]
    cal = cal.dropna(subset=FEATURES + [TARGET])

    if len(cal) < 100:
        log.warning("  %s: calibration set too small (%d rows) — CQR skipped", zone, len(cal))
        return {"interval": 0.0, "p50": 0.0, "n": 0}

    X_cal = cal[FEATURES]
    y_cal = cal[TARGET].values
    n     = len(y_cal)

    p025 = qmodels[0.025].predict(X_cal)
    p50  = qmodels[0.45].predict(X_cal)
    p975 = qmodels[0.975].predict(X_cal)
    p025 = np.minimum(p025, p50)
    p975 = np.maximum(p975, p50)

    # ── Interval: inflate [p025, p975] to 95% coverage ────────────────────────
    # Score = how far y lies outside the current interval (negative = already inside)
    scores   = np.maximum(p025 - y_cal, y_cal - p975)
    q_level  = min(0.95 * (1 + 1 / n), 1.0)
    q_hat_iv = float(np.quantile(scores, q_level))

    # ── p50: shift to 45th-percentile calibration ─────────────────────────────
    # Residuals > 0 mean y > p50 (model is under-forecasting)
    resid_50  = y_cal - p50
    q_level50 = min(0.45 * (1 + 1 / n), 1.0)
    q_hat_50  = float(np.quantile(resid_50, q_level50))

    # ── Diagnostics ───────────────────────────────────────────────────────────
    cov_raw = float(((y_cal >= p025)              & (y_cal <= p975)).mean())              * 100
    cov_cal = float(((y_cal >= p025 - q_hat_iv)   & (y_cal <= p975 + q_hat_iv)).mean())   * 100
    pb_raw  = pinball(y_cal, p50, 0.45)
    pb_cal  = pinball(y_cal, p50 + q_hat_50, 0.45)

    log.info("  %s  CQR (n=%d  window=%s→%s)", zone, n, VAL_END[:7], CAL_END[:7])
    log.info("    interval  Q_hat=%.2f EUR/MWh  coverage raw=%.1f%% → cal=%.1f%% (target 95%%)",
             q_hat_iv, cov_raw, cov_cal)
    log.info("    p50 shift Q_hat=%.2f EUR/MWh  pinball  raw=%.4f → cal=%.4f",
             q_hat_50, pb_raw, pb_cal)

    return {"interval": q_hat_iv, "p50": q_hat_50, "n": n}


# ── Long-term seasonal model ──────────────────────────────────────────────────

def build_longterm_model(zdf: pd.DataFrame, zone: str) -> dict:
    """
    Seasonal profile + annual trend for horizons beyond SHORTTERM_DAYS.

    Profile: mean price per (month, dayofweek, hour) from all historical data
             through CAL_END — 12×7×24 = 2016 cells.
    Trend:   linear regression of year → annual mean price, extrapolated forward.
    Uncertainty: std of (actual − profile_pred), scaled by sqrt(horizon_months)
                 to reflect growing uncertainty at longer horizons.

    p50 uses 45th-percentile residual to stay consistent with the scoring metric.
    """
    hist = zdf[zdf.index < CAL_END].dropna(subset=["price"])

    # ── Seasonal profile ──────────────────────────────────────────────────────
    profile = (
        hist.groupby([hist.index.month, hist.index.dayofweek, hist.index.hour])["price"]
        .mean()
    )
    profile.index.names = ["month", "weekday", "hour"]
    global_mean = float(hist["price"].mean())

    # ── Annual trend ──────────────────────────────────────────────────────────
    annual = hist.groupby(hist.index.year)["price"].mean()
    if len(annual) >= 2:
        years  = annual.index.values.astype(float)
        slope, intercept = np.polyfit(years, annual.values, 1)
    else:
        slope, intercept = 0.0, float(annual.mean())
    anchor_year = int(annual.index[-1])

    # ── Residual uncertainty ──────────────────────────────────────────────────
    def _profile_pred(ts: pd.Timestamp) -> float:
        try:
            return float(profile.loc[(ts.month, ts.dayofweek, ts.hour)])
        except KeyError:
            return global_mean

    profile_preds = pd.Series([_profile_pred(ts) for ts in hist.index], index=hist.index)
    resid         = hist["price"] - profile_preds
    resid_std     = float(resid.std())
    # 45th-percentile residual: p50 should be the 45th pctile, not median, per scoring
    resid_q45     = float(np.quantile(resid.dropna().values, 0.45))

    log.info("  %s LT: slope=%.2f EUR/yr  resid_std=%.2f  q45_bias=%.2f  anchor=%d",
             zone, slope, resid_std, resid_q45, anchor_year)

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

    Interval = 1.96 × resid_std × horizon_scale, where:
      horizon_scale = 1 + sqrt(max(horizon_days − SHORTTERM_DAYS, 0) / 30) × 0.25
    This gives ~base width at 7 days, growing slowly as sqrt(months) beyond.
    """
    # Seasonal component
    try:
        seasonal = float(lt["profile"].loc[(ts.month, ts.dayofweek, ts.hour)])
    except KeyError:
        seasonal = lt["global_mean"]

    # Trend: delta from anchor year
    years_out = (ts.year - lt["anchor_year"]) + (ts.month - 1) / 12.0
    trend_adj = lt["slope"] * years_out

    p50 = seasonal + trend_adj + lt["resid_q45"]

    # Widening interval
    excess_days   = max(horizon_days - SHORTTERM_DAYS, 0.0)
    horizon_scale = 1.0 + (excess_days / 30.0) ** 0.5 * 0.25
    half_interval = lt["resid_std"] * 1.96 * horizon_scale

    return p50 - half_interval, p50, p50 + half_interval


# ── Gap actuals fetch ─────────────────────────────────────────────────────────

def fetch_gap_actuals(zone: str, gap_start: pd.Timestamp, gap_end: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch prices + net_imports from ENTSOE for the period between training data
    and the eval window (typically the last 1–2 days not yet in final_dataset.parquet).

    Only price and net_imports are populated; all other columns are NaN so the
    existing proxy logic in build_eval_row handles generation/weather features.

    Called automatically when --predict is used if a gap exists.
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

    # ENTSOE queries need tz-aware Timestamps; use Brussels (CET/CEST) like ingestion.py
    q_start = pd.Timestamp(gap_start.date().isoformat(), tz="Europe/Brussels")
    q_end   = pd.Timestamp((gap_end + pd.Timedelta(days=1)).date().isoformat(), tz="Europe/Brussels")

    # ── Prices ────────────────────────────────────────────────────────────────
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

    gap_df = prices.to_frame()

    # ── Cross-border flows → net_imports ──────────────────────────────────────
    net_parts: list[pd.Series] = []
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
        net_imports = pd.concat(net_parts, axis=1).sum(axis=1)
        net_imports.name = "net_imports"
        gap_df = gap_df.join(net_imports, how="left")
        log.info("  Gap net_imports %s: %d non-null", zone, gap_df["net_imports"].notna().sum())
    else:
        log.warning("  No cross-border flow data for gap period (%s) — net_imports will be NaN", zone)

    # Clip to exact requested window
    gap_df = gap_df.loc[(gap_df.index >= gap_start) & (gap_df.index < gap_end)]
    return gap_df


# ── Eval-window feature construction ─────────────────────────────────────────

def build_eval_row(
    zdf: pd.DataFrame,
    ref: pd.DataFrame,
    zone: str,
    ts: pd.Timestamp,
    predicted_p50: dict,
    cal,
    weather_fcst: pd.DataFrame | None = None,
) -> dict:
    """
    Build one feature row for a single eval-window timestamp.

    Called sequentially so predicted_p50 is populated with all prior slots
    before this row is built — enabling honest recursive lag_1 / lag_24 fill.

    Generation/load (future unknown): same-weekday-hour mean from last 4 weeks.
    Weather: Open-Meteo 10-day forecast if provided, else same-weekday-hour proxy.
    Lags: actual prices where available; predicted p50 where recursive.
    """
    row: dict = {}

    # ── Generation / proxy features ───────────────────────────────────────────
    same_hw = ref[(ref.index.hour == ts.hour) & (ref.index.dayofweek == ts.dayofweek)]
    proxy_cols = ["load", "wind_generation", "solar_generation", "hydro_generation",
                  "residual_load", "renewable_penetration"]
    if "net_imports" in ref.columns:
        proxy_cols.append("net_imports")
    for col in proxy_cols:
        row[col] = float(same_hw[col].mean()) if len(same_hw) > 0 else float(ref[col].mean())

    # ── Weather: forecast if available, else same-weekday-hour proxy ──────────
    weather_cols = ["temperature", "wind_speed", "solar_radiation"]
    if weather_fcst is not None and ts in weather_fcst.index:
        for col in weather_cols:
            row[col] = float(weather_fcst.loc[ts, col])
    else:
        for col in weather_cols:
            row[col] = float(same_hw[col].mean()) if len(same_hw) > 0 else float(ref[col].mean())

    # ── Fuel: carry forward last known ────────────────────────────────────────
    row["gas_price"]    = float(ref["gas_price"].iloc[-1])
    row["carbon_price"] = float(ref["carbon_price"].iloc[-1])

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
        return float(ref[fallback_col].mean())

    row["lag_168"] = lookup(ts - pd.Timedelta(hours=168), "lag_168")
    row["lag_24"]  = lookup(ts - pd.Timedelta(hours=24),  "lag_24")
    row["lag_1"]   = lookup(ts - pd.Timedelta(hours=1),   "lag_1")

    # Rolling means: use trailing actuals from the known dataset
    row["price_roll_24h"]  = float(zdf["price"].iloc[-24:].mean())
    row["price_roll_168h"] = float(zdf["price"].iloc[-168:].mean())

    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def main(predict: bool = False, pred_start: str | None = None, pred_end: str | None = None) -> None:
    log.info("Loading dataset from %s", DATA_PATH)
    df = pd.read_parquet(DATA_PATH)

    # net_imports only present after ENTSOE pipeline run; drop from FEATURES if absent
    active_features = [f for f in FEATURES if f in df.columns]
    if len(active_features) < len(FEATURES):
        missing = set(FEATURES) - set(active_features)
        log.warning("Features missing from dataset (re-run pipeline): %s", missing)
    globals()["FEATURES"] = active_features  # propagate to train_zone / build_eval_row

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
        zdf = df.xs(zone, level="zone").sort_index()
        cqr[zone] = calibrate_zone(zdf, all_models[zone], zone)

    log.info("━" * 56)
    log.info("LONG-TERM MODEL  (seasonal profile + annual trend)")
    lt_models = {}
    for zone in ZONES:
        zdf = df.xs(zone, level="zone").sort_index()
        lt_models[zone] = build_longterm_model(zdf, zone)

    if not predict:
        return

    log.info("━" * 56)
    log.info("Generating predictions (CQR-adjusted)")

    import holidays as hdays
    _ZONE_COUNTRY = {"DE-LU": "DE", "ES": "ES"}

    # Hackathon eval window defaults; override via --start / --end
    _DEFAULT_START = "2026-05-08 17:00"
    _DEFAULT_END   = "2026-05-09 22:00"
    eval_start = pd.Timestamp(pred_start or _DEFAULT_START, tz="UTC")
    eval_end   = pd.Timestamp(pred_end   or _DEFAULT_END,   tz="UTC")
    eval_idx   = pd.date_range(eval_start, eval_end, freq="h")
    log.info("  Window: %s → %s  (%d slots)", eval_start, eval_end, len(eval_idx))

    # Fetch Open-Meteo forecast — only for the short-term portion (≤16 days)
    # Long-term slots use the seasonal profile; no point fetching a 2-year forecast.
    from ingestion import fetch_weather_forecast as _fetch_wx_fcst
    _WX_FCST_LIMIT_DAYS = 14
    _today = pd.Timestamp.utcnow().normalize()
    fcst_end_capped = min(eval_end, _today + pd.Timedelta(days=_WX_FCST_LIMIT_DAYS))
    zone_weather_fcst: dict[str, pd.DataFrame | None] = {}
    if fcst_end_capped > eval_start:
        fcst_date_start = eval_start.strftime("%Y-%m-%d")
        fcst_date_end   = fcst_end_capped.strftime("%Y-%m-%d")
        for zone in ZONES:
            try:
                raw = _fetch_wx_fcst(zone, fcst_date_start, fcst_date_end)
                raw["time"] = pd.to_datetime(raw["time"], utc=True)
                raw = raw.set_index("time")
                zone_weather_fcst[zone] = raw
                log.info("  Weather forecast %s: %d rows (%s → %s)",
                         zone, len(raw), fcst_date_start, fcst_date_end)
            except Exception as exc:
                log.warning("  Weather forecast fetch failed for %s: %s — proxy", zone, exc)
                zone_weather_fcst[zone] = None
    else:
        log.info("  Window beyond 14-day forecast horizon — using seasonal proxy for all slots")
        for zone in ZONES:
            zone_weather_fcst[zone] = None

    zone_preds = {}
    for zone in ZONES:
        zdf = df.xs(zone, level="zone").sort_index()

        # Patch zdf with gap actuals: prices + net_imports for any hours between
        # training tail and eval start (typically the most recent 1–2 days).
        # This gives lag_24 lookups real values instead of falling back to averages.
        gap_start = zdf.index[-1] + pd.Timedelta(hours=1)
        if gap_start < eval_start:
            log.info("Fetching gap actuals for %s (%s → %s) ...", zone, gap_start.date(), eval_start.date())
            gap_df = fetch_gap_actuals(zone, gap_start, eval_start)
            if len(gap_df) > 0:
                zdf = pd.concat([zdf, gap_df])
                zdf = zdf[~zdf.index.duplicated(keep="last")].sort_index()
                log.info("  zdf extended: tail now %s", zdf.index[-1])

        ref = zdf[eval_start - pd.Timedelta(weeks=4) : eval_start - pd.Timedelta(hours=1)]
        cal = hdays.country_holidays(_ZONE_COUNTRY[zone])
        weather_fcst = zone_weather_fcst[zone]

        predicted_p50 = {}  # populated slot-by-slot for recursive lags
        p025_list, p50_list, p975_list = [], [], []
        regime_log = {"shortterm": 0, "longterm": 0}

        q_iv    = cqr[zone]["interval"]
        q_50    = cqr[zone]["p50"]
        lt      = lt_models[zone]
        zdf_tail = zdf.index[-1]   # reference point for horizon calculation

        for ts in eval_idx:
            horizon_days = (ts - zdf_tail).total_seconds() / 86400.0

            if horizon_days <= SHORTTERM_DAYS:
                # ── Short-term: LightGBM + CQR ────────────────────────────────
                row  = build_eval_row(zdf, ref, zone, ts, predicted_p50, cal, weather_fcst)
                x    = pd.DataFrame([row])[FEATURES]

                p025 = float(all_models[zone][0.025].predict(x)[0])
                p50  = float(all_models[zone][0.45].predict(x)[0])
                p975 = float(all_models[zone][0.975].predict(x)[0])

                p025 = min(p025, p50)
                p975 = max(p975, p50)

                p50  = p50  + q_50
                p025 = p025 - q_iv
                p975 = p975 + q_iv

                p025 = min(p025, p50)
                p975 = max(p975, p50)
                regime_log["shortterm"] += 1

            else:
                # ── Long-term: seasonal profile + annual trend ─────────────────
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

    # Build submission CSV
    # Timestamps: ISO 8601 with CEST offset (+02:00, Europe is on summer time in May)
    cest = pd.DatetimeTZDtype(tz="Europe/Berlin")
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

    out.to_csv(OUT_PATH, index=False, float_format="%.4f")
    log.info("Saved %s  (%d rows)", OUT_PATH, len(out))
    log.info("Preview:\n%s", out.to_string())

    # ── Backtest: if the window overlaps known actuals, report accuracy ────────
    log.info("━" * 56)
    any_backtest = False
    for zone in ZONES:
        zdf = df.xs(zone, level="zone").sort_index()
        known = zdf[(zdf.index >= eval_start) & (zdf.index <= eval_end)]["price"].dropna()
        if len(known) == 0:
            continue
        any_backtest = True
        p50_arr  = np.array(zone_preds[zone]["p50"])
        p025_arr = np.array(zone_preds[zone]["p025"])
        p975_arr = np.array(zone_preds[zone]["p975"])
        # Align: only slots where actuals exist
        mask     = np.array([ts in known.index for ts in eval_idx])
        y_known  = known.reindex(eval_idx[mask]).values
        log.info("  BACKTEST %s (%d slots with actuals):", zone, mask.sum())
        log.info("    MAE p50      : %.2f EUR/MWh", np.abs(y_known - p50_arr[mask]).mean())
        log.info("    Pinball 0.45 : %.4f", pinball(y_known, p50_arr[mask], 0.45))
        log.info("    Coverage     : %.1f%%", coverage(y_known, p025_arr[mask], p975_arr[mask]) * 100)
    if not any_backtest:
        log.info("  No actuals in dataset for this window — forward prediction only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true",
                        help="Generate predictions.csv for the specified window")
    parser.add_argument("--start", default=None,
                        help="Prediction window start, UTC (e.g. '2026-05-10 17:00'). "
                             "Defaults to hackathon eval window 2026-05-08 17:00.")
    parser.add_argument("--end", default=None,
                        help="Prediction window end, UTC (e.g. '2026-05-11 22:00'). "
                             "Defaults to hackathon eval window 2026-05-09 22:00.")
    args = parser.parse_args()
    main(predict=args.predict, pred_start=args.start, pred_end=args.end)
