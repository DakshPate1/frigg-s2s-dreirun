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
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_pinball_loss

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_PATH  = ROOT / "data" / "processed" / "final_dataset.parquet"
OUT_PATH   = ROOT / "predictions.csv"

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


# ── Eval-window feature construction ─────────────────────────────────────────

def build_eval_row(
    zdf: pd.DataFrame,
    ref: pd.DataFrame,
    zone: str,
    ts: pd.Timestamp,
    predicted_p50: dict,
    cal,
) -> dict:
    """
    Build one feature row for a single eval-window timestamp.

    Called sequentially so predicted_p50 is populated with all prior slots
    before this row is built — enabling honest recursive lag_1 / lag_24 fill.

    Generation/load (future unknown): same-weekday-hour mean from last 4 weeks.
    Lags: actual prices where available; predicted p50 where recursive.
    """
    row: dict = {}

    # ── Generation / weather proxies ──────────────────────────────────────────
    same_hw = ref[(ref.index.hour == ts.hour) & (ref.index.dayofweek == ts.dayofweek)]
    proxy_cols = ["load", "wind_generation", "solar_generation", "hydro_generation",
                  "temperature", "wind_speed", "solar_radiation",
                  "residual_load", "renewable_penetration"]
    # net_imports: same weekday-hour proxy (cross-border flows follow weekly patterns)
    if "net_imports" in ref.columns:
        proxy_cols.append("net_imports")
    for col in proxy_cols:
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

def main(predict: bool = False) -> None:
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

    if not predict:
        return

    log.info("━" * 56)
    log.info("Generating eval-window predictions (CQR-adjusted)")

    import holidays as hdays
    _ZONE_COUNTRY = {"DE-LU": "DE", "ES": "ES"}

    eval_start = pd.Timestamp("2026-05-08 17:00", tz="UTC")
    eval_end   = pd.Timestamp("2026-05-09 22:00", tz="UTC")
    eval_idx   = pd.date_range(eval_start, eval_end, freq="h")

    zone_preds = {}
    for zone in ZONES:
        zdf = df.xs(zone, level="zone").sort_index()
        ref = zdf[eval_start - pd.Timedelta(weeks=4) : eval_start - pd.Timedelta(hours=1)]
        cal = hdays.country_holidays(_ZONE_COUNTRY[zone])

        predicted_p50 = {}  # populated slot-by-slot for recursive lags
        p025_list, p50_list, p975_list = [], [], []

        q_iv = cqr[zone]["interval"]
        q_50 = cqr[zone]["p50"]

        for ts in eval_idx:
            row = build_eval_row(zdf, ref, zone, ts, predicted_p50, cal)
            x   = pd.DataFrame([row])[FEATURES]

            p025 = float(all_models[zone][0.025].predict(x)[0])
            p50  = float(all_models[zone][0.45].predict(x)[0])
            p975 = float(all_models[zone][0.975].predict(x)[0])

            # Enforce quantile ordering before CQR
            p025 = min(p025, p50)
            p975 = max(p975, p50)

            # Apply CQR corrections
            p50  = p50  + q_50           # shift p50 toward true 45th percentile
            p025 = p025 - q_iv           # inflate lower bound
            p975 = p975 + q_iv           # inflate upper bound

            # Re-enforce ordering after calibration
            p025 = min(p025, p50)
            p975 = max(p975, p50)

            predicted_p50[ts] = p50      # calibrated p50 for subsequent lag lookups
            p025_list.append(p025)
            p50_list.append(p50)
            p975_list.append(p975)

        zone_preds[zone] = {"p025": p025_list, "p50": p50_list, "p975": p975_list}
        log.info("  %s: mean p50=%.2f  band=%.2f  (CQR: iv±%.2f  p50+%.2f)",
                 zone,
                 np.mean(p50_list),
                 np.mean(np.array(p975_list) - np.array(p025_list)),
                 q_iv, q_50)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true",
                        help="Also generate predictions.csv for eval window")
    args = parser.parse_args()
    main(predict=args.predict)
