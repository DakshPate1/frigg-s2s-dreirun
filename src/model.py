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

# Train on 2021–2024; validate on 2025; eval window is May 2026
TRAIN_END = "2025-01-01"
VAL_END   = "2026-01-01"

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

    if not predict:
        return

    log.info("━" * 56)
    log.info("Generating eval-window predictions")

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

        for ts in eval_idx:
            row = build_eval_row(zdf, ref, zone, ts, predicted_p50, cal)
            x   = pd.DataFrame([row])[FEATURES]

            p025 = float(all_models[zone][0.025].predict(x)[0])
            p50  = float(all_models[zone][0.45].predict(x)[0])
            p975 = float(all_models[zone][0.975].predict(x)[0])

            # Guarantee quantile ordering
            p025 = min(p025, p50)
            p975 = max(p975, p50)

            predicted_p50[ts] = p50   # available for subsequent slots' lags
            p025_list.append(p025)
            p50_list.append(p50)
            p975_list.append(p975)

        zone_preds[zone] = {"p025": p025_list, "p50": p50_list, "p975": p975_list}
        log.info("  %s: mean p50=%.2f  band=%.2f",
                 zone,
                 np.mean(p50_list),
                 np.mean(np.array(p975_list) - np.array(p025_list)))

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
