"""Generate model.ipynb using nbformat."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ── Cell 0: Title + Methodology ──────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
# Frigg S2S — Electricity Price Forecasting

**Zones:** DE-LU (Germany-Luxembourg) · ES (Spain/OMIE)
**Task:** Predict Day-Ahead Auction (DAA) prices (EUR/MWh) at hourly resolution
**Metric:** Pinball loss at q = 0.45 — overestimation penalised ~1.22× more than underestimation

---

## Methodology

### Two-regime forecasting system

| Horizon | Method | Key signals |
|---------|--------|-------------|
| Short-term (≤ 7 days) | Quantile LightGBM (q = 0.025, 0.45, 0.975) | Weather, generation, fuel prices, price lags |
| Long-term (> 7 days → 20 years) | Seasonal profile + YoY trend | Historical hourly/monthly percentiles, annual price trend |

The regime boundary is pragmatic: beyond ~7 days, weather forecast skill collapses and day-ahead generation schedules are unavailable. The long-term model falls back to structural seasonality and trend extrapolation — the only defensible signal at multi-month horizons.

### Short-term model: Quantile LightGBM

**Feature selection** — derived from Tschora (2024, INSA Lyon) SHAP analysis and ENTSO-E live-data feature importance experiments on 2024–2026 DE-LU data:

1. **Fundamental drivers**: `load`, `wind_generation`, `solar_generation`, `hydro_generation`, `temperature`, `wind_speed`, `solar_radiation`
2. **Fuel / carbon**: `gas_price` (TTF spot), `carbon_price` (KRBN EUA proxy)
3. **Derived**: `residual_load` = load − wind − solar; `renewable_penetration` = (wind + solar) / load
4. **Calendar (circular)**: sin/cos encoding of hour, weekday, month, week-of-year; `is_holiday` flag
5. **Price history**: `lag_1`, `lag_24`, `lag_168` (1h / 24h / 7d autoregression); rolling 24h / 168h means

p50 is trained at **q = 0.45** (not 0.5) to match the pinball scoring function — this shifts the point forecast slightly downward, systematically avoiding the more-penalised overestimation side.

**Evaluation window lag-gap** — `lag_1` and `lag_24` for the eval window (May 8–9) require D-1 prices not yet in the dataset. Solved by sequential slot-by-slot prediction: each slot's predicted p50 is inserted into a cache and used as the lag value for subsequent slots.

### Long-term model: Seasonal profile + trend

For multi-week to multi-year horizons:
- **Seasonal profile**: empirical mean and 2.5th/97.5th percentile of historical price per (month × hour) from 2022–2024 training data
- **Trend**: linear year-over-year average price change, extrapolated forward
- **Uncertainty**: direct empirical quantile bands from training data — calibrated without any model assumption

### Why DE-LU and ES models learn differently

| | DE-LU | ES |
|--|-------|-----|
| Top driver | `wind_generation` / `load` | `solar_generation` / `temperature` |
| Negative prices | Frequent (wind surplus + low demand on holidays/weekends) | Rare (less renewable excess, weaker interconnection) |
| Cross-border effect | Strong (NL, FR, AT, CH, DK coupling) | Weak (Pyrenees bottleneck ~2.8 GW to France) |
| Gas price sensitivity | High in thermal-dominant hours | Moderate (more hydro buffer) |

The identical feature vocabulary is intentional — differences in learned feature weights reflect genuine market structure, not data availability.
"""))

# ── Cell 1: Install + imports ─────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
%pip install -q lightgbm scikit-learn pandas numpy matplotlib holidays pyarrow"""))

cells.append(nbf.v4.new_code_cell("""\
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_pinball_loss
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import holidays as hdays

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
"""))

# ── Cell 2: Config ────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
ROOT      = Path("..").resolve()
DATA_PATH = ROOT / "data" / "processed" / "final_dataset.parquet"
OUT_PATH  = ROOT / "predictions.csv"

FEATURES = [
    "load", "wind_generation", "solar_generation", "hydro_generation",
    "temperature", "wind_speed", "solar_radiation",
    "gas_price", "carbon_price",
    "residual_load", "renewable_penetration",
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    "month_sin", "month_cos",
    "week_sin", "week_cos",
    "is_holiday",
    "lag_1", "lag_24", "lag_168",
    "price_roll_24h", "price_roll_168h",
]
TARGET    = "price"
ZONES     = ["DE-LU", "ES"]
QUANTILES = [0.025, 0.45, 0.975]
TRAIN_END = "2025-01-01"
VAL_END   = "2026-01-01"

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
"""))

# ── Cell 3: Load data ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Data Loading and Overview"))

cells.append(nbf.v4.new_code_cell("""\
df = pd.read_parquet(DATA_PATH)

ts_idx = df.index.get_level_values("timestamp")
print(f"Dataset  : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Zones    : {df.index.get_level_values('zone').unique().tolist()}")
print(f"Range    : {ts_idx.min().date()} → {ts_idx.max().date()}")
print()
df.describe().round(2)
"""))

# ── Cell 4: EDA ───────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## Exploratory Data Analysis

### Side-by-side zone comparison

Key contrasts visible from the data:
- **DE-LU** has a heavier left tail (negative prices from renewable surplus) and a strong overnight trough
- **ES** is more symmetric, with a pronounced midday solar dip and higher summer prices (cooling demand)
- Both zones share the morning peak–off-peak–evening peak daily shape, but the magnitudes and drivers differ
"""))

cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
zone_colors = {"DE-LU": "#1565C0", "ES": "#BF360C"}

for i, zone in enumerate(ZONES):
    zdf = df.xs(zone, level="zone").copy()
    c = zone_colors[zone]

    # Price distribution
    ax = axes[i, 0]
    zdf["price"].clip(-100, 300).hist(bins=120, ax=ax, color=c, alpha=0.75, edgecolor="none")
    ax.axvline(0, color="red", lw=1, alpha=0.6, linestyle="--")
    ax.axvline(zdf["price"].median(), color="black", lw=1, linestyle=":", label=f"Median={zdf['price'].median():.0f}")
    ax.set_title(f"{zone} — Price distribution")
    ax.set_xlabel("EUR/MWh (clipped to [-100, 300])")
    ax.legend(fontsize=8)

    # Hourly profile (mean ± 1 std)
    ax = axes[i, 1]
    hp = zdf.groupby(zdf.index.hour)["price"].agg(["mean", "std"])
    ax.fill_between(hp.index, hp["mean"] - hp["std"], hp["mean"] + hp["std"], alpha=0.2, color=c)
    ax.plot(hp.index, hp["mean"], color=c, lw=2, marker="o", markersize=4)
    ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.5)
    ax.set_title(f"{zone} — Hourly profile (mean ± 1σ)")
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("EUR/MWh")
    ax.set_xticks(range(0, 24, 4))

    # Monthly profile
    ax = axes[i, 2]
    mp = zdf.groupby(zdf.index.month)["price"].mean()
    bars = ax.bar(mp.index, mp.values, color=c, alpha=0.8, edgecolor="none")
    ax.set_title(f"{zone} — Monthly mean price")
    ax.set_xlabel("Month")
    ax.set_ylabel("EUR/MWh")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    ax.axhline(mp.mean(), color="black", lw=1, linestyle=":", alpha=0.6)

plt.suptitle("EDA — Price profiles by zone (2021–2025)", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("eda_price_profiles.png", bbox_inches="tight")
plt.show()
"""))

# ── Cell 5: Renewable vs price scatter ───────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for i, zone in enumerate(ZONES):
    zdf = df.xs(zone, level="zone").copy()
    c = zone_colors[zone]
    sample = zdf.sample(min(8000, len(zdf)), random_state=42)

    ax = axes[i, 0]
    ax.scatter(sample["wind_generation"] / 1e3, sample["price"].clip(-100, 300),
               alpha=0.08, s=4, color=c)
    ax.set_xlabel("Wind generation (GW)")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.set_title(f"{zone} — Wind vs price")

    ax = axes[i, 1]
    ax.scatter(sample["solar_generation"] / 1e3, sample["price"].clip(-100, 300),
               alpha=0.08, s=4, color=c)
    ax.set_xlabel("Solar generation (GW)")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.set_title(f"{zone} — Solar vs price")

plt.tight_layout()
plt.savefig("eda_renewable_vs_price.png", bbox_inches="tight")
plt.show()

print("Correlation with price:")
for zone in ZONES:
    zdf = df.xs(zone, level="zone")
    corr = zdf[["wind_generation", "solar_generation", "load", "gas_price", "lag_24"]].corrwith(zdf["price"])
    print(f"  {zone}:")
    for feat, v in corr.items():
        print(f"    {feat:<25} {v:+.3f}")
"""))

# ── Cell 6: Feature engineering overview ─────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## Feature Engineering

All features are derived uniformly for both zones. The same feature names are used — only the learned weights differ.

### Feature groups

| Group | Features | Rationale |
|-------|----------|-----------|
| Generation & load | `load`, `wind_generation`, `solar_generation`, `hydro_generation` | Primary supply/demand balance |
| Weather | `temperature`, `wind_speed`, `solar_radiation` | Drives renewable output and heating/cooling demand |
| Fuel / carbon | `gas_price`, `carbon_price` | Marginal cost of peaking gas plants; regulatory cost |
| Derived | `residual_load`, `renewable_penetration` | Nonlinear interactions: curtailment regime, scarcity signal |
| Calendar (circular) | `hour_sin/cos`, `weekday_sin/cos`, `month_sin/cos`, `week_sin/cos` | Circular sin/cos preserves periodicity continuity (hour 23 adjacent to 0) |
| Holiday | `is_holiday` | Public holidays collapse industrial demand → solar saturation → negative prices |
| Price lags | `lag_1`, `lag_24`, `lag_168` | D-1 same-hour, D-7 same-hour; dominant autoregressive signal |
| Rolling means | `price_roll_24h`, `price_roll_168h` | Recent price level context |

**Not included:** Cross-zone prices (ES isolated; symmetry constraint). D-2 / D-3 lags (<5% SHAP weight per Tschora 2024 — not worth the feature cost). Neighbour network graph (requires full 35-node European topology).
"""))

# ── Cell 7: Model training ────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Model Training"))

cells.append(nbf.v4.new_code_cell("""\
def pinball(y_true, y_pred, q):
    return float(mean_pinball_loss(y_true, y_pred, alpha=q))

def coverage(y_true, lo, hi):
    return float(((y_true >= lo) & (y_true <= hi)).mean())

def train_zone(zdf, zone):
    train = zdf[zdf.index <  TRAIN_END]
    val   = zdf[(zdf.index >= TRAIN_END) & (zdf.index < VAL_END)]
    X_tr, y_tr = train[FEATURES], train[TARGET]
    X_va, y_va = val[FEATURES],   val[TARGET]

    print(f"  {zone}  train={len(X_tr):,}  val={len(X_va):,}")
    qmodels, val_preds = {}, {}

    for q in QUANTILES:
        m = lgb.LGBMRegressor(**{**LGB_BASE, "alpha": q})
        m.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        qmodels[q]   = m
        val_preds[q] = m.predict(X_va)
        print(f"    q={q:.3f}  best_iter={m.best_iteration_:,}  pinball={pinball(y_va.values, val_preds[q], q):.4f}")

    return qmodels, {"preds": val_preds, "actual": y_va, "X": X_va}

all_models, all_val = {}, {}
for zone in ZONES:
    print(f"Training {zone} ...")
    zdf = df.xs(zone, level="zone").sort_index()
    qmodels, val = train_zone(zdf, zone)
    all_models[zone] = qmodels
    all_val[zone]    = val
"""))

# ── Cell 8: Validation metrics ────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Validation (2025)"))

cells.append(nbf.v4.new_code_cell("""\
rows = []
for zone in ZONES:
    y    = all_val[zone]["actual"].values
    p025 = all_val[zone]["preds"][0.025]
    p50  = all_val[zone]["preds"][0.45]
    p975 = all_val[zone]["preds"][0.975]
    lag  = all_val[zone]["X"]["lag_168"].values

    rows.append({
        "Zone"      : zone,
        "MAE p50"   : round(np.abs(y - p50).mean(), 2),
        "Pinball 0.45": round(pinball(y, p50, 0.45), 4),
        "Pinball 0.025": round(pinball(y, p025, 0.025), 4),
        "Pinball 0.975": round(pinball(y, p975, 0.975), 4),
        "Coverage p025-p975": f"{coverage(y, p025, p975)*100:.1f}%",
        "Band width": round((p975 - p025).mean(), 2),
        "Naive MAE (lag_168)": round(np.abs(y - lag).mean(), 2),
    })

pd.DataFrame(rows).set_index("Zone")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Validation residual distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, zone in enumerate(ZONES):
    y   = all_val[zone]["actual"].values
    p50 = all_val[zone]["preds"][0.45]
    err = y - p50
    ax = axes[i]
    ax.hist(err.clip(-100, 100), bins=100, color=zone_colors[zone], alpha=0.75, edgecolor="none")
    ax.axvline(0, color="black", lw=1, linestyle="--")
    ax.axvline(err.mean(), color="red", lw=1.5, linestyle="-", label=f"Mean={err.mean():.1f}")
    ax.set_title(f"{zone} — Validation residuals (actual − p50)")
    ax.set_xlabel("EUR/MWh (clipped)")
    ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
"""))

# ── Cell 9: Feature importance ────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## Cross-Zone Comparison — Feature Importance

Feature importance is measured as split count in the p50 (q=0.45) model. Key observations:
- Both zones: `wind_speed` and `load` dominate — fundamental supply/demand balance drives price
- DE-LU: stronger `wind_generation` importance (offshore + onshore; frequent surplus events)
- ES: `solar_generation` and `temperature` rank higher (solar saturation midday; cooling demand)
- `lag_1` and `lag_24` both appear in top-5 for both zones — short-term price momentum
- `is_holiday` appears lower by split count but carries outsized weight for tail events
"""))

cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for i, zone in enumerate(ZONES):
    m = all_models[zone][0.45]
    imp = pd.Series(m.feature_importances_, index=FEATURES).sort_values(ascending=True).tail(18)
    ax = axes[i]
    bars = ax.barh(imp.index, imp.values, color=zone_colors[zone], alpha=0.8, edgecolor="none")
    ax.set_title(f"{zone} — Feature importance (p50 model, split count)")
    ax.set_xlabel("Split count")

plt.tight_layout()
plt.savefig("feature_importance.png", bbox_inches="tight")
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# Normalised importance side-by-side comparison
imp_data = {}
for zone in ZONES:
    m = all_models[zone][0.45]
    s = pd.Series(m.feature_importances_, index=FEATURES)
    imp_data[zone] = (s / s.sum() * 100).round(2)

comparison = pd.DataFrame(imp_data).sort_values("DE-LU", ascending=False)
comparison.columns = ["DE-LU %", "ES %"]
comparison["Δ (DE-LU − ES)"] = (comparison["DE-LU %"] - comparison["ES %"]).round(2)
comparison
"""))

# ── Cell 10: Long-term forecast ────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## Long-Term Forecast (2 Years)

For horizons beyond ~7 days, granular weather forecasts and generation schedules are unavailable.
The model switches to a **seasonal profile** approach:

1. **Profile construction**: compute empirical mean and 2.5th/97.5th percentile of historical price per (month × hour) from 2022–2024 training data
2. **Trend extrapolation**: estimate year-over-year mean price drift via linear regression on annual averages; apply as a forward adjustment
3. **Uncertainty**: direct empirical quantile bands — no parametric assumptions, calibrated on ~26,000 h of training data per zone

This is intentionally conservative: for 20-year infrastructure financing decisions, the uncertainty should widen substantially with horizon. The empirical 2.5/97.5 bands from historical data serve as a reasonable lower bound on long-run uncertainty.
"""))

cells.append(nbf.v4.new_code_cell("""\
def build_seasonal_profile(df, zone, start_year="2022", end_year="2024"):
    zdf = df.xs(zone, level="zone").sort_index()
    data = zdf[start_year:end_year].copy()
    data["month"] = data.index.month
    data["hour"]  = data.index.hour

    profile = (
        data.groupby(["month", "hour"])["price"]
        .agg(mean="mean", std="std",
             q025=lambda x: x.quantile(0.025),
             q975=lambda x: x.quantile(0.975))
        .reset_index()
    )

    # Linear trend from annual means
    by_year = data.groupby(data.index.year)["price"].mean()
    years   = by_year.index.values.astype(float)
    prices  = by_year.values
    trend   = float(np.polyfit(years, prices, 1)[0]) if len(years) >= 2 else 0.0
    base_year = float(years[-1])

    return profile, trend, base_year


def long_term_forecast(df, zone, start="2026-05-06", periods=17520):
    """Seasonal profile forecast — 2 years (17520 hourly slots)."""
    profile, trend, base_year = build_seasonal_profile(df, zone)

    future_idx = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    p50_list, p025_list, p975_list = [], [], []

    profile_idx = profile.set_index(["month", "hour"])

    for ts in future_idx:
        key = (ts.month, ts.hour)
        row = profile_idx.loc[key] if key in profile_idx.index else None
        years_ahead = (ts.year - base_year) + (ts.month - 1) / 12
        adj = trend * years_ahead

        if row is not None:
            p50_list.append(float(row["mean"]) + adj)
            p025_list.append(float(row["q025"]) + adj)
            p975_list.append(float(row["q975"]) + adj)
        else:
            mid = profile["mean"].mean() + adj
            p50_list.append(mid)
            p025_list.append(mid - 30)
            p975_list.append(mid + 30)

    return pd.DataFrame({"p025": p025_list, "p50": p50_list, "p975": p975_list},
                        index=future_idx)


print("Seasonal trend (EUR/MWh / year):")
for zone in ZONES:
    _, trend, _ = build_seasonal_profile(df, zone)
    print(f"  {zone}: {trend:+.2f}")
"""))

cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

for i, zone in enumerate(ZONES):
    lt = long_term_forecast(df, zone)
    lt_daily = lt.resample("D").mean()
    lt_monthly = lt.resample("ME").mean()

    ax = axes[i]
    ax.fill_between(lt_daily.index, lt_daily["p025"], lt_daily["p975"],
                    alpha=0.2, color=zone_colors[zone], label="p025–p975 (empirical quantile band)")
    ax.plot(lt_monthly.index, lt_monthly["p50"],
            lw=2, color=zone_colors[zone], label="p50 — monthly mean (seasonal + trend)")
    ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.4)
    ax.set_title(f"{zone} — 2-Year Long-Term Forecast (daily band, monthly p50 line)")
    ax.set_ylabel("EUR/MWh")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

plt.tight_layout()
plt.savefig("long_term_forecast.png", bbox_inches="tight")
plt.show()
"""))

# ── Cell 11: Eval window prediction ──────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## Evaluation Window Prediction

**Window:** 2026-05-08 18:00 CEST → 2026-05-09 23:00 CEST (30 hourly slots)

Context: Friday evening into Saturday. Renewables saturation risk on Saturday midday (solar peak, reduced industrial demand). May 9 is not a public holiday, but weekend dynamics apply.

**Lag-gap handling:** lag_168 (7 days back) is fully in the historical dataset. lag_24 / lag_1 require D-1 prices (May 7–8). For early slots in the window, D-1 actuals are available from the dataset; for subsequent slots, predicted p50 values are used recursively.
"""))

cells.append(nbf.v4.new_code_cell("""\
_ZONE_COUNTRY = {"DE-LU": "DE", "ES": "ES"}

def build_eval_row(zdf, ref, zone, ts, predicted_p50, cal):
    row = {}

    same_hw = ref[(ref.index.hour == ts.hour) & (ref.index.dayofweek == ts.dayofweek)]
    for col in ["load", "wind_generation", "solar_generation", "hydro_generation",
                "temperature", "wind_speed", "solar_radiation",
                "residual_load", "renewable_penetration"]:
        row[col] = float(same_hw[col].mean()) if len(same_hw) > 0 else float(ref[col].mean())

    row["gas_price"]    = float(ref["gas_price"].iloc[-1])
    row["carbon_price"] = float(ref["carbon_price"].iloc[-1])

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

    def lookup(lag_ts, fallback_col):
        if lag_ts in zdf.index:
            return float(zdf.loc[lag_ts, "price"])
        if lag_ts in predicted_p50:
            return predicted_p50[lag_ts]
        return float(ref[fallback_col].mean())

    row["lag_168"] = lookup(ts - pd.Timedelta(hours=168), "lag_168")
    row["lag_24"]  = lookup(ts - pd.Timedelta(hours=24),  "lag_24")
    row["lag_1"]   = lookup(ts - pd.Timedelta(hours=1),   "lag_1")
    row["price_roll_24h"]  = float(zdf["price"].iloc[-24:].mean())
    row["price_roll_168h"] = float(zdf["price"].iloc[-168:].mean())

    return row


eval_start = pd.Timestamp("2026-05-08 17:00", tz="UTC")
eval_end   = pd.Timestamp("2026-05-09 22:00", tz="UTC")
eval_idx   = pd.date_range(eval_start, eval_end, freq="h")

zone_preds = {}
for zone in ZONES:
    zdf = df.xs(zone, level="zone").sort_index()
    ref = zdf[eval_start - pd.Timedelta(weeks=4) : eval_start - pd.Timedelta(hours=1)]
    cal = hdays.country_holidays(_ZONE_COUNTRY[zone])

    predicted_p50 = {}
    p025_list, p50_list, p975_list = [], [], []

    for ts in eval_idx:
        row  = build_eval_row(zdf, ref, zone, ts, predicted_p50, cal)
        x    = pd.DataFrame([row])[FEATURES]
        p025 = float(all_models[zone][0.025].predict(x)[0])
        p50  = float(all_models[zone][0.45].predict(x)[0])
        p975 = float(all_models[zone][0.975].predict(x)[0])
        p025 = min(p025, p50)
        p975 = max(p975, p50)

        predicted_p50[ts] = p50
        p025_list.append(p025); p50_list.append(p50); p975_list.append(p975)

    zone_preds[zone] = {"p025": p025_list, "p50": p50_list, "p975": p975_list}
    print(f"{zone}: mean p50={np.mean(p50_list):.2f}  band={np.mean(np.array(p975_list)-np.array(p025_list)):.2f}")
"""))

cells.append(nbf.v4.new_code_cell("""\
cest = eval_idx.tz_convert("Europe/Berlin")

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

for i, zone in enumerate(ZONES):
    ax = axes[i]
    p025 = zone_preds[zone]["p025"]
    p50  = zone_preds[zone]["p50"]
    p975 = zone_preds[zone]["p975"]
    c = zone_colors[zone]

    ax.fill_between(cest, p025, p975, alpha=0.25, color=c, label="p025–p975 band")
    ax.plot(cest, p50,  lw=2.5, color=c, label="p50 — point forecast")
    ax.plot(cest, p025, lw=0.8, color=c, alpha=0.6, linestyle="--")
    ax.plot(cest, p975, lw=0.8, color=c, alpha=0.6, linestyle="--")
    ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.5)
    ax.set_title(f"{zone} — Evaluation window forecast (8–9 May 2026, CEST)")
    ax.set_ylabel("EUR/MWh")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))

plt.tight_layout()
plt.savefig("eval_window_forecast.png", bbox_inches="tight")
plt.show()
"""))

# ── Cell 12: Save predictions.csv ─────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
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
print(f"Saved: {OUT_PATH}  ({len(out)} rows)")
print()
print(out.to_string(index=False))
"""))

nb.cells = cells

with open("model.ipynb", "w") as f:
    nbf.write(nb, f)

print("Generated model.ipynb")
