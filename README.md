# Frigg S2S — Electricity Price Forecasting

End-to-end pipeline for forecasting Day-Ahead Auction (DAA) electricity prices for two European bidding zones: **DE-LU** (Germany-Luxembourg) and **ES** (Spain).

**Evaluation target:** 2026-05-11 02:00 CEST → 2026-05-12 01:00 CEST (**24 hourly slots**, May 11 00:00–23:00 UTC)
**Scoring:** asymmetric pinball loss at q=0.45 — overestimation penalised ~1.22× more than underestimation.
**Deadline:** Saturday 9 May 2026 23:59 CEST

---

## Validation results (updated_maiya, 2025 holdout)

| Zone | MAE p50 | Pinball 0.45 | Coverage p025–p975 | Naive MAE |
|------|--------:|-------------:|-------------------:|----------:|
| DE-LU | **7.14 EUR/MWh** | 3.45 | 88.0% | 32.82 |
| ES | **6.47 EUR/MWh** | 3.25 | 79.7% | 29.22 |

vs original main branch: DE-LU 8.53 / ES 6.95 — **16% / 7% improvement** from expanded feature set.

**CQR calibration (Jan–May 2026, n≈3047/zone):**

| Zone | Coverage raw → CQR | Mondrian b=0 Q_hat | Mondrian b=1 Q_hat | p50 shift |
|------|-------------------:|-------------------:|-------------------:|----------:|
| DE-LU | 88.4% → 95.0% | 2.37 (normal weekday) | 4.34 (weekend/holiday) | +1.42 |
| ES | 72.4% → 95.0% | 4.31 (normal weekday) | 3.72 (weekend/holiday) | −1.62 |

---

## Setup

```bash
pip install -r requirements.txt

# Add your ENTSOE token to .env:
cp .env.example .env
# edit .env: ENTSOE_TOKEN=<your-token>
# Free key: https://transparency.entsoe.eu/usrm/user/createPublicUser

cd src
python pipeline.py              # full pipeline
python pipeline.py --from-clean # skip ingestion
python pipeline.py --from-align # skip ingestion + cleaning
python pipeline.py --validate-only

# Generate eval window predictions (24 slots, May 11 2026):
python model.py --predict

# Predict any window:
python model.py --predict --start "2026-05-10 17:00" --end "2026-05-12 22:00"
python model.py --predict --start "2026-05-10 00:00" --end "2028-05-10 00:00"  # 2-year LT

# Backtest (actuals auto-reported if window overlaps dataset):
python model.py --predict --start "2025-06-01 00:00" --end "2025-06-07 23:00"
```

---

## Data sources

| Source | Data | Files |
|---|---|---|
| ENTSOE Transparency Platform | DAA prices, load, generation per type, cross-border flows, day-ahead forecasts | `prices_{zone}.csv`, `generation_{zone}.csv`, `crossborder_{zone}.csv`, `forecasts_{zone}.csv` |
| ENTSOE (neighbors) | FR, NL, CH, DK day-ahead prices | `prices_neighbors.csv` |
| ENTSOE (nuclear) | ES nuclear REMIT unavailability | `nuclear_ES.csv` |
| Open-Meteo archive | Multi-city weather: 26 stations, 9 groups | `weather_{zone}.csv` |
| Open-Meteo ensemble | ECMWF ensemble spread (wind/solar uncertainty) | `ensemble_{zone}.csv` |
| yfinance `TTF=F` | TTF gas futures EUR/MWh | `fuel_prices.csv` |
| yfinance `KRBN` | EUA carbon proxy USD | `fuel_prices.csv` |
| yfinance `MTF=F` | API2 coal futures USD/t | `fuel_prices.csv` |
| energy-charts.info | Fallback prices + generation (rate-limited) | retained in `ingestion.py` |

Training window: **2023-05-01 → 2026-05-07** (post-energy-crisis only).

---

## Features (41 total)

| Group | Features |
|---|---|
| Supply/demand | `load`, `wind_generation`, `solar_generation`, `hydro_generation`, `nuclear_generation` |
| Single-city weather (fallback) | `temperature`, `wind_speed`, `solar_radiation` |
| Multi-city weather | `wind_speed_agg`, `wind_speed_cubed`, `solar_radiation_agg`, `solar_hour_interaction`, `temperature_agg`, `temperature_sq` |
| DE-LU cross-border signals | `DK_wind_speed`, `DK_wind_speed_cubed`, `CH_precipitation`, `CH_precip_7d_sum` |
| ES hydro + nuclear | `ES_hydro_precipitation`, `ES_hydro_precip_7d_sum`, `nuclear_available_mw` |
| Fuel / carbon | `gas_price`, `carbon_price`, `coal_price` |
| Derived generation | `residual_load`, `renewable_penetration`, `residual_load_ramp` |
| Day-ahead forecasts | `residual_load_forecast`, `renewable_penetration_forecast`, `residual_load_ramp_forecast` |
| Cross-border flows | `net_imports` |
| Neighbor prices | `FR_price_lag24`, `NL_price_lag24`, `CH_price_lag24`, `DK_price_lag24` |
| Transmission spreads | `DE_LU_FR_spread`, `DE_LU_NL_spread`, `DE_LU_CH_spread`, `ES_FR_spread` |
| Cross-zone | `cross_zone_lag24` |
| Calendar (circular) | `hour_sin/cos`, `weekday_sin/cos`, `month_sin/cos`, `week_sin/cos`, `is_holiday`, `days_to_holiday`, `days_from_holiday` |
| Regime flags | `crisis_period`, `is_peak`, `negative_price_lag24` |
| Price history | `lag_1`, `lag_24`, `lag_168`, `price_roll_24h`, `price_roll_168h`, `price_roll_std_168h` |
| Ensemble uncertainty | `wind_ensemble_std`, `solar_ensemble_std` |

Same feature vocabulary across both zones — zone-specific columns are NaN for the "wrong" zone; LightGBM handles NaN natively.

---

## Forecasting regimes

| Horizon | Model | Uncertainty |
|---|---|---|
| ≤ 7 days from training tail | Quantile LightGBM (q=0.025/0.45/0.975) + Mondrian CQR | Q_hat per regime (bucket 0/1) |
| > 7 days | Recency-weighted seasonal median + post-crisis trend | resid_std × 1.96 × sqrt-scaled |

**Long-term model:** profile = per `(month, dayofweek, hour)` recency-weighted median (2021×1 → 2025×4 weighting to down-weight energy-crisis outliers). Trend anchored on 2023+ annual means only. Interval grows with sqrt(excess_days/30).

---

## Project structure

```
src/
  config.py       zones, paths, date ranges, weather station coordinates
  ingestion.py    ENTSOE + neighbor prices + nuclear + Open-Meteo + ensemble + yfinance
  cleaning.py     UTC parsing, hourly reindex, interpolation; multi-city weather support
  alignment.py    join on (timestamp, zone) MultiIndex; optional crossborder join
  features.py     all 41 features; lags per-zone to prevent cross-zone leakage
  validation.py   tiered gap/NaN/range checks
  pipeline.py     orchestrator: loads all DataFrames and passes to engineer_features()
  model.py        train + CQR calibrate + predict; dual-regime; backtest if actuals exist

gen_notebook.py    generates model.ipynb submission notebook
gen_gnn_notebook.py  alternative notebook generator
gen_playground.py  generates playground.ipynb (EDA + analysis)
playground.ipynb   exploration notebook

alpine-arbitrage_predictions.csv   current submission output
```
