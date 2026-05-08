# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

Frigg S2S hackathon — electricity price forecasting for two European bidding zones: **DE-LU** (Germany-Luxembourg) and **ES** (Spain). Predict Day-Ahead Auction prices (EUR/MWh) for the evaluation window: **2026-05-11 00:00 UTC → 2026-05-11 23:00 UTC** (24 hourly slots = 02:00–25:00 CEST May 11).

Deliverables: `{team_name}_predictions.csv`, `{team_name}_model.ipynb`, `{team_name}_data.zip`. Team name: **alpine-arbitrage**. Deadline: **2026-05-09 23:59 CEST**.

## Running the pipeline

All scripts are in `src/` and must be run from within `src/`:

```bash
cd src

python pipeline.py              # full pipeline
python pipeline.py --from-clean # skip ingestion (raw CSVs exist)
python pipeline.py --from-align # skip ingestion + cleaning
python pipeline.py --validate-only
```

## Data flow

```
data/raw/entsoe/*.csv          ← prices, generation, crossborder, forecasts, neighbors, nuclear
data/raw/openmeteo/*.csv       ← multi-city weather (26 stations) + ensemble
data/raw/fuel/fuel_prices.csv  ← gas, carbon, coal
        ↓  Stage 1: ingestion.py
data/clean/*.parquet            ← Stage 2: cleaning.py
        ↓
data/aligned/base_dataset.parquet   ← Stage 3: alignment.py  (timestamp, zone) MultiIndex
        ↓
data/processed/final_dataset.parquet ← Stage 4: features.py  41 features, ~30k rows
```

## Architecture

**`config.py`** — single source of truth: zone names, paths (`ROOT`, `DATA_RAW`, `DATA_CLEAN`, `DATA_ALIGNED`, `DATA_PROCESSED`), weather station groups + weights, date ranges. Training window: `TRAIN_START="2023-05-01"` → `TRAIN_END="2026-05-07"` (post-energy-crisis only). Val: 2025, Cal: Jan–May 2026. Eval: 24 slots May 11 2026.

**`ingestion.py`** — fetches raw data. Global `socket.setdefaulttimeout(120)` prevents entsoe-py hanging. Key functions:
- ENTSOE: prices, generation, crossborder, forecasts, neighbor prices (FR/NL/CH/DK), nuclear ES
- Open-Meteo: multi-city archive (26 stations, 9 groups) + ensemble (ECMWF spread) + `fetch_weather_forecast()` for ≤14d ahead
- yfinance: `TTF=F` gas, `KRBN` carbon, `MTF=F` coal (daily, ffill to hourly)
- energy-charts.info: retained as fallback (`fetch_prices_ec()` / `fetch_generation_ec()`)

**`cleaning.py`** — UTC timestamp parsing, hourly reindex, linear interpolation for gaps ≤ 24h. `clean_weather()` detects multi-city format and creates legacy aliases (`temperature`, `wind_speed`, `solar_radiation`) for alignment compatibility. `clean_fuel_prices()` dynamically handles all fuel columns present in the raw CSV.

**`alignment.py`** — joins cleaned sources on UTC hourly index per zone. Crossborder joined with `left` if parquet exists. Drops rows missing critical columns (`price`, `load`, `wind_generation`, `solar_generation`, `temperature`). Produces `(timestamp, zone)` MultiIndex.

**`features.py`** — adds all 41 features. `add_weather_features()` joins only columns not already in zone_df (avoids duplicate error if alignment already brought them in). Lags computed per zone independently to prevent cross-zone leakage. First 168 rows per zone dropped after lag_168 (burn-in).

**`pipeline.py`** — orchestrator. Stage 4 loads all DataFrames from disk and passes explicitly to `engineer_features(weather_de, weather_es, neighbor_df, fuel_df, nuclear_df, ensemble_de, ensemble_es)`.

**`validation.py`** — tiered gap check: warns < 50 missing timestamps (real-world outages, e.g. Apr 2025 ES blackout), errors ≥ 200 (systematic failure). Hard-fails on missing columns, NaN, wrong zones.

## model.py usage

```bash
cd src

# Train + validate
python model.py

# Predict eval window (default: 2026-05-11 00:00 → 23:00 UTC, 24 slots)
python model.py --predict

# Predict any window
python model.py --predict --start "2026-06-01 00:00" --end "2028-06-01 00:00"

# Backtest (accuracy auto-reported if window has actuals)
python model.py --predict --start "2025-06-01 00:00" --end "2025-06-07 23:00"
```

When `--predict` runs it automatically:
1. `fetch_gap_actuals` — fetches ENTSOE prices + crossborder from training tail to pred start for real lag_24 values
2. `fetch_weather_forecast` — Open-Meteo 14d forecast replaces proxy for near-term slots
3. `fetch_entsoe_gen_forecast` — ENTSOE day-ahead load/wind/solar replaces proxy if published; falls back gracefully
4. Routes each slot: **≤ 7d** from training tail → LightGBM + Mondrian CQR; **> 7d** → long-term seasonal model
5. Outputs `alpine-arbitrage_predictions.csv`

## CQR calibration (Mondrian)

`calibrate_zone()` uses Jan–May 2026 calibration set (n≈3047/zone). `dropna(subset=[TARGET])` only — LightGBM handles NaN features natively (zone-specific columns like `ES_hydro_precipitation` are NaN for DE-LU).

- **Bucket 0** (normal weekday): Q_hat=2.37 DE-LU, 4.31 ES
- **Bucket 1** (weekend / holiday / bridge day): Q_hat=4.34 DE-LU, 3.72 ES
- p50 shift: +1.42 DE-LU, −1.62 ES

`train_zone()` also uses `dropna(subset=[TARGET])` only for the same reason.

Row count assert fires only for default eval window (`pred_start is None and pred_end is None`): `assert len(out) == 24`.

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
| Cross-border | `net_imports` |
| Neighbor prices | `FR_price_lag24`, `NL_price_lag24`, `CH_price_lag24`, `DK_price_lag24` |
| Transmission spreads | `DE_LU_FR_spread`, `DE_LU_NL_spread`, `DE_LU_CH_spread`, `ES_FR_spread` |
| Cross-zone | `cross_zone_lag24` |
| Calendar (circular) | `hour_sin/cos`, `weekday_sin/cos`, `month_sin/cos`, `week_sin/cos`, `is_holiday`, `days_to_holiday`, `days_from_holiday` |
| Regime flags | `crisis_period`, `is_peak`, `negative_price_lag24` |
| Price history | `lag_1`, `lag_24`, `lag_168`, `price_roll_24h`, `price_roll_168h`, `price_roll_std_168h` |
| Ensemble uncertainty | `wind_ensemble_std`, `solar_ensemble_std` |

Zone-specific features are NaN for the "wrong" zone — LightGBM handles this natively, never `dropna` on FEATURES.

## Forecasting regimes

| Horizon | Model | Uncertainty |
|---|---|---|
| ≤ 7 days | Quantile LightGBM (q=0.025/0.45/0.975) + Mondrian CQR | Per-bucket Q_hat |
| > 7 days | Recency-weighted seasonal median + post-crisis trend | resid_std × 1.96 × sqrt-scaled |

Long-term model: recency weights 2021×1 → 2025×4 (down-weights energy-crisis outliers). Trend from 2023+ annual means only. Interval grows as `1 + sqrt(excess_days/30) × 0.25`.

## Key constraints

- **Same feature vocabulary across both zones** — zone-specific columns are NaN for the wrong zone, not absent.
- **No future leakage** — lags always `price.shift(n)` on per-zone sorted series; rolling windows use `min_periods`.
- **Scoring**: `pinball_loss(q=0.45)` — p50 trained at q=0.45, not 0.5. Overestimation penalised 1.22×.
- **Lag fill**: lag_1/lag_24 filled recursively slot-by-slot from model's own p50. ENTSOE gap actuals fetched for real lag_24 values where available.
- **ENTSOE chunk boundary**: `_chunk_dates` yields 180-day windows; all fetch functions add `+1 day` to end timestamp (ENTSOE API end is exclusive at midnight Brussels — without this, 21h gaps appear at each boundary).
