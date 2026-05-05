# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

Frigg S2S hackathon — electricity price forecasting for two European bidding zones: **DE-LU** (Germany-Luxembourg) and **ES** (Spain). The goal is to predict Day-Ahead Auction prices (EUR/MWh) for the evaluation window: **2026-05-08 17:00 UTC → 2026-05-09 22:00 UTC** (30 hourly slots).

Deliverables: `predictions.csv`, `model.ipynb`, `data.zip`.

## Running the pipeline

All scripts are in `src/` and must be run from within `src/` (imports are relative to that directory):

```bash
cd src

# Full pipeline (ingestion takes ~20 min due to API rate limits)
python pipeline.py

# Skip ingestion if raw CSVs already exist in data/raw/
python pipeline.py --from-clean

# Skip ingestion + cleaning (parquets in data/clean/ already exist)
python pipeline.py --from-align

# Validate the final dataset only
python pipeline.py --validate-only

# Run individual stages
python ingestion.py
python cleaning.py
python alignment.py
python features.py
python validation.py
```

## Data flow

```
data/raw/{energycharts,openmeteo,fuel}/*.csv   ← Stage 1: ingestion.py
        ↓
data/clean/*.parquet                            ← Stage 2: cleaning.py
        ↓
data/aligned/base_dataset.parquet              ← Stage 3: alignment.py  (timestamp, zone) MultiIndex
        ↓
data/processed/final_dataset.parquet           ← Stage 4: features.py   20 columns, 93k rows
```

## Architecture

**`config.py`** — single source of truth for all constants: zone names, API parameters, file paths (`ROOT`, `DATA_RAW`, `DATA_CLEAN`, `DATA_ALIGNED`, `DATA_PROCESSED`), weather station coordinates, date ranges (`TRAIN_START`, `TRAIN_END`), and evaluation window (`EVAL_START`, `EVAL_END`). Import from here rather than hardcoding.

**`ingestion.py`** — fetches raw data from three APIs, writes unmodified CSVs:
- `energy-charts.info` API: prices (`/price?bzn=`) and generation+load (`/public_power?country=`) in 180-day chunks due to API limits
- Open-Meteo archive API: weather (no API key) in 365-day chunks; also has `fetch_weather_forecast()` for the forward-looking period
- yfinance: TTF gas futures (`TTF=F`) and KRBN carbon ETF as EUA proxy (daily, forward-filled to hourly)

**`cleaning.py`** — per-dataset cleaning: UTC timestamp parsing (unix seconds or ISO strings), hourly reindex, linear interpolation for gaps ≤ 3h, clamp negatives. Generation columns are collapsed: wind = offshore + onshore (summed, clamped ≥ 0), hydro = run-of-river + reservoir + pumped storage.

**`alignment.py`** — joins the four cleaned sources on a common UTC hourly index per zone, intersecting on the narrowest shared window. Drops rows missing any critical column (`price`, `load`, `wind_generation`, `solar_generation`, `temperature`). Produces a `(timestamp, zone)` MultiIndex DataFrame.

**`features.py`** — adds derived features to the aligned dataset. **Lags are computed per zone independently** (call `df.xs(zone, level="zone")` before shifting) to prevent cross-zone data leakage. First 168 rows per zone are dropped after lag_168 is added (burn-in).

**`validation.py`** — hard-fails on: missing required columns, any NaN, wrong zones, timestamp gaps; warns on out-of-range values. Run after every pipeline change.

## Final dataset schema

Index: `(timestamp [UTC hourly], zone ["DE-LU" | "ES"])`

| Column | Source | Notes |
|---|---|---|
| `price` | energy-charts | Target variable, EUR/MWh |
| `load` | energy-charts | MW |
| `wind_generation` | energy-charts | MW, offshore+onshore summed |
| `solar_generation` | energy-charts | MW |
| `hydro_generation` | energy-charts | MW, three types summed |
| `temperature` | Open-Meteo | °C at Frankfurt / Madrid |
| `wind_speed` | Open-Meteo | m/s |
| `solar_radiation` | Open-Meteo | W/m² |
| `gas_price` | yfinance TTF=F | EUR/MWh, daily ffill to hourly |
| `carbon_price` | yfinance KRBN | USD, EUA proxy, daily ffill |
| `residual_load` | derived | load − wind − solar, clipped ≥ 0 |
| `renewable_penetration` | derived | (wind+solar)/load, clipped [0,1] |
| `hour`, `weekday`, `month` | derived | temporal features |
| `lag_1`, `lag_24`, `lag_168` | derived | price lags, per-zone only |
| `price_roll_24h`, `price_roll_168h` | derived | rolling means |

## Key constraints

- **Same feature vocabulary across both zones** — no zone-specific columns. Models may differ in weights but must use identical feature names.
- **No future leakage** — lags are always `price.shift(n)` on the per-zone sorted series; rolling windows use `min_periods` to avoid partial-window NaN propagation.
- **Evaluation scoring**: `pinball_loss(y_true, y_pred, q=0.45)` — p50 should be trained at q=0.45, not 0.5 (penalises overestimation ~1.22× more than underestimation).
- **Evaluation window lag problem**: lag_1 and lag_24 for the evaluation period (May 8-9) require prices from May 7-8 which are in the future. These must be filled via recursive/iterative forecasting of the gap period before generating evaluation predictions.

## Data sources (no API keys required)

- `https://api.energy-charts.info/price?bzn={DE-LU|ES}&start=YYYY-MM-DD&end=YYYY-MM-DD`
- `https://api.energy-charts.info/public_power?country={de|es}&start=...&end=...`
- `https://archive-api.open-meteo.com/v1/archive` (historical) / `https://api.open-meteo.com/v1/forecast` (≤16 days ahead)
- yfinance tickers: `TTF=F` (gas), `KRBN` (carbon proxy)
