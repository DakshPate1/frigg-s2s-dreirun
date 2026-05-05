# Frigg Electricity Price Forecasting — Data Pipeline

Data pipeline for the Frigg S2S hackathon challenge: forecasting Day-Ahead Auction (DAA) electricity prices for two European bidding zones — **DE-LU** (Germany-Luxembourg) and **ES** (Spain).

## What this does

Pulls raw data from public APIs, cleans and standardises it, joins all sources on a common UTC hourly index, and engineers a model-ready feature set. Output is a single parquet file with identical schema for both zones.

## Data sources

| Source | Data | API |
|---|---|---|
| [energy-charts.info](https://api.energy-charts.info) | DAA prices, load, wind/solar/hydro generation | Free, no key |
| [Open-Meteo](https://open-meteo.com) | Temperature, wind speed, solar radiation | Free, no key |
| yfinance `TTF=F` | TTF natural gas futures (EUR/MWh) | Free, no key |
| yfinance `KRBN` | EU carbon price proxy (EUA ETF) | Free, no key |

Weather stations: Frankfurt (DE-LU), Madrid (ES). Training data: 2021–2026.

## Pipeline stages

```
Stage 1  ingestion.py    → data/raw/          raw CSVs, no modification
Stage 2  cleaning.py     → data/clean/        UTC timestamps, hourly freq, gap interpolation
Stage 3  alignment.py    → data/aligned/      joined on common (timestamp, zone) index
Stage 4  features.py     → data/processed/    derived features + price lags
Stage 5  validation.py                        asserts no NaN, continuous timestamps, sane ranges
```

## Final dataset

`data/processed/final_dataset.parquet` — **93,152 rows × 20 columns**, MultiIndex `(timestamp UTC, zone)`.

| Column | Description |
|---|---|
| `price` | Day-ahead price EUR/MWh (target) |
| `load` | Total grid load MW |
| `wind_generation` | Wind output MW (offshore + onshore) |
| `solar_generation` | Solar output MW |
| `hydro_generation` | Hydro output MW (run-of-river + reservoir + pumped) |
| `temperature` | °C |
| `wind_speed` | m/s |
| `solar_radiation` | W/m² |
| `gas_price` | TTF gas EUR/MWh |
| `carbon_price` | EUA proxy (KRBN ETF) |
| `residual_load` | load − wind − solar |
| `renewable_penetration` | (wind + solar) / load |
| `hour`, `weekday`, `month` | Temporal features |
| `lag_1`, `lag_24`, `lag_168` | Price lags 1h / 24h / 7d |
| `price_roll_24h`, `price_roll_168h` | Rolling means |

## Setup

```bash
pip install -r requirements.txt
```

## Running

All scripts run from `src/`:

```bash
cd src

# Full pipeline (ingestion ~20 min due to API chunking)
python pipeline.py

# Resume from existing raw files
python pipeline.py --from-clean

# Resume from existing clean parquets
python pipeline.py --from-align

# Validate only
python pipeline.py --validate-only
```

## Loading the dataset

```python
import pandas as pd

df = pd.read_parquet("data/processed/final_dataset.parquet")

delu = df.xs("DE-LU", level="zone")
es   = df.xs("ES",    level="zone")
```

## Project structure

```
src/
  config.py       zones, paths, API params, date ranges
  ingestion.py    fetch raw data from APIs
  cleaning.py     parse timestamps, enforce hourly freq, interpolate gaps
  alignment.py    join sources, drop rows with missing critical fields
  features.py     derived features, temporal encoding, price lags
  validation.py   data quality assertions
  pipeline.py     orchestrator with --from-* flags
```

## Evaluation window

Predictions required for **2026-05-08 18:00 CEST → 2026-05-09 23:00 CEST** (30 hourly slots), covering DE-LU and ES at quantiles p025 / p50 / p975.

Scoring: asymmetric pinball loss at q=0.45 (overestimation penalised ~1.22× more than underestimation).
