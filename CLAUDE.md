# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

Frigg S2S hackathon — electricity price forecasting for two European bidding zones: **DE-LU** (Germany-Luxembourg) and **ES** (Spain). The goal is to predict Day-Ahead Auction prices (EUR/MWh) for the evaluation window: **2026-05-08 17:00 UTC → 2026-05-09 22:00 UTC** (30 hourly slots).

Deliverables: `predictions.csv`, `model.ipynb`, `data.zip`.

## Running the pipeline

All scripts are in `src/` and must be run from within `src/` (imports are relative to that directory):

```bash
cd src

# Full pipeline — ENTSOE primary source, much faster than energy-charts
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
- **ENTSOE Transparency Platform** (primary): prices, load, generation per type, cross-border physical flows. API key in `config.ENTSOE_TOKEN`. Returns 15-min data → resampled to hourly. Raw CSVs in `data/raw/entsoe/`.
- **energy-charts.info** (fallback, rate-limited): functions retained in `ingestion.py` as `fetch_prices_ec()` / `fetch_generation_ec()` — not called by default pipeline.
- Open-Meteo archive API: weather (no API key) in 365-day chunks; also has `fetch_weather_forecast()` for the forward-looking period
- yfinance: TTF gas futures (`TTF=F`) and KRBN carbon ETF as EUA proxy (daily, forward-filled to hourly)

**`cleaning.py`** — per-dataset cleaning: UTC timestamp parsing, hourly reindex, linear interpolation for gaps ≤ `MAX_INTERP_GAP` (currently 24h — covers 21h chunk-boundary gaps from ENTSOE's exclusive-end query behaviour). ENTSOE generation columns arrive pre-aggregated from ingestion (wind/solar/hydro already collapsed). Cross-border parquet is optional — alignment skips it gracefully if absent.

**`alignment.py`** — joins cleaned sources on a common UTC hourly index per zone. Cross-border parquet (`crossborder_{zone}.parquet`) is joined with `left` if it exists. Drops rows missing any critical column (`price`, `load`, `wind_generation`, `solar_generation`, `temperature`). Produces a `(timestamp, zone)` MultiIndex DataFrame.

**`features.py`** — adds derived features to the aligned dataset. **Lags are computed per zone independently** (call `df.xs(zone, level="zone")` before shifting) to prevent cross-zone data leakage. First 168 rows per zone are dropped after lag_168 is added (burn-in).

**`validation.py`** — tiered gap check: warns for < 50 missing timestamps per zone (real-world outages, e.g. Apr 2025 ES blackout), errors for ≥ 200 (systematic ingestion failure). Hard-fails on missing required columns, any NaN, wrong zones. Warns on out-of-range values.

## Final dataset schema

Index: `(timestamp [UTC hourly], zone ["DE-LU" | "ES"])`

| Column | Source | Notes |
|---|---|---|
| `price` | ENTSOE | Target variable, EUR/MWh |
| `load` | ENTSOE | MW |
| `wind_generation` | ENTSOE | MW, offshore+onshore summed |
| `solar_generation` | ENTSOE | MW |
| `hydro_generation` | ENTSOE | MW, three types summed |
| `net_imports` | ENTSOE | MW, net cross-border imports (positive = importing); 8 neighbors for DE-LU, 2 for ES |
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

## model.py usage

```bash
cd src

# Train + validate (no predictions)
python model.py

# Predict the hackathon eval window (default: 2026-05-08 17:00 → 2026-05-09 22:00 UTC)
python model.py --predict

# Predict any arbitrary window — short-term, long-term, or mixed
python model.py --predict --start "2026-05-10 17:00" --end "2026-05-11 22:00"
python model.py --predict --start "2026-05-10 00:00" --end "2028-05-10 00:00"  # 2-year

# Backtest: if --start/--end overlap actuals in the dataset, accuracy is reported automatically
python model.py --predict --start "2025-06-01 00:00" --end "2025-06-02 23:00"
```

When `--predict` runs it automatically:
1. Fetches ENTSOE prices + cross-border flows for gap between training tail and eval start (`fetch_gap_actuals`) — gives lag_24 real values instead of averages
2. Fetches Open-Meteo forecast (capped at 14 days) — replaces seasonal proxy for temperature/wind/solar_radiation on near-term slots
3. Fetches ENTSOE day-ahead load+wind+solar generation forecasts (`fetch_entsoe_gen_forecast`) — replaces proxy for generation/load on near-term slots if published; falls back to proxy gracefully
4. Routes each slot by horizon:
   - **≤ `SHORTTERM_DAYS` (7d)** from training tail → LightGBM + Mondrian CQR
   - **> 7d** → long-term seasonal model (no CQR; own uncertainty)
5. Outputs `alpine-arbitrage_predictions.csv`

## CQR calibration (Mondrian)

`calibrate_zone()` computes both global and per-regime Q_hats (Jan–May 2026, n≈2998/zone):
- **Bucket 0** (normal weekday, not adjacent to holiday): larger Q_hat — weekday prediction was harder to calibrate in Jan–May 2026
- **Bucket 1** (weekend OR holiday OR bridge day): smaller Q_hat — weekend patterns were more predictable

`_mondrian_bucket(ts, cal) → int` classifies each timestamp. Bridge days = any weekday within 1 day of a public holiday.

ENTSOE forecast pipeline infrastructure is fully implemented (`fetch_entsoe_forecasts` in ingestion.py, `clean_entsoe_forecasts` in cleaning.py, forecast join in alignment.py, `add_derived_forecasts` in features.py). Run a full pipeline to get `_forecast` columns. To use them in training, add `load_forecast`, `wind_generation_forecast`, `solar_generation_forecast`, `residual_load_forecast`, `renewable_penetration_forecast`, `residual_load_ramp_forecast` to `FEATURES` in model.py — the `active_features` filter handles graceful fallback to actuals if the columns are absent.

## Forecasting regimes

| Horizon | Model | Features | Uncertainty |
|---|---|---|---|
| ≤ 7 days | Quantile LightGBM | All 29 features + recursive lags | Mondrian CQR ±Q_hat per regime |
| > 7 days | Seasonal profile + trend | month × weekday × hour median + post-crisis annual trend | resid_std × 1.96 × sqrt-scale |

**Long-term model (`build_longterm_model`):**
- Profile: **recency-weighted median** per `(month, dayofweek, hour)` — years resampled proportional to recency (2021×1 → 2025×4) so energy-crisis 2022 outliers are down-weighted
- Trend: **post-crisis anchor** — linear fit on 2023+ annual means only (falls back to all years if < 2 post-crisis years). Avoids importing the 2022 crisis spike/reversion into the slope.
- p50 bias: 45th-percentile residual (matches scoring metric)
- Interval: `resid_std × 1.96 × (1 + sqrt(excess_days/30) × 0.25)` — grows with sqrt(months)
- Structural factors documented in docstring: DE-LU nuclear phase-out (Apr 2023), ES solar growth (~8 GW/yr), DE-LU wind build-out (~5 GW/yr)

## Features (29 total)

Three features added beyond the original 26:

| Feature | Where added | Rationale |
|---|---|---|
| `residual_load_ramp` | `features.py add_lags()` | Hour-over-hour diff of residual_load per zone. Fast positive ramp = gas peakers spinning up = spike risk. |
| `days_to_holiday` | `features.py add_holidays()` | Days until next public holiday, capped at 7. Continuous — avoids binary bridge-day edge cases. |
| `days_from_holiday` | `features.py add_holidays()` | Days since last public holiday, capped at 7. Model learns gradual demand recovery pattern. |

`residual_load_ramp` is per-zone (computed inside the zone loop in `add_lags` to avoid cross-zone leakage). Holiday distance uses vectorised numpy `searchsorted` on sorted holiday ordinals — fast even for 5-year datasets.

## Key constraints

- **Same feature vocabulary across both zones** — no zone-specific columns. Models may differ in weights but must use identical feature names.
- **No future leakage** — lags are always `price.shift(n)` on the per-zone sorted series; rolling windows use `min_periods` to avoid partial-window NaN propagation.
- **Evaluation scoring**: `pinball_loss(y_true, y_pred, q=0.45)` — p50 should be trained at q=0.45, not 0.5 (penalises overestimation ~1.22× more than underestimation).
- **Lag fill for future windows**: lag_1 / lag_24 for the prediction window are filled recursively (slot-by-slot using model's own calibrated p50). ENTSOE gap actuals are fetched for the period between training tail and prediction start so lag_24 uses real prices where available.
- **ENTSOE chunk boundary**: `_chunk_dates` yields 180-day windows; all three ENTSOE fetch functions add `+1 day` to the end timestamp because ENTSOE's API end is exclusive at midnight Brussels — without this each chunk boundary has a 21h gap.

## Data sources (no API keys required)

- `https://api.energy-charts.info/price?bzn={DE-LU|ES}&start=YYYY-MM-DD&end=YYYY-MM-DD`
- `https://api.energy-charts.info/public_power?country={de|es}&start=...&end=...`
- `https://archive-api.open-meteo.com/v1/archive` (historical) / `https://api.open-meteo.com/v1/forecast` (≤16 days ahead)
- yfinance tickers: `TTF=F` (gas), `KRBN` (carbon proxy)
