# alpine-arbitrage — Electricity Price Forecasting

> Frigg S2S × EC EPFL × ACE Hackathon · May 2026

A price forecasting system that covers **any time horizon** — from the next hour to the next 20 years — for two European electricity markets: **DE-LU** (Germany-Luxembourg) and **ES** (Spain).

---

## Results

| Zone | MAE (p50) | Pinball q=0.45 | vs Naive baseline | Coverage (95% CI) |
|------|----------:|---------------:|------------------:|------------------:|
| DE-LU | **6.01 EUR/MWh** | 2.85 | 5.5× better | 95.0% |
| ES | **5.15 EUR/MWh** | 2.51 | 5.8× better | 95.0% |

Holdout: Q4 2025 (Oct–Dec). Naive baseline: same hour, one week ago.
Coverage is post-CQR calibration — the 95% intervals contain the true price 95% of the time.

---

## How it works

```
[ENTSOE prices/gen/flows]  [Open-Meteo weather]  [yfinance fuels]  [GPR index]
            │                       │                    │              │
            └───────────┬───────────┴────────────┬───────┘              │
                        ▼                        ▼                      ▼
               [Pipeline A: hourly]      [Pipeline B: monthly]
                        │                        │
                        ▼                        ▼
            [Quantile LightGBM 0–7d]    [Merit-order MC 2026–2045]
                        │                        │
                        └─────────┬──────────────┘
                                  ▼
                          [Horizon router]
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
     slot ≤ 7d from training tail         slot > 7d
     LightGBM + Mondrian CQR         Seasonal median + trend
     Calibrated 95% intervals        MC fan, sqrt-widening
```

**Short-term (0–7 days):** Three quantile LightGBM models per zone (q=0.025, 0.45, 0.975) trained on 20 engineered features. Mondrian CQR post-calibration splits by regime (weekday vs weekend/holiday) to guarantee 95% coverage.

**Long-term (>7 days):** Structural merit-order model — stacks power plants by ascending marginal cost, finds where the stack meets demand. Forward fuel curves via Schwartz mean-reversion. 200-draw Monte Carlo fan charts to 2045. Zero historical price leakage.

**Router:** Every prediction call is automatically dispatched based on horizon. One entry point, any date range.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your ENTSOE API token
cp .env.example .env
# edit .env: ENTSOE_TOKEN=your_token_here
# Free key: https://transparency.entsoe.eu/usrm/user/createPublicUser

# 3. Run the full data pipeline (only needed once)
cd src
python pipeline.py

# 4. Generate predictions
python model.py --predict                                         # eval window (May 11 2026)
python model.py --predict --start "2026-06-01 00:00" --end "2026-06-07 23:00"  # any window
python model.py --predict --start "2025-06-01 00:00" --end "2025-06-07 23:00"  # backtest (auto-reports MAE)
```

Output: `alpine-arbitrage_predictions.csv` in the repo root.
Runtime: ~5 minutes (trains both zones, fetches live lag actuals + weather forecast).

---

## Data pipeline

```
data/raw/entsoe/*.csv          ← prices, generation, crossborder, forecasts, neighbors, nuclear
data/raw/openmeteo/*.csv       ← 26-station weather + ECMWF ensemble spread
data/raw/fuel/fuel_prices.csv  ← gas (TTF), carbon (KRBN), coal (MTF)
         ↓  src/ingestion.py
data/clean/*.parquet
         ↓  src/cleaning.py + alignment.py
data/aligned/base_dataset.parquet    (timestamp, zone) MultiIndex
         ↓  src/features.py
data/processed/final_dataset.parquet   52,593 rows × 80 columns
```

Pipeline stages can be skipped if clean data already exists:

```bash
python pipeline.py --from-clean   # skip ingestion, start from raw CSVs
python pipeline.py --from-align   # skip ingestion + cleaning
python pipeline.py --validate-only
```

---

## Features

20 features selected from 41 candidates via **SHAP + gain Borda rank fusion** on the 2025 holdout, independently per zone. Union of top-15 per zone = 20-feature shared schema. Zone-specific extras are NaN for the wrong zone — LightGBM skips NaN-only columns at split time.

| Group | Features |
|-------|----------|
| Price history | `lag_1`, `lag_24`, `lag_168`, `price_roll_24h`, `price_roll_168h`, `price_roll_std_168h` |
| Grid state | `residual_load`, `residual_load_ramp`, `residual_load_forecast`, `residual_load_ramp_forecast`, `renewable_penetration_forecast` |
| Generation | `wind_generation`, `solar_generation`, `hydro_generation` |
| Cross-border | `net_imports`, `cross_zone_lag24`, `NL_price_lag24`*, `CH_price_lag24`* |
| Fuel / carbon | `carbon_price` |
| Load | `load`* |

*DE-LU only (NaN for ES, handled natively by LightGBM)

---

## Uncertainty calibration

Raw quantile models don't reliably achieve their nominal coverage. **Mondrian CQR** corrects this using a held-out calibration set (Jan–May 2026, ~3,047 rows per zone), split by regime:

| Zone | Bucket | Regime | Q_hat | Coverage |
|------|--------|--------|------:|---------:|
| DE-LU | 0 | Normal weekday | 2.37 | 95.0% |
| DE-LU | 1 | Weekend / holiday / bridge | 4.34 | 95.0% |
| ES | 0 | Normal weekday | 4.31 | 95.0% |
| ES | 1 | Weekend / holiday / bridge | 3.72 | 95.0% |

May 11 2026 is a Sunday → bucket 1 for all 24 eval slots.

---

## Market structure — why the two zones differ

| | DE-LU | ES |
|--|-------|-----|
| Dominant renewable | Wind (north Germany) | Solar (Andalusia, Murcia) |
| Nuclear | Zero (phase-out 2023) | ~7 GW active |
| Interconnection | 8 neighbours, tightly coupled | 1 neighbour (France), partially isolated |
| Negative prices | Frequent (wind surplus weekends) | Less common |
| Key model driver | `lag_24`, `residual_load_ramp` | `lag_1`, `solar_generation` |

Both zones use the same feature vocabulary — zone-specific dynamics emerge from learned weights, not different architectures.

---

## Data sources

| Source | What | How |
|--------|------|-----|
| [ENTSOE Transparency Platform](https://transparency.entsoe.eu) | DAA prices, load, generation, crossborder flows, day-ahead forecasts, neighbor prices, nuclear REMIT | `entsoe-py` library |
| [Open-Meteo](https://open-meteo.com) | 26 weather stations, 9 regional groups; ECMWF ensemble spread | Archive + forecast API |
| [yfinance](https://github.com/ranaroussi/yfinance) | TTF gas (`TTF=F`), carbon proxy (`KRBN`), coal (`MTF=F`) | Daily spot, forward-filled to hourly |
| [energy-charts.info](https://energy-charts.info) | Fallback prices + generation | REST API (retained in `ingestion.py`) |

---

## Project structure

```
src/
├── config.py        zones, paths, date ranges, 26 weather stations + weights
├── ingestion.py     ENTSOE + neighbor prices + nuclear + Open-Meteo + ensemble + yfinance
├── cleaning.py      UTC parsing, hourly reindex, gap interpolation, multi-city weather aliases
├── alignment.py     join sources on (timestamp, zone) MultiIndex
├── features.py      all 41 feature candidates; lags computed per-zone (no cross-zone leakage)
├── validation.py    tiered gap/NaN/range checks; warns <50 missing, errors ≥200
├── pipeline.py      orchestrator — loads DataFrames, calls engineer_features()
└── model.py         train · CQR calibrate · predict · backtest · dual-regime routing

longterm/
├── src/             Pipeline B — merit-order MC, fuel curves, capacity roadmap
└── data/processed/  marginal_costs_monthly.parquet, structural_extended.parquet

gen_final_notebook.py      generates alpine-arbitrage_model.ipynb
gen_playground.py          generates playground.ipynb (feature selection experiment)

alpine-arbitrage_predictions.csv   submission output (24 rows, 7 columns)
alpine-arbitrage_model.ipynb       submission notebook
README.txt                         data zip documentation
```

---

## Submission

```
alpine-arbitrage_predictions.csv   24 hourly slots, May 11 2026 00:00–23:00 UTC
alpine-arbitrage_model.ipynb       end-to-end reproducible notebook
alpine-arbitrage_data.zip          training data + README.txt
```

Deadline: **Saturday 9 May 2026 23:59 CEST**
