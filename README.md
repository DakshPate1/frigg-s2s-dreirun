# Frigg S2S — Electricity Price Forecasting

End-to-end pipeline for forecasting Day-Ahead Auction (DAA) electricity prices for two European bidding zones: **DE-LU** (Germany-Luxembourg) and **ES** (Spain).

Covers data ingestion through feature engineering, with modelling and probabilistic forecasting to follow.

**Evaluation target:** 2026-05-08 18:00 CEST → 2026-05-09 23:00 CEST (30 hourly slots), quantiles p025 / p50 / p975.
**Scoring:** asymmetric pinball loss at q=0.45 — overestimation penalised ~1.22× more than underestimation.

---

## Data sources

| Source | Data | API |
|---|---|---|
| [energy-charts.info](https://api.energy-charts.info) | DAA prices, load, wind/solar/hydro generation | Free, no key |
| [Open-Meteo](https://open-meteo.com) | Temperature, wind speed, solar radiation | Free, no key |
| yfinance `TTF=F` | TTF natural gas futures (EUR/MWh) | Free, no key |
| yfinance `KRBN` | EU carbon price proxy (EUA ETF) | Free, no key |

Weather stations: Frankfurt (DE-LU), Madrid (ES). Training window: 2021–2026.

---

## Data pipeline

```
Stage 1  ingestion.py    → data/raw/          raw CSVs, no modification
Stage 2  cleaning.py     → data/clean/        UTC timestamps, hourly freq, gap interpolation
Stage 3  alignment.py    → data/aligned/      joined on common (timestamp, zone) index
Stage 4  features.py     → data/processed/    derived features, circular encoding, lags
Stage 5  validation.py                        asserts no NaN, continuous timestamps, sane ranges
```

### Running

All scripts run from `src/`:

```bash
pip install -r requirements.txt
cd src

python pipeline.py              # full pipeline (~20 min, API rate limits)
python pipeline.py --from-clean # skip ingestion (raw CSVs exist)
python pipeline.py --from-align # skip ingestion + cleaning
python pipeline.py --validate-only
```

### Loading the dataset

```python
import pandas as pd

df = pd.read_parquet("data/processed/final_dataset.parquet")

delu = df.xs("DE-LU", level="zone")
es   = df.xs("ES",    level="zone")
```

---

## Dataset schema

`data/processed/final_dataset.parquet` — **93,152 rows × 30 columns**, MultiIndex `(timestamp UTC, zone)`.

| Column | Description |
|---|---|
| `price` | Day-ahead price EUR/MWh — target variable |
| `load` | Total grid load MW |
| `wind_generation` | Wind output MW (offshore + onshore summed) |
| `solar_generation` | Solar output MW |
| `hydro_generation` | Hydro output MW (run-of-river + reservoir + pumped) |
| `temperature` | °C at representative city (Frankfurt / Madrid) |
| `wind_speed` | m/s |
| `solar_radiation` | W/m² |
| `gas_price` | TTF natural gas EUR/MWh (daily, forward-filled hourly) |
| `carbon_price` | EUA proxy via KRBN ETF (daily, forward-filled hourly) |
| `residual_load` | load − wind − solar, clipped ≥ 0 |
| `renewable_penetration` | (wind + solar) / load, clipped [0, 1] |
| `hour`, `weekday`, `month`, `week_of_year` | Temporal features (raw integers) |
| `hour_sin/cos`, `weekday_sin/cos`, `month_sin/cos`, `week_sin/cos` | Circular encoding — makes periodicity continuous for tree/NN models |
| `is_holiday` | Public holiday flag (zone-specific: DE for DE-LU, ES for ES) |
| `lag_1`, `lag_24`, `lag_168` | Price lags 1h / 24h / 7d — computed per zone, no cross-zone leakage |
| `price_roll_24h`, `price_roll_168h` | Rolling price means 24h / 168h |

---

## Key findings from prior analysis

Feature importance is consistent across the literature (Tschora 2024, ENTSO-E run on 2024–2026 DE-LU data):

**Permutation / SHAP ranking for DE-LU:**
1. `load` / `load_forecast` — single largest signal; demand directly sets the clearing level
2. `lag_24` (D-1 price, same hour) — price momentum; strongest autoregressive effect
3. `wind_generation` — high wind → low or negative prices
4. `solar_generation` — solar saturation drives the worst negative-price spikes
5. `lag_168` (D-7), `hour`, `weekday` — weekly seasonality and daily shape
6. `gas_price` — matters structurally but noisy in high-volatility regimes
7. `is_holiday`, `month` — minor on average; `is_holiday` critical for tail events

**German vs Spanish market dynamics:**
- DE-LU is thermal + renewable driven. Renewables intermittency causes negative prices (saw -€500 floor on 2026-05-01 Labor Day). Gas price and renewable forecast matter most.
- ES is more hydro-modulated. Consumption is heat-sensitive (cooling load in summer). Renewable penetration is high but network is less coupled to central Europe.

**On the lag structure:**
D-1 (lag_24h) and D-7 (lag_168h) carry almost all autocorrelation signal. D-2 and D-3 lags contribute <5% of SHAP weight — not worth the feature cost.

**On probabilistic calibration:**
Raw quantile regression is structurally undercovered (~53% empirical coverage claiming 80%). Conformalized Quantile Regression (CQR) on a held-out calibration slice pulled coverage to ~71%. Remaining gap is seasonal distribution shift between calibration and evaluation windows — Mondrian conformal or adaptive conformal closes it.

**Tail risk:**
The five worst single-hour errors are all renewables-saturation events on holidays/weekends (solar ~50 GW, depressed industrial demand). The model correctly identifies the direction (negative) but underestimates the magnitude. This is a known structural limitation of median-honest models trained on MAE/pinball loss.

---

## Project structure

```
src/
  config.py       zones, paths, API params, date ranges, eval window
  ingestion.py    fetch raw data from three APIs
  cleaning.py     UTC timestamps, hourly reindex, linear interpolation
  alignment.py    join sources on (timestamp, zone) MultiIndex
  features.py     derived features, circular encoding, lags, holiday flag
  validation.py   hard assertions on final dataset quality
  pipeline.py     orchestrator with --from-* resume flags

notebooks/        exploratory analysis, model development (to come)
```

---

## Next steps

- [ ] Quantile model (LightGBM at q=0.025, 0.45, 0.975) per zone
- [ ] Conformal calibration (CQR) on a held-out calibration slice
- [ ] Recursive lag-gap fill for evaluation window (lag_1, lag_24 require D-1 actuals)
- [ ] Neighbor price features (FR, PT for ES; NL, CH for DE-LU) — thesis finds ~15% gain potential
- [ ] Gas price feature: consider switching from D-2 TTF to a spot proxy less susceptible to crisis volatility
- [ ] Holiday-aware conformal band (Mondrian) to close coverage gap in tail events
