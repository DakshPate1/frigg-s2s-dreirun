# Frigg S2S — Electricity Price Forecasting

End-to-end pipeline for forecasting Day-Ahead Auction (DAA) electricity prices for two European bidding zones: **DE-LU** (Germany-Luxembourg) and **ES** (Spain).

**Evaluation target:** 2026-05-08 18:00 CEST → 2026-05-09 23:00 CEST (30 hourly slots), quantiles p025 / p50 / p975.
**Scoring:** asymmetric pinball loss at q=0.45 — overestimation penalised ~1.22× more than underestimation.

---

## Project status

### Done

**Data pipeline (Stages 1–5) — complete and validated**
- ENTSOE Transparency Platform switched in as primary source (prices, load, generation, cross-border flows). energy-charts retained as fallback for rate-limit situations.
- Full hourly dataset 2021–present for both zones: ~93k rows × 31 columns.
- Cross-border net import/export flows added as a new feature (`net_imports`, MW): 8 neighbors for DE-LU, FR + PT for ES. This captures interconnection pressure that energy-charts couldn't provide.

**Model (src/model.py) — trained, CQR-calibrated, predictions generated**
- Quantile LightGBM at q = 0.025 / 0.45 / 0.975 per zone. p50 trained at q=0.45 to match scoring function (underforecast bias by design).
- 26 features: generation + weather + fuel/carbon + circular calendar + is_holiday + price lags + net_imports.
- Recursive lag fill for evaluation window: lag_1 / lag_24 populated slot-by-slot from model's own calibrated p50 predictions.
- **CQR calibration applied** (Jan–May 2026 calibration set, n≈2992/zone): inflates p025/p975 symmetrically to hit 95% coverage; shifts p50 to correct systematic zone bias.

**Validation results (2025 holdout, pre-CQR):**

| Zone | MAE p50 | Pinball 0.45 | Coverage p025–p975 | Band width | Naive MAE |
|------|--------:|-------------:|-------------------:|-----------:|----------:|
| DE-LU | 8.53 EUR/MWh | 4.12 | 88.0% | 44.97 EUR/MWh | 33.10 |
| ES | 6.95 EUR/MWh | 3.50 | 82.3% | 29.33 EUR/MWh | 28.97 |

**CQR calibration results (Jan–May 2026 calibration set):**

| Zone | Coverage (raw → CQR) | Pinball 0.45 (raw → CQR) | Band width | CQR interval shift | p50 shift |
|------|---------------------:|-------------------------:|-----------:|-------------------:|----------:|
| DE-LU | 88% → 95% | 4.12 → ~4.1 | ~59 EUR/MWh | ±8.11 EUR/MWh | +0.72 |
| ES | 74.8% → 95% | 3.64 → 3.43 | ~40 EUR/MWh | ±5.21 EUR/MWh | −2.08 |

**Eval window predictions (predictions.csv) — CQR-calibrated candidate**
- 30 slots, May 8–9 2026 in CEST.
- DE-LU: mean p50 ~€81, strong solar dip midday (p50 → −€8), evening recovery to €101.
- ES: mean p50 ~€63, midday near zero, evening €100.

**Playground notebook (`notebooks/playground.ipynb`) — ready**
- 36-cell notebook covering every pipeline layer: data health → EDA → features → training → validation metrics (with explanations) → CQR before/after → feature importance → error analysis → eval window preview.
- All sections independently runnable. Run `python notebooks/gen_playground.py` to regenerate.

**Feature selection rationale (from Tschora 2024 + ENTSOE analysis)**
- Documented in Key findings section below.
- Cross-zone features deliberately excluded: ES is isolated (Pyrenees bottleneck ~2.8 GW to FR), and the feature-vocabulary symmetry constraint would add noise not signal.

---

### What needs to happen next

**Must-do before submission:**
- [ ] Re-run full pipeline with ENTSOE as source to pull in `net_imports` for all training years, then retrain model — current predictions.csv was generated without this feature
- [ ] Build `model.ipynb` submission notebook — `notebooks/gen_notebook.py` is ready, run it to generate the notebook skeleton, then fill in output cells
- [ ] Package `data.zip` with all data files used + README.txt describing each file

**High-value improvements:**
- [x] ~~Conformalized Quantile Regression (CQR)~~ — **done**. Coverage corrected to 95% on both zones. Calibration set Jan–May 2026 (n≈2992/zone). See `calibrate_zone()` in `src/model.py`.
- [ ] Open-Meteo weather forecast for eval window — currently using seasonal same-weekday-hour proxy; actual 10-day forecast available free. Replace proxy in `build_eval_row` for better eval-window accuracy.
- [ ] Eval window: fetch ENTSOE actual cross-border flows for May 1–7 (they exist now) so `net_imports` lags are real values not proxies.

**Worth exploring if time allows:**
- [ ] Mondrian conformal bands — holiday-aware calibration to tighten intervals on normal hours without widening tail-event hours. Closes the remaining coverage gap from seasonal distribution shift.
- [ ] Day-ahead load + wind/solar forecasts from ENTSOE as features (ENTSOE provides these; thesis recommends using forecasts not actuals to avoid minor leakage). Available via `client.query_load_forecast()` and `client.query_wind_and_solar_forecast()`.
- [ ] Longer training tail: ENTSOE data goes back to 2015; adding 2015–2020 may improve rare-event coverage (energy crisis periods, COVID demand collapse).
- [ ] Gas price: TTF front-month futures (TTF=F) can be a lagged signal. Consider switching to day-ahead TTF spot if a reliable free source exists.

---

## Setup

```bash
pip install -r requirements.txt

# Add your ENTSOE token to .env (copy from .env.example):
cp .env.example .env
# edit .env and set ENTSOE_TOKEN=<your-token>
# Free registration: https://transparency.entsoe.eu/usrm/user/createPublicUser

cd src
python pipeline.py              # full pipeline
python pipeline.py --from-clean # skip ingestion (raw CSVs already exist)
python pipeline.py --from-align # skip ingestion + cleaning
python pipeline.py --validate-only

# Train model and generate predictions:
python model.py --predict
```

---

## Data sources

| Source | Data | API |
|---|---|---|
| [ENTSOE Transparency Platform](https://transparency.entsoe.eu) | DAA prices, load, generation per type, cross-border physical flows | Free, key required — set `ENTSOE_TOKEN` in `.env` |
| [Open-Meteo](https://open-meteo.com) | Temperature, wind speed, solar radiation | Free, no key |
| yfinance `TTF=F` | TTF natural gas futures (EUR/MWh) | Free, no key |
| yfinance `KRBN` | EU carbon price proxy (EUA ETF) | Free, no key |
| [energy-charts.info](https://api.energy-charts.info) | Fallback for prices + generation (rate-limited, no key) | Free, no key |

Weather stations: Frankfurt (DE-LU), Madrid (ES). Training window: 2021–present.

---

## Dataset schema

`data/processed/final_dataset.parquet` — MultiIndex `(timestamp UTC, zone)`.

| Column | Source | Description |
|---|---|---|
| `price` | ENTSOE | Target variable, EUR/MWh |
| `load` | ENTSOE | Total grid load, MW |
| `wind_generation` | ENTSOE | Wind output MW (offshore + onshore summed) |
| `solar_generation` | ENTSOE | Solar output, MW |
| `hydro_generation` | ENTSOE | Hydro output MW (run-of-river + reservoir + pumped) |
| `net_imports` | ENTSOE | Net cross-border imports MW — positive = net importer |
| `temperature` | Open-Meteo | °C at Frankfurt / Madrid |
| `wind_speed` | Open-Meteo | m/s |
| `solar_radiation` | Open-Meteo | W/m² |
| `gas_price` | yfinance TTF=F | EUR/MWh, daily forward-filled to hourly |
| `carbon_price` | yfinance KRBN | EUA proxy USD, daily forward-filled to hourly |
| `residual_load` | derived | load − wind − solar, clipped ≥ 0 |
| `renewable_penetration` | derived | (wind + solar) / load, clipped [0, 1] |
| `hour_sin/cos`, `weekday_sin/cos`, `month_sin/cos`, `week_sin/cos` | derived | Circular encoding of cyclic calendar features |
| `is_holiday` | derived | Public holiday flag, zone-specific (DE / ES) |
| `lag_1`, `lag_24`, `lag_168` | derived | Price lags 1h / 24h / 7d — per-zone only, no cross-zone leakage |
| `price_roll_24h`, `price_roll_168h` | derived | Rolling price means 24h / 168h |

---

## Key findings from prior analysis

Feature importance is consistent across the literature (Tschora 2024, ENTSO-E run on 2024–2026 DE-LU data):

**Permutation / SHAP ranking for DE-LU:**
1. `load` — single largest signal; demand directly sets the clearing level
2. `lag_24` (D-1 price, same hour) — price momentum; strongest autoregressive effect
3. `wind_generation` — high wind → low or negative prices
4. `solar_generation` — solar saturation drives the worst negative-price spikes
5. `lag_168` (D-7), `hour`, `weekday` — weekly seasonality and daily shape
6. `gas_price` — matters structurally but noisy in high-volatility regimes
7. `is_holiday`, `month` — minor on average; `is_holiday` critical for tail events (e.g. May 1 solar saturation: −€500)

**German vs Spanish market dynamics:**
- DE-LU is thermal + renewable driven. Renewables intermittency causes negative prices. Gas price and wind forecast matter most.
- ES is more hydro-modulated. Consumption is heat-sensitive (cooling load in summer). High solar penetration but isolated grid — Pyrenees bottleneck ~2.8 GW to France means cross-border smoothing is limited.

**On the lag structure:**
D-1 (lag_24h) and D-7 (lag_168h) carry almost all autocorrelation signal. D-2 and D-3 lags contribute <5% of SHAP weight — not included.

**On probabilistic calibration:**
Raw quantile regression empirical coverage is ~82–88% against a nominal p025–p975 band (target 95%). Gap is explained by seasonal distribution shift between training and evaluation windows. CQR on a held-out calibration slice is the planned fix.

**On cross-border flows:**
`net_imports` captures interconnection pressure: high net imports signal demand exceeds local supply (upward price pressure), high net exports signal local surplus (downward). For DE-LU this is a strong signal (8 neighbors, significant MW flows). For ES it is weaker (2 neighbors, constrained capacity) but still informative at the margin.

---

## Project structure

```
src/
  config.py       zones, paths, ENTSOE token (from env), date ranges
  ingestion.py    ENTSOE primary + energy-charts fallback + Open-Meteo + yfinance
  cleaning.py     UTC timestamps, hourly reindex, interpolation, ENTSOE + EC cleaners
  alignment.py    join sources on (timestamp, zone) MultiIndex; net_imports joined if present
  features.py     derived features, circular encoding, lags, holiday flag
  validation.py   hard assertions on final dataset quality
  pipeline.py     orchestrator with --from-* resume flags; auto-loads .env
  model.py        quantile LightGBM training, validation, eval-window prediction

notebooks/
  gen_notebook.py run this to generate model.ipynb (submission notebook)

predictions.csv   current best submission (30 rows, eval window)
.env.example      copy to .env and set ENTSOE_TOKEN
```
