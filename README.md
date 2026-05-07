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

**Model (src/model.py) — trained, CQR-calibrated, dual-regime predictions**
- Quantile LightGBM at q = 0.025 / 0.45 / 0.975 per zone. p50 trained at q=0.45 to match scoring function (underforecast bias by design).
- 26 features: generation + weather + fuel/carbon + circular calendar + is_holiday + price lags + net_imports.
- Recursive lag fill for evaluation window: lag_1 / lag_24 populated slot-by-slot from model's own calibrated p50 predictions.
- **CQR calibration applied** (Jan–May 2026 calibration set, n≈2992/zone): inflates p025/p975 symmetrically to hit 95% coverage; shifts p50 to correct systematic zone bias.
- **Dual-regime forecasting**: ≤7 days → LightGBM + CQR; >7 days → long-term seasonal model (recency-weighted median profile + post-crisis 2023+ trend + sqrt-scaled uncertainty). Handles any horizon including 2-year notebook forecast.
- **3 new features** (29 total): `residual_load_ramp` (gas ramp-rate spike trigger), `days_to_holiday` + `days_from_holiday` (continuous bridge-day context replacing binary flag).

**Validation results (2025 holdout, pre-CQR):**

| Zone | MAE p50 | Pinball 0.45 | Coverage p025–p975 | Band width | Naive MAE |
|------|--------:|-------------:|-------------------:|-----------:|----------:|
| DE-LU | 8.53 EUR/MWh | 4.12 | 88.0% | 44.97 EUR/MWh | 33.10 |
| ES | 6.95 EUR/MWh | 3.50 | 82.3% | 29.33 EUR/MWh | 28.97 |

**CQR + Mondrian calibration results (Jan–May 2026 calibration set, n≈2998/zone):**

| Zone | Coverage (raw → CQR) | Global Q_hat | Bucket 0 (weekday) Q_hat | Bucket 1 (weekend/holiday) Q_hat | p50 shift |
|------|---------------------:|-------------:|-------------------------:|---------------------------------:|----------:|
| DE-LU | 90.4% → 95% | 3.71 EUR/MWh | 4.15 EUR/MWh | 2.75 EUR/MWh | +1.24 |
| ES | 77.9% → 95% | 3.46 EUR/MWh | 4.03 EUR/MWh | 3.04 EUR/MWh | −1.88 |

*Weekday Q_hat > weekend Q_hat because Jan–May 2026 weekday slots were harder to calibrate (unusual demand patterns). Mondrian applies tighter bands on weekend/holiday slots where the model was already well-calibrated.*

**Eval window predictions (alpine-arbitrage_predictions.csv) — Mondrian CQR-calibrated**
- 30 slots, May 8–9 2026 in CEST.
- DE-LU: mean p50 ~€53, solar dip May 9 (p50 → −€51 at 14:00), evening recovery to €80.
- ES: mean p50 ~€57, midday trough (~€0), evening €78.
- Bands: tighter on May 9 Saturday (bucket 1), wider on May 8 Friday evening (bucket 0).

**Playground notebook (`notebooks/playground.ipynb`) — ready**
- 36-cell notebook covering every pipeline layer: data health → EDA → features → training → validation metrics (with explanations) → CQR before/after → feature importance → error analysis → eval window preview.
- All sections independently runnable. Run `python notebooks/gen_playground.py` to regenerate.

**Feature selection rationale (from Tschora 2024 + ENTSOE analysis)**
- Documented in Key findings section below.
- Cross-zone features deliberately excluded: ES is isolated (Pyrenees bottleneck ~2.8 GW to FR), and the feature-vocabulary symmetry constraint would add noise not signal.

---

### What needs to happen next

**Must-do before submission:**
- [x] ~~Re-run full pipeline with ENTSOE as source~~ — **done**. `net_imports` now in all training years. Pipeline re-run with chunk-boundary gap fix (21h gaps at 180-day boundaries now interpolated).
- [ ] Build `model.ipynb` submission notebook — `notebooks/gen_notebook.py` is ready, run it to generate the notebook skeleton, then fill in output cells
- [ ] Package `data.zip` with all data files used + README.txt describing each file

**High-value improvements:**
- [x] ~~Conformalized Quantile Regression (CQR)~~ — **done**. Coverage corrected to 95% on both zones. Calibration set Jan–May 2026 (n≈2992/zone). See `calibrate_zone()` in `src/model.py`.
- [x] ~~Open-Meteo weather forecast for eval window~~ — **done**. `fetch_weather_forecast()` called automatically by `model.py --predict`; replaces seasonal proxy for temperature/wind/solar_radiation on all eval slots.
- [x] ~~ENTSOE actual cross-border flows for gap period~~ — **done**. `fetch_gap_actuals()` in `model.py` fetches prices + net_imports for the gap between training tail and eval start; lag_24 lookups use real values.
- [x] ~~Mondrian conformal bands~~ — **done**. Per-regime CQR calibration: bucket 0 (normal weekday) vs bucket 1 (weekend/holiday/bridge day). Each bucket gets its own Q_hat, tightening intervals where the model is already well-calibrated. Results: DE-LU bucket 0 Q_hat=4.15 / bucket 1 Q_hat=2.75; ES bucket 0 Q_hat=4.03 / bucket 1 Q_hat=3.04.
- [x] ~~Day-ahead generation forecasts for eval window~~ — **done**. `fetch_entsoe_gen_forecast()` in `model.py` fetches ENTSOE load+wind+solar forecasts for the eval slots (what operators publish before the auction). Falls back to proxy if not yet published. Pipeline infrastructure also added (full historical forecast training requires a pipeline rerun).

**Recent feature improvements:**
- [x] `residual_load_ramp` — hour-over-hour grid ramp rate; triggers gas-peaker spike predictions
- [x] `days_to_holiday` / `days_from_holiday` — continuous holiday proximity; fixes bridge-day misses
- [x] Long-term model: recency-weighted median profile + post-crisis (2023+) trend anchor

**Worth exploring if time allows:**
- [ ] Longer training tail: ENTSOE data goes back to 2015; adding 2015–2020 may improve rare-event coverage (energy crisis periods, COVID demand collapse).
- [ ] ENTSOE historical forecast pipeline rerun: `fetch_entsoe_forecasts()` is implemented in `ingestion.py`; running `python pipeline.py` will fetch load/wind/solar forecast columns for training. Then add `_forecast` variants to FEATURES in `model.py` to train on pre-auction information instead of actuals.
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

# Train model and generate predictions (hackathon eval window):
python model.py --predict

# Predict any window — model auto-selects regime per slot:
python model.py --predict --start "2026-05-10 17:00" --end "2026-05-12 22:00"  # short-term
python model.py --predict --start "2026-05-10 00:00" --end "2028-05-10 00:00"  # 2-year LT

# Backtest against known prices (actuals auto-reported if window overlaps dataset):
python model.py --predict --start "2025-06-01 00:00" --end "2025-06-02 23:00"
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
Raw quantile regression empirical coverage is ~82–88% against a nominal p025–p975 band (target 95%). Gap is explained by seasonal distribution shift between training and evaluation windows. CQR on a Jan–May 2026 calibration set corrects this to exactly 95% on both zones — see `calibrate_zone()` in `src/model.py`.

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
  model.py        quantile LightGBM (short-term) + seasonal profile (long-term), CQR calibration
                  (--start/--end for any window; auto regime split at 7d; backtest if actuals exist)

notebooks/
  gen_notebook.py  run this to generate model.ipynb (submission notebook)
  gen_playground.py run this to regenerate playground.ipynb
  playground.ipynb  36-cell exploration notebook covering every pipeline layer

alpine-arbitrage_predictions.csv   current best submission (30 rows, eval window, CQR-calibrated)
.env.example      copy to .env and set ENTSOE_TOKEN
```
