alpine-arbitrage — Submission Package
======================================

Team: alpine-arbitrage
Hackathon: Frigg S2S x EC EPFL x ACE, May 2026
Zones: DE-LU (Germany-Luxembourg), ES (Spain)
Scoring: Pinball loss q=0.45

---

CONTENTS OF THIS ZIP
---------------------

  alpine-arbitrage_predictions.csv       — final submission (24 rows, 7 columns)
  alpine-arbitrage_model.ipynb           — full reproducible notebook
  data/processed/final_dataset.parquet   — engineered training dataset (52,593 rows)
  longterm/data/processed/
    marginal_costs_monthly.parquet       — monthly merit-order costs (LT model)
    structural_extended.parquet          — capacity roadmap 2018-2045 (LT model)
  longterm/notebooks/
    hero_long_term_forecast.png          — pre-generated LT forecast chart
  src/model.py                           — training + prediction entry point
  src/config.py                          — paths, zones, date ranges
  requirements.txt                       — Python dependencies
  README.txt                             — this file

---

WHY THERE IS NO MODEL CHECKPOINT
----------------------------------

We chose not to ship a serialised model file for three reasons:

  1. Training is fast. Both zones train in under 3 minutes on a laptop CPU
     (LightGBM with early stopping on a ~28,000-row training set per zone).
     Loading a checkpoint saves less time than it costs in complexity.

  2. Live data fetching happens at predict time. When you run --predict, the
     script fetches real ENTSOE price actuals up to the eval window start (for
     honest lag_24 values) and a live Open-Meteo weather forecast. A frozen
     checkpoint cannot do this — it would need to be re-run anyway to get
     fresh features. Training and fetching together is the correct single step.

  3. Reproducibility over convenience. A checkpoint captures one specific run.
     Training from the parquet on your machine reproduces our exact numbers
     (fixed random seed, same data, same hyperparameters) and is more
     trustworthy for evaluation than a binary blob we serialised on ours.

---

QUICK START — REPRODUCE PREDICTIONS IN ONE COMMAND
----------------------------------------------------

Step 1 — Install dependencies

    pip install -r requirements.txt

Step 2 — Set your ENTSOE API token

    Create a file called .env in the repo root with one line:

        ENTSOE_TOKEN=your_token_here

    Free token: https://transparency.entsoe.eu/usrm/user/createPublicUser
    (needed to fetch live gap actuals and day-ahead generation forecasts)

Step 3 — Run the model

    cd src
    python model.py --predict

    That's it. The script will:
      - Load data/processed/final_dataset.parquet
      - Train 6 LightGBM models (3 quantiles x 2 zones) — ~3 min
      - Run Mondrian CQR calibration on the Jan-May 2026 calibration set
      - Fetch ENTSOE price actuals from training tail to May 11 (real lag values)
      - Fetch Open-Meteo 14-day weather forecast for May 11
      - Fetch ENTSOE day-ahead generation forecast for May 11 (if published)
      - Generate 24 predictions for May 11 2026 00:00-23:00 UTC
      - Write alpine-arbitrage_predictions.csv to the repo root

    Total runtime: approximately 5 minutes.

---

OTHER USEFUL COMMANDS
----------------------

Predict a different window:

    python model.py --predict --start "2026-06-01 00:00" --end "2026-06-07 23:00"

    --start and --end are UTC timestamps (YYYY-MM-DD HH:MM format).
    The horizon router automatically applies the right model per slot:
      - slot within 7 days of training tail  ->  LightGBM + Mondrian CQR
      - slot beyond 7 days                   ->  seasonal median + trend

Backtest on a past window (MAE and pinball auto-reported if actuals exist):

    python model.py --predict --start "2025-06-01 00:00" --end "2025-06-07 23:00"

Train only (no predictions, just validation metrics):

    python model.py

---

OUTPUT FORMAT
--------------

alpine-arbitrage_predictions.csv has exactly 24 rows and 7 columns:

    timestamp          ISO 8601 with +01:00 offset (CEST)
    DE-LU p025         2.5th percentile EUR/MWh
    DE-LU p50          point forecast (median) EUR/MWh
    DE-LU p975         97.5th percentile EUR/MWh
    ES p025            2.5th percentile EUR/MWh
    ES p50             point forecast (median) EUR/MWh
    ES p975            97.5th percentile EUR/MWh

---

DATA FILES — WHAT IS IN EACH
------------------------------

data/processed/final_dataset.parquet
  The fully engineered hourly dataset that model.py trains on.
  52,593 rows x 80 columns. MultiIndex: (timestamp UTC, zone).
  Date range: 2023-05-08 to 2026-05-07.
  Sources: ENTSOE prices + generation + crossborder flows + day-ahead forecasts,
           Open-Meteo 26-station weather + ECMWF ensemble spread,
           yfinance gas/carbon/coal daily spot prices.

longterm/data/processed/marginal_costs_monthly.parquet
  Monthly technology-level marginal costs for the merit-order supply stack.
  1,632 rows. Zones: DE-LU, ES. Date range: 2017-12 to 2026-05.

longterm/data/processed/structural_extended.parquet
  Annual installed capacity per technology per zone, observed 2018-2025
  and projected 2026-2045. 56 rows. Used by the long-term model.

---

QUESTIONS
----------

See alpine-arbitrage_model.ipynb for full methodology, feature selection
rationale, validation analysis, and long-term model documentation.
