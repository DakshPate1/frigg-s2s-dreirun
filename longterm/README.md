# Frigg long-term forecast

Long-horizon (months → 2045) electricity-price model for **DE-LU** and **ES**, complementing the short-term GNN in `team_repo/`. Built for the S2S × EC EPFL × ACE Hackathon.

**Read [`METHODOLOGY.md`](METHODOLOGY.md) first** — that's the document explaining what we built, why, and how it scores.

---

## What's in here

```
longterm/
├── src/                                  # Sequential data pipeline
│   ├── config.py                         # All constants in one place
│   ├── ingestion.py                      # Stage 1 — fetch raw data
│   ├── cleaning.py                       # Stage 2 — clean per source
│   ├── alignment.py                      # Stage 3 — join into panels
│   ├── features.py                       # Stage 4 — derive marginal costs etc.
│   ├── validation.py                     # Stage 5 — schema + range checks
│   └── pipeline.py                       # Orchestrator with --from-* flags
├── notebooks/
│   ├── merit_order_v1.ipynb              # The model, end-to-end (~40 cells)
│   └── *.png                             # Generated plots, including hero_long_term_forecast.png
├── data/
│   ├── raw/                              # ENTSOE CSVs, yfinance fuels, GPR
│   ├── clean/                            # Per-source parquets
│   ├── aligned/                          # Joined annual + monthly panels
│   └── processed/                        # Marginal costs, structural projections
├── METHODOLOGY.md                        # Judge-facing document — start here
├── README.md                             # This file
└── requirements.txt                      # Python deps
```

---

## Running the pipeline

The pipeline shares a venv with the short-term repo at `/Users/josefloresramos/Desktop/Frigg/.venv`. If starting fresh on a new machine:

```bash
python3 -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt

# ENTSOE token must be in Frigg/.env (one level above this folder):
echo "ENTSOE_API_TOKEN=your_token_here" > ../.env
```

Then from `longterm/src/`:

```bash
# Full pipeline — fetches everything, ~30 minutes
python pipeline.py

# Skip the slow ENTSOE re-fetch if data/raw/ is already populated
python pipeline.py --from-clean

# Skip ingestion + cleaning; just rebuild aligned panels and features
python pipeline.py --from-align

# Just rebuild marginal-cost panel + structural projection from existing aligned data
python pipeline.py --from-features

# Re-run only the validation checks
python pipeline.py --validate-only
```

Validation should print **8 ✓ lines** with no warnings.

---

## Running the model

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace merit_order_v1.ipynb
```

Or open `merit_order_v1.ipynb` in your editor and run cells interactively.

The notebook produces 7 PNG plots, including the centerpiece **`hero_long_term_forecast.png`**.

---

## Data sources

| Source | What | License |
|---|---|---|
| ENTSOE Transparency Platform | Installed capacity (annual), day-ahead prices, load, generation per type | Free, requires API key |
| yfinance | TTF gas (`TTF=F`), KRBN carbon, coal API2 (`MTF=F`), Brent oil (`BZ=F`) | Free |
| Iacoviello GPR | Monthly geopolitical risk index | Free, [matteoiacoviello.com/gpr.htm](https://www.matteoiacoviello.com/gpr.htm) |
| Static (`config.py`) | Heat rates, emissions factors, capacity roadmaps, gas-cap parameters | IPCC/EEA defaults + EU policy targets, hand-tuned to observed ENTSOE values |

**No historical electricity prices are used as model input** — they appear only as a calibration target.
