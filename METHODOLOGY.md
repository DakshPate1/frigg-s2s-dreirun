# Methodology — Alpine Arbitrage

---

## What we are predicting

The Day-Ahead Auction (DAA) sets electricity prices for every hour of tomorrow. Grid operators, generators, and traders submit bids and offers; the market clears at the price where supply meets demand for each hour. We predict 24 of these prices — one per hour of Sunday 11 May 2026 — for two European bidding zones: **DE-LU** (Germany-Luxembourg) and **ES** (Spain).

The competition scores predictions using **pinball loss at q=0.45**. This is not a symmetric metric. Overestimating the price is penalised approximately 1.22 times more than underestimating by the same amount. This shifts the optimal prediction slightly below the true expected value, which is why our point forecast targets the 45th percentile, not the 50th.

---

## The challenge

Electricity price forecasting is structurally harder than most time series problems. Understanding why shapes every decision we made.

**Prices are not a time series — they are market clearings.** Each hourly price is the outcome of a simultaneous auction involving hundreds of participants with private information. The price is the intersection of a supply curve (who is willing to generate at what cost) and a demand curve (who needs power at what price). A model that treats prices as a smooth signal to be extrapolated will fail because it does not understand the mechanism producing the numbers.

**The distribution has heavy tails in both directions.** Normal market hours see prices in a 20–100 EUR/MWh range. Calm conditions with abundant solar can push prices below zero — generators pay the grid to take their power rather than shut down and restart. Tight supply with a cold snap and low wind can push prices above 500 EUR/MWh. The model must learn from a distribution where the 99th percentile event is qualitatively different from the median, not just larger.

**The relevant training history is short.** The 2021–2022 European energy crisis (Russia curtailing gas flows, TTF gas futures at 10× their pre-crisis level) distorted prices for 18 months in ways that have not recurred. Including that data in training teaches the model a regime that no longer exists. Excluding it reduces the effective training window to roughly 3 years. That is not much data for a model trying to learn patterns that repeat at daily, weekly, seasonal, and annual frequencies simultaneously.

**Two zones, two physics.** DE-LU is dominated by wind (north) and solar (south) with gas peakers setting the marginal price on calm days. Spain has higher solar penetration, meaningful hydro swing supply, active nuclear, and is nearly isolated from Central European markets — the single France-Spain interconnector runs at roughly 3 GW capacity. Patterns that explain DE-LU prices often have no analog in ES and vice versa.

**The scoring metric directly penalises the most natural error.** When a model is uncertain, its safe default is to predict conservatively — high, to avoid underestimating a spike. But the q=0.45 metric penalises overestimation more. Every design choice involving uncertainty — the quantile target, the calibration shift, the interval inflation — has to work against the model's natural conservative instinct.

**At prediction time, the future is partially known.** Unlike a purely blind forecast, we have access to ENTSOE's published day-ahead load and generation forecasts (what the grid operator expects), Open-Meteo's 14-day weather forecast, and neighbor zone prices from the previous day. These are real signals that market participants used when placing their bids. Not using them would deliberately handicap the model.

**Lags are the most important features and also the hardest to fill.** SHAP confirms that yesterday-same-hour price (`lag_24`) and last-week-same-hour price (`lag_168`) dominate all other features. For the first slot of the eval window, `lag_24` is the price from 24 hours earlier — a real number we can fetch. But as we predict further ahead recursively, `lag_1` for slot 2 is the model's own prediction for slot 1. Error compounds. The lag strategy must account for this.

---

## What we tried

The solution did not arrive fully formed. It was built iteratively, with each experiment informing the next. The sequence matters as much as the final result.

### Starting point: the naive baseline

The simplest possible forecast is the same-hour price from one week ago (`lag_168`). Electricity prices have strong weekly seasonality — Monday 09:00 looks roughly like last Monday 09:00 — so this is a meaningful baseline, not a trivial one.

| Zone | Naive MAE (lag_168) |
|------|--------------------:|
| DE-LU | 32.82 EUR/MWh |
| ES | 29.22 EUR/MWh |

Any model we build needs to beat this substantially to justify its complexity.

### First model: LightGBM with basic features

The original codebase used a single-city weather reading per zone, 10-metre wind speed, no neighbor prices, no fuel prices beyond gas, and no nuclear data. The feature set was around 25 variables.

| Zone | MAE (original) |
|------|---------------:|
| DE-LU | 8.53 EUR/MWh |
| ES | 6.95 EUR/MWh |

This was already 4× better than naive. But it left identifiable gaps. Feature importance showed weather variables ranking highly for DE-LU while the single weather station (Berlin) was not representative of either where German wind turbines are or where German solar panels are. The model was learning a noisy proxy.

### What the GNN explored

Our teammate built a **GATv2 Graph Attention Network** that treats European bidding zones as nodes in a graph — DE-LU, ES, France, Belgium, Netherlands, Austria, Switzerland, Portugal. Edges represent physical interconnections. Each node carries 25 features encoding its own generation mix, weather, and market state. The network runs two graph attention layers that let each zone aggregate information from its neighbors before making a prediction.

The architecture is elegant for a fundamentally graph-structured problem: electricity markets are literally a network, and the network topology determines what signals can transmit where.

| Zone | GNN MAE (5-seed ensemble) |
|------|---------------------------:|
| DE-LU | 8.21 EUR/MWh |
| ES | 6.75 EUR/MWh |

The GNN underperformed the expanded LightGBM (8.21 vs 7.14 for DE-LU, 6.75 vs 6.47 for ES) for two reasons. First, our dataset only contains DE-LU and ES as full rows with complete features. France, Belgium, Netherlands, Austria, Switzerland, and Portugal exist only as neighbor price lag columns — they are not full graph nodes with their own generation, weather, and load data. The GNN's graph topology assumed all 8 zones had equal information; the actual data had 2 well-observed zones and 6 sparsely observed ones. Second, the GNN required significantly more infrastructure (PyTorch Geometric, multi-seed training, separate joblib model files) for a result that was worse. The simpler model won.

### Feature expansion: where LightGBM improved

With the GNN result as context, we invested in expanding the LightGBM feature set rather than the model architecture. The expansions, in the order they were added:

**Multi-city capacity-weighted weather.** Germany has 60 GW of onshore wind concentrated in the north (Schleswig-Holstein, Lower Saxony) and 60 GW of solar concentrated in the south (Bavaria, Baden-Württemberg). A single weather station in Berlin represents neither. We expanded to 26 stations across 9 location groups, capacity-weighted by installed generation at each location. This alone improved DE-LU MAE by roughly 10%.

**Wind at hub height.** The original feature used wind speed at 10 metres (standard meteorological measurement). Turbines operate at 80–150 metres. Wind speed increases non-linearly with height (follows a power law), and wind *power* scales with the cube of wind speed. Switching to 100-metre measurements and adding the cubed wind feature captured the actual physical relationship.

**Neighbor prices with lag.** France (nuclear-heavy) and the Netherlands (gas hub) set the reference price for Central European markets. Switzerland (alpine hydro) dampens DE-LU spikes. Denmark exports North Sea wind into northern Germany. We added the previous day's settled price from each of these zones as features, along with the spread between DE-LU and each neighbor — a large spread indicates the interconnector is congested and arbitrage is limited. This gave the model a view of what the interconnected market had actually cleared the day before.

**Nuclear unavailability for ES.** Spain runs approximately 7 GW of nuclear capacity. ENTSOE publishes REMIT notices — legally required disclosures of planned and unplanned generation unavailability. We compute the remaining available nuclear MW after all active notices. A 2 GW nuclear outage on a low-wind day creates a price spike that neither weather nor load can explain alone. Adding this feature disproportionately improved ES performance.

**Coal price.** Germany still operates coal power plants as peakers. API2 coal futures (USD/tonne) are a second fuel price signal relevant for DE-LU hours when gas is expensive. Minor improvement individually; meaningful in combination.

**ECMWF ensemble spread.** The European Centre for Medium-Range Weather Forecasts publishes not just a single weather forecast but an ensemble of 50 member runs. The spread between ensemble members is a measure of forecast uncertainty — when the models disagree about tomorrow's wind, the price interval should be wider. We added the standard deviation of wind and solar forecasts across ensemble members as features.

**Regime flags.** Three binary indicators that capture structural market states: whether the zone was in the energy crisis period, whether the current hour is a peak demand hour (07–09 or 17–20), and whether the zone's own price was negative 24 hours earlier (a leading indicator of solar oversupply regime).

The cumulative effect:

| Zone | Original MAE | Expanded MAE | Improvement |
|------|-------------:|-------------:|------------:|
| DE-LU | 8.53 EUR/MWh | 7.14 EUR/MWh | −16% |
| ES | 6.95 EUR/MWh | 6.47 EUR/MWh | −7% |

DE-LU improved more because it benefited most from the multi-city weather and neighbor price additions — both are harder to approximate with simple proxies in a large, interconnected market. ES was already better-constrained by its relative isolation.

---

## Why LightGBM

### The initial reasoning — before any experiment ran

The choice of LightGBM was made before writing a single training loop, by matching the problem's structural properties against what each model class is actually good at. This section explains that reasoning from first principles, then contrasts it with the alternatives we ruled out.

The problem has five properties that, taken together, point strongly toward gradient-boosted trees:

**1. The input is a feature table, not a sequence.** Each prediction is: given a row of 41 numbers describing the current state of the grid, the weather, the market, and the calendar — what is the price for this hour? The temporal structure (daily seasonality, weekly seasonality, price momentum) is already encoded in the lag features and rolling statistics before the model ever sees the data. The model does not need to process raw time series; it needs to learn a mapping from a feature vector to a price. This is a regression problem on a table, and for that class of problem, gradient boosting is the empirically dominant method.

**2. The relationships are non-linear but not compositional.** Price does not increase smoothly with wind speed — at high wind penetration, prices collapse non-linearly as supply overwhelms demand. Temperature affects heating demand quadratically around a comfort threshold. Solar output interacts with time of day in ways that change with season. These are non-linear interactions, but they are not the kind of deep hierarchical compositions that neural networks are designed to learn. A decision tree naturally captures a threshold like "when wind penetration exceeds 60% AND it is a weekend, prices collapse" without needing to be told where the threshold is. Stacking thousands of such trees through gradient boosting builds up a flexible, non-parametric function approximator that handles exactly this kind of local, regime-dependent non-linearity.

**3. Missing values are structural, not a data quality problem.** Zone-specific features are always absent for the wrong zone by design. ES hydro precipitation is a meaningful feature for Spain; it is undefined for Germany-Luxembourg. Danish wind speed matters for DE-LU; it is irrelevant for ES. A model that requires a complete input vector — which includes most neural networks, linear models, and SVMs — forces a choice: either train two separate models with different feature sets, or impute values that carry no information. LightGBM routes NaN observations to whichever branch of the tree produces better predictions without any preprocessing. This allows a single unified feature schema for both zones, which simplifies the code and avoids the risk of inadvertently encoding zone identity through imputed constants.

**4. The training set is moderate, not large.** Effective training data covers May 2023 to early 2026 — roughly 18 months per zone after burn-in, approximately 13,000 hourly rows. Neural networks benefit from scale: transformers and LSTMs built for time series typically need at least 50,000–100,000 observations to learn generalised temporal patterns. With 13,000 rows, a neural network with sufficient capacity to model the non-linear feature interactions will overfit unless heavily regularised — and that regularisation introduces its own hyperparameter search. Gradient boosted trees with early stopping regularise naturally at this data scale.

**5. Interpretability is a development tool, not a nice-to-have.** We did not know at the start which features would matter most or in what direction they would push prices. SHAP values for LightGBM run in seconds on the full validation set. After adding a feature, we could immediately see whether it was doing something meaningful or being ignored, and whether its direction of effect was physically plausible. This feedback loop is how we discovered that single-city weather was inadequate (the Berlin weather reading had high SHAP variance that suggested compensation for noise rather than a clean signal), and how we identified that nuclear unavailability was worth fetching for Spain (a high-gain interaction that appeared as soon as the feature was added). A model that is slower to train or opaque to interpret would have permitted far fewer experiments within the available time.

### Why not the alternatives

**Time series models (ARIMA, SARIMA, SARIMAX)**

Classical time series models are designed for a single univariate series with stationary noise. Electricity prices are not stationary — variance shifts between seasons, between market regimes, and between crisis and non-crisis periods. SARIMAX with 41 exogenous variables is heavily under-identified with 13,000 observations, and even fully specified it cannot represent the non-linear interactions between features. The model also has no mechanism for handling missing values in the exogenous inputs. These models are appropriate for stable, low-feature forecasting problems. This is not that problem.

**Linear quantile regression**

Linear quantile regression can directly optimise the pinball loss, which is an advantage. But it cannot represent any of the non-linear relationships that matter here — wind cubed, temperature squared, the interaction between solar penetration and hour of day, the threshold behaviour of interconnector congestion — without those features being constructed by hand in advance and every interaction term specified explicitly. We already hand-constructed wind cubed and temperature squared; a linear model would have required hundreds of additional interaction terms to approximate what LightGBM learns automatically. The representational gap is too large.

**Random forests**

Random forests and LightGBM are both tree ensembles. The structural difference is in how they build trees. Random forests train trees independently in parallel, each on a random subsample, and average their outputs. LightGBM uses gradient boosting: each tree is trained to correct the residual errors of all prior trees. For structured tabular data, boosting consistently outperforms bagging (random forests) because it focuses computational effort on the hardest-to-predict observations rather than training identical-distribution trees that average to a smooth function. Random forests also tend to require more trees to reach the same accuracy and offer no native quantile loss objective.

**XGBoost and CatBoost**

Both are gradient boosting frameworks like LightGBM. XGBoost grows trees level-by-level (symmetric splits across all leaves at a depth); LightGBM grows leaf-by-leaf, always expanding the leaf with the highest gain. Leaf-wise growth converges faster on the same dataset and typically reaches lower loss at the same number of trees. LightGBM also uses a histogram-based split-finding algorithm that is substantially faster than XGBoost's exact split search on datasets with many features. CatBoost is strongest when the feature set is dominated by categorical variables — it handles ordered categorical encoding natively. Our features are almost entirely numeric (continuous measurements, circular encodings, and binary flags). The LightGBM advantage in speed and the nature of our features made it the natural choice over the other two.

**LSTMs and recurrent neural networks**

LSTMs process raw sequences and learn temporal dependencies directly, which superficially sounds well-suited to hourly price data. But the key dependency structure we care about — same-hour-last-day and same-hour-last-week — is not a continuous dependency that an LSTM would naturally capture with a hidden state evolved over hundreds of hours of intervening data. We encode those dependencies explicitly as lag features, which is both more interpretable and more computationally efficient. LSTMs also require complete sequences (no NaN in inputs without masking layers), cannot natively output quantiles without a specialised output head, are slow to train compared to gradient boosting, and typically require at least 50,000 sequences to learn generalisable temporal patterns.

**Transformers and Temporal Fusion Transformers**

The Temporal Fusion Transformer (TFT) is a state-of-the-art architecture for tabular time series that incorporates attention mechanisms, variable selection networks, and explicit handling of known future inputs (like calendar features). It is the closest neural network alternative to our setup. The reasons we did not use it: first, it requires substantially more data than LightGBM to outperform it — published benchmarks show TFT surpassing gradient boosting only at scale. Second, its training time is an order of magnitude longer, making feature iteration slow. Third, interpreting which inputs are driving a given prediction requires careful attention analysis rather than the immediate SHAP readout that guided our feature development. Fourth, quantile outputs require a custom loss modification. The architecture is appropriate when data is abundant and the temporal pattern is complex enough that explicit lag features cannot capture it. Neither condition holds here.

**Neural networks generally**

The general case against neural networks for this problem is not that they are bad models — they are not — but that they optimise for a different problem shape. Neural networks excel when: (a) the input is raw (images, text, audio) and features must be learned rather than constructed; (b) the dataset is large enough to support the parameter count; (c) the training budget allows many epochs of experimentation. Our inputs are already engineered features, our dataset is moderate, and our iteration speed requirement is high. Gradient boosting dominates this operating point across competition benchmarks, academic electricity forecasting comparisons, and our own experiments.

### What LightGBM specifically provides

Having ruled out alternatives, the specific properties of LightGBM that make it the right gradient boosting implementation:

- **Native quantile loss:** `objective='quantile', alpha=q` trains each model to minimise pinball loss directly. No post-processing, no Gaussian assumption, no delta-method confidence intervals. Each of the three quantile models is a dedicated specialist.
- **Native NaN handling:** splits route NaN observations to whichever child node produced better predictions on the training set. No imputation required.
- **Leaf-wise tree growth:** converges faster than level-wise growth for the same number of leaves, reaching lower loss in fewer boosting rounds.
- **Early stopping:** training halts when validation pinball stops improving for 100 rounds. The model cannot overfit past the point where it stops generalising.
- **SHAP integration:** exact TreeSHAP values in seconds, not an approximation. This is the feedback mechanism that guided every feature addition.
- **Training speed:** under five minutes for six models (3 quantiles × 2 zones) on a standard laptop. This made feature iteration possible within a single working session.

---

## The data problem

Electricity prices are not generated by a simple formula. They emerge from the simultaneous interaction of physical generation capacity, weather, fuel costs, cross-border flows, and calendar effects. Getting all of this into a consistent, hourly, UTC-indexed table is most of the engineering work.

### Training window choice

We train on **May 2023 → May 2026** only. The 2021–2022 European energy crisis (Russian gas supply disruption, TTF gas at 10× normal) created a price regime that does not exist in 2026. Training on those years teaches the model to respond to gas prices that will not recur. We keep a binary `crisis_period` flag so the window can be extended for sensitivity analysis without corrupting the model.

---

## How the data pipeline works

The pipeline has four sequential stages before any model code runs. Each stage saves its output to disk so any stage can be restarted without redoing earlier work.

```
ingestion.py  →  data/raw/
cleaning.py   →  data/clean/
alignment.py  →  data/aligned/base_dataset.parquet
features.py   →  data/processed/final_dataset.parquet
                        ↓
               model.py  (train → calibrate → predict)
```

### Stage 1 — Ingestion (`ingestion.py`)

Fetches raw data from three external sources.

**ENTSOE Transparency Platform** is the primary source. It publishes actual DAA prices, total grid load, generation by source (wind, solar, hydro, nuclear), cross-border flows, and day-ahead generation forecasts that market participants could see before the auction closed. For DE-LU, neighbor zone prices from France, Netherlands, Switzerland, and Denmark are fetched separately. For ES, nuclear unavailability (REMIT) notices are fetched and aggregated into available MW.

ENTSOE data is fetched in 180-day chunks because the API rejects large windows. Each chunk's end timestamp is pushed forward by one day to avoid a known API behaviour where the last few hours of each window are dropped.

**Open-Meteo** provides 26-station weather across 9 location groups. Wind is fetched at 100 metres; solar and temperature at surface. An ensemble API returns ECMWF spread (standard deviation across 50 ensemble members) for wind and solar.

**yfinance** provides daily fuel prices: TTF gas, KRBN carbon proxy, API2 coal. These are forward-filled to hourly because fuel markets close daily.

A global socket timeout of 120 seconds prevents any call from hanging the pipeline. All fetch functions catch their own exceptions so a failure in one source does not abort the run.

### Stage 2 — Cleaning (`cleaning.py`)

Every source arrives in a different format. Cleaning normalises everything to UTC, reindexes to a full hourly grid, and linearly interpolates gaps up to 24 hours. Gaps longer than 24 hours remain as NaN. The 24-hour interpolation threshold specifically covers the 21-hour API artefact that appears at ENTSOE chunk boundaries.

Multi-city weather arrives with one column per station. Cleaning computes the capacity-weighted aggregate and creates legacy alias columns (`temperature`, `wind_speed`, `solar_radiation`) so alignment works regardless of which format was ingested.

### Stage 3 — Alignment (`alignment.py`)

Joins all cleaned sources per zone on a shared UTC hourly index, then stacks both zones into a `(timestamp, zone)` MultiIndex.

Cross-border flows join with `left` because ENTSOE crossborder coverage has gaps; missing values become NaN. Rows missing any of five critical columns — price, load, wind generation, solar generation, temperature — are dropped. These are rows where the target variable or its most essential drivers are unknowable.

### Stage 4 — Feature engineering (`features.py`)

Adds all 41 model features and saves `data/processed/final_dataset.parquet`.

The feature schema is intentionally identical for both zones. Zone-specific features (e.g. `DK_wind_speed` for DE-LU, `ES_hydro_precipitation` for ES) are present as columns for every row but are NaN for the zone that does not use them. LightGBM learns to ignore NaN branches — no performance penalty, no implicit zone encoding.

The first 168 rows per zone are dropped after feature engineering. `lag_168` requires a full week of prior data; keeping early rows would teach the model corrupted lag relationships.

---

## Feature design: what we included and why

Every feature answers a specific question about what determines the price for that hour.

**Supply and demand fundamentals** (`load`, `wind_generation`, `solar_generation`, `hydro_generation`, `nuclear_generation`): what is actually running on the grid. These are actuals for training, proxied by ENTSOE day-ahead forecasts at inference time.

**Multi-city weather** (`wind_speed_agg`, `wind_speed_cubed`, `solar_radiation_agg`, `solar_hour_interaction`, `temperature_agg`, `temperature_sq`): capacity-weighted aggregates across generation centres and demand centres separately. Wind is cubed because power output scales with the cube of wind speed. Solar is interacted with hour because the same irradiance in the morning has different generation consequences from the same irradiance at noon (panel angle, ambient temperature effects). Temperature is squared to capture nonlinear heating and cooling demand thresholds.

**Cross-border physical signals** (`DK_wind_speed`, `DK_wind_speed_cubed`, `CH_precipitation`, `CH_precip_7d_sum`): Danish North Sea wind exports into northern Germany; Swiss alpine precipitation fills reservoirs that export cheap hydro into DE-LU. The 7-day rolling sum captures the lag between rain falling and water reaching the turbine.

**ES-specific hydro and nuclear** (`ES_hydro_precipitation`, `ES_hydro_precip_7d_sum`, `nuclear_available_mw`): Spain's hydro is a major swing supply source that neither weather nor generation actuals fully capture without the reservoir filling lag. Nuclear unavailability from REMIT notices explains price spikes that have no weather or load explanation.

**Fuel and carbon** (`gas_price`, `carbon_price`, `coal_price`): set the marginal cost of the most expensive plant that must run. Gas peakers dominate in DE-LU; coal is a secondary signal relevant when gas is expensive.

**Day-ahead generation forecasts** (`residual_load_forecast`, `renewable_penetration_forecast`, `residual_load_ramp_forecast`): what market participants knew before they placed bids. Including forecast alongside actuals teaches the model the gap between expectation and outcome, which is a real market signal.

**Neighbor prices with lag** (`FR_price_lag24`, `NL_price_lag24`, `CH_price_lag24`, `DK_price_lag24`): price transmission between coupled markets takes roughly one auction cycle — 24 hours. Traders see yesterday's French nuclear clearance price when setting today's bids for DE-LU. Transmission spreads (`DE_LU_FR_spread`, etc.) signal whether interconnectors are congested; a large spread means arbitrage is limited and prices in each zone are more decoupled.

**Cross-zone price** (`cross_zone_lag24`): the other zone's settled price from 24 hours earlier. Captures the degree of Iberian isolation (ES) versus Central European coupling (DE-LU). When the spread between zones is large, they are operating in different regimes; when it is small, common drivers are dominating.

**Calendar features** (circular sine/cosine encoding): hour, weekday, month, week of year. Circular encoding means the model sees no discontinuity between 23:00 and 00:00, or between December and January. Holiday proximity (`is_holiday`, `days_to_holiday`, `days_from_holiday`) captures bridge-day demand suppression — the Friday between a Thursday holiday and the weekend behaves more like a weekend than a workday.

**Regime flags** (`crisis_period`, `is_peak`, `negative_price_lag24`): `is_peak` marks morning (07–09) and evening (17–20) demand peaks. `negative_price_lag24` is a binary leading indicator that the zone was in solar oversupply 24 hours ago — a signal that conditions for negative prices may persist.

**Price history** (`lag_1`, `lag_24`, `lag_168`, rolling means and standard deviation): SHAP analysis consistently identifies `lag_24` and `lag_168` as the two strongest features. The rolling standard deviation (`price_roll_std_168h`) captures volatility regime — in high-volatility periods, the model widens its uncertainty implicitly through its training distribution.

**Ensemble uncertainty** (`wind_ensemble_std`, `solar_ensemble_std`): ECMWF ensemble spread across 50 weather model runs. High spread means the weather forecast itself is uncertain, which should propagate into a wider price interval. This is the only feature that directly informs the model about its own second-order uncertainty.

---

## Training

The dataset is split by time into three non-overlapping windows:

- **Training:** May 2023 → January 2025 (~18 months per zone after burn-in)
- **Validation:** January 2025 → January 2026 (full year 2025, unseen during training)
- **Calibration:** January 2026 → May 10 2026 (CQR adjustment only, not model fitting)

Both zones are trained independently. The feature schema is shared, but a separate set of three LightGBM models is fit for each zone. Training drops rows where the target price is NaN but never drops on feature columns — LightGBM handles NaN features natively.

Training runs in `model.py` → `train_zone()`. It returns both the fitted models and the validation predictions, which feed directly into reporting and calibration.

---

## Uncertainty calibration

Raw quantile regression does not guarantee that the predicted 97.5th percentile actually contains 97.5% of outcomes in practice — the training distribution and the test distribution differ. We apply **Conformalized Quantile Regression (CQR)** to correct this.

CQR works as follows: take a calibration set the model has never trained on, run the model's predictions on it, compute how wrong each prediction is (the non-conformity score), and find the threshold that would have covered 95% of those errors. At test time, inflate the prediction intervals by that threshold. This is a distribution-free guarantee: regardless of what the true price distribution looks like, CQR achieves at least 95% coverage on exchangeable data.

We use **Mondrian CQR**, which computes separate thresholds per regime:

- **Bucket 0 — normal weekday:** no holiday within one day. Demand and generation mix are predictable. Intervals are tighter.
- **Bucket 1 — weekend, holiday, or bridge day:** demand drops unpredictably. Industrial load shuts down. Prices move more freely in either direction. Intervals are wider.

The separate correction for p50 is distinct from the interval correction. We compute the 45th percentile of calibration residuals and add this constant to every p50 prediction. This directly minimises the pinball loss at q=0.45 on held-out data, rather than indirectly hoping that the trained quantile hits the right value.

For the May 11 2026 eval window (a Sunday), every slot falls in bucket 1.

| Zone | Q_hat bucket 0 | Q_hat bucket 1 | p50 shift |
|------|---------------:|---------------:|----------:|
| DE-LU | 2.37 EUR/MWh | 4.34 EUR/MWh | +1.42 |
| ES | 4.31 EUR/MWh | 3.72 EUR/MWh | −1.62 |

The DE-LU p50 is shifted up, meaning the model underestimated prices in the Jan–May 2026 calibration window. The ES p50 is shifted down — overestimation consistent with continued solar capacity growth suppressing midday prices beyond what the training data would predict.

CQR calibration runs in `model.py` → `calibrate_zone()`.

---

## Prediction for the eval window

For each of the 24 hourly slots, the prediction is assembled in `model.py` → `build_eval_row()`, pulling from three live sources:

**1. Gap actuals** (`fetch_gap_actuals()`): ENTSOE prices and crossborder flows from the training tail (May 7) to the eval start (May 11). This gives honest `lag_24` values for the first eval slots rather than historical proxies. Without this, the most important feature for the first 24 slots would be wrong.

**2. Weather forecast** (`_fetch_weather()`): Open-Meteo's 14-day forecast API. For a prediction made on May 9, May 11 weather is a forecast rather than a measurement. We use the forecast API rather than the historical archive for these slots.

**3. ENTSOE generation forecast** (`fetch_entsoe_gen_forecast()`): ENTSOE publishes day-ahead wind, solar, and load forecasts the morning before each auction. These are exactly what market participants used when placing their bids and are strictly better than historical proxies.

Slots are built sequentially in time order. Price lags that fall inside the prediction window are filled with the model's own prior outputs — recursive inference. Using historical prices for in-window lags would introduce look-ahead bias. The loop is in `model.py` near line 1005, structured specifically for this sequential dependency.

**Horizon routing:** slots more than 7 days from the training tail route to the long-term seasonal model (`predict_longterm_slot()`) instead of LightGBM. The seasonal model uses a recency-weighted median profile per `(month, dayofweek, hour)` plus a post-crisis linear trend. For the May 11 2026 window (4 days from the May 7 training tail), the fallback is not triggered — all 24 slots go through LightGBM.

---

## Full model comparison

All approaches evaluated on the 2025 holdout year:

| Approach | DE-LU MAE | ES MAE | Notes |
|----------|----------:|-------:|-------|
| Naive baseline (lag_168) | 32.82 | 29.22 | Same-hour-last-week price |
| LightGBM — original features | 8.53 | 6.95 | ~25 features, single-city weather, 10m wind |
| GATv2 GNN (5-seed ensemble) | 8.21 | 6.75 | Graph attention, 8-zone graph topology |
| **LightGBM — expanded features** | **7.14** | **6.47** | 41 features, 26-station weather, 100m wind |

The feature expansion contributed more than the architectural alternative. The GNN's graph structure is conceptually well-suited to the problem but is constrained by what data is actually available per zone — France, Belgium, Netherlands, Austria, Switzerland, and Portugal are present as neighbor price lag columns, not as full graph nodes with generation, weather, and load. The GNN architecture assumed 8 well-observed nodes; the reality was 2 well-observed and 6 sparsely observed. LightGBM required no such assumption.

After CQR calibration on Jan–May 2026:

| Zone | Raw coverage | CQR coverage | Pinball q=0.45 |
|------|-------------:|-------------:|---------------:|
| DE-LU | 88.0% | 95.0% | 3.45 |
| ES | 79.7% | 95.0% | 3.25 |

Raw coverage before calibration is below 95% for both zones. CQR brings both exactly to target, confirming the calibration is working as intended.

---

## Validation results

On the 2025 holdout year, the final model is approximately **4.5× better than the naive baseline** on both zones, and 16% / 7% better than the pre-expansion LightGBM.

A full-year 2025 breakdown reveals the model's difficulty distribution:

- **Winter months (Jan–Feb):** consistently the easiest. Load is predictable, wind is the dominant variable, and price ranges are moderate.
- **Spring months (Mar–May):** hardest for DE-LU due to solar ramp-up volatility. Negative prices begin appearing as solar penetration exceeds midday demand.
- **Summer months (Jun–Aug):** hard for ES due to deep solar troughs; easier for DE-LU as the daily pattern stabilises.
- **Autumn (Sep–Nov):** transitional; moderate difficulty for both.

The hour × weekday error heatmap shows that solar ramp hours (09:00–13:00 UTC) are consistently harder than other hours for both zones, and that Sundays are harder than any weekday — both consistent with the physics of low industrial demand meeting high renewable generation.

---

## What causes the model to fail

These are documented from backtests, not hypothetical.

**Extreme solar oversupply (negative prices).** On 11 May 2025, DE-LU prices went below −200 EUR/MWh during peak solar hours. The model predicted around 20–40 EUR/MWh for those slots. MAE for that day was 25.69 EUR/MWh versus a holdout average of 7.14. The failure is not a code bug — it is a training distribution problem. The model has seen relatively few negative-price hours; it cannot extrapolate the magnitude of extreme events it has not been trained on. The `negative_price_lag24` flag partially captures the regime but cannot predict the depth.

**Post-crisis market anomalies.** In the days following the April 28 2025 Spain blackout, ES prices stayed near-zero for extended periods as grid operators tested restoration. MAE for ES in that window was 16.42 EUR/MWh. The blackout itself has no electricity price feature representation — it is not a weather signal or a generation signal or a fuel signal. No model trained on price history can predict a grid restoration event.

**New structural regime.** DE-LU completed its nuclear phase-out in April 2023, captured implicitly by the training window starting in May 2023. Any future structural change — large battery deployment, a new interconnector, demand-side response policy — would not be in the training data and would degrade performance.

**Feature staleness at inference time.** `lag_24` for the first eval slots is fetched from ENTSOE at run time. If the ENTSOE API is unavailable, these lags fall back to historical proxies — the mean of the same slot across prior weeks. This is a significant degradation; the most important feature for the first 24 slots becomes an approximation.

**Calibration window mismatch.** CQR is calibrated on Jan–May 2026 (~3,000 observations). If May 11 2026 falls into a regime unlike anything in that calibration window — another extreme solar event, another grid incident — the interval corrections will be too small and actual coverage will fall below 95%.

---

## Code map

| File | Role |
|------|------|
| `config.py` | Single source of truth: paths, zone EIC codes, weather station coordinates and weights, date ranges, scoring parameters. All other modules import from here. Changing `TRAIN_END` or `EVAL_START` here propagates automatically. |
| `ingestion.py` | Fetches raw data. Key functions: `fetch_entsoe_prices()`, `fetch_entsoe_generation()`, `fetch_neighbor_prices()`, `fetch_entsoe_nuclear()`, `fetch_weather_archive()`, `fetch_weather_ensemble()`, `fetch_fuel_prices()`. Global socket timeout set here. |
| `cleaning.py` | UTC normalisation, hourly reindex, gap interpolation. Key functions: `clean_prices()`, `clean_generation()`, `clean_weather()` (detects multi-city format), `clean_fuel_prices()` (handles any fuel columns present). |
| `alignment.py` | Joins cleaned sources per zone on UTC hourly index. `align_zone()` does the join; `align_all()` stacks both zones into the MultiIndex. Drops rows missing critical columns. |
| `features.py` | Adds all 41 features. `add_weather_features()` skips columns already present (avoids duplication if alignment brought them in). `add_lag_features()` computes lags per-zone independently to prevent cross-zone leakage. |
| `validation.py` | Dataset quality gate before training. Tiered gap check: warns on < 50 missing hours, errors on ≥ 200. Hard-fails on missing required columns or NaN in price. |
| `pipeline.py` | Orchestrator. Calls stages 1–4 in order. `--from-clean` skips ingestion; `--from-align` skips ingestion and cleaning. |
| `model.py` | All model logic. Flow: `train_zone()` → `report_zone()` → `feature_importance()` → `calibrate_zone()` → `build_longterm_model()` → `fetch_gap_actuals()` → `_fetch_weather()` → `fetch_entsoe_gen_forecast()` → `build_eval_row()` → main prediction loop. |

Within `model.py`, the `for ts in eval_idx` loop (near line 1005) is the critical inference path. It assembles each feature vector via `build_eval_row()`, routes to LightGBM or the seasonal model by horizon, applies Mondrian CQR corrections, and fills recursive lags from its own prior outputs. The loop is sequential by design — parallelising it would break the lag dependency.

---

## What we would do with more time

**Retrain on May 9 data.** Our training tail is May 7. Running ingestion again on May 9 would give the model two more days of real prices, making `lag_24` for the eval window exact rather than gap-fetched. The most important feature would be real.

**Ensemble LightGBM and GNN.** The GNN makes different errors than LightGBM — it sees the graph topology while LightGBM sees individual features. A weighted average of the two would likely outperform either individually, even though the GNN alone is weaker. Model diversity is the condition for ensembling to help, and the two models are architecturally diverse.

**Zone-specific calibration windows.** Currently both zones use the same Jan–May 2026 calibration window. A sliding window calibrated on the most recent 90 days would better capture the current price regime, particularly for ES where solar capacity growth is changing the midday price profile faster than the calibration window can absorb.
