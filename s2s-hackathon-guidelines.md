[Skip to content](https://friggeco.notion.site/Frigg-Challenge-Content-S2S-x-ETH-Analytics-Club-x-EPFL-UNIL-Entrepreneur-Club-3480ce323d3e802eb3bbf388ddd10b7b#main)

![🕹️ Page icon](<Base64-Image-Removed>)![🕹️ Page icon](https://notion-emojis.s3-us-west-2.amazonaws.com/prod/svg-twitter/1f579-fe0f.svg)

# Frigg Challenge Content \| S2S x ETH Analytics Club x EPFL-UNIL Entrepreneur Club

Contents:

[Electricity Price Forecasting Challenge](https://friggeco.notion.site/Frigg-Challenge-Content-S2S-x-ETH-Analytics-Club-x-EPFL-UNIL-Entrepreneur-Club-3480ce323d3e802eb3bbf388ddd10b7b?pvs=25#3510ce323d3e80068ce2ec6a055c9877)

[The ProblemNo](https://friggeco.notion.site/Frigg-Challenge-Content-S2S-x-ETH-Analytics-Club-x-EPFL-UNIL-Entrepreneur-Club-3480ce323d3e802eb3bbf388ddd10b7b?pvs=25#3510ce323d3e80d0857ac93fae9e2839)

[What You Need to Deliver](https://friggeco.notion.site/Frigg-Challenge-Content-S2S-x-ETH-Analytics-Club-x-EPFL-UNIL-Entrepreneur-Club-3480ce323d3e802eb3bbf388ddd10b7b?pvs=25#3510ce323d3e80e58416db1386793902)

[Why This is Hard (and Interesting)](https://friggeco.notion.site/Frigg-Challenge-Content-S2S-x-ETH-Analytics-Club-x-EPFL-UNIL-Entrepreneur-Club-3480ce323d3e802eb3bbf388ddd10b7b?pvs=25#3510ce323d3e805aa157dec18a83a334)

[The Freedom (and Responsibility)](https://friggeco.notion.site/Frigg-Challenge-Content-S2S-x-ETH-Analytics-Club-x-EPFL-UNIL-Entrepreneur-Club-3480ce323d3e802eb3bbf388ddd10b7b?pvs=25#3510ce323d3e80db95bbed6208c9b377)

[Deliverables](https://friggeco.notion.site/Frigg-Challenge-Content-S2S-x-ETH-Analytics-Club-x-EPFL-UNIL-Entrepreneur-Club-3480ce323d3e802eb3bbf388ddd10b7b?pvs=25#3510ce323d3e80c2b87bccd08a882118)

[1\. Predictions — {team\_name}\_predictions.csv](https://friggeco.notion.site/Frigg-Challenge-Content-S2S-x-ETH-Analytics-Club-x-EPFL-UNIL-Entrepreneur-Club-3480ce323d3e802eb3bbf388ddd10b7b?pvs=25#3510ce323d3e80bd81f7d36609d01f9b)

[2\. Notebook — {team\_name}\_model.ipynb](https://friggeco.notion.site/Frigg-Challenge-Content-S2S-x-ETH-Analytics-Club-x-EPFL-UNIL-Entrepreneur-Club-3480ce323d3e802eb3bbf388ddd10b7b?pvs=25#3510ce323d3e808c864de0634bd49469)

[3\. Training data — {team\_name}\_data.zip](https://friggeco.notion.site/Frigg-Challenge-Content-S2S-x-ETH-Analytics-Club-x-EPFL-UNIL-Entrepreneur-Club-3480ce323d3e802eb3bbf388ddd10b7b?pvs=25#3510ce323d3e808aa9c2d5c47bc7bb23)

[Evaluation](https://friggeco.notion.site/Frigg-Challenge-Content-S2S-x-ETH-Analytics-Club-x-EPFL-UNIL-Entrepreneur-Club-3480ce323d3e802eb3bbf388ddd10b7b?pvs=25#3510ce323d3e80ceb2afd447ce548382)

[Why This Matters](https://friggeco.notion.site/Frigg-Challenge-Content-S2S-x-ETH-Analytics-Club-x-EPFL-UNIL-Entrepreneur-Club-3480ce323d3e802eb3bbf388ddd10b7b?pvs=25#3510ce323d3e80fc9beec405a767dab2)

[Prize](https://friggeco.notion.site/Frigg-Challenge-Content-S2S-x-ETH-Analytics-Club-x-EPFL-UNIL-Entrepreneur-Club-3480ce323d3e802eb3bbf388ddd10b7b?pvs=25#3480ce323d3e80cab362c41a7df875f2)

## Electricity Price Forecasting Challenge

### The ProblemNo

Europe's wholesale electricity markets are among the most complex and volatile in the world. Prices can swing from negative values during sunny, windy weekends to €500+/MWh during energy crises. For energy infrastructure developers, accurate price forecasting is the difference between a profitable project and a financial disaster.

Your challenge: Build a predictive system that forecasts Day Ahead Auction (DAA) electricity prices (EUR/MWh) for two European bidding zones over any requested time horizon:

DE-LU — the German-Luxembourg zone, Europe's largest market, with deep interconnections, high renewable penetration, and frequent negative prices

ES — the Spanish zone (OMIE), a more isolated market with a different generation mix (high solar, significant hydro) and distinct seasonal dynamics

You will train separate models for each zone, but both models must use the same types of features (e.g. if you use a wind forecast feature for Germany, you must also use the equivalent for Spain). The models may differ in architecture, hyperparameters, and learned feature weights — and a core part of your submission is to explain why they differ.

### What You Need to Deliver

A model — or a pair of models — that takes as input:

A time frame — start and end timestamp, at hourly granularity

A bidding zone — DE-LU or ES

Any features you choose to drive the prediction (historical prices, weather, generation mix, fuel prices, etc.)

And produces as output:

Predicted electricity price (EUR/MWh) for each hourly slot in the requested window

Confidence interval — the 2.5th and 97.5th percentile of your predicted price distribution for each hour

Your system must handle any time horizon — from tomorrow to twenty years out. The key design challenge is that the appropriate forecasting logic changes drastically depending on how far ahead you are looking:

Short-term (next few days): You can exploit rich, granular signals — weather forecasts, scheduled generation, fuel spot prices, recent price patterns. High-resolution, feature-driven models are appropriate here.

Long-term (months to years ahead): Granular signals are unavailable or unreliable. Your model should fall back to structural approaches — trend extrapolation, seasonal indexation, long-run marginal cost reasoning. For uncertainty at these horizons, assume historical volatility applies and widen your intervals accordingly.

Your model should explicitly reason about which regime it is operating in and apply the appropriate methodology for each part of the requested window.

### Why This is Hard (and Interesting)

Electricity prices are driven by a complex interplay of factors:

Weather patterns: Wind speed, solar irradiance, temperature

Supply dynamics: Nuclear availability, renewable penetration, fuel prices (gas, coal, carbon)

Demand fluctuations: Industrial activity, seasonal patterns, time-of-day cycles

Market structure: Cross-border flows, transmission constraints, bidding zone configurations

Macro events: Policy changes, geopolitical shocks, economic cycles

The two zones are structurally very different, and this is precisely what makes the challenge interesting:

|  | DE-LU | ES |
| --- | --- | --- |
| Key renewables | Wind (onshore + offshore), solar | Solar (very high irradiance), wind, hydro |
| Fossil peaking | Gas, residual coal | Gas |
| Interconnection | Highly connected (FR, AT, CH, NL, DK, PL) | Relatively isolated (mainly FR) |
| Negative prices | Frequent (high wind + solar weekends) | Less common |
| Demand seasonality | Heating-driven winter peaks | Cooling-driven summer peaks (AC) |

A model that works well for Germany will not automatically work well for Spain — the relative importance of solar irradiance, hydro availability, and the peninsula's weaker cross-border flows will shift the learned feature weights substantially. Your job is to discover and explain those differences.

The methodology is entirely up to you. Will you use physics-informed neural networks? Transformer architectures for temporal patterns? Gaussian processes for uncertainty? Ensemble methods that blend a short-term forecaster with a long-term scenario model? We don't care — surprise us.

### The Freedom (and Responsibility)

You decide:

Which features to use (historical prices, weather forecasts, generation mix, fuel prices, political indicators, social media sentiment, satellite imagery... sometimes less is more)

How to engineer those features (temporal lags, rolling statistics, nonlinear transformations...)

What model architecture to employ for each forecasting regime

How to transition between short-term and long-term logic

How to handle missing data, outliers, and regime changes

How to calibrate your 2.5th/97.5th percentile estimates — and how to fall back gracefully when the horizon is too far out for data-driven uncertainty quantification

One hard constraint: the feature vocabulary must be the same across both zones. You cannot use a feature for one zone that has no equivalent for the other. This ensures that differences in model behavior reflect genuine market structure rather than data availability.

Starting point:

Historical DAA prices and power generation data for DE-LU, ES, and other European bidding zones are available at [https://energy-charts.info/index.html?l=en&c=DE](https://energy-charts.info/index.html?l=en&c=DE)

For the avoidance of doubt: you are free to obtain and use any other type of data to train your model. However, any data that you use MUST be publicly accessible, legal to download, and legal to use for non-commercial purposes such as this hackathon.

### Deliverables

Submit a single zip archive containing the three items below.

#### 1\. Predictions —   {team\_name}\_predictions.csv

A CSV file with 30 rows (one per hourly slot) after the header row and 7 columns covering the evaluation window:

> From: Friday 8 May 2026 at 18:00 CEST (17:00 UTC)
>
> To: Saturday 9 May 2026 at 23:00 CEST (22:00 UTC) — inclusive

The file must have exactly these seven columns, in this order:

| Column | Format | Example |
| --- | --- | --- |
| timestamp | ISO 8601 with UTC offset | 2026-05-08T18:00:00+01:00 |
| DE-LU p025 | float, EUR/MWh | 52.40 |
| DE-LU p50 | float, EUR/MWh | 68.15 |
| DE-LU p975 | float, EUR/MWh | 94.70 |
| ES p025 | float, EUR/MWh | 45.20 |
| ES p50 | float, EUR/MWh | 61.80 |
| ES p975 | float, EUR/MWh | 88.30 |

p50

is your point forecast (median);

p025

and

p975

are your 2.5th and 97.5th percentile bounds.

No missing values accepted. Timestamps must be hourly and monotonically increasing; no gaps, no duplicates.

Example (with random values):

predictions.csv

1.9 KiB

#### 2\. Notebook —   {team\_name}\_model.ipynb

A single Jupyter notebook covering both zones that a reviewer can run end-to-end (given your data zip) to reproduce your predictions. It should be clearly structured along these lines:

Methodology overview: this part is the most important. Here, please provide a concise overview of your work; the methodology you used, the features you decided to pick, the architecture you built, and the results that you got. You can either do this with text, or link a video, a slide deck, an audio file, a song.. whatever you like!

Data loading and preprocessing (for both zones)

Feature selection and EDA — including a side-by-side comparison of the two zones

Feature engineering (same feature types applied to each zone's data)

Model training and validation (separate models per zone, or a single model)

Cross-zone comparison — which features dominate in each zone, and why? Where do the models agree or diverge?

Prediction generation and visualization for a long-term timespan (e.g. starting tomorrow for the next 2 years)

Prediction generation for the evaluation window

The notebook should be self-contained: reading from relative paths within your data zip, with all dependencies installable via a

%pip install ...

cell at the top.

#### 3\. Training data —   {team\_name}\_data.zip

A zip archive of all data files you used to train and validate your models — for both zones. Include a short

README.txt

inside the zip describing each file (source, variables, date range). There is no size limit, but only include what you actually used.

### Evaluation

Primary metric: Pinball loss averaged across both zones over the live evaluation window, applied to your p50 predictions versus the actual values. Overestimations are penalized more strongly than underestimations, reflecting the conservative approach required in financial infrastructure modeling. Here is the loss function that will be applied to your predictions:

defscoring\_loss(y\_true, y\_pred, q=0.45):"""
Asymmetric pinball loss at quantile q.

Penalty for underestimation (actual > pred) : q × \|error\|
Penalty for overestimation (actual < pred) : (1 − q) × \|error\|

q < 0.5 makes overestimation costlier than underestimation.
At q = 0.45 the overestimation penalty is (0.55 / 0.45) ≈ 1.22× the underestimation penalty.
"""
r = np.asarray(y\_true,float)- np.asarray(y\_pred,float)returnfloat(np.mean(np.where(r >=0, q \* r,(q -1)\* r)))

​

Secondary metric: Qualitative score assessing the reasoning and creativity of your methodology — how you selected and processed features, how you justified your short-term vs. long-term split, how you approached uncertainty quantification, and how clearly and insightfully you explained the structural differences between the DE-LU and ES models.

### Why This Matters

Your work could directly influence investment decisions. Energy infrastructure projects require 20-year price forecasts to secure financing. A 10% improvement in forecast accuracy can mean millions of euros in NPV and determine whether critical renewable energy projects get built.

This is machine learning where accuracy has immediate, tangible impact.

Ready to compete? Show us what you can build.

## Prize

Spend a day with the Frigg team in Zug

Interview for a software engineering internship with us

Cake!