# Notebook Handoff — Final Push Before Submission

Hey — I went through the full notebook (44 cells) and re-read the Frigg Intelligence product page. The model and methodology are genuinely strong, the bottleneck now is that a Frigg juror reading this won't immediately see how it plugs into *their* product. This doc is the full edit list, ordered by impact. Skim P1–P3 first; P4 onwards is polish.

The big picture: right now we're a price-forecasting notebook that mentions Frigg in the intro. We need to be a price-forecasting notebook where every section quietly answers "and here's how a Frigg user would actually use this." That single shift is the difference between "cool model" and "we want to talk to these guys after the demo."

---

## Quick refresher on what Frigg Intelligence actually is

So we don't lose the thread. Frigg Intelligence is an automated overlay that scores renewable energy projects on **risk-adjusted excess return** vs the risk-free rate. The headline number is the **Frigg Score (0–10)**, benchmarked against 20,000+ projects, adjusted for:

- return volatility
- project stage
- offtake risk

It also produces projection charts for production data and financial metrics, text-based project assessments, and suggested debt structuring. Two user types: **developers** (optimize Capex/Opex/timeline/revenue, get debt-structuring suggestions) and **asset managers/investors** (compare deals, evaluate risk-return).

Every recommendation below is grounded in: does this make our model more useful for those workflows, and is it obvious to the juror that it does.

---

## P1 — The "so what" gap (this is the whole game)

The notebook never explicitly maps our outputs to Frigg's primitives. We compute exactly the things they need (a price level, a volatility, a tail, a regime classifier) but we leave it to the juror to connect the dots. Don't make them work for it.

### P1.1 — Add a new section between §1 and §2: "How This Feeds Frigg Intelligence"

This is the single highest-leverage edit in the notebook. ~1 page of markdown, no new code. Structure it as a table plus a worked example.

The table should pair every Frigg Intelligence input with the model output that produces it:

| Frigg Intelligence input | What we provide | Section |
|---|---|---|
| Unlevered IRR — revenue numerator | LT median price path 2026–2045, monthly, per zone | §10 |
| Frigg Score — return volatility adjustment | 200-draw MC fan width + Mondrian CQR p025/p975 bands | §7, §10 |
| Frigg Score — offtake risk adjustment | Negative-price frequency + P5 tail by regime bucket | §6, §8 |
| Frigg Score — project stage adjustment | Capacity roadmap to 2045 (when does the merit order shift?) | §10 |
| Merchant tail / spot revenue modeling | Quantile LightGBM 0–7d with calibrated 95% bands | §6 |
| Debt sizing — DSCR stress test | P5 long-term price scenario from MC fan | §10 |
| Cross-project benchmarking (the 20k pipeline) | Two-zone methodology that generalizes to any EU bidding zone | §8 |

Then a **worked example** — this is the part that lands. Pick a hypothetical 100 MW solar farm in Andalusia, COD 2028. In ~10 lines of code or a markdown calculation:

1. Take our ES P50 monthly path 2028–2048
2. Compute unlevered IRR at P50, P25, P75
3. Show how the **IRR distribution width** is what feeds the Frigg Score's volatility adjustment
4. End with one sentence: "A Frigg user drops project Capex/Opex into this template and gets a price-driven IRR fan in seconds, not days."

That's the takeaway they walk away with. That's the "I get it now."

### P1.2 — Reframe every section title around the outcome, not the method

Method-first titles like "Mondrian CQR" and "Merit-order Monte Carlo" make the juror translate. Do the translation for them. Keep the method name (we want credit for the technical depth) but add an outcome subtitle:

- §6 → "Short-term Model — *Day-ahead trading and merchant revenue (0–7d)*"
- §7 → "Uncertainty Calibration — *95% bands that actually hold up out-of-sample*"
- §8 → "Cross-zone Comparison — *One methodology, any European market*"
- §9 → "Horizon Routing — *One API call, any time horizon*"
- §10 → "Long-term Model — *2045 IRR and debt-sizing curves, no historical price leakage*"
- §11 → "Evaluation Window — *Submission*"

Five-minute edit, big clarity gain. The "no historical price leakage" line on §10 is especially worth keeping because it's a real differentiator — most LT forecasters cheat by autoregressing on price; we don't.

### P1.3 — Add an "Implication for Frigg Score" row to the §8 cross-zone table

The DE-LU vs ES table is already great. Add one row at the bottom:

> **Implication for Frigg Score:** ES projects show materially lower negative-price frequency and tighter return volatility → smaller offtake-risk and volatility penalties → typically higher scores than equivalent DE-LU projects, all else equal.

That single sentence proves we understand their product, not just price forecasting. It's the kind of thing a Frigg PM would screenshot.

---

## P2 — Executive summary up top

You said yes to this — here's the proposed text. Drop it as a markdown block **above** the current SITUATION/TASK/ACTION/RESULT block (don't replace it, the STAR structure has its own value, just don't make it the first thing the juror reads).

```markdown
> **TL;DR — alpine-arbitrage**
>
> We built a price forecaster that goes from 1 hour to 20 years in a single API call,
> for DE-LU and ES. Short-term MAE is 6.01 (DE) and 5.15 (ES) EUR/MWh — about
> 5× better than the same-hour-last-week baseline. After Mondrian CQR calibration
> our 95% intervals actually contain the true price 95% of the time (most quantile
> models miss this badly). Long-term MAE is 33 EUR/MWh in non-crisis regimes
> with zero historical price leakage — the LT model is structural, built from
> capacity stacks and forward fuel curves.
>
> **Why Frigg should care:** the LT median path drops straight into unlevered
> IRR projections, the MC fan width feeds the Frigg Score's volatility adjustment,
> and the negative-price tail feeds the offtake-risk adjustment. The ST model
> is what a Frigg user needs for merchant-tail revenue and dispatch optimization.
> Same model, two products: developer pricing and investor risk-return.
```

Also: **Cell 1's RESULT paragraph appears to cut off mid-sentence at "Main limita..."** in what GitHub renders. Open the notebook, scroll to the end of §1, make sure the limitations sentence actually completes. If it doesn't, finish it with something honest like *"Main limitations: long-term model assumes no major regulatory shocks beyond what's encoded in the GPR index, and weather forecasts past 7 days are climatological rather than NWP-driven."*

---

## P3 — Add business context around technical artifacts

Several cells produce great visuals with zero markdown context. A juror who isn't a quant will scroll past them. Each of these is a 2–4 sentence markdown cell.

### P3.1 — Cell 38 (LT hero chart) currently has no surrounding markdown

Right now we just `display(Image(...))`. Before the cell, add:

> The chart below is the headline long-term forecast: monthly P50 with 200-draw MC fans for DE-LU and ES through 2045. Read it like a Frigg user: a project commissioning in 2030 sees a P50 around €X, with a P25–P75 band of €Y–€Z. That band width is what feeds the Frigg Score's return volatility adjustment. The fan visibly narrows in the late 2030s as the merit order saturates with renewables — an effect that's structurally invisible to any model trained only on historical prices.

(Fill in actual numbers from the chart.)

### P3.2 — Capacity roadmap (Cell 39) needs a "why this matters" intro

Add before the cell:

> The capacity roadmap is the **structural prior** that lets us forecast 20 years out without leaking historical prices into the model. Knowing how much gas CCGT vs solar vs nuclear is on the system in 2035 mechanically determines which technology sets the marginal price most hours of the year — and that's the price. This is also what maps directly to Frigg's "project stage" axis: a 2028 COD project lives in a different merit order than a 2040 COD project.

### P3.3 — Submission summary (Cell 43) — add a "what a Frigg user does with this" closer

After the print output, add a final markdown cell:

> **What a Frigg user would do with this submission:** the 24 hourly P50/P95 values are the granular merchant-revenue inputs for an 11 May 2026 dispatch decision. For project finance, the same model rolled forward 20 years (§10) drops into the IRR engine. Same architecture, two products. We think this is the right shape for Frigg Intelligence to extend — start with two zones, generalize to all 27 EU bidding zones with the same code path.

Closer is everything. Don't end on a `print()` statement.

### P3.4 — §9 horizon routing — add the Frigg lens

The ASCII routing diagram is fine for engineers. Add one sentence after it:

> **From Frigg's perspective:** short-term routes feed spot/PPA pricing modules and merchant-tail revenue; long-term routes feed the IRR engine and debt-structuring suggestions. The user makes one call; we route to the right model.

---

## P4 — Trim the noise

The notebook has some debug artifacts and duplicate views that dilute the signal. Be ruthless — every cell competes for the juror's attention.

- **Cell 12 (feature-availability heatmap)** — this is a debug artifact. Either drop it entirely or replace with a single sentence: "20-feature shared schema across zones; 5 zone-specific features (e.g., NL/CH neighbor prices for DE-LU, hydro precipitation for ES) are NaN-skipped by LightGBM at split time." The heatmap doesn't tell a story.
- **Cell 29 (printed rank table at the end of cross-zone SHAP)** — we already have the bar chart above it. Two views of the same data is one too many. Drop the print, keep the chart.
- **Cell 0 (`%pip install`)** — fine to keep but move to an appendix or wrap in a `try/except ImportError` so judges don't have to wait through it on a re-run.
- **The `try: import shap except ImportError: subprocess install` block in Cell 22** — same deal. Move installs to Cell 0.

If a cell doesn't either (a) produce a number that goes in the TL;DR, (b) produce a chart that tells a story, or (c) explain what's happening — cut it.

---

## P5 — Add a single architecture diagram at the top

This is the highest-impact visual we don't currently have. One boxes-and-arrows schematic, ideally hand-drawn or done in something like Excalidraw, placed right under the TL;DR in §1:

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
                                  ▼
                  ┌───────────────┴────────────────┐
                  ▼                                ▼
        [Spot / PPA / merchant       [Frigg Score volatility,
         revenue — developer use]     offtake risk, IRR fan —
                                      investor use]
```

A non-technical juror reads this in 10 seconds and gets the entire architecture. They will not get there from reading code.

---

## P6 — Reproducibility signals

Judges love evidence it actually runs end-to-end. Add near the top of §3:

- A one-liner: *"End-to-end run: `make all` — ~12 min on M-series Mac, no GPU required."*
- A `requirements.txt` lock or a note that `pip install -r requirements.txt` reproduces our environment.
- A `.env.example` file in the repo so the ENTSOE token setup is obvious.

Also: the `plt.savefig` calls all dump PNGs into the repo root. Move them to a `figs/` subfolder. It's a 30-second find-and-replace and makes the repo look professional.

---

## P7 — Things to actively defend, not hide

The notebook has three things that are genuinely impressive and should be louder:

1. **No historical price leakage in the LT model.** Most long-term forecasters autoregress on price and look great in backtest because they're cheating. We don't. Say this explicitly in §10's intro and in the TL;DR. This is a real moat.
2. **Mondrian CQR achieving exactly 95% coverage out of sample.** Coverage that holds up on a holdout is rare. Most teams will report nominal coverage and miss by 10+ points. Frame this as a reliability story: "an interval that says 95% and means it is what makes risk-adjusted scoring possible."
3. **Two-zone methodology that generalizes.** DE-LU and ES are the two structurally most-different markets in Europe (wind-dominated/8-neighbor vs solar-dominated/1-neighbor). If our architecture works for both, it works for all 27. This is the scaling story for Frigg's 20k-project pipeline.

Give each of these its own callout box (`> **Why this matters:** ...`) in the relevant section.

---

## P8 — Final pre-submission checklist

Before we hit submit:

- [ ] Cell 1 RESULT paragraph completes (not truncated)
- [ ] All 44 cells run top-to-bottom on a fresh kernel without errors
- [ ] No `pip install` cells silently fail in CI
- [ ] All saved figures (`nb_*.png`) are committed to the repo
- [ ] `alpine-arbitrage_predictions.csv` exists at the path Cell 41 reads from
- [ ] README in the repo points to the notebook as the entry point
- [ ] Team name spelled consistently ("alpine-arbitrage" — confirm this is what's on the leaderboard)
- [ ] One final read-through specifically asking "would a Frigg PM understand this?"

---

## Suggested edit order for you

If you only have a few hours, do these in order and stop when you run out of time. The first three are 80% of the value.

1. **Write the "How This Feeds Frigg Intelligence" section** (P1.1) — 45 min, biggest single win
2. **Add the TL;DR block at the top** (P2) — 15 min
3. **Add outcome subtitles to every section** (P1.2) — 10 min
4. **Add markdown context around Cell 38, Cell 39, and Cell 43** (P3) — 30 min
5. **Add the Frigg Score implication row to §8 table** (P1.3) — 5 min
6. **Drop Cell 12 and the Cell 29 print table** (P4) — 5 min
7. **Add the architecture diagram** (P5) — 30 min if you have a tool ready
8. **Polish: figs/ subfolder, README, reproducibility note** (P6) — 20 min

Everything else is gravy. Ping me when you've got a draft and I'll do a final pass.
