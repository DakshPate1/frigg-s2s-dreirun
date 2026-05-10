# How to Read the Quantile Bloom

**Alpine Arbitrage · DE-LU & ES · 11 May 2026**

---

## What is it?

The Quantile Bloom is a circular visualization of day-ahead electricity price forecasts for two European bidding zones — **Germany-Luxembourg (DE-LU)** and **Spain (ES)** — produced by the Alpine Arbitrage LightGBM quantile regression model. Each forecast covers all 24 hours of the day. Instead of two line charts, the predictions are folded into a single mandala: one petal per hour per zone, three concentric tiers per petal.

---

## The Basics: Angle, Length, Tier

### Angle → Hour of day

The bloom reads **clockwise from the top**, like a clock.

```
         00:00
    23:00     01:00
  ...               ...
    18:00     06:00
         12:00
```

- Top (12 o'clock) = **00:00 UTC**
- Right (3 o'clock) = **06:00 UTC**
- Bottom (6 o'clock) = **12:00 UTC**
- Left (9 o'clock) = **18:00 UTC**

### Length → Price in €/MWh

A longer petal = a higher forecasted price. The mapping is linear and globally normalised across all hours and both zones, so you can compare lengths directly.

- **Minimum scale** (shortest possible petal): ES q=0.025 at 15:00 UTC ≈ **−€1.5/MWh**
- **Maximum scale** (longest possible petal): DE-LU q=0.975 at 18:00 UTC ≈ **€185.7/MWh**

### Tier → Quantile level (uncertainty band)

Each petal is made of three nested layers:

| Tier | Quantile | Colour | Meaning |
|------|----------|--------|---------|
| **Inner** | q = 0.025 | Green / Cyan | Lower bound — 97.5% of outcomes expected above this |
| **Middle** | q = 0.45 ✶ | Lighter green / White | Point forecast — asymmetric pinball loss (1.22×) biases this slightly below the median |
| **Outer** | q = 0.975 | Amber / Gold | Upper bound — 97.5% of outcomes expected below this |

The **gap between inner and outer** = the model's uncertainty at that hour. A wide gap means the model is unsure; a narrow gap means it is confident.

---

## Two Zones, One Flower

Each hour has **two petals** — one per zone — fanned slightly apart:

| Zone | Fan direction | Colour family |
|------|--------------|---------------|
| **DE-LU** | Fans left (counter-clockwise) from the hour mark | Green / Violet |
| **ES** | Fans right (clockwise) from the hour mark | Amber / Coral |

At every hour position you will see a pair of petals opening like a small V. The left petal is Germany, the right is Spain.

This gives **48 petals total**: 24 hours × 2 zones.

---

## The Price Curves (the rings)

Three closed curves trace the tips of each tier across all 24 hours:

- **Inner ring** (q=0.025) — the lower-bound contour. A bumpy circle whose radius at each hour equals the lower-bound forecast.
- **Middle ring** (q=0.45) — the thickest line; the point-forecast contour.
- **Outer ring** (q=0.975) — the upper-bound contour, widest and most diffuse.

Because prices vary by hour, these rings are **not circular** — they bulge where prices are high and contract where prices are low. The shape encodes the full 24-hour price profile at a glance.

---

## Reading Uncertainty

The **band width** (distance between inner and outer ring at any hour) represents model uncertainty:

- **Narrow band** → model is confident (consistent historical patterns at that hour)
- **Wide band** → model is uncertain (volatile hour, e.g. morning ramp, evening peak)

The outer petal is drawn translucent for this reason — the more diffuse it looks, the less certain the upper bound.

> **Sunday effect**: The model uses Mondrian Conformalized Quantile Regression (CQR) stratified by day-of-week. Sunday (bucket 1) produces systematically wider intervals than weekdays.

---

## Hover Tooltip

Hover anywhere on the bloom to read exact values. The tooltip shows one quantile at a time, updating as you move radially:

- Move **inward** → snap to q=0.025
- Move to the **middle tier** → snap to q=0.45
- Move **outward** → snap to q=0.975

The tooltip shows:
```
DE-LU · 18:00 UTC
q=0.975   €185.7 / MWh
upper bound
```

---

## What to Look For

### Morning ramp (04:00–08:00 UTC)
Petals grow noticeably longer — especially DE-LU — as demand rises. The band also widens, reflecting uncertainty around the speed of the ramp.

### Solar trough (10:00–15:00 UTC)
ES petals shrink dramatically in the middle hours as Spanish solar generation suppresses prices. The inner tier (q=0.025) can go **negative** (ES 15:00 UTC: −€1.5/MWh). The bloom narrows on the ES side during this window.

### Evening peak (17:00–19:00 UTC)
The longest petals in the entire bloom. DE-LU hits its maximum at **18:00 UTC** (q=0.975 ≈ €185.7/MWh). The outer ring bulges sharply. This is when the model is also most uncertain — the widest bands of the day.

### DE-LU vs ES spread
Compare the left (DE-LU, green) and right (ES, amber) petals at the same hour. A large difference in petal length = a potential arbitrage opportunity between the two zones.

### Seed variation
The small noise applied by the seed parameter adds organic texture to petal width and length. Different seeds produce visually distinct flowers from the same underlying data — the **price information is identical** across seeds; only the decorative perturbation changes.

---

## Model Details

| Property | Value |
|----------|-------|
| Model | LightGBM quantile regression |
| Calibration | Conformalized Quantile Regression (CQR) |
| Stratification | Mondrian by day-of-week bucket |
| Target date | Sunday 11 May 2026 |
| Zones | DE-LU (Germany–Luxembourg), ES (Spain) |
| Horizon | 24-hour day-ahead, UTC 00:00–23:00 |
| Quantiles | 0.025 · 0.45 · 0.975 |
| Global range | −€1.5/MWh (ES 15:00, q=0.025) to €185.7/MWh (DE-LU 18:00, q=0.975) |

---

## Quick Reference

```
Clockwise angle  →  hour of day (00:00 at top)
Petal length     →  forecasted price in €/MWh
Left petal       →  DE-LU (Germany–Luxembourg)
Right petal      →  ES (Spain)
Inner tier       →  q=0.025  lower bound
Middle tier      →  q=0.45 ✶ point forecast
Outer tier       →  q=0.975  upper bound
Band width       →  model uncertainty at that hour
```
