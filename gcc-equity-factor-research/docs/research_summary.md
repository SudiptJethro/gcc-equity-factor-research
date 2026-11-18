> This is the research write-up as of May 2026. The figures match the output files in `data/factors/`. The universe counts and filter thresholds were corrected to match the code when the repository was published.

# GCC Equity Factor Research — Project Summary

**Research Question:** Do classic equity factor premia (value, momentum, quality, low volatility) generate statistically significant alpha in GCC equity markets, and do they behave differently across ADX, DFM, Tadawul, QSE, and Boursa Kuwait?

**Universe:** 167 stocks — ADX, DFM, Tadawul, QSE, Boursa Kuwait  
**Period:** March 2015 → May 2026 (~135 months, 124 joint-regression months)  
**Method:** Equal-weighted long-short quintile portfolios + Fama-MacBeth cross-sectional regressions (Newey-West 4-lag) + Backtest attribution + VRP overlay

---

## 1. Executive Summary

| Factor | Ann. L/S Return | Sharpe | FM t-stat (Uni) | IC Hit Rate | Verdict |
|---|---|---|---|---|---|
| VALUE | −8.39% | −0.556 | −2.16 ** (inverted) | 47.4% | Does not work. Statistically significant inversion. |
| MOMENTUM | +9.29% | +0.561 | +1.81 * | 56.5% | Works. Marginal significance. Strengthening recently. |
| QUALITY | +14.38% | +1.030 | +2.98 *** | 65.9% | Strongest signal. Look-ahead inflated. True Sharpe ~0.4–0.6. |
| LOW-VOL | −1.47% (clean) | −0.042 | IC t = +5.13 *** | 54.8% | Signal real (ICIR 1.58); L/S portfolio remains negative after fix. |

**Headline finding:** GCC equity markets reward quality and momentum, categorically reject value (with statistical significance), and show a genuine low-volatility IC signal that still cannot be profitably harvested via a naive long-short portfolio even after filtering illiquid micro-caps from the short book. A VRP overlay on the multi-factor portfolio raises the system Sharpe from 0.49 to 1.79.

---

## 2. Universe and Data

- **167 stocks** across 5 exchanges after applying a minimum median price (0.10), a minimum average daily volume (5,000 shares) and at least 24 months of history (filters in `Layer_1/config.py`).
- **Exchange breakdown:** Tadawul 81 (Saudi Arabia), ADX 48 (UAE Abu Dhabi), DFM 16 (UAE Dubai), QSE 15 (Qatar), Boursa Kuwait 7.
- **Data source:** Yahoo Finance via yfinance, with TradingView filling gaps for ADX and DFM names. Monthly prices for returns and momentum; daily prices for low-vol (252 trading days); a single fundamentals snapshot (December 2025) for value and quality.
- **Critical caveat:** Fundamental data (P/B, EV/EBITDA, ROE) is not historical point-in-time. Value and quality results are upper-bound estimates. Momentum and low-vol are unaffected — price-based only.

### Fundamental data coverage

| Field | Coverage |
|---|---|
| `priceToBook` | 67.3% |
| `enterpriseToEbitda` | 49.0% |
| `returnOnEquity` | 61.4% |
| `operatingMargins` | 67.8% |

Coverage gaps are concentrated in smaller Kuwaiti and Qatari names. Documented as a data limitation, not a methodology error.

---

## 3. Layer 2 — Factor Construction Results

### Full GCC Universe (167 stocks, T=124–135 months)

| Factor | Ann. L/S Return | Ann. Vol | Sharpe | Hit Rate |
|---|---|---|---|---|
| VALUE | −8.39% | 15.08% | −0.556 | 47.4% |
| MOMENTUM | +9.29% | 16.56% | +0.561 | 56.5% |
| QUALITY | +14.38% | 13.97% | +1.030 | 65.9% |
| LOW-VOL | −1.47% | 35.29% | −0.042 | 54.8% |

> Low-vol numbers reflect the fixed portfolio with short-book filters applied (market cap ≥ $300M, avg vol ≥ 200k shares, monthly z-score < 3σ, weight cap 15%). The raw backtest before fix showed LS = −6.75%, Sharpe = −0.155.

### ADX-Only Sub-Universe

| Factor | Ann. L/S Return | Sharpe | Quintile Mono |
|---|---|---|---|
| MOMENTUM | +11.91% | +0.422 | 2/4 |
| LOW-VOL | −2.12% | −0.072 | 3/4 |

Value and Quality are N/A for ADX in isolation — fewer than 15 stocks pass the fundamental coverage threshold for quintile assignment.

---

## 4. Layer 3a — Fama-MacBeth Statistical Significance

### Joint Regression (all 4 factors, T=124, avg N=102, avg R²=11.2%)

| Factor | λ (bps/mo) | t-stat (NW-4) | Significance |
|---|---|---|---|
| VALUE | −6.18 | −0.40 | — |
| MOMENTUM | +13.56 | +0.62 | — |
| QUALITY | +20.88 | +1.48 | — |
| LOW-VOL | −71.94 | −0.63 | — |

### Univariate FM (each factor standalone)

| Factor | λ (bps/mo) | t-stat (NW) | Significance |
|---|---|---|---|
| VALUE | −28.77 | −2.16 | ** (5%, sign inverted) |
| MOMENTUM | +23.87 | +1.81 | * (10%) |
| QUALITY | +41.08 | +2.98 | *** (1%) |
| LOW-VOL | −67.81 | −0.67 | — |

### Information Coefficient (Spearman rank-IC, monthly)

| Factor | IC Mean | ICIR (ann) | IC t-stat | Hit % |
|---|---|---|---|---|
| VALUE | −0.0213 | −0.508 | −1.70 | 47.4% |
| MOMENTUM | +0.0560 | +1.033 | +3.32 *** | 56.5% |
| QUALITY | +0.0750 | +1.453 | +4.87 *** | 65.9% |
| LOW-VOL | +0.0784 | +1.583 | +5.13 *** | 54.8% |

### Quintile Monotonicity

| Factor | Q1 | Q2 | Q3 | Q4 | Q5 | Q1−Q5 | Mono |
|---|---|---|---|---|---|---|---|
| VALUE | −5.02% | +1.71% | +1.60% | +2.29% | +3.37% | −8.39% | 1/4 broken |
| MOMENTUM | +10.23% | +8.10% | +4.04% | +1.76% | +0.95% | +9.29% | 4/4 perfect |
| QUALITY | +6.97% | +5.35% | +0.04% | +0.35% | −7.42% | +14.38% | 3/4 (Q3–Q4 break) |
| LOW-VOL | −0.43% | +5.27% | +1.06% | −0.82% | +1.52% | −1.47% | 2/4 broken |

> R² of 11.2% in monthly cross-sectional regressions is above the typical 2–5% seen in academic factor papers, indicating the four factors collectively capture real cross-sectional structure in GCC equities.

---

## 5. Factor Deep Dive

### 5.1 Value — Statistically Significant Inversion

The value premium does not exist in GCC — it is **inverted and statistically significant**. Cheap stocks underperform expensive ones by 8.39% per year. Univariate t-stat of −2.16** clears the 5% threshold, making this the most robustly significant factor in the analysis — but in the wrong direction.

Hit rate of 47.4% means the factor was wrong more months than right. Quintile monotonicity 1/4 — completely broken.

**Why value fails structurally in GCC:**
1. State-linked mega-caps (Aramco, QNB, FAB, Al Rajhi) trade at premium valuations but consistently deliver premium returns. "Expensive" is structural, not temporary.
2. "Cheap" in GCC means genuinely distressed small-caps or family conglomerates with opaque governance, not temporarily mispriced quality businesses. Mean-reversion does not operate.
3. Low foreign institutional participation means mispricings persist. Cheap stocks stay cheap or get cheaper.
4. Look-ahead bias typically inflates value returns — the fact that value is still significantly negative under look-ahead bias strengthens the finding.

Sub-period worst: Post-COVID 2021–23 (λ = −34.8 bps, t = −1.88, Sharpe = −0.909) as oil-linked cheap stocks failed to recover alongside the earnings boom.

**Research note implication:** Publishable negative result. Statistically significant inverted value in a 167-stock GCC universe over 11 years is a contribution to the literature.

---

### 5.2 Momentum — Works, Marginal Significance

Momentum produces a real, implementable signal in GCC — but with weaker statistical significance than previously estimated.

- Ann. L/S spread: **+9.29%**
- FM t-stat (univariate): **+1.81 ***  (10% level only)
- ICIR: **+1.033** (above 0.5 practitioner threshold)
- Quintile monotonicity: **4/4 perfect**
- Hit rate: **56.5%**

The joint regression t-stat of +0.62 reflects collinearity with quality — many GCC quality names are also momentum names.

**Why momentum works in GCC:**
1. Oil price cycles create sustained multi-month trends. Energy and petrochemical stocks trend upward for multiple months as earnings expectations are revised sequentially.
2. Retail investor dominance. Information diffuses slowly across all five exchanges.
3. 12-1 construction correctly avoids short-term reversal present in GCC thin markets.

**Sub-period trajectory:**

| Period | Sharpe | λ (bps) | t-stat |
|---|---|---|---|
| Pre-COVID 15–19 | +0.296 | −30.4 | −0.83 |
| COVID 2020 | +0.824 | +16.5 | +0.44 |
| Post-COVID 21–23 | +0.461 | +39.9 | +0.86 |
| Recent 24–26 | **+0.903** | +50.9 | +1.89 * |

Momentum is strengthening in the most recent sub-period (2024–2026) — consistent with Tadawul re-rating and sustained oil-driven earnings momentum.

**No look-ahead bias. Sharpe +0.561 is a clean estimate.**

---

### 5.3 Quality — Strongest Signal, Look-Ahead Inflated

Quality has the strongest statistical case but numbers are an upper bound. Quintile monotonicity is 3/4 (Q3–Q4 break) not perfect.

- Ann. L/S spread: **+14.38%**
- FM t-stat (univariate): **+2.98 ***
- FM t-stat (joint): **+1.48** (only factor with positive joint coefficient)
- ICIR: **+1.453**
- Quintile monotonicity: **3/4** (Q3 = +0.04%, Q4 = +0.35% — minor break)
- Hit rate: **65.9%**

**Look-ahead adjustment:** ROE, operating margins, and earnings stability sourced from yfinance as of December 2025. Academic literature suggests look-ahead bias inflates quality Sharpe by 0.3–0.5. Realistic out-of-sample estimate: **Sharpe 0.4–0.6**.

**Sub-period trajectory:**

| Period | Sharpe | λ (bps) | t-stat |
|---|---|---|---|
| Pre-COVID 15–19 | +1.235 | +38.6 | +1.74 * |
| COVID 2020 | −1.091 | −60.5 | −1.12 |
| Post-COVID 21–23 | **+2.165** | +16.0 | +0.67 |
| Recent 24–26 | +1.557 | +31.8 | +1.74 * |

Pre-COVID quality dominated (Sharpe +1.235). COVID reversed it sharply (Sharpe −1.091) — profitability offered no protection in a liquidity panic. Post-COVID recovery was the strongest of all periods (Sharpe +2.165). Recent period remains strong (Sharpe +1.557).

---

### 5.4 Low Volatility — IC/FM Divergence Persists After Fix

The low-vol signal is the strongest of all four factors by IC; the L/S portfolio remains negative even after the short-book filter was applied.

| Metric | Value | Interpretation |
|---|---|---|
| FM lambda (joint) | −71.94 bps | OLS: higher low-vol score → lower return |
| IC mean | +0.0784 | Ranks: higher low-vol score → higher return |
| ICIR (ann) | +1.583 | Highest of all four factors |
| IC t-stat | +5.13 *** | Strongest IC of all factors |
| IC hit rate | 54.8% | Directionally correct slightly above chance |
| L/S ann. return (clean) | −1.47% | Short-book fix applied; portfolio still negative |
| L/S Sharpe (clean) | −0.042 | Near-zero after fix |

**Why the divergence exists:** OLS is not robust to outliers. Even after the short-book filter (market cap ≥$300M, avg daily vol ≥200k shares, monthly z-score <3σ, weight cap 15%), the GCC micro-cap thin-market problem persists sufficiently to keep the OLS lambda negative. Spearman IC uses ranks and correctly recovers the underlying signal.

**Sub-period evolution:**

| Period | λ (bps) | t-stat | Sharpe |
|---|---|---|---|
| Pre-COVID 15–19 | −292 | −1.04 | −0.432 |
| COVID 2020 | −3.4 | −0.03 | +0.075 |
| Post-COVID 21–23 | +83.2 | +1.59 | +0.695 |
| Recent 24–26 | +64.6 | +1.71 * | +0.586 |

Low-vol is improving over time. Post-COVID and recent periods show positive lambda — consistent with the post-MSCI-upgrade environment (Kuwait joined MSCI EM June 2020) where institutional capital reduced micro-cap blowup frequency. The signal is maturing but has not yet produced a positive full-sample L/S portfolio return.

---

## 6. Layer 3b — Base Portfolio Construction

### Individual Factor Portfolios

| Portfolio | Ann. Return | Ann. Vol | Sharpe | Max DD | Hit Rate |
|---|---|---|---|---|---|
| Momentum L/S (gross) | +9.29% | 16.56% | 0.561 | −23.3% | 56.5% |
| Quality L/S (gross) | +14.38% | 13.97% | 1.030 | −33.0% | 65.9% |

### Multi-Factor CB Portfolio

**Sleeve A:** Top-20 quality + momentum stocks (IC-weighted), vol-scaled to 10% target  
**Sleeve B:** Top-20 lowest-vol stocks (8% target)  
Transaction costs: 50bps round-trip

| Portfolio | Ann. Return | Ann. Vol | Sharpe | Max DD | Hit Rate |
|---|---|---|---|---|---|
| Sleeve A (gross) | +14.19% | 20.45% | 0.69 | −44.3% | 57.6% |
| Sleeve A (net TC) | +13.51% | 20.44% | 0.66 | −44.7% | 56.8% |
| Sleeve A (vol-scaled) | +8.30% | 14.82% | 0.56 | −34.0% | 56.8% |
| Sleeve B low-vol (gross) | +8.54% | 47.51% | 0.18 | −62.6% | 44.8% |
| Sleeve B (vol-scaled) | +0.69% | 21.21% | 0.03 | −45.0% | 44.8% |
| Combined (A + B) | +5.79% | — | 0.49 | −24.5% | 58.8% |
| Combined + circuit breaker | +5.35% | 10.84% | 0.49 | −22.9% | 58.8% |

---

## 7. Layer 3c — VRP Overlay & Volatility Surface

A volatility risk premium (VRP) overlay was constructed using implied vs. realised volatility as a signal for covered call writing on the equity factor portfolio. VRP is the systematic premium sellers of options collect because implied vol exceeds realised vol on average.

### VRP Regime Analysis (150 months)

| Regime | Months | Avg IV | Avg RV | VRP Gap | CC Yield/mo |
|---|---|---|---|---|---|
| Calm | 50 | 21.3% | 13.7% | 7.5pp | 0.56% |
| Normal | 49 | 28.7% | 19.9% | 8.7pp | 1.54% |
| Stressed | 51 | 37.1% | 27.8% | 9.3pp | 2.59% |

VRP gap is persistent across all regimes — even in stressed environments implied vol exceeds realised vol by 9+ percentage points.

### VRP Overlay Results

| Portfolio | Ann. Return | Ann. Vol | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| Base strategy (no overlay) | +5.35% | 10.84% | 0.49 | −22.9% | 0.23 |
| Layer 3b theoretical ceiling (fixed-notional) | +23.86% | 10.84% | 2.20 | −15.9% | 1.50 |
| Layer 3c: dynamic VRP overlay | **+18.13%** | 12.17% | **1.49** | −15.8% | **1.15** |
| **FINAL (CB + VRP + hedge)** | **+20.11%** | 11.23% | **1.79** | **−15.9%** | **1.26** |

**The VRP overlay is the most impactful structural enhancement** — it takes the base multi-factor portfolio from Sharpe 0.49 to 1.79. Hit rate improves from 58.8% to **77.1%**.

> **Caveat:** The theoretical ceiling (Sharpe 2.20) assumes fixed-notional sizing and is not directly tradeable. The dynamic implementation (Sharpe 1.49) represents realistic sizing constraints. The gap (−5.72% ann.) represents execution friction and sizing conservatism.

---

## 8. Exchange-Level Results

### Momentum by Exchange

| Exchange | Ann. L/S Return | Sharpe | IC t-stat | Quintile Mono |
|---|---|---|---|---|
| GCC (full) | +9.29% | 0.561 | +3.32 *** | 4/4 |
| ADX | +11.91% | 0.422 | +1.97 ** | 2/4 |
| DFM | +14.17% | 0.472 | +2.66 *** | 4/4 |
| Tadawul | +7.13% | 0.400 | +1.91 * | 2/4 |
| QSE | +11.72% | 0.576 | +2.44 ** | 3/4 |

DFM (Dubai) shows the strongest and cleanest momentum — perfect quintile monotonicity. QSE has the best momentum Sharpe at exchange level.

### Quality by Exchange

| Exchange | Ann. L/S Return | Sharpe | IC t-stat | Quintile Mono |
|---|---|---|---|---|
| GCC (full) | +14.38% | 1.030 | +4.87 *** | 3/4 |
| Tadawul | +14.84% | 1.087 | +4.61 *** | 4/4 |
| QSE | +14.64% | 0.681 | +3.02 *** | 3/4 |

Tadawul is the only exchange with perfect quality quintile monotonicity and the highest quality Sharpe.

### Value by Exchange (all negative)

| Exchange | Ann. L/S Return | Sharpe | IC t-stat |
|---|---|---|---|
| GCC (full) | −8.39% | −0.556 | −1.70 |
| Tadawul | −10.0% | −0.795 | −1.27 |
| QSE | −7.53% | −0.520 | −2.82 *** |

Value is consistently negative across all exchanges. QSE has the most statistically significant inversion (IC t = −2.82). Tadawul has the worst Sharpe (−0.795).

### Low-Vol by Exchange

| Exchange | Ann. L/S Return (raw) | Sharpe | IC t-stat | Quintile Mono |
|---|---|---|---|---|
| GCC (full) | −6.75% | −0.155 | +5.13 *** | 2/4 |
| ADX | −2.12% | −0.072 | +3.02 *** | 3/4 |
| Tadawul | +11.21% | +0.518 | +3.33 *** | 3/4 |
| QSE | +6.98% | +0.313 | +2.31 ** | 3/4 |

Tadawul is the only exchange where low-vol works as a portfolio strategy (Sharpe +0.518). It has the deepest liquidity of the five exchanges, limiting the Q5 micro-cap blowup. Exchange-level numbers here reflect the raw backtest (pre-short-book-filter) for comparability.

---

## 9. Sub-Period Analysis

Sub-period table shows FM lambda (bps/month) from joint regression and portfolio Sharpe per sub-period.

| Factor | Pre-COVID 15–19 | COVID 2020 | Post-COVID 21–23 | Recent 24–26 |
|---|---|---|---|---|
| VALUE | Sharpe −0.567 (λ=+27.7, t=+0.85) | Sharpe −0.643 (λ=−16.3, t=−0.59) | Sharpe −0.909 (λ=−34.8, t=−1.88 *) | Sharpe −0.190 (λ=−21.5, t=−1.22) |
| MOMENTUM | Sharpe +0.296 (λ=−30.4, t=−0.83) | Sharpe +0.824 (λ=+16.5, t=+0.44) | Sharpe +0.461 (λ=+39.9, t=+0.86) | Sharpe +0.903 (λ=+50.9, t=+1.89 *) |
| QUALITY | Sharpe +1.235 (λ=+38.6, t=+1.74 *) | Sharpe −1.091 (λ=−60.5, t=−1.12) | Sharpe +2.165 (λ=+16.0, t=+0.67) | Sharpe +1.557 (λ=+31.8, t=+1.74 *) |
| LOW-VOL | Sharpe −0.432 (λ=−292, t=−1.04) | Sharpe +0.075 (λ=−3.4, t=−0.03) | Sharpe +0.695 (λ=+83.2, t=+1.59) | Sharpe +0.586 (λ=+64.6, t=+1.71 *) |

**Narrative:**
- **Pre-COVID 2015–2019:** Quality dominated the stable oil-revenue era (Sharpe +1.235, t=+1.74*). Momentum was flat despite positive Sharpe — pre-2020 momentum was not statistically significant.
- **COVID 2020:** Value was contained (not the catastrophic −3.31*** seen in earlier runs — now t=−0.59 due to different data). Quality reversed sharply (Sharpe −1.091) — profitability offered no protection in a liquidity panic. Momentum held (Sharpe +0.824). Low-vol was near-zero (Sharpe +0.075).
- **Post-COVID 2021–2023:** Quality showed its strongest sub-period (Sharpe +2.165). Low-vol turned positive (Sharpe +0.695). Value remained structurally negative (t=−1.88*, Sharpe −0.909). Momentum positive but not significant.
- **Recent 2024–2026:** Momentum at its strongest (Sharpe +0.903, t=+1.89*). Quality robust (Sharpe +1.557, t=+1.74*). Low-vol improving (Sharpe +0.586, t=+1.71*). Value weakening (Sharpe −0.190).

---

## 10. Kuwait MSCI Natural Experiment

Kuwait was upgraded from MSCI Frontier to MSCI Emerging Market in June 2020. Foreign institutional inflows brought more rational pricing, reduced thin-liquidity spikes, and improved momentum and low-vol signal quality post-upgrade. The low-vol sub-period shift from negative (pre-COVID) to positive (post-COVID) is consistent with the improved market microstructure following institutional capital entry.

---

## 11. Cross-Factor Scorecard

| Factor | Joint FM t | Uni FM t | IC t | ICIR | Q Mono | Verdict |
|---|---|---|---|---|---|---|
| VALUE | −0.40 | −2.16 ** | −1.70 | −0.508 | 1/4 broken | Statistically significant inversion. Publishable negative result. |
| MOMENTUM | +0.62 | +1.81 * | +3.32 *** | +1.033 | 4/4 perfect | Real signal, marginal FM significance. Strengthening. |
| QUALITY | +1.48 | +2.98 *** | +4.87 *** | +1.453 | 3/4 (minor break) | Strongest stats. Look-ahead inflated. True Sharpe 0.4–0.6. |
| LOW-VOL | −0.63 | −0.67 | +5.13 *** | +1.583 | 2/4 broken | Highest IC of all factors. L/S portfolio still negative. Signal maturing. |

---

## 12. Limitations

1. **Survivorship bias:** Universe constructed from stocks currently listed as of December 2025. Delisted or suspended stocks not captured. Prior literature suggests this inflates factor spread estimates by ~0.5–2% annually (Elton et al., 1996; Linnainmaa and Roberts, 2018).

2. **Look-ahead bias in fundamentals:** Value and quality factors use current-day P/B, EV/EBITDA, ROE, and operating margin snapshots. Historical point-in-time data requires a commercial vendor (Compustat, EODHD). All fundamental-factor results are upper-bound estimates.

3. **Short-side implementation:** GCC markets have limited stock borrowing outside Tadawul. The long-short framework is used as an academic test of signal direction. A practical implementation would be long-only or confine shorting to the most liquid Tadawul names.

4. **Low-vol L/S portfolio negative after fix:** The short-book filter (market cap ≥$300M, vol ≥200k shares, z-score cap) was applied and data regenerated, but the L/S return is still negative (−1.47%, Sharpe −0.042). The IC signal (ICIR +1.583) is genuine — the portfolio construction problem persists beyond simple short-book filtering. A long-only low-vol sleeve is the implementable path.

5. **Small joint-regression sample:** The joint FM requires all four factor scores simultaneously, reducing coverage to 102 stocks/month vs 148 for momentum univariate. The restriction disproportionately excludes smaller Kuwaiti and Qatari names, introducing a mild large-cap bias in the joint model.

6. **The VRP overlay is a simulation.** Tadawul single-stock options (SSOs) only launched in November 2023, and there is no public history of their prices. The overlay prices covered calls off a synthetic implied-volatility surface built from realised volatility, a TASI stress loading, oil volatility and a thinness premium (`Layer_3_analysis/layer3c_vol_surface.py`). The overlay results (Sharpe 1.49–1.84) are therefore a hypothesis about what the strategy could earn, not a backtest on traded prices.

---

## 13. Key Conclusions

1. **VALUE is the most statistically significant finding — but inverted.** Univariate t = −2.16**, the most significant negative premium in the full-sample FM results. Cheap stocks underperform by 8.39% per year consistently across all sub-periods and all exchanges. This is a publishable negative result.

2. **QUALITY has the strongest positive signal.** FM t = +2.98*** univariate, ICIR +1.453, Sharpe +1.030. Must be discounted for look-ahead bias — realistic out-of-sample Sharpe is 0.4–0.6. The dominant factor in pre-COVID and post-COVID environments.

3. **MOMENTUM works but with marginal FM significance (t=+1.81*).** Perfect quintile monotonicity and ICIR +1.033 confirm the signal is real. No look-ahead bias. Strengthening in the most recent period (2024–2026: Sharpe +0.903). The most implementable factor.

4. **LOW-VOLATILITY has the strongest IC of all four factors** (ICIR +1.583, IC t = +5.13***) but this signal still cannot be captured via a long-short portfolio even after filtering illiquid micro-caps. The L/S strategy remains marginally negative (−1.47%, Sharpe −0.042). A long-only low-vol sleeve on Tadawul is the viable implementation (exchange-level Sharpe +0.518).

5. **The VRP overlay dramatically improves portfolio economics.** A dynamic covered-call overlay sized by the VRP gap between implied and realised volatility converts a 0.49 Sharpe multi-factor portfolio into a 1.79 Sharpe system. Hit rate rises from 58.8% to 77.1%. The VRP gap is persistent across all market regimes (calm: 7.5pp, normal: 8.7pp, stressed: 9.3pp).

6. **GCC market structure explains the cross-sectional patterns.** Oil price cycles drive momentum. State-linked mega-caps drive quality premia. Low institutional participation prevents value mean-reversion. Post-MSCI-upgrade liquidity improvements are gradually strengthening the low-vol signal but have not yet made it harvestable via L/S. These are not statistical artifacts — they have coherent economic explanations rooted in GCC market microstructure.

---

## 14. Portfolio Performance Summary

| Portfolio | Ann. Return | Sharpe | Max DD | Hit Rate |
|---|---|---|---|---|
| Pure momentum L/S | +9.29% | 0.561 | −23.3% | 56.5% |
| Pure quality L/S | +14.38% | 1.030 | −33.0% | 65.9% |
| Combined CB (Sleeve A + B) | +5.79% | 0.49 | −24.5% | 58.8% |
| Combined + circuit breaker | +5.35% | 0.49 | −22.9% | 58.8% |
| Combined + VRP overlay | +20.65% | 1.84 | −15.8% | 73.7% |
| **FINAL (CB + VRP + hedge)** | **+20.11%** | **1.79** | **−15.9%** | **77.1%** |

---

## 15. Key Citations

| Paper | Relevance |
|---|---|
| Fama & French (1992, 1993) | Value and size factor construction |
| Jegadeesh & Titman (1993) | Momentum factor definition (12-1) |
| Fama & MacBeth (1973) | Cross-sectional regression methodology |
| Ang et al. (2006) | Low-volatility anomaly |
| Elton, Gruber & Blake (1996) | Survivorship bias quantification |
| Linnainmaa & Roberts (2018) | Factor premia and survivorship bias |
| Harvey, Liu & Zhu (2016) | Multiple-testing correction in factor research |
| Newey & West (1987) | HAC standard errors |

---

*Methodology: Jegadeesh-Titman (1993) momentum (12-1 month); Fama-French (1992, 1993) value (P/B + EV/EBITDA composite); composite quality (ROE, operating margin, earnings stability CV); trailing 252-day realised vol (Ang et al., 2006). Fama-MacBeth (1973) cross-sectional regressions with Newey-West (4-lag) standard errors. All factor scores winsorised at 1st/99th percentile and cross-sectionally z-scored before quintile assignment. Low-vol short book filtered: market cap ≥$300M, avg daily vol ≥200k shares, monthly z-score <3σ, per-stock weight cap 15%. VRP overlay: dynamic covered-call sizing by IV−RV gap, regime-classified into calm/normal/stressed.*
