> This is the original project plan, written before the data was collected. Planned figures such as the ~202-stock universe differ from what was finally built. See `research_summary.md` and the main README for the actual results.

# GCC Equity Factor Research Engine — Project Specification

> Hand this document to Claude Code to implement the full project.  
> Every architectural decision, design choice, and methodology constraint is captured here.

---

## 1. What I am trying to achieve

I am building a **systematic equity factor research engine** covering GCC capital markets as a proof-of-work project to enter the asset management / hedge fund industry in the UAE. The project must do two things simultaneously:

1. **Demonstrate quantitative research ability** — not just "I can code", but "I can think like a quant analyst and build something an AM firm would recognise as rigorous."
2. **Be a publishable research study** — the underlying question ("do classic equity factor premia hold in GCC markets?") has not been answered in the published literature at the stock level. This is genuinely original work.

The deliverable is a **live public dashboard** (Streamlit) with a 4-page research note (PDF) and a clean GitHub repo. These three together go on my CV and into every cold email I send to Mubadala, Arqaam, Wamda, and similar firms in DIFC.

The project also serves as my MSc application thesis proposal. The exact research question:

> *"Do classic equity factor premia (value, momentum, quality, low volatility) generate statistically significant alpha in GCC equity markets, and do they behave differently across ADX, DFM, Tadawul, QSE, and Boursa Kuwait?"*

---

## 2. Project context

- **Background:** Undergraduate degree completed. One year out of university (as of April 2025). Builder mindset — combining finance and technology.
- **Career target:** Asset management roles in UAE (DIFC). Longer-term: hedge fund.
- **Previous work:** FortuneFlo (personal finance platform) — established that I can build financial data pipelines.
- **Why GCC specifically:** This is the market the hiring firms operate in. Showing you understand GCC equity structure, liquidity constraints, and market microstructure beats any generic quant project.

---

## 3. Universe — 202 stocks final

The investable universe after applying the liquidity and data quality filters described below.

### Exchange breakdown

| Exchange | Country | MSCI Status | Yahoo Finance suffix | Target stocks |
|---|---|---|---|---|
| ADX | UAE – Abu Dhabi | Emerging Market | `.AD` | ~35 |
| DFM | UAE – Dubai | Emerging Market | `.DU` | ~25 |
| Tadawul | Saudi Arabia | Emerging Market | `.SR` | ~100 |
| QSE | Qatar | Emerging Market | `.QA` | ~25 |
| Boursa Kuwait | Kuwait | Emerging Market (upgraded June 2020) | `.KW` | ~17 |
| **Total** | | | | **~202** |

Oman (MSM) and Bahrain (BSE) are excluded — insufficient liquidity for implementable factor portfolios. This exclusion is documented in the methodology as a deliberate decision, not a gap.

### Liquidity filters applied during pipeline (Layer 1)

```python
MIN_PRICE       = 0.10        # exclude suspended / penny stocks
MIN_AVG_VOLUME  = 50_000      # average daily shares traded
MIN_MONTHS_DATA = 12          # minimum months of price history
START_DATE      = "2015-01-01"
END_DATE        = "2025-12-31"
```

**Why 2015 as the start date:**
- Saudi Arabia opened Tadawul to direct foreign institutional investment: June 2015
- UAE (ADX/DFM) and Qatar upgraded from MSCI Frontier to Emerging Market: May 2014
- yfinance coverage for GCC stocks is reliably consistent from 2015 onwards

### GCC-specific data handling

- **Trading calendar:** ADX, DFM, Tadawul trade Sunday–Thursday. Resample to month-end using `.resample('ME').last()` — do not try to build a custom GCC trading calendar.
- **Corporate actions:** Use `yf.download(ticker, auto_adjust=True)` always. Etisalat → EAND is the most common rename that breaks pipelines.
- **Kuwait MSCI natural experiment:** Save separate Parquet files split at June 2020 (`prices_kuwait_pre_msci.parquet`, `prices_kuwait_post_msci.parquet`). Test factor premia pre vs post MSCI upgrade as a sub-study.

---

## 4. Layer 1 — Data pipeline (complete)

**Status: built. Do not rebuild unless there is a bug.**

### Output files written to `/data/`

```
data/
├── prices_adx.parquet          # month-end prices, ADX stocks
├── prices_dfm.parquet
├── prices_tadawul.parquet
├── prices_qse.parquet
├── prices_kuwait.parquet
├── prices_kuwait_pre_msci.parquet   # split at June 2020
├── prices_kuwait_post_msci.parquet
├── returns_adx.parquet         # monthly log returns
├── returns_dfm.parquet
├── returns_tadawul.parquet
├── returns_qse.parquet
├── returns_kuwait.parquet
├── daily_adx.parquet           # daily prices (for vol factor)
├── daily_dfm.parquet
├── daily_tadawul.parquet
├── daily_qse.parquet
├── daily_kuwait.parquet
└── universe.csv                # filtered universe with metadata
```

### `universe.csv` schema

| Column | Description |
|---|---|
| `ticker` | Yahoo Finance ticker (e.g. `FAB.AD`) |
| `exchange` | ADX / DFM / Tadawul / QSE / Kuwait |
| `company_name` | Human-readable name |
| `first_valid_date` | First date with valid price data |
| `avg_daily_volume` | Average daily volume over full period |
| `market_cap` | Market cap in local currency (from yfinance .info) |
| `months_of_data` | Total months with valid price data |

---

## 5. Layer 2 — Factor construction (implement this)

### Overview

Compute four factors monthly across all 202 stocks. For each factor:
1. Compute the raw signal per stock
2. Winsorise at 1st/99th percentile (suppress GCC outliers — small-cap ratios are extreme)
3. Cross-sectional z-score (zero mean, unit variance, within each month)
4. Assign to quintile Q1 (best) through Q5 (worst)
5. Build equal-weighted long-short portfolio: long Q1, short Q5

Monthly rebalancing. Returns shifted by t+1 (rank at end of month t, earn returns in month t+1).

---

### Factor 1: Momentum (12-1)

**Source data:** `prices_{exchange}.parquet` (monthly)

**Definition:** Jegadeesh-Titman (1993) — cumulative return from t-12 to t-2, skipping the most recent month.

```
MOM(t) = (Price[t-2] / Price[t-12]) - 1
```

**Skip logic:** The most recent month (t-1) is excluded because GCC equities show short-term reversal — stocks that rallied last month give back gains in the current month. Including it would contaminate the signal.

**Minimum data requirement:** Stock must have at least `0.80 × (12 - 1) = 8.8` valid monthly observations in the 11-month window (t-12 to t-2), i.e. round up to 9. Stock gets `NaN` otherwise.

**New IPO handling:** A stock listed in March 2022 cannot receive a momentum score until March 2023. The `NaN` propagates correctly through pandas — no special handling needed beyond the minimum history check.

**Direction:** Higher momentum score = better. Q1 = top momentum stocks (long). Q5 = worst momentum (short).

---

### Factor 2: Low Volatility

**Source data:** `daily_{exchange}.parquet`

**Definition:** Trailing 252 trading-day realised volatility, annualised.

```
VOL(t) = std(daily_log_returns over past 252 days) × √252
LOW_VOL_SCORE(t) = -z_score(VOL(t))   ← inverted: lower vol = higher score
```

**Minimum data requirement:** At least 180 daily observations in the 252-day window. Otherwise `NaN`.

**Fallback (if daily files missing):** Use trailing 12-month monthly return std × √12. Flag this in the output with a warning — it is less accurate.

**Direction:** Invert before z-scoring. Lower volatility = higher score = long side of the portfolio. Do not forget the inversion — this is the most common implementation error.

**Why daily, not monthly:** 252 daily observations give a precise vol estimate. 12 monthly observations give a noisy one. The momentum and value/quality factors can use monthly data because they capture slow-moving signals. Vol is sensitive to estimation frequency.

---

### Factor 3: Value (composite)

**Source data:** `yfinance.Ticker(ticker).info` — point-in-time snapshot

**Signals:**
- `priceToBook` → invert: `1 / P/B`
- `enterpriseToEbitda` → invert: `1 / EV/EBITDA`

**Composite construction:**
1. Invert each ratio (lower ratio = more value = higher signal)
2. Exclude negative values (negative P/B or EV/EBITDA = distressed / unreliable) — set to `NaN`
3. Winsorise each signal at 1/99
4. Z-score each signal cross-sectionally
5. Average the two z-scores (equal weight). If only one is available, use that one alone.

**Expected coverage:** ~60% of the 202-stock universe will have at least one valid signal. The remaining ~40% (mostly smaller names) are excluded from value portfolios but still receive momentum and low-vol scores.

**Look-ahead bias note (required in research note):**
> yfinance provides today's fundamental values, not historical point-in-time values. This introduces look-ahead bias into the Value and Quality factors. We acknowledge this limitation, document it in the methodology, and note that it would require a commercial data vendor (e.g. EODHD, Compustat) to resolve. The Momentum and Low-Vol factors are unaffected.

**Direction:** Higher composite value z-score = cheaper stock = better. Q1 = cheapest stocks (long). Q5 = most expensive (short).

---

### Factor 4: Quality (composite)

**Source data:** `yfinance.Ticker(ticker).info` + `yfinance.Ticker(ticker).quarterly_earnings`

**Signals and weights:**

| Signal | Weight | Source | Notes |
|---|---|---|---|
| ROE (return on equity) | 50% | `.info['returnOnEquity']` | Exclude negative ROE (distressed) |
| Earnings stability | 30% | `.quarterly_earnings` | Inverse CV of quarterly EPS over available history |
| Operating margin | 20% | `.info['operatingMargins']` | Exclude < -100% (extreme losses) |

**Earnings stability calculation:**
```python
eps = quarterly_earnings['Earnings'].dropna()
if len(eps) < 4:
    return NaN
cv = eps.std() / abs(eps.mean())   # coefficient of variation
stability = 1 / cv                  # invert: lower CV = higher stability
```

**Composite construction:** Same as Value — winsorise, z-score, weighted average with partial-NaN handling (use available signals, adjust weights accordingly).

**Direction:** Higher quality score = better. Q1 = highest quality (long). Q5 = lowest quality (short).

---

### Statistical pre-processing (applies to all factors)

```python
def winsorise(series, lower=0.01, upper=0.99):
    lo, hi = series.quantile(lower), series.quantile(upper)
    return series.clip(lower=lo, upper=hi)

def cross_sectional_zscore(series):
    w = winsorise(series)
    return (w - w.mean()) / w.std()
```

**Why winsorise:** GCC universe contains small-cap names with extreme fundamental ratios. A P/B of 0.001 or 150 is data noise, not signal. Winsorising at 1/99 caps these without discarding the observation entirely.

**Cross-sectional, not time-series:** The z-score is computed across all stocks in a given month, not across time for a given stock. This removes any common market-wide drift from the factor scores and makes quintile assignments comparable month-to-month.

---

### Quintile assignment

```python
# Q1 = best score, Q5 = worst score
quintile = pd.qcut(
    scores.rank(method='first'),
    q=5,
    labels=[5, 4, 3, 2, 1]   # reverse so Q1 = highest score
)
```

Minimum 15 valid stocks in a given month for quintile assignment to proceed. If fewer, skip that month.

---

### Portfolio return calculation

```python
# For each month t:
# - Rank stocks at end of month t → assign quintiles
# - Compute equal-weighted returns in month t+1 for each quintile
# - Long-short = Q1_return - Q5_return

for date in dates[:-1]:
    next_date = dates[i + 1]
    q1_stocks = quintiles.loc[date][quintiles.loc[date] == 1].index
    q5_stocks = quintiles.loc[date][quintiles.loc[date] == 5].index
    ls_return = returns.loc[next_date, q1_stocks].mean() \
              - returns.loc[next_date, q5_stocks].mean()
```

**All five quintile returns are saved** (Q1–Q5), not just the long-short spread. This allows quintile monotonicity analysis — a properly-working factor shows monotonic return improvement from Q5 to Q1.

---

### Output files written to `/data/`

```
data/
├── fundamentals.parquet           # cached yfinance .info snapshot
├── earnings_stability.parquet     # cached quarterly EPS stability scores
├── factor_scores.parquet          # MultiIndex: (factor_name, ticker) × date
├── quintile_assignments.parquet   # MultiIndex: (factor_name, ticker) × date
├── portfolio_returns.parquet      # columns: value_Q1..Q5, value_LS, mom_Q1..Q5, mom_LS, etc.
├── portfolio_returns.csv          # same, CSV format for easy inspection
└── factor_snapshot.csv            # most recent month cross-sectional scores for all stocks
```

### `portfolio_returns.parquet` schema

| Column | Description |
|---|---|
| `value_Q1` through `value_Q5` | Monthly equal-weighted return of each value quintile |
| `value_LS` | Long-short spread: Q1 minus Q5 |
| `mom_Q1` through `mom_Q5` | Momentum quintile returns |
| `mom_LS` | Momentum long-short |
| `quality_Q1` through `quality_Q5` | Quality quintile returns |
| `quality_LS` | Quality long-short |
| `lowvol_Q1` through `lowvol_Q5` | Low-vol quintile returns |
| `lowvol_LS` | Low-vol long-short |

---

### Caching strategy

Fundamentals and earnings stability fetches are slow (~5 minutes for 202 tickers). Cache immediately after first fetch.

```python
fund_cache = "data/fundamentals.parquet"
if os.path.exists(fund_cache):
    fundamentals = pd.read_parquet(fund_cache)
else:
    fundamentals = fetch_all_fundamentals(tickers)
    fundamentals.to_parquet(fund_cache)
```

Same pattern for `earnings_stability.parquet`. Add a `--refresh-fundamentals` CLI flag to force a re-fetch when you want to update the snapshot.

---

### Run command

```bash
python factor_construction.py
# Options:
#   --refresh-fundamentals    Force re-fetch from yfinance (slow)
#   --exchange ADX            Run for one exchange only (debugging)
#   --verbose                 Print per-stock factor assignment
```

### Expected terminal output on completion

```
=============================== FACTOR CONSTRUCTION COMPLETE ===
Period: 2016-01-31 → 2025-12-31   |   117 months of L/S returns

Factor      Ann. Return   Ann. Vol   Sharpe   Hit Rate
──────────────────────────────────────────────────────
VALUE          +X.XX%      X.XX%     X.XX     XX.X%
MOMENTUM       +X.XX%      X.XX%     X.XX     XX.X%
QUALITY        +X.XX%      X.XX%     X.XX     XX.X%
LOW-VOL        +X.XX%      X.XX%     X.XX     XX.X%

Files written to: data/
Next: python fama_macbeth.py
```

---

## 6. Layer 3 — Analysis (build after Layer 2)

### 6a. Fama-MacBeth regressions (`fama_macbeth.py`)

Two-step procedure (Fama & MacBeth, 1973):

**Step 1:** For each month t, run a cross-sectional OLS regression of stock returns on factor scores:
```
R_i,t+1 = α_t + γ_value,t × VALUE_i,t + γ_mom,t × MOM_i,t
         + γ_quality,t × QUALITY_i,t + γ_lowvol,t × LOWVOL_i,t + ε_i,t
```

**Step 2:** Average the monthly slope coefficients `γ_k,t` across all T months. Test whether the time-series mean is significantly different from zero using a t-test with Newey-West standard errors (4 lags).

The t-statistic on `γ̄_k` is the factor's statistical significance. A |t| > 2.0 is the threshold for a "significant" factor premium in the literature.

**Output:** Table of `{factor: mean_gamma, t_stat, p_value}` for all four factors, separately for the full GCC universe and for each exchange.

### 6b. Performance attribution (`backtest.py`)

For each factor's L/S portfolio:

| Metric | Formula |
|---|---|
| Annualised return | `mean(monthly_LS) × 12` |
| Annualised volatility | `std(monthly_LS) × √12` |
| Sharpe ratio | `ann_return / ann_vol` |
| Maximum drawdown | `min(cumulative_drawdown_series)` |
| Hit rate | `fraction of months where LS > 0` |
| Information coefficient (IC) | `spearman_corr(factor_scores_t, returns_t+1)` — monthly, then mean |
| Turnover | `fraction of portfolio that changes on each rebalance date` |
| Quintile monotonicity | Do Q1 through Q5 show monotonically increasing returns? |

**Transaction cost adjustment:** Assume 50bps round-trip per trade (realistic for mid-cap GCC names). Apply to turnover to compute net-of-cost Sharpe.

**Sub-period analysis:**
- Pre-COVID: 2015–2019
- COVID shock: 2020
- Post-COVID / rate cycle: 2021–2023
- Recent: 2024–2025
- Kuwait pre/post MSCI upgrade: before/after June 2020

---

## 7. Layer 4 — Dashboard (Streamlit)

File: `app.py`. Deploy free on Streamlit Community Cloud.

### Pages

1. **Universe overview** — map of GCC exchanges, stock count per exchange, market cap breakdown, sector composition
2. **Factor scores (live)** — table of all 202 stocks with current z-scores for all four factors and their quintile assignment. Filterable by exchange and sector.
3. **Factor performance** — cumulative L/S return chart per factor, quintile return bar chart (monotonicity check), rolling 12-month Sharpe
4. **Research note** — embedded PDF or rendered markdown
5. **Methodology** — one-page plain-language explanation of what each factor measures and how it is constructed

### Tech stack

```
streamlit          # UI
plotly             # interactive charts
pandas             # data
pyarrow            # parquet reading
scipy              # stats
```

No database required. All data lives in Parquet files read at startup.

---

## 8. Research note outline (4 pages, PDF)

The research note is sent instead of a cover letter to quant roles. Write it alongside the code.

1. **Abstract** (half page) — research question, method, key finding
2. **Introduction** (half page) — why GCC, why factors, what gap this fills in literature
3. **Data and universe construction** (half page) — exchanges, date range, liquidity filters, survivorship bias disclosure
4. **Methodology** (1 page) — factor definitions, Fama-MacBeth procedure, transaction cost assumption
5. **Results** (1 page) — factor premia table with t-stats, quintile monotonicity charts, sub-period analysis
6. **Conclusion** (quarter page) — answer the research question, limitations, extensions

**The survivorship bias paragraph (include verbatim):**
> "Our universe is constructed from stocks currently listed on each exchange as of December 2025, introducing potential survivorship bias. Stocks delisted, acquired, or suspended during the study period are not captured. Prior literature suggests survivorship bias inflates annualised factor spread estimates by approximately 0.5–2% (Elton et al., 1996; Linnainmaa and Roberts, 2018). We mitigate this by focusing on comparative cross-market analysis rather than claiming absolute implementable alpha. Newly listed securities are included from their first trading month but require a minimum 12-month price history before receiving factor scores, consistent with standard practice."

**The look-ahead bias paragraph (include verbatim):**
> "Fundamental data (P/B, EV/EBITDA, ROE) is sourced from Yahoo Finance as a point-in-time snapshot as of December 2025. Historical point-in-time fundamental data would require a commercial vendor (Compustat, EODHD). This introduces look-ahead bias into the Value and Quality factor backtests; these results should be interpreted as upper-bound estimates of true out-of-sample performance. Momentum and Low-Volatility factors, which are constructed exclusively from price data, are unaffected."

---

## 9. Key citations

| Paper | Why it is cited |
|---|---|
| Fama & French (1992, 1993) | Value and size factor construction |
| Jegadeesh & Titman (1993) | Momentum factor definition (12-1) |
| Fama & MacBeth (1973) | Cross-sectional regression methodology |
| Ang et al. (2006) | Low-volatility anomaly — idiosyncratic vol and expected returns |
| Elton, Gruber & Blake (1996) | Survivorship bias quantification |
| Linnainmaa & Roberts (2018) | Factor premia and data mining / survivorship bias |
| Harvey, Liu & Zhu (2016) | Multiple-testing correction in factor research |
| Newey & West (1987) | Heteroskedasticity / autocorrelation-consistent standard errors |

---

## 10. File structure

```
gcc-factor-engine/
├── data/                          # all Parquet and CSV outputs
├── gcc_data_pipeline.py           # Layer 1 — complete
├── gcc_tickers_updated.py         # universe ticker lists (202 stocks)
├── update_gcc_universe.py         # utility to check for new listings
├── factor_construction.py         # Layer 2 — implement this
├── fama_macbeth.py                # Layer 3a
├── backtest.py                    # Layer 3b
├── app.py                         # Layer 4 — Streamlit dashboard
├── research_note.md               # 4-page research note source
├── requirements.txt
└── README.md
```

### `requirements.txt`

```
yfinance>=0.2.40
pandas>=2.0
numpy>=1.26
scipy>=1.12
pyarrow>=15.0
streamlit>=1.35
plotly>=5.20
beautifulsoup4>=4.12
requests>=2.31
statsmodels>=0.14     # for Newey-West standard errors
```

---

## 11. Common failure modes to anticipate

| Problem | Cause | Fix |
|---|---|---|
| `yf.download()` returns empty DataFrame | Ticker string wrong or delisted | Log to dead_tickers list, skip, continue |
| Negative P/B or EV/EBITDA | Distressed firm or data error | Set to NaN — do not invert |
| `quarterly_earnings` returns None | No coverage in yfinance | Return NaN for stability score |
| Factor scores all identical in a month | Insufficient valid stocks | Skip month — log a warning |
| Kuwait returns look abnormal pre-2016 | Thin yfinance coverage | Filter: only use Kuwait data from 2016 onwards |
| `priceToBook` missing for 40%+ of universe | Small-cap GCC coverage gap | Expected — document in research note, not a bug |
| Tadawul tickers returning USD prices | Wrong ticker format | Tadawul uses numeric codes: `2222.SR`, not company names |

---

## 12. What success looks like

The project is complete when:

1. `factor_construction.py` runs end-to-end on 202 stocks and prints the summary table
2. `fama_macbeth.py` outputs a factor significance table with t-stats
3. `backtest.py` outputs Sharpe ratios, max drawdowns, and IC for all four factors
4. `app.py` runs locally and shows the live factor scores and performance charts
5. The Streamlit app is deployed to a public URL
6. The 4-page research note PDF is in the repo

The project is interview-ready when I can answer, without notes:
- Why Fama-MacBeth rather than panel OLS?
- What does survivorship bias do to my factor spread estimates?
- Why does momentum tend to be weaker in high oil-beta markets?
- What would I do differently with Bloomberg data?