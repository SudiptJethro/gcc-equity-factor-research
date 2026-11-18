
# ─────────────────────────────────────────────────────────────
# 4.  PIPELINE PER EXCHANGE
# ─────────────────────────────────────────────────────────────

import time
import pandas as pd
import yfinance as yf

from config import START_DATE, END_DATE
from universe import EXCHANGE_META, TWELVE_DATA_MAP
from downloader import download_ticker
from filters import passes_filters


def build_exchange_data(exchange_name: str, tickers: list) -> tuple:
    """
    Downloads, filters, and processes one exchange.
    Returns (monthly_prices, daily_prices, metadata_list).
    """
    meta = EXCHANGE_META.get(exchange_name, {})
    print(f"\n{'='*56}")
    print(f"  {exchange_name}  ({meta.get('country','')})  "
          f"MSCI: {meta.get('msci_status','')}  "
          f"suffix: {meta.get('suffix','')}")
    print(f"  Downloading {len(tickers)} tickers  |  {START_DATE} -> {END_DATE}")
    print(f"{'='*56}")

    raw_close  = {}
    raw_volume = {}
    metadata   = []
    dead       = []

    tickers = list(dict.fromkeys(tickers))  # de-duplicate, preserve order

    for i, ticker in enumerate(tickers):
        td_exchange, td_symbol = TWELVE_DATA_MAP.get(ticker, ("", ""))
        df = download_ticker(ticker, START_DATE, END_DATE,
                             td_exchange=td_exchange, td_symbol=td_symbol)
        passed, reason = passes_filters(ticker, df)

        if not passed:
            dead.append((ticker, reason))
            continue

        avg_vol = df["Volume"].mean()
        raw_close[ticker]  = df["Close"]
        raw_volume[ticker] = df["Volume"]
        print(f"  [OK]   {ticker:22s}  {len(df):>4d} rows  vol {avg_vol:>10,.0f}")

        if i % 5 == 4:
            time.sleep(1.5)  # rate-limit buffer

    print(f"\n  Passed: {len(raw_close)} / {len(tickers)} | Dropped: {len(dead)}")
    if dead:
        for t, reason in dead:
            print(f"    dropped {t}: {reason}")

    if not raw_close:
        print(f"  WARNING: No usable data for {exchange_name}")
        return pd.DataFrame(), pd.DataFrame(), []

    daily = pd.DataFrame(raw_close)
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    monthly = daily.resample("ME").last()
    monthly = monthly[monthly.notna().mean(axis=1) >= 0.5]  # drop sparse months

    for ticker in raw_close:
        try:
            info    = yf.Ticker(ticker).fast_info
            mkt_cap = getattr(info, "market_cap", None)
        except Exception:
            mkt_cap = None

        metadata.append({
            "ticker":       ticker,
            "exchange":     exchange_name,
            "country":      meta.get("country", ""),
            "msci_status":  meta.get("msci_status", ""),
            "daily_rows":   int(raw_close[ticker].dropna().__len__()),
            "monthly_rows": int(monthly[ticker].notna().sum()),
            "first_date":   str(monthly[ticker].first_valid_index()),
            "last_date":    str(monthly[ticker].last_valid_index()),
            "avg_volume":   float(raw_volume[ticker].mean()),
            "market_cap":   float(mkt_cap) if mkt_cap else None,
        })

    return monthly, daily, metadata
