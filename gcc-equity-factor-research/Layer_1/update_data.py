"""
Incremental data updater for GCC Equity Factor Research.

Logic per exchange:
  Existing tickers  / extend from last date in parquet to END_DATE
  New tickers       / download full history, apply quality filters
  Removed tickers   / kept in parquet (data is still valid research history)

Run:  python update_data.py
"""

import os
import time
import pandas as pd
import numpy as np
import yfinance as yf

from config import START_DATE, END_DATE, DATA_DIR, KUWAIT_MSCI_UPGRADE
from universe import ALL_EXCHANGES, EXCHANGE_META, TWELVE_DATA_MAP
from downloader import download_ticker
from filters import passes_filters
from returns import log_returns, kuwait_split


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_parquet(exchange: str, kind: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{kind}_{exchange.lower()}.parquet")
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print(f"  [WARN] Could not load {path}: {e}")
    return pd.DataFrame()


def save_parquet(df: pd.DataFrame, exchange: str, kind: str):
    path = os.path.join(DATA_DIR, f"{kind}_{exchange.lower()}.parquet")
    df.to_parquet(path)
    print(f"  Saved: {path}  ({df.shape[0]} rows × {df.shape[1]} cols)")


# ─────────────────────────────────────────────────────────────────────────────
# Per-exchange update
# ─────────────────────────────────────────────────────────────────────────────

def update_exchange(exchange_name: str, tickers: list) -> tuple:
    meta = EXCHANGE_META.get(exchange_name, {})
    print(f"\n{'='*62}")
    print(f"  {exchange_name}  |  {meta.get('country','')}  |  {START_DATE} -> {END_DATE}")
    print(f"{'='*62}")

    tickers = list(dict.fromkeys(tickers))  # deduplicate, preserve order

    existing_daily = load_parquet(exchange_name, "daily")
    existing_cols  = set(existing_daily.columns.tolist()) if not existing_daily.empty else set()

    universe_set   = set(tickers)
    new_tickers    = [t for t in tickers if t not in existing_cols]
    old_tickers    = [t for t in tickers if t in existing_cols]
    # Tickers in existing parquet but dropped from universe / retain their data
    orphan_tickers = [t for t in existing_cols if t not in universe_set]

    print(f"  Universe: {len(tickers)} tickers  |  "
          f"Existing: {len(old_tickers)}  |  "
          f"New: {len(new_tickers)}  |  "
          f"Orphaned (kept): {len(orphan_tickers)}")

    close_data      = {}   # ticker -> pd.Series (daily Close)
    avg_vol_map     = {}   # ticker -> float (for metadata)

    # ── 1. Incremental extension of existing tickers ──────────────────────────
    for i, ticker in enumerate(old_tickers):
        col = existing_daily[ticker].dropna()
        last_date = col.index[-1] if not col.empty else pd.Timestamp(START_DATE)

        if last_date >= pd.Timestamp(END_DATE):
            close_data[ticker] = existing_daily[ticker]
            print(f"  [CURR] {ticker:26s}  already at {last_date.date()}")
            continue

        fetch_from = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        td_exchange, td_symbol = TWELVE_DATA_MAP.get(ticker, ("", ""))
        df_new = download_ticker(ticker, fetch_from, END_DATE,
                                 td_exchange=td_exchange, td_symbol=td_symbol)

        # yfinance failed / try Twelve Data directly when mapping exists
        if df_new.empty and td_exchange and td_symbol:
            from downloader import download_twelve_data
            time.sleep(8)
            df_new = download_twelve_data(ticker, fetch_from, END_DATE, td_exchange, td_symbol)

        if df_new.empty:
            close_data[ticker] = existing_daily[ticker]
            print(f"  [KEEP] {ticker:26s}  no new data, kept existing ({last_date.date()})")
        else:
            close_col = "Close" if "Close" in df_new.columns else df_new.columns[0]
            merged = pd.concat([existing_daily[ticker], df_new[close_col]]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            close_data[ticker] = merged
            rows_added = len(df_new)
            print(f"  [UPD]  {ticker:26s}  +{rows_added} rows -> last {merged.dropna().index[-1].date()}")

        if i % 5 == 4:
            time.sleep(1.5)

    # ── 2. Full download for new tickers ──────────────────────────────────────
    for i, ticker in enumerate(new_tickers):
        td_exchange, td_symbol = TWELVE_DATA_MAP.get(ticker, ("", ""))
        df = download_ticker(ticker, START_DATE, END_DATE,
                             td_exchange=td_exchange, td_symbol=td_symbol)

        if df.empty:
            print(f"  [MISS] {ticker:26s}  no data from any source")
            continue

        passed, reason = passes_filters(ticker, df)
        if not passed:
            continue

        close_data[ticker]  = df["Close"]
        avg_vol_map[ticker] = float(df["Volume"].mean())
        print(f"  [NEW]  {ticker:26s}  {len(df)} rows  vol {avg_vol_map[ticker]:>10,.0f}")

        if i % 5 == 4:
            time.sleep(1.5)

    # ── 3. Carry forward orphaned tickers (history preserved) ─────────────────
    for ticker in orphan_tickers:
        close_data[ticker] = existing_daily[ticker]

    if not close_data:
        print(f"  WARNING: No usable data for {exchange_name}")
        return pd.DataFrame(), pd.DataFrame(), []

    # ── 4. Build daily + monthly frames ───────────────────────────────────────
    daily   = pd.DataFrame(close_data)
    daily.index = pd.to_datetime(daily.index)
    daily   = daily.sort_index()

    monthly = daily.resample("ME").last()
    monthly = monthly[monthly.notna().mean(axis=1) >= 0.5]

    # ── 5. Metadata ───────────────────────────────────────────────────────────
    metadata = []
    for ticker in close_data:
        try:
            info    = yf.Ticker(ticker).fast_info
            mkt_cap = getattr(info, "market_cap", None)
        except Exception:
            mkt_cap = None

        col = daily[ticker].dropna()
        metadata.append({
            "ticker":       ticker,
            "exchange":     exchange_name,
            "country":      meta.get("country", ""),
            "msci_status":  meta.get("msci_status", ""),
            "daily_rows":   int(len(col)),
            "monthly_rows": int(monthly[ticker].notna().sum()) if ticker in monthly.columns else 0,
            "first_date":   str(col.index[0])  if len(col) else "",
            "last_date":    str(col.index[-1]) if len(col) else "",
            "avg_volume":   avg_vol_map.get(ticker, float("nan")),
            "market_cap":   float(mkt_cap) if mkt_cap else None,
            "in_universe":  ticker in universe_set,
        })

    return monthly, daily, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nGCC Factor Research / Incremental Data Update")
    print(f"Period: {START_DATE}  ->  {END_DATE}\n")

    all_metadata     = []
    exchange_monthly = {}

    for exchange, tickers in ALL_EXCHANGES.items():
        monthly, daily, meta = update_exchange(exchange, tickers)

        if monthly.empty:
            continue

        exchange_monthly[exchange] = monthly
        all_metadata.extend(meta)

        save_parquet(daily,   exchange, "daily")
        save_parquet(monthly, exchange, "prices")

        ret_path = os.path.join(DATA_DIR, f"returns_{exchange.lower()}.parquet")
        log_returns(monthly).to_parquet(ret_path)
        print(f"  Saved: {ret_path}")

        if exchange == "Kuwait":
            pre, post = kuwait_split(monthly)
            pre.to_parquet(os.path.join(DATA_DIR, "prices_kuwait_pre_msci.parquet"))
            post.to_parquet(os.path.join(DATA_DIR, "prices_kuwait_post_msci.parquet"))
            print(f"  Kuwait MSCI split: {len(pre)} pre / {len(post)} post months")

    # ── Universe CSV ──────────────────────────────────────────────────────────
    if all_metadata:
        univ = (pd.DataFrame(all_metadata)
                  .sort_values(["exchange", "market_cap"],
                               ascending=[True, False], na_position="last"))
        univ_path = os.path.join(DATA_DIR, "universe.csv")
        univ.to_csv(univ_path, index=False)

        print(f"\n{'='*62}")
        print(f"  Universe saved: {univ_path}")
        in_universe = univ[univ["in_universe"]]
        print(f"  Active universe: {len(in_universe)} stocks  |  "
              f"Total in parquet (incl orphans): {len(univ)}")
        for ex in in_universe["exchange"].unique():
            rows = in_universe[in_universe["exchange"] == ex]
            print(f"    {ex:10s}  {len(rows):>3d} stocks")
        print(f"{'='*62}")

    print("\nDone. Next: python factor_construction.py")
