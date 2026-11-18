"""
GCC Equity Factor Research Engine  /  v2
Layer 1: Data Pipeline
-----------------------
Five-country GCC universe:
  Saudi Arabia  (Tadawul)  .SR   ~80 stocks
  UAE Abu Dhabi (ADX)      .AD   ~20 stocks
  UAE Dubai     (DFM)      .DU   ~20 stocks
  Qatar         (QSE)      .QA   ~15 stocks
  Kuwait        (BK)       .KW   ~10 stocks

Run:    python main.py
Output: data/prices_{exchange}.parquet   (monthly, adjusted)
        data/returns_{exchange}.parquet  (monthly log returns)
        data/daily_{exchange}.parquet    (daily, for vol computation)
        data/universe.csv               (filtered universe + metadata)
"""

import os
import pandas as pd

from config import DATA_DIR, START_DATE, END_DATE, KUWAIT_MSCI_UPGRADE
from universe import ALL_EXCHANGES, EXCHANGE_META
from pipeline import build_exchange_data
from returns import log_returns, kuwait_split


if __name__ == "__main__":
    print("\nGCC Factor Research Engine  /  v2  (Five-country universe)")
    print(f"Period: {START_DATE}  ->  {END_DATE}\n")
    print("Universe coverage:")
    for ex, meta in EXCHANGE_META.items():
        print(f"  {ex:10s} | {meta['country']:14s} | MSCI: {meta['msci_status']}")

    all_metadata     = []
    exchange_monthly = {}
    exchange_daily   = {}

    for exchange, tickers in ALL_EXCHANGES.items():
        monthly, daily, meta = build_exchange_data(exchange, tickers)

        if monthly.empty:
            continue

        exchange_monthly[exchange] = monthly
        exchange_daily[exchange]   = daily
        all_metadata.extend(meta)

        # Save monthly prices
        path = os.path.join(DATA_DIR, f"prices_{exchange.lower()}.parquet")
        monthly.to_parquet(path)
        print(f"\n  Saved: {path}  "
              f"({monthly.shape[0]} months × {monthly.shape[1]} stocks)")

        # Save monthly log returns
        ret_path = os.path.join(DATA_DIR, f"returns_{exchange.lower()}.parquet")
        log_returns(monthly).to_parquet(ret_path)
        print(f"  Saved: {ret_path}")

        # Save daily prices (for vol factor computation)
        d_path = os.path.join(DATA_DIR, f"daily_{exchange.lower()}.parquet")
        daily.to_parquet(d_path)
        print(f"  Saved: {d_path}  ({daily.shape[0]} days × {daily.shape[1]} stocks)")

        # Kuwait: save pre/post MSCI split
        if exchange == "Kuwait":
            pre, post = kuwait_split(monthly)
            pre.to_parquet(os.path.join(DATA_DIR, "prices_kuwait_pre_msci.parquet"))
            post.to_parquet(os.path.join(DATA_DIR, "prices_kuwait_post_msci.parquet"))
            print(f"  Kuwait MSCI split: {len(pre)} pre-upgrade months, "
                  f"{len(post)} post-upgrade months")

    if all_metadata:
        univ = (pd.DataFrame(all_metadata)
                  .sort_values(["exchange", "market_cap"],
                               ascending=[True, False],
                               na_position="last"))
        univ_path = os.path.join(DATA_DIR, "universe.csv")
        univ.to_csv(univ_path, index=False)

        print(f"\n{'='*56}")
        print(f"  Universe saved: {univ_path}")
        print(f"  Total stocks in universe: {len(univ)}")
        print()
        for ex in univ["exchange"].unique():
            rows = univ[univ["exchange"] == ex]
            print(f"    {ex:10s} | {EXCHANGE_META[ex]['country']:14s} | "
                  f"{len(rows):>3d} stocks | "
                  f"MSCI: {EXCHANGE_META[ex]['msci_status']}")
        print(f"{'='*56}")

    print("\nResearch note statistics (include in methodology section):")
    print(f"  Study period:      {START_DATE} to {END_DATE}")
    print(f"  Exchanges covered: {len(ALL_EXCHANGES)} (Oman & Bahrain excluded: "
          f"insufficient liquidity)")
    print(f"  Exclusion rationale: published in methodology per Harvey et al. (2016)")
    print(f"  Kuwait natural experiment: MSCI upgrade {KUWAIT_MSCI_UPGRADE}")
    print()
    print("Next step: python factor_construction.py")
