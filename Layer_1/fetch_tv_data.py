"""
Supplemental data fetch via TradingView (tvDatafeed).
Fills gaps for ADX and DFM stocks that yfinance / Twelve Data cannot reach.
Run AFTER main.py / reads existing parquets and merges new data in.

Usage:  python fetch_tv_data.py
"""

import sys
import os
import time
import pandas as pd
import numpy as np

try:
    from tvDatafeed import TvDatafeed, Interval
except ImportError:
    raise SystemExit(
        "tvDatafeed not found. Install via:\n"
        "  pip install git+https://github.com/rongardF/tvdatafeed.git"
    )

from config import DATA_DIR, START_DATE, END_DATE, MIN_MONTHS_DATA, MIN_AVG_VOLUME, MIN_PRICE
from returns import log_returns

# ──────────────────────────────────────────────────────────────
# TV symbol map: (original_ticker, exchange, tv_exchange, tv_symbol)
# ──────────────────────────────────────────────────────────────
TV_SYMBOLS = [
    # ── ADX / all tickers (yfinance dead, Twelve Data paywalled) ─────────────
    # Existing in parquet but stale at 2025-12-30 → TV will refresh to today
    ("IHC.AD",           "ADX", "ADX", "IHC"),
    ("TAQA.AD",          "ADX", "ADX", "TAQA"),
    ("ADNOCGAS.AD",      "ADX", "ADX", "ADNOCGAS"),
    ("FAB.AD",           "ADX", "ADX", "FAB"),
    ("EAND.AD",          "ADX", "ADX", "EAND"),
    ("ADCB.AD",          "ADX", "ADX", "ADCB"),
    ("ADNOCDRILL.AD",    "ADX", "ADX", "ADNOCDRILL"),
    ("ALPHADHABI.AD",    "ADX", "ADX", "ALPHADHABI"),
    ("BOROUGE.AD",       "ADX", "ADX", "BOROUGE"),
    ("ADIB.AD",          "ADX", "ADX", "ADIB"),
    ("ALDAR.AD",         "ADX", "ADX", "ALDAR"),
    ("ADNOCDIST.AD",     "ADX", "ADX", "ADNOCDIST"),
    ("ADNOCLS.AD",       "ADX", "ADX", "ADNOCLS"),
    ("FERTIGLB.AD",      "ADX", "ADX", "FERTIGLB"),
    ("PUREHEALTH.AD",    "ADX", "ADX", "PUREHEALTH"),
    ("ADPORTS.AD",       "ADX", "ADX", "ADPORTS"),
    ("PRESIGHT.AD",      "ADX", "ADX", "PRESIGHT"),
    ("RAKBANK.AD",       "ADX", "ADX", "RAKBANK"),
    ("NMDC.AD",          "ADX", "ADX", "NMDC"),
    ("AMR.AD",           "ADX", "ADX", "AMR"),
    ("AGILITY.AD",       "ADX", "ADX", "AGILITY"),
    ("BURJEEL.AD",       "ADX", "ADX", "BURJEEL"),
    ("DANA.AD",          "ADX", "ADX", "DANA"),
    ("ADNH.AD",          "ADX", "ADX", "ADNH"),
    ("ADNIC.AD",         "ADX", "ADX", "ADNIC"),
    ("WAHA.AD",          "ADX", "ADX", "WAHA"),
    ("AGTHIA.AD",        "ADX", "ADX", "AGTHIA"),
    ("RAKCEC.AD",        "ADX", "ADX", "RAKCEC"),
    ("ADSB.AD",          "ADX", "ADX", "ADSB"),
    # New ADX listings
    ("MODON.AD",         "ADX", "ADX", "MODON"),
    ("LULU.AD",          "ADX", "ADX", "LULU"),
    ("NMDCENR.AD",       "ADX", "ADX", "NMDCENR"),
    ("SPACE42.AD",       "ADX", "ADX", "SPACE42"),
    ("ALEFEDT.AD",       "ADX", "ADX", "ALEFEDT"),
    ("EMSTEEL.AD",       "ADX", "ADX", "EMSTEEL"),
    ("APEX.AD",          "ADX", "ADX", "APEX"),
    ("PHX.AD",           "ADX", "ADX", "PHX"),
    ("ALPHADATA.AD",     "ADX", "ADX", "ALPHADATA"),
    ("ADNHC.AD",         "ADX", "ADX", "ADNHC"),
    ("NBF.AD",           "ADX", "ADX", "NBF"),
    ("INVICTUS.AD",      "ADX", "ADX", "INVICTUS"),
    ("PALMS.AD",         "ADX", "ADX", "PALMS"),
    ("SAGA.AD",          "ADX", "ADX", "SAGA"),
    ("MBME.AD",          "ADX", "ADX", "MBME"),
    # ── DFM / stocks still missing after yfinance .AE alias attempts ──────────
    ("DAMAC.DU",         "DFM", "DFM", "DAMAC"),
    ("EMAARMALLS.DU",    "DFM", "DFM", "EMAARMALLS"),
    ("GGICO.DU",         "DFM", "DFM", "GGICO"),
    ("UNIONPROPERTIES.DU","DFM","DFM", "UNIONPROPERTIES"),
    ("GFIN.DU",          "DFM", "DFM", "GFIN"),
    ("EMAARIP.DU",       "DFM", "DFM", "EMAARIP"),
    ("ALMADINA.DU",      "DFM", "DFM", "ALMADINA"),
    ("PARKIN.DU",        "DFM", "DFM", "PARKIN"),
    ("DUCAB.DU",         "DFM", "DFM", "DUCAB"),
    ("DUBAIHOLDING.DU",  "DFM", "DFM", "DUBAIHOLDING"),
]


def fetch_tv(tv: TvDatafeed, tv_exchange: str, tv_symbol: str,
             start: str, end: str, n_bars: int = 5000,
             retries: int = 3) -> pd.DataFrame:
    """Download daily OHLCV from TradingView and slice to [start, end]."""
    for attempt in range(retries):
        try:
            df = tv.get_hist(tv_symbol, tv_exchange,
                             interval=Interval.in_daily, n_bars=n_bars)
            if df is not None and not df.empty:
                break
            if attempt < retries - 1:
                time.sleep(5)
        except Exception as e:
            print(f"\n    [TV retry {attempt+1}] {e}", end="")
            if attempt < retries - 1:
                time.sleep(5)
    else:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    try:
        # Normalize: strip time component so index aligns with yfinance date-only index
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = "Date"
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]  # collapse any same-day duplicates
        df = df.loc[start:end]

        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if "close" in cl:
                col_map[c] = "Close"
            elif "volume" in cl:
                col_map[c] = "Volume"
        df = df.rename(columns=col_map)

        if "Close" not in df.columns:
            return pd.DataFrame()
        if "Volume" not in df.columns:
            df["Volume"] = 0.0

        return df[["Close", "Volume"]].copy()

    except Exception as e:
        print(f"    [TV error] {tv_exchange}:{tv_symbol}: {e}")
        return pd.DataFrame()


def passes_filters(ticker: str, df: pd.DataFrame) -> tuple[bool, str | None]:
    min_rows = MIN_MONTHS_DATA * 20
    if df.empty or len(df) < min_rows:
        return False, f"too few rows ({len(df)} < {min_rows})"
    if df["Close"].median() < MIN_PRICE:
        return False, f"price too low (median {df['Close'].median():.4f})"
    if df["Volume"].mean() < MIN_AVG_VOLUME:
        return False, f"volume too low (avg {df['Volume'].mean():,.0f})"
    return True, None


def update_exchange_parquets(exchange: str,
                             new_daily: dict[str, pd.Series],
                             new_volume: dict[str, pd.Series],
                             metadata_rows: list[dict]) -> None:
    """Merge new ticker data into existing exchange parquets."""
    # Load existing
    daily_path   = os.path.join(DATA_DIR, f"daily_{exchange.lower()}.parquet")
    monthly_path = os.path.join(DATA_DIR, f"prices_{exchange.lower()}.parquet")
    ret_path     = os.path.join(DATA_DIR, f"returns_{exchange.lower()}.parquet")

    if os.path.exists(daily_path):
        daily_existing = pd.read_parquet(daily_path)
    else:
        daily_existing = pd.DataFrame()

    # Merge: overwrite full column (TV covers full history; refreshes stale data)
    for ticker, series in new_daily.items():
        if ticker in daily_existing.columns and not daily_existing[ticker].dropna().empty:
            # Concat existing + TV, deduplicate keeping TV value on overlap
            merged = pd.concat([daily_existing[ticker], series]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            daily_existing[ticker] = merged
        else:
            daily_existing[ticker] = series

    daily_existing.index = pd.to_datetime(daily_existing.index)
    daily_existing = daily_existing.sort_index()

    # Resample to monthly / no sparse-month filter here;
    # each ticker's NaN window reflects its actual listing date.
    monthly = daily_existing.resample("ME").last()

    # Save
    daily_existing.to_parquet(daily_path)
    monthly.to_parquet(monthly_path)
    log_returns(monthly).to_parquet(ret_path)
    print(f"  [{exchange}] daily={daily_existing.shape}  monthly={monthly.shape}")


def main():
    print("TradingView supplemental fetch")
    print(f"Period: {START_DATE} -> {END_DATE}\n")

    tv = TvDatafeed()
    time.sleep(2)

    exchanges = list(dict.fromkeys(e for _, e, _, _ in TV_SYMBOLS))
    new_daily:   dict[str, dict[str, pd.Series]] = {e: {} for e in exchanges}
    new_volume:  dict[str, dict[str, pd.Series]] = {e: {} for e in exchanges}
    meta_rows:   list[dict] = []

    for orig_ticker, exchange, tv_exch, tv_sym in TV_SYMBOLS:
        print(f"  {tv_exch}:{tv_sym:15s}", end="  ", flush=True)
        df = fetch_tv(tv, tv_exch, tv_sym, START_DATE, END_DATE)
        time.sleep(1.5)

        if df.empty:
            print("EMPTY")
            continue

        passed, reason = passes_filters(orig_ticker, df)
        if not passed:
            print(f"FILTERED ({reason})")
            continue

        print(f"OK  {len(df):>4d} rows  vol {df['Volume'].mean():>10,.0f}")
        new_daily[exchange][orig_ticker]  = df["Close"]
        new_volume[exchange][orig_ticker] = df["Volume"]

        monthly_series = df["Close"].resample("ME").last().dropna()
        meta_rows.append({
            "ticker":       orig_ticker,
            "exchange":     exchange,
            "source":       "TradingView",
            "daily_rows":   int(df["Close"].dropna().__len__()),
            "monthly_rows": int(monthly_series.__len__()),
            "first_date":   str(monthly_series.index[0]) if len(monthly_series) else "",
            "last_date":    str(monthly_series.index[-1]) if len(monthly_series) else "",
            "avg_volume":   float(df["Volume"].mean()),
            "market_cap":   None,
        })

    print()
    for exchange in exchanges:
        if new_daily[exchange]:
            print(f"Updating {exchange} parquets ({len(new_daily[exchange])} new tickers)...")
            update_exchange_parquets(exchange, new_daily[exchange],
                                     new_volume[exchange], meta_rows)

    # Append to universe.csv
    if meta_rows:
        univ_path = os.path.join(DATA_DIR, "universe.csv")
        if os.path.exists(univ_path):
            existing = pd.read_csv(univ_path)
            new_df   = pd.DataFrame(meta_rows)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["ticker"], keep="last")
            combined.to_csv(univ_path, index=False)
        else:
            pd.DataFrame(meta_rows).to_csv(univ_path, index=False)

        print(f"\nAdded {len(meta_rows)} tickers via TradingView.")

    # Summary
    univ = pd.read_csv(os.path.join(DATA_DIR, "universe.csv"))
    print(f"\nTotal universe: {len(univ)} stocks")
    for ex in univ["exchange"].unique():
        n = len(univ[univ["exchange"] == ex])
        print(f"  {ex:10s} {n:>3d} stocks")


if __name__ == "__main__":
    main()
