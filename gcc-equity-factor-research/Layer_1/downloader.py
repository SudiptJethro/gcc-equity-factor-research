
# ─────────────────────────────────────────────────────────────
# 3.  DOWNLOAD  (with retry + rate-limit backoff)
# ─────────────────────────────────────────────────────────────

import os
import time
import requests
import pandas as pd
import yfinance as yf

# Set via environment:
#   $env:TWELVE_DATA_API_KEY = "your_key"   (get free key at twelvedata.com)
#   $env:AV_API_KEY          = "your_key"   (Alpha Vantage / doesn't cover UAE)
TD_API_KEY  = os.environ.get("TWELVE_DATA_API_KEY", "")
AV_API_KEY  = os.environ.get("AV_API_KEY", "")
TD_BASE_URL = "https://api.twelvedata.com/time_series"
AV_BASE_URL = "https://www.alphavantage.co/query"


def download_twelve_data(
    ticker: str,
    start: str,
    end: str,
    td_exchange: str = "",
    td_symbol: str = "",
) -> pd.DataFrame:
    """
    Fetch adjusted daily OHLCV from Twelve Data.
    Free tier: 800 credits/day, 8 calls/min / caller must throttle.
    td_exchange / td_symbol override the default (stripped) symbol if supplied.
    """
    if not TD_API_KEY:
        return pd.DataFrame()

    symbol = td_symbol if td_symbol else ticker.split(".")[0]
    params = {
        "symbol":     symbol,
        "interval":   "1day",
        "start_date": start,
        "end_date":   end,
        "outputsize": 5000,
        "order":      "ASC",
        "apikey":     TD_API_KEY,
    }
    if td_exchange:
        params["exchange"] = td_exchange

    try:
        resp = requests.get(TD_BASE_URL, params=params, timeout=15)
        data = resp.json()

        if data.get("status") == "error":
            print(f"  [TD]   {ticker}: {data.get('message','')[:80]}")
            return pd.DataFrame()

        values = data.get("values", [])
        if not values:
            return pd.DataFrame()

        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        df = df.rename(columns={"close": "Close", "volume": "Volume"})
        df["Close"]  = pd.to_numeric(df["Close"],  errors="coerce")
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
        return df[["Close", "Volume"]].copy()

    except Exception as e:
        print(f"  [TD]   {ticker}: exception {e}")
        return pd.DataFrame()


def download_alpha_vantage(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fallback via Alpha Vantage (TIME_SERIES_DAILY_ADJUSTED).
    Note: this endpoint is premium-only on AV; UAE/GCC stocks are not covered.
    Kept here for completeness / unlikely to return data for GCC names.
    """
    if not AV_API_KEY:
        return pd.DataFrame()

    symbol = ticker.split(".")[0]
    params = {
        "function":   "TIME_SERIES_DAILY_ADJUSTED",
        "symbol":     symbol,
        "outputsize": "full",
        "apikey":     AV_API_KEY,
    }
    try:
        resp = requests.get(AV_BASE_URL, params=params, timeout=15)
        data = resp.json()
        ts   = data.get("Time Series (Daily)", {})
        if not ts:
            note = data.get("Note") or data.get("Information") or data.get("Error Message", "")
            if note:
                print(f"  [AV]   {ticker}: {note[:80]}")
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(ts, orient="index", dtype=float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().loc[start:end]
        close_col = "5. adjusted close" if "5. adjusted close" in df.columns else "4. close"
        vol_col   = "6. volume"         if "6. volume"         in df.columns else "5. volume"
        df = df[[close_col, vol_col]].rename(columns={close_col: "Close", vol_col: "Volume"})
        return df

    except Exception as e:
        print(f"  [AV]   {ticker}: exception {e}")
        return pd.DataFrame()


# Yahoo Finance uses different suffixes than our canonical universe for some exchanges.
# DFM canonical suffix is .DU (stockanalysis / Bloomberg convention) but yfinance
# serves DFM under .AE. Try the alias before falling back to Twelve Data.
_YF_SUFFIX_ALIAS = {".DU": ".AE"}


def _yf_download(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Try yfinance for ticker, then for any known suffix alias."""
    candidates = [ticker]
    for old_sfx, new_sfx in _YF_SUFFIX_ALIAS.items():
        if ticker.endswith(old_sfx):
            candidates.append(ticker[: -len(old_sfx)] + new_sfx)

    for sym in candidates:
        try:
            df = yf.download(sym, start=start, end=end,
                             auto_adjust=True, progress=False, timeout=10)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df[["Close", "Volume"]].copy()
        except Exception:
            pass
    return pd.DataFrame()


def download_ticker(
    ticker: str,
    start: str,
    end: str,
    retries: int = 2,
    td_exchange: str = "",
    td_symbol: str = "",
) -> pd.DataFrame:
    """
    Download adjusted OHLCV.
    Order: yfinance (+ suffix alias) -> Twelve Data -> Alpha Vantage (last resort).
    td_exchange / td_symbol are forwarded to Twelve Data when provided.
    """
    for attempt in range(retries + 1):
        df = _yf_download(ticker, start, end)
        if not df.empty:
            return df
        if attempt < retries:
            time.sleep(3 + attempt * 2)

    # yfinance returned nothing / try Twelve Data
    if TD_API_KEY:
        print(f"  [TD]   {ticker}: yfinance empty, trying Twelve Data...")
        time.sleep(8)   # stay under 8 req/min free-tier limit
        df = download_twelve_data(ticker, start, end, td_exchange, td_symbol)
        if not df.empty:
            return df

    # Last resort: Alpha Vantage (premium endpoint / usually fails for GCC)
    if AV_API_KEY:
        time.sleep(12)
        return download_alpha_vantage(ticker, start, end)

    return pd.DataFrame()
