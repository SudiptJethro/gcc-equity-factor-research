import pandas as pd
from config import MIN_MONTHS_DATA, MIN_AVG_VOLUME, MIN_PRICE


def passes_filters(ticker: str, df: pd.DataFrame) -> tuple[bool, str | None]:
    """
    Run all three quality gates on a raw OHLCV dataframe.
    Returns (passed, reason_for_rejection).
    reason is None when passed=True.
    """
    if df.empty or len(df) < MIN_MONTHS_DATA * 20:
        print(f"  [SKIP] {ticker:22s}  no data or < {MIN_MONTHS_DATA * 20} rows")
        return False, "no data / short history"

    if df["Close"].median() < MIN_PRICE:
        print(f"  [SKIP] {ticker:22s}  median price below {MIN_PRICE}")
        return False, "price below minimum"

    avg_vol = df["Volume"].mean()
    if avg_vol < MIN_AVG_VOLUME:
        print(f"  [SKIP] {ticker:22s}  avg vol {avg_vol:,.0f}")
        return False, f"avg vol {avg_vol:,.0f} < {MIN_AVG_VOLUME}"

    return True, None
