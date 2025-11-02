import os
import socket
import pandas as pd
import yfinance as yf
from utils.logger import get_logger
from config import YFINANCE_AUTO_ADJUST

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
logger = get_logger("finance", "finance.log")

# -----------------------------------------------------------------------------
# Disable all YFinance caching (must happen before any network calls)
# -----------------------------------------------------------------------------
os.environ["YFINANCE_CACHE_DIR"] = "/tmp/yf_cache_disabled"
os.environ["YFINANCE_NO_CACHE"] = "1"
os.makedirs("/tmp/yf_cache_disabled", exist_ok=True)
print(f"📁 Using cache dir: {os.environ['YFINANCE_CACHE_DIR']}")

# -----------------------------------------------------------------------------
# Network check helper
# -----------------------------------------------------------------------------
def _check_network(host="query1.finance.yahoo.com", port=443, timeout=3):
    """Check if Yahoo Finance endpoint is reachable."""
    try:
        socket.create_connection((host, port), timeout=timeout)
        print(f"✅ Network OK: {host}:{port}")
        return True
    except Exception as e:
        print(f"❌ Network unreachable: {e}")
        return False

# -----------------------------------------------------------------------------
# Fetch data
# -----------------------------------------------------------------------------
def fetch_equity_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    print("=" * 80)
    print(f"🔍 Fetching data for {ticker} ({start} → {end})")

    _check_network()
    logger.info(f"Fetching data for {ticker} between {start} and {end}")

    try:
        df = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=YFINANCE_AUTO_ADJUST,
            threads=False
        )

        print(f"📊 Type: {type(df)}, Shape: {df.shape}")
        print(df.head(5))

        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            print(f"⚠️ No data returned for {ticker}.")
            raise ValueError(f"No data found for {ticker} between {start} and {end}")

        df.reset_index(inplace=True)
        df.rename(columns={"Date": "date"}, inplace=True)
        df = df[["date", "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["date", "open", "high", "low", "close", "volume"]

        df["daily_return"] = df["close"].pct_change().fillna(0)
        df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1

        print(f"✅ Done! Rows: {len(df)}")
        print(df.head(3))

        logger.info(f"Fetched {len(df)} rows for {ticker}")
        return df

    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        print(f"💥 Error fetching {ticker}: {e}")
        raise RuntimeError(f"Error fetching data for {ticker}: {e}")
