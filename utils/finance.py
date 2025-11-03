import os
import socket
import pathlib
import pandas as pd
import yfinance as yf
from utils.logger import get_logger
from config import YFINANCE_AUTO_ADJUST

# =============================================================================
# 📦 YFinance Cache Initialization
# =============================================================================
def _init_yfinance_cache(enable_cache: bool = False):
    """Initialize a safe, writable yfinance cache path."""
    try:
        if enable_cache:
            cache_dir = pathlib.Path("/tmp/yfinance_cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(cache_dir, 0o777)
            os.environ["XDG_CACHE_HOME"] = str(cache_dir)
            os.environ["YFINANCE_CACHE_DIR"] = str(cache_dir)
            os.environ["YFINANCE_NO_CACHE"] = "0"
            print(f"📦 YFinance cache ENABLED at: {cache_dir}")
        else:
            disable_dir = pathlib.Path("/tmp/yf_cache_disabled")
            disable_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(disable_dir, 0o777)
            os.environ["YFINANCE_CACHE_DIR"] = str(disable_dir)
            os.environ["YFINANCE_NO_CACHE"] = "1"
            print(f"🧹 YFinance cache DISABLED. Using: {disable_dir}")

        version = getattr(yf, "__version__", "unknown")
        print(f"🔢 yfinance version: {version}")

    except Exception as e:
        print(f"💥 Cache init failed: {e}")
        os.environ["YFINANCE_NO_CACHE"] = "1"

# Initialize cache before logger
_init_yfinance_cache(enable_cache=False)

# =============================================================================
# 🧾 Logger Initialization
# =============================================================================
logger = get_logger("finance", "finance.log")

# =============================================================================
# 🌐 Network Connectivity Check
# =============================================================================
def _check_network(host="query1.finance.yahoo.com", port=443, timeout=3):
    """Verify Yahoo Finance endpoint is reachable."""
    try:
        socket.create_connection((host, port), timeout=timeout)
        print(f"✅ Network OK: {host}:{port}")
        return True
    except Exception as e:
        print(f"❌ Network unreachable: {e}")
        return False

# =============================================================================
# 📈 Fetch Equity Data (Robust)
# =============================================================================
def fetch_equity_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch and clean equity OHLCV data from Yahoo Finance.
    Handles single- and multi-index formats robustly.
    """
    print("=" * 80)
    print(f"🔍 Fetching data for {ticker} ({start} → {end})")

    _check_network()
    logger.info(f"Fetching data for {ticker} between {start} and {end}")

    try:
        # --- Fetch raw data ---
        df = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=YFINANCE_AUTO_ADJUST,
            threads=False,
            group_by="ticker"
        )

        print(f"📊 Raw df type: {type(df)}, shape: {df.shape}")
        print(f"📄 Raw columns: {df.columns}")

        if df.empty:
            raise ValueError(f"No data found for {ticker} between {start} and {end}")

        # --- Flatten MultiIndex columns ---
        if isinstance(df.columns, pd.MultiIndex):
            if df.columns.nlevels == 2:
                df.columns = [col[1] if col[0] == ticker else col[0] for col in df.columns]
            else:
                df.columns = df.columns.get_level_values(-1)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # --- Ensure date column exists ---
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.rename(columns={df.columns[0]: "date"}, inplace=True)
        elif "date" in df.columns:
            pass
        elif "datetime" in df.columns:
            df.rename(columns={"datetime": "date"}, inplace=True)
        else:
            raise ValueError("No valid date column or index found.")

        # --- Identify OHLCV columns dynamically ---
        candidates = {
            "open": next((c for c in df.columns if "open" in c), None),
            "high": next((c for c in df.columns if "high" in c), None),
            "low": next((c for c in df.columns if "low" in c), None),
            "close": next((c for c in df.columns if "close" in c and "adj" not in c), None),
            "volume": next((c for c in df.columns if "volume" in c), None),
        }

        missing = [k for k, v in candidates.items() if v is None]
        if missing:
            raise ValueError(f"Missing expected OHLCV columns: {missing}. Found: {df.columns.tolist()}")

        # --- Standardize final columns ---
        df = df[["date", candidates["open"], candidates["high"],
                 candidates["low"], candidates["close"], candidates["volume"]]]
        df.columns = ["date", "open", "high", "low", "close", "volume"]

        # --- Derived columns ---
        df["daily_return"] = df["close"].pct_change().fillna(0)
        df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1

        print(f"✅ Done! Rows: {len(df)}")
        print(df.head(3))
        logger.info(f"✅ Successfully fetched {len(df)} rows for {ticker}")

        return df

    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        print(f"💥 Error fetching {ticker}: {e}")
        raise RuntimeError(f"Error fetching data for {ticker}: {e}")
