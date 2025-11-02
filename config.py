# config.py
import os

# -------------------------------
# Database Configuration
# -------------------------------
DB_NAME = os.environ.get("DB_NAME", "equity_analyst.db")
DATABASE_URI = f"sqlite:///{DB_NAME}"  # For production, you can switch to PostgreSQL/MySQL

# -------------------------------
# Default Table Suffixes
# -------------------------------
EQUITIES_SUFFIX = "_data"
METRICS_SUFFIX = "_metrics"
BACKTEST_SUFFIX = "_backtest"
PREDICTIONS_SUFFIX = "_predictions"

# -------------------------------
# App Settings
# -------------------------------
FLASK_DEBUG = True
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000

# Refresh rates for streaming/auto-fetch (milliseconds)
AUTO_REFRESH_MS = 10000

# -------------------------------
# Other Global Constants
# -------------------------------
YFINANCE_AUTO_ADJUST = True  # Makes sure 'Close' is adjusted
