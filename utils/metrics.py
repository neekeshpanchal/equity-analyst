# utils/metrics.py
import pandas as pd
import numpy as np

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute advanced equity metrics from raw OHLCV data.

    Args:
        df (pd.DataFrame): Raw data with columns ['date', 'open', 'high', 'low', 'close', 'volume']

    Returns:
        pd.DataFrame: Metrics DataFrame with same index and columns:
            ['date', 'open', 'high', 'low', 'close', 'volume',
             'daily_return', 'cumulative_return', 'volatility', 'sharpe_ratio', 'drawdown',
             'rolling_avg_5', 'rolling_avg_20']
    """
    df = df.copy()
    
    # Ensure proper sorting by date
    df.sort_values('date', inplace=True)

    # Daily return
    df['daily_return'] = df['close'].pct_change()

    # Cumulative return
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1

    # Rolling volatility (standard deviation of daily returns)
    df['volatility'] = df['daily_return'].rolling(window=20).std() * np.sqrt(252)  # annualized

    # Sharpe ratio (assuming risk-free rate ~0)
    df['sharpe_ratio'] = df['daily_return'].rolling(window=20).mean() / df['daily_return'].rolling(window=20).std() * np.sqrt(252)

    # Drawdown (peak-to-trough decline)
    df['rolling_max'] = df['close'].cummax()
    df['drawdown'] = (df['close'] - df['rolling_max']) / df['rolling_max']

    # Simple moving averages
    df['rolling_avg_5'] = df['close'].rolling(window=5).mean()
    df['rolling_avg_20'] = df['close'].rolling(window=20).mean()

    # Cleanup intermediate columns
    df.drop(columns=['rolling_max'], inplace=True)

    # Fill initial NaNs with 0 or reasonable defaults
    df.fillna(0, inplace=True)

    return df
