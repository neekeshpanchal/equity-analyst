# ml/backtest.py
import pandas as pd
import numpy as np

def generate_signal_from_prediction(predicted_price, current_price, threshold_pct=0.5):
    """
    Generate buy/sell/hold signal based on predicted vs current price.
    
    Args:
        predicted_price: float, next predicted price
        current_price: float, current last known price
        threshold_pct: float, minimum % difference to trigger action
    
    Returns:
        int: 1 = buy, -1 = sell, 0 = hold
    """
    diff_pct = ((predicted_price - current_price) / current_price) * 100
    if diff_pct > threshold_pct:
        return 1
    elif diff_pct < -threshold_pct:
        return -1
    else:
        return 0

def backtest_strategy(df: pd.DataFrame, capital=10000):
    """
    Backtest a simple strategy using 'signal' column in df.
    
    Args:
        df: pd.DataFrame with columns ['date', 'close', 'signal']
        capital: initial capital
    
    Returns:
        df: pd.DataFrame with equity curve
        metrics: dict with return %, volatility %, Sharpe, drawdown
    """
    df = df.copy()
    df['position'] = df['signal'].shift(1).fillna(0)  # Apply signal to next day
    df['daily_return'] = df['close'].pct_change().fillna(0)
    df['strategy_return'] = df['position'] * df['daily_return']
    
    # Equity curve
    df['equity_curve'] = (1 + df['strategy_return']).cumprod() * capital
    
    # Metrics
    total_return_pct = ((df['equity_curve'].iloc[-1] / capital) - 1) * 100
    volatility_pct = df['strategy_return'].std() * np.sqrt(252) * 100  # annualized
    sharpe_ratio = (df['strategy_return'].mean() / df['strategy_return'].std()) * np.sqrt(252) if df['strategy_return'].std() != 0 else 0
    
    # Max drawdown
    running_max = df['equity_curve'].cummax()
    drawdown = (df['equity_curve'] - running_max) / running_max
    max_drawdown = drawdown.min() * 100  # as percentage
    
    metrics = {
        'total_return_pct': round(total_return_pct, 2),
        'volatility_pct': round(volatility_pct, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'max_drawdown_pct': round(max_drawdown, 2)
    }
    
    return df, metrics
