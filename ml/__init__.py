# ml/__init__.py

# Expose key functions for convenience
from .lstm_model import train_lstm, predict_lstm, save_model, load_lstm_model
from .backtest import backtest_strategy, generate_signal_from_prediction
