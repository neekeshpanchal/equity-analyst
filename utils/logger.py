import logging
from logging.handlers import RotatingFileHandler
import os

# -------------------------------
# Log Directory Setup
# -------------------------------
# Always resolve to /app/logs inside the container
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Ensure /app/logs exists
os.makedirs(LOG_DIR, exist_ok=True)

# -------------------------------
# Logger Factory
# -------------------------------
def get_logger(name, filename, level=logging.INFO, max_bytes=5 * 1024 * 1024, backup_count=3):
    """
    Returns a configured rotating logger that is safe for Docker usage.
    - name: logger name
    - filename: log file name (e.g. 'finance.log')
    - level: logging level
    - max_bytes: maximum file size before rotation
    - backup_count: number of rotated logs to keep
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Compose full log file path (always inside /app/logs)
    log_path = os.path.join(LOG_DIR, filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Prevent duplicate handlers (important for repeated imports)
    if not logger.handlers:
        logger.addHandler(handler)
        print(f"[LOGGER] Initialized '{name}' logger at: {log_path}")

    return logger

# -------------------------------
# Predefined App Loggers
# -------------------------------
data_logger = get_logger('data_logger', 'data_operations.log')
backtest_logger = get_logger('backtest_logger', 'backtest_operations.log')
