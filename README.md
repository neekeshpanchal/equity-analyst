# Equity Analysis Dashboard

A modern web-based application for fetching, storing, and analyzing equity data.
It provides a responsive, database-backed dashboard with data visualization, performance metrics, and optional AI-based analytical summaries.

---

## Overview

The **Equity Analysis Dashboard** is a Flask-based web app that allows users to:

* Fetch historical stock data directly from Yahoo Finance.
* Store and organize data in an internal SQLite database.
* View structured tables of results and computed metrics.
* Optionally generate short AI-driven interpretations of performance statistics.
* Operate within a modern, dark-themed, interactive UI optimized for browser use.

---

## Key Features

* **Automated Data Retrieval** – Query stock data for any ticker and date range using Yahoo Finance.
* **Database Management** – Save data to existing or new tables with a single form submission.
* **Interactive Data Preview** – View results instantly in a formatted, scrollable table.
* **Backtesting Metrics** – Compute and display total return, volatility, Sharpe ratio, and max drawdown.
* **AI Summary Integration (Optional)** – Generate concise insights using an open-source language model (Hugging Face).
* **Containerized Deployment** – Fully Dockerized setup for consistent, portable execution across environments.
* **GPU Compatibility** – Supports GPU acceleration if available, otherwise defaults to CPU execution.

---

## System Architecture

| Layer                | Technology                    |
| -------------------- | ----------------------------- |
| Frontend             | HTML, TailwindCSS, JavaScript |
| Backend              | Flask                         |
| Data Layer           | yfinance, pandas              |
| Database             | SQLite                        |
| AI Engine (optional) | Hugging Face Transformers     |
| Containerization     | Docker                        |

---

## Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/equity-analysis-dashboard.git
cd equity-analysis-dashboard
```

### 2. Build the Docker Image

```bash
docker build -t equity-analysis .
```

### 3. Run the Container

**With GPU:**

```bash
docker run --gpus all -p 5000:5000 equity-analysis
```

**Without GPU:**

```bash
docker run -p 5000:5000 equity-analysis
```

### 4. Access the Dashboard

Open your browser and navigate to:

```
http://localhost:5000
```

Enter a ticker (e.g., `AAPL`), select start and end dates, specify a table name, and run **Fetch & Save** to view and store data.

---

## Example Usage

```text
Ticker: MSFT
Start: 2024-01-01
End: 2025-01-01
Table: msft_2024
```

The app retrieves Microsoft’s price data for the specified range, stores it in SQLite, and displays:

* Daily open, high, low, close, and volume data.
* Computed statistics (total return, volatility, Sharpe ratio, max drawdown).
* AI summary (if enabled).

---

## Logs

Operational logs are stored within the container:

```
/app/logs/
├── data_operations.log
├── backtest_operations.log
└── finance.log
```

---

## Local Execution (Without Docker)

```bash
pip install -r requirements.txt
python app.py
```

Then visit [http://localhost:5000](http://localhost:5000).

---

## Notes

* Requires Python 3.10+
* GPU mode depends on proper CUDA and PyTorch installation.
* For AI summaries, ensure the selected Hugging Face model is publicly accessible.
