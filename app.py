# app.py
from flask import Flask, render_template, request, redirect, url_for, jsonify
from utils.finance import fetch_equity_data
from utils.metrics import compute_metrics
from utils.db import (
    list_tables, get_dynamic_table, create_table_if_not_exists, bulk_insert
)
from ml.lstm_model import train_lstm, predict_lstm, save_model, load_lstm_model
from ml.backtest import backtest_strategy, generate_signal_from_prediction
from config import DATABASE_URI
import pandas as pd
from datetime import datetime
from utils.logger import data_logger, backtest_logger

app = Flask(__name__)

# -------------------------------
# DASHBOARD ROUTES
# -------------------------------

@app.route('/')
def home():
    """Landing page redirect to data dashboard"""
    return redirect(url_for('data_dashboard'))

@app.route('/data_dashboard', methods=['GET', 'POST'])
def data_dashboard():
    """
    Dashboard for pulling equity data, computing metrics, storing into DB.
    Users can select the table to append data or create a new one.
    """
    tables = list_tables()
    message = None
    df_preview = None

    if request.method == 'POST':
        ticker = request.form['ticker']
        start = request.form['start']
        end = request.form['end']
        selected_table = request.form.get('table')
        new_table = request.form.get('new_table')

        # Determine table name
        table_name = new_table.strip() if new_table else selected_table
        if not table_name:
            message = "Error: You must select or enter a table name."
        else:
            try:
                # Fetch raw equity data
                df = fetch_equity_data(ticker, start, end)
                df_preview = df.copy()

                # Compute metrics (optional to merge)
                metrics_df = compute_metrics(df)

                # Dynamic table handling
                TableClass = get_dynamic_table(BaseEquities, table_name)
                create_table_if_not_exists(TableClass)

                # Insert raw data (optionally merge metrics)
                bulk_insert(TableClass, df.to_dict(orient='records'))

                message = f"✅ Data for {ticker} saved to table '{table_name}'!"

            except Exception as e:
                message = f"Error: {str(e)}"

    return render_template(
        'data_dashboard.html',
        tables=tables,
        message=message,
        df=df_preview
    )


@app.route('/backtest_dashboard', methods=['GET', 'POST'])
def backtest_dashboard():
    """
    Dashboard for predictive ML models and backtesting.
    Users select ticker, table, model, date range.
    """
    tables = list_tables()
    result = None
    chart_data = None

    if request.method == 'POST':
        ticker = request.form['ticker']
        table_name = request.form['table']
        start = request.form['start']
        end = request.form['end']
        model_name = request.form['model']

        if not table_name:
            result = {'error': "Please select a valid table."}
        else:
            try:
                # Get table dynamically
                TableClass = get_dynamic_table(BaseEquities, table_name)

                # Query data
                records = query_table(TableClass, filters={'ticker': ticker})
                df = pd.DataFrame([{
                    'date': r.date,
                    'open': r.open,
                    'high': r.high,
                    'low': r.low,
                    'close': r.close,
                    'volume': r.volume
                } for r in records])
                
                # Filter by date range
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= pd.to_datetime(start)) & (df['date'] <= pd.to_datetime(end))]
                df.sort_values('date', inplace=True)

                if df.empty:
                    raise ValueError("No data available for this selection.")

                # Train or load LSTM model
                model_path = f"ml_models/{model_name}.h5"
                if not os.path.exists(model_path):
                    model, scaler = train_lstm(df)
                    save_model(model, model_name)
                else:
                    model = load_lstm_model(model_name)
                    scaler = None  # Ideally save/load scaler too

                # Generate prediction & signal
                last_price = df['close'].iloc[-1]
                predicted_price = predict_lstm(model, scaler, df)
                signal = generate_signal_from_prediction(predicted_price, last_price)
                df['signal'] = signal

                # Backtest
                backtested_df, metrics = backtest_strategy(df)

                chart_data = backtested_df.to_dict(orient='records')
                result = metrics

            except Exception as e:
                result = {'error': str(e)}

    return render_template(
        'backtest_dashboard.html',
        tables=tables,
        result=result,
        chart_data=chart_data
    )


# -------------------------------
# API Endpoints (Optional)
# -------------------------------
@app.route('/api/fetch_metrics', methods=['POST'])
def api_fetch_metrics():
    """
    Example API endpoint for fetching and storing data via JSON
    """
    data = request.json
    ticker = data.get('ticker')
    start = data.get('start')
    end = data.get('end')
    table_name = data.get('table')

    df = fetch_equity_data(ticker, start, end)
    metrics_df = compute_metrics(df)

    TableClass = get_dynamic_table(BaseEquities, table_name)
    create_table_if_not_exists(TableClass)
    bulk_insert(TableClass, df.to_dict(orient='records'))

    return jsonify({'status': 'success', 'rows': len(df)})


# -------------------------------
# Run Flask
# -------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
