import os
import joblib
import logging
from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
from datetime import datetime
from sqlalchemy import String, Float, Column
from utils.finance import fetch_equity_data
from utils.db import (
    list_tables, get_dynamic_table, create_table_if_not_exists,
    bulk_insert, query_table, BaseEquities
)
from ml.lstm_model import train_lstm, predict_lstm, save_model, load_lstm_model
from ml.backtest import backtest_strategy, generate_signal_from_prediction
import numpy as np

# Hugging Face LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# -------------------------------
# FLASK APP SETUP
# -------------------------------
app = Flask(__name__)

# -------------------------------
# LOGGING CONFIGURATION
# -------------------------------
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------------
# CONFIG
# -------------------------------
ML_MODELS_DIR = "ml_models"
os.makedirs(ML_MODELS_DIR, exist_ok=True)
HF_MODEL_NAME = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"  
tokenizer = None
llm_model = None

def init_llm_model():
    global tokenizer, llm_model
    if tokenizer is None or llm_model is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, use_fast=False)
            device = "cpu"  
            llm_model = AutoModelForCausalLM.from_pretrained(
                HF_MODEL_NAME,
                device_map=None,          
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to(device)
            logger.info(f"✅ LLM loaded successfully on {device}")
        except Exception as e:
            logger.error("Failed to load LLM", exc_info=True)
            tokenizer, llm_model = None, None


def interpret_backtest_metrics(total_return, volatility, sharpe, max_drawdown):
    global tokenizer, llm_model
    try:
        if tokenizer is None or llm_model is None:
            init_llm_model()

        # fallback if still unavailable
        if tokenizer is None or llm_model is None:
            return [
                f"⚠️ AI model not loaded. Here is a summary:",
                f"- Total return: {total_return}%, decent performance",
                f"- Volatility: {volatility}%, shows risk level",
                f"- Sharpe ratio: {sharpe}, risk-adjusted performance",
                f"- Max drawdown: {max_drawdown}%, maximum risk exposure"
            ]

        prompt = f"""
Explain these backtest metrics in concise, simple English with 3-5 bullet points:

Total Return %: {total_return}
Volatility %: {volatility}
Sharpe Ratio: {sharpe}
Max Drawdown %: {max_drawdown}
"""
        inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device)
        with torch.no_grad():
            output_ids = llm_model.generate(
                inputs["input_ids"],
                max_new_tokens=150,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        insights = [s.strip() for s in output_text.split("\n") if s.strip()]
        return insights if insights else ["⚠️ AI did not generate meaningful insights."]
    except Exception as e:
        logger.error(f"AI interpretation failed", exc_info=True)
        return [f"⚠️ AI interpretation failed: {e}"]



# -------------------------------
# DATA DASHBOARD
# -------------------------------
@app.route('/', methods=['GET'])
def home():
    return redirect(url_for('data_dashboard'))

@app.route('/data_dashboard', methods=['GET', 'POST'])
def data_dashboard():
    tables = list_tables()
    message = None
    df_preview = None

    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        start = request.form['start']
        end = request.form['end']
        selected_table = request.form.get('table')
        new_table = request.form.get('new_table')

        table_name = new_table.strip() if new_table else selected_table
        if not table_name:
            message = "⚠️ Select or enter a table name."
        else:
            try:
                df = fetch_equity_data(ticker, start, end)
                if df.empty:
                    raise ValueError("No data fetched for this ticker/date range.")

                df['ticker'] = ticker
                df['daily_return'] = df['close'].pct_change().fillna(0)
                df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
                df['date'] = pd.to_datetime(df['date']).dt.to_pydatetime()

                extra_columns = {
                    'ticker': Column('ticker', String, nullable=False),
                    'daily_return': Column('daily_return', Float),
                    'cumulative_return': Column('cumulative_return', Float)
                }

                TableClass = get_dynamic_table(BaseEquities, table_name, extra_columns=extra_columns)
                create_table_if_not_exists(TableClass)

                valid_cols = set(TableClass.__table__.columns.keys())
                data_to_insert = [{k: v for k, v in row.items() if k in valid_cols} for row in df.to_dict(orient='records')]
                bulk_insert(TableClass, data_to_insert)

                df_preview = df.copy()
                message = f"✅ Data for {ticker} saved to '{table_name}' successfully."
                logger.info(message)

            except Exception as e:
                message = f"💥 Error: {str(e)}"
                logger.error(message)

    return render_template('data_dashboard.html', tables=tables, message=message, df=df_preview)

# -------------------------------
# BACKTEST DASHBOARD
# -------------------------------
@app.route('/backtest_dashboard', methods=['GET', 'POST'])
def backtest_dashboard():
    tables = list_tables()
    result = None
    chart_data = None
    ai_insights = []

    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        table_name = request.form['table']
        start = request.form['start']
        end = request.form['end']
        model_name = request.form['model']

        if not table_name:
            result = {'error': "⚠️ Please select a table."}
            return render_template('backtest_dashboard.html',
                                   tables=tables,
                                   result=result,
                                   chart_data=chart_data,
                                   ai_insights=ai_insights)

        try:
            # --- Load dynamic table ---
            extra_columns = {'ticker': Column('ticker', String, nullable=False)}
            TableClass = get_dynamic_table(BaseEquities, table_name, extra_columns=extra_columns)
            create_table_if_not_exists(TableClass)

            records = query_table(TableClass, filters={'ticker': ticker})
            if not records:
                raise ValueError(f"No data found for ticker '{ticker}' in table '{table_name}'.")

            df_rows = []
            for r in records:
                row = {col: getattr(r, col) for col in TableClass.__table__.columns.keys()}
                if 'date' in row and not isinstance(row['date'], datetime):
                    row['date'] = pd.to_datetime(row['date'])
                df_rows.append(row)
            df = pd.DataFrame(df_rows)

            df = df[(df['date'] >= pd.to_datetime(start)) & (df['date'] <= pd.to_datetime(end))]
            df.sort_values('date', inplace=True)

            if df.empty:
                raise ValueError("No data available for this selection.")

            # --- Check dataset size for LSTM ---
            MIN_LSTM_ROWS = 20
            if len(df) < MIN_LSTM_ROWS:
                logger.warning(f"Not enough data for LSTM: need {MIN_LSTM_ROWS}, got {len(df)}")
                predicted_price = np.array([df['close'].iloc[-1]] * len(df))  # fallback: last price
            else:
                # --- LSTM model paths ---
                model_path = os.path.join(ML_MODELS_DIR, f"{model_name}.h5")
                scaler_path = os.path.join(ML_MODELS_DIR, f"{model_name}_scaler.pkl")

                if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                    model_obj, scaler = train_lstm(df)
                    save_model(model_obj, model_name)
                    joblib.dump(scaler, scaler_path)
                else:
                    model_obj = load_lstm_model(model_name)
                    scaler = joblib.load(scaler_path)

                # --- Make prediction ---
                last_price = df['close'].iloc[-1]
                predicted_price = predict_lstm(model_obj, scaler, df)

                # Ensure prediction is 1D
                predicted_price = np.array(predicted_price).flatten()
                if predicted_price.size != len(df):
                    # fallback: repeat last value
                    predicted_price = np.array([last_price] * len(df))

            # --- Generate signals safely ---
            last_price = df['close'].iloc[-1]
            signals = np.where(predicted_price > last_price, 1, -1)
            df['signal'] = signals

            # --- Backtest ---
            backtested_df, metrics = backtest_strategy(df)
            chart_data = backtested_df.to_dict(orient='records')
            result = metrics

            # --- AI interpretation ---
            ai_insights = interpret_backtest_metrics(
                total_return=result.get('total_return_pct', 0),
                volatility=result.get('volatility_pct', 0),
                sharpe=result.get('sharpe_ratio', 0),
                max_drawdown=result.get('max_drawdown_pct', 0)
            )

        except Exception as e:
            result = {'error': str(e)}
            logger.error(f"Backtest error: {e}")

    return render_template('backtest_dashboard.html',
                           tables=tables,
                           result=result,
                           chart_data=chart_data,
                           ai_insights=ai_insights)


# -------------------------------
# API ENDPOINT
# -------------------------------
@app.route('/api/fetch_metrics', methods=['POST'])
def api_fetch_metrics():
    data = request.json
    ticker = data.get('ticker', '').upper()
    start = data.get('start')
    end = data.get('end')
    table_name = data.get('table')

    if not ticker or not start or not end or not table_name:
        return jsonify({'status': 'error', 'message': 'Missing required parameters'}), 400

    try:
        df = fetch_equity_data(ticker, start, end)
        if df.empty:
            return jsonify({'status': 'error', 'message': 'No data fetched.'}), 400

        df['ticker'] = ticker
        df['daily_return'] = df['close'].pct_change().fillna(0)
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
        df['date'] = pd.to_datetime(df['date']).dt.to_pydatetime()

        extra_columns = {
            'ticker': Column('ticker', String, nullable=False),
            'daily_return': Column('daily_return', Float),
            'cumulative_return': Column('cumulative_return', Float)
        }

        TableClass = get_dynamic_table(BaseEquities, table_name, extra_columns=extra_columns)
        create_table_if_not_exists(TableClass)

        valid_cols = set(TableClass.__table__.columns.keys())
        data_to_insert = [{k: v for k, v in row.items() if k in valid_cols} for row in df.to_dict(orient='records')]
        bulk_insert(TableClass, data_to_insert)

        logger.info(f"Fetched {len(df)} rows for {ticker} into table {table_name}")
        return jsonify({'status': 'success', 'rows': len(df)})

    except Exception as e:
        logger.error(f"API fetch_metrics error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    

# -------------------------------
# API ENDPOINT
# -------------------------------
@app.route('/api/interpret_metrics', methods=['POST'])
def api_interpret_metrics():
    data = request.json
    insights = interpret_backtest_metrics(
        total_return=data.get('total_return', 0),
        volatility=data.get('volatility', 0),
        sharpe=data.get('sharpe', 0),
        max_drawdown=data.get('max_drawdown', 0)
    )
    return jsonify({'messages': insights})


# -------------------------------
# RUN FLASK
# -------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
