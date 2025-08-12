# algo_trading.py

import os
import datetime
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import gspread
from google.oauth2.service_account import Credentials
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging

# ---------------------------
# CONFIG
# ---------------------------
DEMO_MODE = True  # <<< Set to False for real trading run

TICKERS = ["TCS.NS", "INFY.NS", "RELIANCE.NS"]
PERIOD = "6mo" if DEMO_MODE else "2y"
INTERVAL = "1d"
RSI_WINDOW = 14
SMA_SHORT = 20
SMA_LONG = 50
OUTPUT_DIR = "AlgoTrading"
SHEET_NAME = "AlgoTradingLogs"

# Google Sheets
SERVICE_ACCOUNT_FILE = "service_account.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)

# ---------------------------
# TELEGRAM ALERT FUNCTION
# ---------------------------
def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        r = requests.post(url, json=payload)
        if r.status_code != 200:
            logging.error(f"Telegram error: {r.text}")
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")

# ---------------------------
# DATA FUNCTIONS
# ---------------------------
def fetch_data(ticker: str) -> pd.DataFrame:
    logging.info(f"Fetching {ticker} for period={PERIOD} interval={INTERVAL}")
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    df.index = pd.to_datetime(df.index)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in df.columns:
            df[col] = df[col].astype(float).squeeze()
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=RSI_WINDOW).rsi()
    df[f'SMA_{SMA_SHORT}'] = df['Close'].rolling(SMA_SHORT).mean()
    df[f'SMA_{SMA_LONG}'] = df['Close'].rolling(SMA_LONG).mean()
    df['short_gt_long'] = df[f'SMA_{SMA_SHORT}'] > df[f'SMA_{SMA_LONG}']
    df['short_gt_long_prev'] = df['short_gt_long'].shift(1).fillna(False).astype(bool)
    df['cross_up'] = df['short_gt_long'] & (~df['short_gt_long_prev'])
    df['cross_down'] = (~df['short_gt_long']) & (df['short_gt_long_prev'])
    return df

# ---------------------------
# BACKTESTING
# ---------------------------
def backtest_df(df: pd.DataFrame, ticker: str):
    trades = []
    position = False
    entry_price, entry_date = None, None
    
    if DEMO_MODE:
        df['Buy_Signal'] = df['RSI_14'] < 60  # Looser for demo
    else:
        df['Buy_Signal'] = (df['RSI_14'] < 40) & (df['cross_up'])
    
    for i in range(len(df) - 1):
        row, next_row = df.iloc[i], df.iloc[i+1]
        
        # Buy
        if not position and row['Buy_Signal']:
            if not np.isnan(next_row['Open']):
                position = True
                entry_price = float(next_row['Open'])
                entry_date = df.index[i+1]
                send_telegram_message(f"📈 BUY: {ticker} at {entry_price} on {entry_date.date()}")
        
        # Sell
        elif position:
            if (row['RSI_14'] > 60) or row['cross_down']:
                exit_price = float(next_row['Open']) if not np.isnan(next_row['Open']) else float(row['Close'])
                exit_date = df.index[i+1]
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                trades.append({
                    'Ticker': ticker,
                    'Entry_Date': entry_date.date().isoformat(),
                    'Entry_Price': round(entry_price, 2),
                    'Exit_Date': exit_date.date().isoformat(),
                    'Exit_Price': round(exit_price, 2),
                    'PnL_%': round(pnl_pct, 2)
                })
                send_telegram_message(f"📉 SELL: {ticker} at {exit_price} on {exit_date.date()} | PnL: {pnl_pct:.2f}%")
                position = False
                entry_price, entry_date = None, None

    if position:
        last_close = float(df.iloc[-1]['Close'])
        pnl_pct = ((last_close - entry_price) / entry_price) * 100
        trades.append({
            'Ticker': ticker,
            'Entry_Date': entry_date.date().isoformat(),
            'Entry_Price': round(entry_price, 2),
            'Exit_Date': df.index[-1].date().isoformat(),
            'Exit_Price': round(last_close, 2),
            'PnL_%': round(pnl_pct, 2)
        })
    return trades

def summarize_trades(trades_df: pd.DataFrame):
    if trades_df.empty:
        return {'Trades': 0, 'Win%': 0, 'Total_PnL%': 0}
    win_rate = (trades_df['PnL_%'] > 0).mean() * 100
    return {
        'Trades': len(trades_df),
        'Win%': round(win_rate, 2),
        'Total_PnL%': round(trades_df['PnL_%'].sum(), 2)
    }

# ---------------------------
# GOOGLE SHEETS
# ---------------------------
def init_gsheets():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    gc = gspread.authorize(creds)
    try:
        sh = gc.open(SHEET_NAME)
    except gspread.SpreadsheetNotFound:
        sh = gc.create(SHEET_NAME)
    return sh

def ensure_worksheet(sh, title):
    try:
        ws = sh.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=1000, cols=20)
    return ws

def append_trades_to_sheet(ws, trades_df):
    if trades_df.empty:
        logging.info("No trades to append to sheet.")
        return
    if not ws.get_all_values():
        ws.append_row(list(trades_df.columns))
    ws.append_rows(trades_df.astype(str).values.tolist())

def append_summary_to_sheet(ws, summary):
    if not ws.get_all_values():
        ws.append_row(["Run_Time"] + list(summary.keys()))
    ws.append_row([datetime.datetime.now().isoformat()] + list(summary.values()))

# ---------------------------
# ML MODEL
# ---------------------------
def train_ml_model(df: pd.DataFrame, ticker: str):
    try:
        df = df.dropna().copy()
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        features = ['RSI_14', f'SMA_{SMA_SHORT}', f'SMA_{SMA_LONG}', 'Volume']
        X = df[features]
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logging.info(f"ML Model Accuracy for {ticker}: {acc:.2%}")
        logging.info(f"\n{classification_report(y_test, preds)}")
    except Exception as e:
        logging.error(f"ML training failed for {ticker}: {e}")

# ---------------------------
# MAIN SCRIPT
# ---------------------------
if __name__ == "__main__":
    try:
        logging.info("=== DEMO MODE ===" if DEMO_MODE else "=== LIVE MODE ===")
        send_telegram_message("🚀 Starting backtest run in DEMO MODE" if DEMO_MODE else "🚀 Starting LIVE backtest run")
        
        all_trades = []
        for ticker in TICKERS:
            df = fetch_data(ticker)
            df = add_indicators(df)
            df.to_csv(f"{OUTPUT_DIR}/{ticker}_with_indicators.csv")
            trades = backtest_df(df, ticker)
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv(f"{OUTPUT_DIR}/{ticker}_trades.csv", index=False)
            all_trades.append(trades_df)
            plt.figure(figsize=(10,5))
            plt.plot(df['Close'], label='Close')
            plt.plot(df[f'SMA_{SMA_SHORT}'], label=f'SMA{SMA_SHORT}')
            plt.plot(df[f'SMA_{SMA_LONG}'], label=f'SMA{SMA_LONG}')
            plt.legend()
            plt.savefig(f"{OUTPUT_DIR}/{ticker}_chart.png")
            plt.close()

            train_ml_model(df, ticker)

        combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        combined.to_csv(f"{OUTPUT_DIR}/Backtest_Results.csv", index=False)
        summary = summarize_trades(combined)

        logging.info("\n=== BACKTEST RESULTS ===")
        logging.info(combined if not combined.empty else "No trades found.")
        logging.info("\n=== SUMMARY ===")
        logging.info(summary)

        sh = init_gsheets()
        trade_ws = ensure_worksheet(sh, "TradeLog")
        summary_ws = ensure_worksheet(sh, "P&L Summary")
        append_trades_to_sheet(trade_ws, combined)
        append_summary_to_sheet(summary_ws, summary)

        send_telegram_message("✅ Backtest complete and results pushed to Google Sheets.")
        logging.info("Backtest complete and results pushed to Google Sheets.")

    except Exception as e:
        send_telegram_message(f"❌ ERROR: {e}")
        logging.error(f"ERROR: {e}")
