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

# ---------------------------
# CONFIG
# ---------------------------
TICKERS = ["TCS.NS", "INFY.NS", "RELIANCE.NS"]
PERIOD = "2y"
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
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8357325778:AAFmFSYlcHFAmngAiBTMqjKL4uiZ-sEtvF8")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1279793310")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# TELEGRAM ALERT FUNCTION
# ---------------------------
def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Skipping Telegram message: Missing token or chat ID.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        r = requests.post(url, json=payload)
        if r.status_code != 200:
            print(f"Telegram error: {r.text}")
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

# ---------------------------
# DATA FUNCTIONS
# ---------------------------
def fetch_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    df.index = pd.to_datetime(df.index)

    # Validate required columns
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker} data missing columns: {missing} | Got: {list(df.columns)}")

    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=RSI_WINDOW).rsi()
    df[f'SMA_{SMA_SHORT}'] = df['Close'].rolling(SMA_SHORT).mean()
    df[f'SMA_{SMA_LONG}'] = df['Close'].rolling(SMA_LONG).mean()
    df['short_gt_long'] = df[f'SMA_{SMA_SHORT}'] > df[f'SMA_{SMA_LONG}']
    df['short_gt_long_prev'] = df['short_gt_long'].shift(1).fillna(False)
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

    # Close open position
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
        print("No trades to append.")
        return
    if not ws.get_all_values():
        ws.append_row(list(trades_df.columns))
    ws.append_rows(trades_df.astype(str).values.tolist())

def append_summary_to_sheet(ws, summary):
    if not ws.get_all_values():
        ws.append_row(["Run_Time"] + list(summary.keys()))
    ws.append_row([datetime.datetime.now().isoformat()] + list(summary.values()))

# ---------------------------
# MAIN SCRIPT
# ---------------------------
if __name__ == "__main__":
    try:
        all_trades = []
        for ticker in TICKERS:
            try:
                df = fetch_data(ticker)
                df = add_indicators(df)
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                send_telegram_message(f"❌ Error fetching {ticker}: {e}")
                continue

            df.to_csv(f"{OUTPUT_DIR}/{ticker}_with_indicators.csv")
            trades = backtest_df(df, ticker)
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv(f"{OUTPUT_DIR}/{ticker}_trades.csv", index=False)
            all_trades.append(trades_df)

            plt.figure(figsize=(10, 5))
            plt.plot(df['Close'], label='Close')
            plt.plot(df[f'SMA_{SMA_SHORT}'], label=f'SMA{SMA_SHORT}')
            plt.plot(df[f'SMA_{SMA_LONG}'], label=f'SMA{SMA_LONG}')
            plt.legend()
            plt.savefig(f"{OUTPUT_DIR}/{ticker}_chart.png")
            plt.close()

        combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        combined.to_csv(f"{OUTPUT_DIR}/Backtest_Results.csv", index=False)
        summary = summarize_trades(combined)

        print("\n=== BACKTEST RESULTS ===")
        print(combined if not combined.empty else "No trades found.")
        print("\n=== SUMMARY ===")
        print(summary)

        # Push to Google Sheets
        try:
            sh = init_gsheets()
            trade_ws = ensure_worksheet(sh, "TradeLog")
            summary_ws = ensure_worksheet(sh, "P&L Summary")
            append_trades_to_sheet(trade_ws, combined)
            append_summary_to_sheet(summary_ws, summary)
        except Exception as e:
            print(f"Google Sheets error: {e}")
            send_telegram_message(f"❌ Google Sheets error: {e}")

        send_telegram_message("✅ Backtest complete and results pushed to Google Sheets.")
        print("\nBacktest complete and results pushed to Google Sheets.")

    except Exception as e:
        send_telegram_message(f"❌ ERROR: {e}")
        print(f"ERROR: {e}")
