import os
import datetime
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import yfinance as yf
import logging
import gspread
from google.oauth2.service_account import Credentials

# ---------------------------
# Config
# ---------------------------
TICKERS = ["TCS.NS", "INFY.NS", "RELIANCE.NS"]   # Change if needed
PERIOD = "6mo"
INTERVAL = "1d"
RSI_WINDOW = 14
SMA_SHORT = 20
SMA_LONG = 50
OUTPUT_DIR = "AlgoTrading"
RESULTS_CSV = f"{OUTPUT_DIR}/Backtest_Results.csv"
SHEET_NAME = "AlgoTradingLogs"   # Must match your Google Sheet name

# ---------------------------
# Setup
# ---------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Google Sheets API Auth
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]
SERVICE_ACCOUNT_FILE = "service_account.json"  # Created from GitHub Actions secret

creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
gc = gspread.authorize(creds)

try:
    sh = gc.open(SHEET_NAME)
except gspread.SpreadsheetNotFound:
    sh = gc.create(SHEET_NAME)
    logging.info(f"Created sheet '{SHEET_NAME}'.")

def ensure_worksheet(title, rows=1000, cols=20):
    try:
        ws = sh.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=rows, cols=cols)
    return ws

trade_ws = ensure_worksheet("TradeLog")
summary_ws = ensure_worksheet("P&L Summary")
win_ws = ensure_worksheet("WinRatio")

# ---------------------------
# Functions
# ---------------------------
def fetch_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    df.index = pd.to_datetime(df.index)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in df.columns:
            df[col] = df[col].astype(float).squeeze()
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=RSI_WINDOW).rsi()
    df[f'SMA_{SMA_SHORT}'] = df['Close'].rolling(SMA_SHORT).mean()
    df[f'SMA_{SMA_LONG}'] = df['Close'].rolling(SMA_LONG).mean()
    df['short_gt_long'] = df[f'SMA_{SMA_SHORT}'] > df[f'SMA_{SMA_LONG}']
    df['short_gt_long_prev'] = df['short_gt_long'].shift(1).fillna(False)
    df['cross_up'] = df['short_gt_long'] & (~df['short_gt_long_prev'])
    df['cross_down'] = (~df['short_gt_long']) & df['short_gt_long_prev']
    return df

def backtest_df(df: pd.DataFrame, ticker: str):
    trades = []
    position = False
    entry_price, entry_date = None, None
    df['Buy_Signal'] = (df['RSI_14'] < 30) & (df['cross_up'])
    for i in range(len(df) - 1):
        row, next_row = df.iloc[i], df.iloc[i+1]
        if not position and row['Buy_Signal']:
            if not np.isnan(next_row['Open']):
                position = True
                entry_price = float(next_row['Open'])
                entry_date = df.index[i+1]
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

def append_trades_to_sheet(trades_df: pd.DataFrame, worksheet):
    if trades_df is None or trades_df.empty:
        logging.info("No trades to append.")
        return
    if not worksheet.get_all_values():
        worksheet.append_row(list(trades_df.columns), value_input_option='USER_ENTERED')
    worksheet.append_rows(trades_df.astype(str).values.tolist(), value_input_option='USER_ENTERED')
    logging.info(f"Appended {len(trades_df)} trades to '{worksheet.title}'.")

def update_summary_to_sheet(summary_dict: dict, worksheet):
    if not worksheet.get_all_values():
        worksheet.append_row(["Run_Time"] + list(summary_dict.keys()), value_input_option='USER_ENTERED')
    row = [datetime.datetime.now().isoformat()] + list(summary_dict.values())
    worksheet.append_row(row, value_input_option='USER_ENTERED')
    logging.info(f"Summary updated in '{worksheet.title}'.")

def update_winratio_to_sheet(summary_dict: dict, worksheet):
    win_dict = {
        "Win%": summary_dict.get("Win%"),
        "Trades": summary_dict.get("Trades")
    }
    if not worksheet.get_all_values():
        worksheet.append_row(["Run_Time"] + list(win_dict.keys()), value_input_option='USER_ENTERED')
    row = [datetime.datetime.now().isoformat()] + list(win_dict.values())
    worksheet.append_row(row, value_input_option='USER_ENTERED')
    logging.info(f"Win ratio updated in '{worksheet.title}'.")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    all_trades = []
    for ticker in TICKERS:
        df = fetch_data(ticker)
        df = add_indicators(df)
        df.to_csv(f"{OUTPUT_DIR}/{ticker}_with_indicators.csv")
        trades = backtest_df(df, ticker)
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(f"{OUTPUT_DIR}/{ticker}_trades.csv", index=False)
        all_trades.append(trades_df)

        # Plot chart
        plt.figure(figsize=(10,5))
        plt.plot(df['Close'], label='Close')
        plt.plot(df[f'SMA_{SMA_SHORT}'], label=f'SMA{SMA_SHORT}')
        plt.plot(df[f'SMA_{SMA_LONG}'], label=f'SMA{SMA_LONG}')
        if not trades_df.empty:
            buy_dates = pd.to_datetime(trades_df['Entry_Date'])
            plt.scatter(buy_dates, df.loc[buy_dates]['Close'], marker='^', color='g', label='Buy', s=100)
        plt.title(f"{ticker} Signals")
        plt.legend()
        plt.savefig(f"{OUTPUT_DIR}/{ticker}_chart.png")
        plt.close()

    # Combine and summarize
    combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    combined.to_csv(RESULTS_CSV, index=False)
    summary = summarize_trades(combined)

    # Update Google Sheets
    append_trades_to_sheet(combined, trade_ws)
    update_summary_to_sheet(summary, summary_ws)
    update_winratio_to_sheet(summary, win_ws)

    logging.info("\n=== Backtest Summary ===")
    logging.info(summary)
