# algo_trading.py
"""
Algo-Trading mini-prototype (assignment-complete)

Features:
- Fetches 6 months daily data for 3 NIFTY50 tickers via yfinance
- Indicators: RSI(14), SMA20, SMA50, MACD, SMA_diff, returns
- Strategy: BUY when RSI < 30 AND 20-DMA just crossed above 50-DMA
  Entry at next day's Open. Exit when 20-DMA crosses below 50-DMA OR RSI>60 OR end of data.
- Backtest 6 months, log trades, compute P&L and analytics
- ML: LogisticRegression to predict next-day up/down, outputs accuracy & report
- Google Sheets logging: TradeLog, P&L Summary, WinRatio, ML Summary
- Telegram alerts for each buy/sell and final summary
- Uses env vars for secrets; supports GitHub Actions via env secret GCP_SERVICE_ACCOUNT_JSON
"""

import os
import json
import logging
import datetime
import requests
from typing import List, Dict, Any

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import gspread
from google.oauth2.service_account import Credentials

# ---------------------------
# CONFIG
# ---------------------------
TICKERS = ["TCS.NS", "INFY.NS", "RELIANCE.NS"]  # change if needed
PERIOD = "6mo"    # exactly 6 months per assignment
INTERVAL = "1d"
RSI_WINDOW = 14
SMA_SHORT = 20
SMA_LONG = 50
OUTPUT_DIR = "AlgoTrading"
SHEET_NAME = "AlgoTradingLogs"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# ENV / Secrets (must be provided)
GCP_JSON_ENV = os.getenv("GCP_SERVICE_ACCOUNT_JSON", "").strip()   # GitHub Actions secret (raw JSON)
SERVICE_ACCOUNT_FILE = "service_account.json"  # fallback local file
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("algo_trading")

# ---------------------------
# Google credentials loader
# ---------------------------
def load_creds():
    if GCP_JSON_ENV:
        try:
            info = json.loads(GCP_JSON_ENV)
            creds = Credentials.from_service_account_info(info, scopes=SCOPES)
            logger.info("Loaded GCP credentials from env var.")
            return creds
        except Exception as e:
            logger.error("Invalid GCP_SERVICE_ACCOUNT_JSON environment variable: %s", e)
            return None
    elif os.path.exists(SERVICE_ACCOUNT_FILE):
        try:
            creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
            logger.info("Loaded GCP credentials from local file.")
            return creds
        except Exception as e:
            logger.error("Failed to load local service_account.json: %s", e)
            return None
    else:
        logger.warning("No Google credentials provided (will skip Sheets updates).")
        return None

CREDS = load_creds()

# ---------------------------
# Telegram helper
# ---------------------------
def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info("Telegram not configured; skipping message: %s", text[:80])
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        if r.status_code != 200:
            logger.warning("Telegram API returned %s: %s", r.status_code, r.text)
    except Exception as e:
        logger.exception("Failed sending Telegram message: %s", e)

# ---------------------------
# Data & Indicator functions
# ---------------------------
def fetch_data(ticker: str, period=PERIOD, interval=INTERVAL) -> pd.DataFrame:
    logger.info("Fetching %s data for period=%s", ticker, period)
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    df.index = pd.to_datetime(df.index)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ensure numeric
    for c in ['Open','High','Low','Close','Adj Close','Volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # RSI
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=RSI_WINDOW).rsi()
    # SMAs
    df[f"SMA_{SMA_SHORT}"] = df['Close'].rolling(SMA_SHORT).mean()
    df[f"SMA_{SMA_LONG}"] = df['Close'].rolling(SMA_LONG).mean()
    # MACD for ML (fast=12, slow=26, signal=9)
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    # SMA diff and cross flags
    df['sma_diff'] = df[f"SMA_{SMA_SHORT}"] - df[f"SMA_{SMA_LONG}"]
    df['short_gt_long'] = df[f"SMA_{SMA_SHORT}"] > df[f"SMA_{SMA_LONG}"]
    df['short_gt_long_prev'] = df['short_gt_long'].shift(1).fillna(False).astype(bool)
    df['cross_up'] = df['short_gt_long'] & (~df['short_gt_long_prev'])
    df['cross_down'] = (~df['short_gt_long']) & (df['short_gt_long_prev'])
    # daily returns
    df['ret'] = df['Close'].pct_change()
    return df

# ---------------------------
# Backtest: entry/exit rules exactly as assignment
# ---------------------------
def backtest_df(df: pd.DataFrame, ticker: str) -> List[Dict[str,Any]]:
    trades = []
    position = False
    entry_price = None
    entry_date = None
    # Signals: RSI < 30 and cross_up True on that day
    df['Buy_Signal'] = (df['RSI_14'] < 30) & (df['cross_up'] == True)
    for i in range(len(df)-1):  # use next row's open for entry/exit
        row = df.iloc[i]
        nxt = df.iloc[i+1]
        # entry
        if (not position) and bool(row.get('Buy_Signal', False)):
            if not np.isnan(nxt['Open']):
                position = True
                entry_price = float(nxt['Open'])
                entry_date = df.index[i+1]
                logger.info("%s ENTRY %s @ %.2f", ticker, entry_date.date(), entry_price)
                send_telegram_message(f"📈 BUY {ticker} @ {entry_price:.2f} on {entry_date.date()}")
        # exit
        elif position:
            exit_rsi = row.get('RSI_14', 0) > 60
            exit_cross = bool(row.get('cross_down', False))
            if exit_rsi or exit_cross:
                exit_price = float(nxt['Open']) if not np.isnan(nxt['Open']) else float(row['Close'])
                exit_date = df.index[i+1]
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                trades.append({
                    'Ticker': ticker,
                    'Entry_Date': entry_date.date().isoformat(),
                    'Entry_Price': round(entry_price,2),
                    'Exit_Date': exit_date.date().isoformat(),
                    'Exit_Price': round(exit_price,2),
                    'PnL_%': round(pnl_pct,2)
                })
                logger.info("%s EXIT %s @ %.2f PnL=%.2f%%", ticker, exit_date.date(), exit_price, pnl_pct)
                send_telegram_message(f"📉 SELL {ticker} @ {exit_price:.2f} on {exit_date.date()} | PnL: {pnl_pct:.2f}%")
                position = False
                entry_price = None
                entry_date = None
    # force exit at end
    if position and entry_price is not None:
        last_close = float(df.iloc[-1]['Close'])
        last_date = df.index[-1]
        pnl_pct = ((last_close - entry_price) / entry_price) * 100.0
        trades.append({
            'Ticker': ticker,
            'Entry_Date': entry_date.date().isoformat(),
            'Entry_Price': round(entry_price,2),
            'Exit_Date': last_date.date().isoformat(),
            'Exit_Price': round(last_close,2),
            'PnL_%': round(pnl_pct,2)
        })
        logger.info("%s FORCE EXIT %s @ %.2f PnL=%.2f%%", ticker, last_date.date(), last_close, pnl_pct)
        send_telegram_message(f"📉 FORCE EXIT {ticker} @ {last_close:.2f} (end) | PnL: {pnl_pct:.2f}%")
    return trades

# ---------------------------
# Portfolio analytics
# ---------------------------
def summarize_trades(trades_df: pd.DataFrame) -> Dict[str,Any]:
    if trades_df.empty:
        return {'n_trades':0, 'n_wins':0, 'win_rate_pct':0.0, 'total_pnl_pct':0.0, 'avg_pnl_pct':0.0, 'max_drawdown_pct':0.0}
    n_trades = len(trades_df)
    n_wins = (trades_df['PnL_%'] > 0).sum()
    win_rate = float(n_wins) / n_trades * 100.0
    total_pnl = float(trades_df['PnL_%'].sum())
    avg_pnl = float(trades_df['PnL_%'].mean())
    # equity curve approximate
    equity = [1.0]
    for pnl in trades_df['PnL_%']:
        equity.append(equity[-1] * (1.0 + float(pnl)/100.0))
    equity = np.array(equity[1:])
    if len(equity) > 0:
        peak = np.maximum.accumulate(equity)
        drawdowns = (equity - peak) / peak
        max_dd = float(np.min(drawdowns) * 100.0)
    else:
        max_dd = 0.0
    return {'n_trades':int(n_trades), 'n_wins':int(n_wins), 'win_rate_pct':round(win_rate,2),
            'total_pnl_pct':round(total_pnl,2), 'avg_pnl_pct':round(avg_pnl,2), 'max_drawdown_pct':round(max_dd,2)}

# ---------------------------
# ML: simple Logistic Regression for next-day up/down
# ---------------------------
def prepare_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # features: RSI, MACD, MACD_signal, Volume (log), sma_diff, today's return
    df['vol_log'] = np.log1p(df['Volume'].fillna(0))
    features = df[['RSI_14','MACD','MACD_signal','vol_log','sma_diff','ret']].copy()
    features = features.dropna()
    # target: next-day up (1) if next close > today close else 0
    target = (df['Close'].shift(-1) > df['Close']).astype(int)
    features['target'] = target
    features = features.dropna()
    return features

def run_ml_model(df: pd.DataFrame) -> Dict[str,Any]:
    feat = prepare_ml_features(df)
    if feat.empty or feat.shape[0] < 30:
        logger.warning("Not enough rows for ML training. Skipping ML.")
        return {'trained':False}
    X = feat.drop(columns=['target'])
    y = feat['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds).tolist()
    logger.info("ML model accuracy: %.3f", acc)
    return {'trained':True, 'accuracy':float(acc), 'report':report, 'confusion_matrix':cm}

# ---------------------------
# Google Sheets helpers
# ---------------------------
def init_gsheets(creds):
    if creds is None:
        logger.warning("No Google credentials; skipping Sheets operations.")
        return None
    gc = gspread.authorize(creds)
    try:
        sh = gc.open(SHEET_NAME)
    except gspread.SpreadsheetNotFound:
        sh = gc.create(SHEET_NAME)
        logger.info("Created sheet %s in service account Drive.", SHEET_NAME)
    return sh

def ensure_worksheet(sh, title):
    try:
        ws = sh.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=1000, cols=20)
    return ws

def append_unique_trades(ws, trades_df: pd.DataFrame):
    """Append trades while avoiding duplicates by (Ticker, Entry_Date, Entry_Price)."""
    if trades_df.empty:
        logger.info("No trades to append to sheet.")
        return 0
    existing = ws.get_all_records()
    existing_set = set()
    for r in existing:
        key = (r.get('Ticker'), r.get('Entry_Date'), r.get('Entry_Price'))
        existing_set.add(key)
    rows = []
    for _, r in trades_df.iterrows():
        key = (r['Ticker'], r['Entry_Date'], str(r['Entry_Price']))
        if key in existing_set:
            continue
        rows.append([r['Ticker'], r['Entry_Date'], r['Entry_Price'], r['Exit_Date'], r['Exit_Price'], r['PnL_%']])
    if rows:
        # ensure header
        if not existing:
            ws.append_row(['Ticker','Entry_Date','Entry_Price','Exit_Date','Exit_Price','PnL_%'])
        ws.append_rows(rows, value_input_option='USER_ENTERED')
    return len(rows)

def append_summary(ws, summary: Dict[str,Any]):
    if ws is None:
        return
    # header if empty
    if not ws.get_all_values():
        ws.append_row(['Run_Time'] + list(summary.keys()))
    ws.append_row([datetime.datetime.now().isoformat()] + list(summary.values()))

# ---------------------------
# MAIN workflow
# ---------------------------
def main():
    logger.info("Starting backtest run for tickers: %s", TICKERS)
    all_trades_list = []
    ml_results = {}

    for ticker in TICKERS:
        try:
            df = fetch_data(ticker, period=PERIOD, interval=INTERVAL)
            df = add_indicators(df)
            trades = backtest_df(df, ticker)
            trades_df = pd.DataFrame(trades)
            # save local artifacts
            df.to_csv(f"{OUTPUT_DIR}/{ticker}_with_indicators.csv", index=True)
            trades_df.to_csv(f"{OUTPUT_DIR}/{ticker}_trades.csv", index=False)
            # chart
            plt.figure(figsize=(10,4))
            plt.plot(df['Close'], label='Close')
            plt.plot(df[f"SMA_{SMA_SHORT}"], label=f"SMA{SMA_SHORT}")
            plt.plot(df[f"SMA_{SMA_LONG}"], label=f"SMA{SMA_LONG}")
            if not trades_df.empty:
                buy_dates = pd.to_datetime(trades_df['Entry_Date'])
                buy_dates = buy_dates.intersection(df.index)
                if len(buy_dates):
                    plt.scatter(buy_dates, df.loc[buy_dates]['Close'], marker='^', s=80, label='Buys', zorder=5)
            plt.legend()
            plt.title(f"{ticker} - Close & SMAs")
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/{ticker}_chart.png")
            plt.close()
            all_trades_list.append(trades_df)

            # ML per ticker (optional) - we collect results for last ticker only for summary; you may aggregate
            ml_results[ticker] = run_ml_model(df)
        except Exception as e:
            logger.exception("Error processing %s: %s", ticker, e)
            send_telegram_message(f"❌ ERROR processing {ticker}: {e}")

    combined = pd.concat(all_trades_list, ignore_index=True, sort=False) if all_trades_list else pd.DataFrame()
    combined.to_csv(f"{OUTPUT_DIR}/Backtest_Results.csv", index=False)
    summary = summarize_trades(combined)

    # Console output
    logger.info("=== BACKTEST RESULTS ===")
    if combined.empty:
        logger.info("No trades found in 6-month window.")
    else:
        logger.info("\n%s", combined.to_string(index=False))

    logger.info("=== SUMMARY ===")
    logger.info(summary)
    send_telegram_message(f"✅ Backtest finished. Trades: {summary['n_trades']}, Win%: {summary['win_rate_pct']}%, Total PnL%: {summary['total_pnl_pct']}")

    # ML aggregate: compute average accuracy where available
    ml_summary = {}
    accs = []
    for t, res in ml_results.items():
        if res.get('trained'):
            ml_summary[t] = {'accuracy': res['accuracy'], 'cm': res['confusion_matrix']}
            accs.append(res['accuracy'])
    if accs:
        mean_acc = float(np.mean(accs))
        send_telegram_message(f"🔬 ML average accuracy across tickers: {mean_acc:.3f}")
    else:
        logger.info("ML not trained for any ticker (insufficient data).")

    # Google Sheets update
    if CREDS is not None:
        sh = init_gsheets(CREDS)
        trade_ws = ensure_worksheet(sh, "TradeLog")
        pnl_ws = ensure_worksheet(sh, "P&L Summary")
        win_ws = ensure_worksheet(sh, "WinRatio")
        ml_ws = ensure_worksheet(sh, "ML Summary")

        # append trades uniquely
        appended = append_unique_trades(trade_ws, combined)
        logger.info("Appended %d new trades to Google Sheet.", appended)

        # P&L summary row
        pnl_row = {
            'Trades': summary['n_trades'],
            'Wins': summary['n_wins'],
            'Win%': summary['win_rate_pct'],
            'Total_PnL%': summary['total_pnl_pct'],
            'Avg_PnL%': summary['avg_pnl_pct'],
            'Max_Drawdown%': summary['max_drawdown_pct']
        }
        append_summary(pnl_ws, pnl_row)

        # win ratio
        winrow = {'Win%': summary['win_rate_pct'], 'Wins': summary['n_wins'], 'Trades': summary['n_trades']}
        append_summary(win_ws, winrow)

        # ML summary
        if accs:
            ml_summary_row = {'Avg_Accuracy': float(np.mean(accs)), 'Tickers_Trained': len(accs)}
            append_summary(ml_ws, ml_summary_row)
        logger.info("Google Sheets updated.")

    else:
        logger.warning("Google Sheets not updated (no creds).")

    # final logs
    logger.info("Run complete.")

if __name__ == "__main__":
    main()
