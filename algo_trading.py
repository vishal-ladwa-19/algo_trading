#!/usr/bin/env python3
"""
algo_trading.py

Assignment-complete script:
- Fetches 6 months daily data for 3 NIFTY50 tickers via yfinance
- Indicators: RSI(14), SMA20, SMA50, MACD (for ML)
- Strategy: BUY when RSI < 30 AND 20-DMA just crossed above 50-DMA
  Entry: next trading day's Open after signal
  Exit: when 20-DMA crosses below 50-DMA OR RSI > 60 OR end of data
- Backtest horizon: last 6 months (daily)
- ML: DecisionTreeClassifier to predict next-day up/down; outputs accuracy
- Google Sheets: writes TradeLog, P&L Summary, WinRatio, ML Summary tabs
- Telegram: sends BUY/SELL alerts and summary/errors
- Robust against yfinance MultiIndex columns and missing columns
- Uses environment variables for sensitive data (or local service_account.json)
"""

import os
import json
import logging
import datetime
from typing import Optional, Dict, Any, List

import requests
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

import gspread
from google.oauth2.service_account import Credentials

# ---------------------------
# CONFIG
# ---------------------------
TICKERS = ["TCS.NS", "INFY.NS", "RELIANCE.NS"]  # change if needed (3 NIFTY50 tickers)
PERIOD = "6mo"    # EXACT 6 months per assignment
INTERVAL = "1d"
RSI_WINDOW = 14
SMA_SHORT = 20
SMA_LONG = 50
OUTPUT_DIR = "AlgoTrading"
SHEET_NAME = "AlgoTradingLogs"  # created in service account Drive if missing

# Google scopes
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Credentials sources:
# - Use env var GCP_SERVICE_ACCOUNT_JSON (the entire JSON string) in GitHub Actions
# - Or place service_account.json file next to this script (Colab/local)
GCP_JSON_ENV = os.getenv("GCP_SERVICE_ACCOUNT_JSON", "").strip()
SERVICE_ACCOUNT_FILE = "service_account.json"

# Telegram (set via env / secrets)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# LOGGING
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("algo_trading")

# ---------------------------
# Telegram helper
# ---------------------------
def send_telegram_message(text: str) -> None:
    """
    Send Telegram message. If Telegram credentials are missing or invalid, log and skip.
    NOTE: This function keeps your existing BUY/SELL message text exactly as required.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram token/chat id not set. Skipping message: %s", text[:120])
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            logger.warning("Telegram error: %s", r.text)
    except Exception as e:
        logger.exception("Failed to send telegram message: %s", e)

# ---------------------------
# Google credentials loader
# ---------------------------
def load_gcp_creds() -> Optional[Credentials]:
    """
    Load Google service account credentials.
    Prefers env var GCP_SERVICE_ACCOUNT_JSON (useful in CI), otherwise local file.
    Returns google.oauth2 Credentials or None.
    """
    try:
        if GCP_JSON_ENV:
            data = json.loads(GCP_JSON_ENV)
            creds = Credentials.from_service_account_info(data, scopes=SCOPES)
            logger.info("Loaded GCP credentials from environment variable.")
            return creds
        if os.path.exists(SERVICE_ACCOUNT_FILE):
            creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
            logger.info("Loaded GCP credentials from local service_account.json.")
            return creds
        logger.warning("No Google credentials found (GCP_SERVICE_ACCOUNT_JSON or service_account.json).")
        return None
    except Exception as e:
        logger.exception("Failed to load Google credentials: %s", e)
        return None

CREDS = load_gcp_creds()

# ---------------------------
# Google Sheets helpers
# ---------------------------
def init_gsheets(creds: Optional[Credentials]):
    if creds is None:
        logger.warning("No credentials provided; Google Sheets operations will be skipped.")
        return None
    try:
        gc = gspread.authorize(creds)
        try:
            sh = gc.open(SHEET_NAME)
        except gspread.SpreadsheetNotFound:
            sh = gc.create(SHEET_NAME)
            logger.info("Created Google Sheet '%s' in service account Drive.", SHEET_NAME)
        return sh
    except Exception as e:
        logger.exception("Failed to authorize gspread: %s", e)
        return None

def ensure_worksheet(sh, title: str):
    try:
        ws = sh.worksheet(title)
    except Exception:
        ws = sh.add_worksheet(title=title, rows=2000, cols=20)
    return ws

def append_rows_dedup(ws, df: pd.DataFrame, key_cols: List[str]) -> int:
    """
    Append rows to worksheet while avoiding duplicates based on key_cols.
    Returns number of appended rows.
    """
    if df.empty:
        return 0
    existing = ws.get_all_records()  # list of dicts
    existing_keys = set()
    for r in existing:
        key = tuple(str(r.get(k)) for k in key_cols)
        existing_keys.add(key)
    rows_to_append = []
    for _, r in df.iterrows():
        key = tuple(str(r[k]) for k in key_cols)
        if key in existing_keys:
            continue
        rows_to_append.append([r[col] for col in df.columns])
    if rows_to_append:
        # ensure header existence
        if not existing:
            ws.append_row(list(df.columns), value_input_option='USER_ENTERED')
        ws.append_rows(rows_to_append, value_input_option='USER_ENTERED')
    return len(rows_to_append)

def append_summary(ws, summary: Dict[str, Any]):
    if ws is None:
        return
    header = ["Run_Time"] + list(summary.keys())
    if not ws.get_all_values():
        ws.append_row(header, value_input_option='USER_ENTERED')
    row = [datetime.datetime.now().isoformat()] + list(summary.values())
    ws.append_row(row, value_input_option='USER_ENTERED')

# ---------------------------
# Data & Indicators
# ---------------------------
def fetch_data(ticker: str, period: str = PERIOD, interval: str = INTERVAL) -> pd.DataFrame:
    """
    Download OHLCV data using yfinance.
    Flattens MultiIndex columns and validates that required columns exist.
    """
    logger.info("Fetching %s for period=%s interval=%s", ticker, period, interval)
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    # flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if x and str(x) != ""]) for col in df.columns.values]
        # try to normalize e.g. "Close" vs "Close_Adj"
    # standardize column names: Accept several possibilities
    # try to coerce to behave: prefer 'Close', 'Open', 'High', 'Low', 'Volume'
    cols = list(df.columns)
    # map variants to canonical names if necessary
    rename_map = {}
    for c in cols:
        c_low = c.lower()
        if "close" in c_low and "adj" not in c_low:
            rename_map[c] = "Close"
        elif "open" in c_low:
            rename_map[c] = "Open"
        elif "high" in c_low:
            rename_map[c] = "High"
        elif "low" in c_low:
            rename_map[c] = "Low"
        elif "volume" in c_low:
            rename_map[c] = "Volume"
        elif "adj close" in c_low or "adj_close" in c_low or "adjclose" in c_low:
            # prefer plain Close if not present
            if "Close" not in rename_map.values():
                rename_map[c] = "Close"
    if rename_map:
        df = df.rename(columns=rename_map)
    # final validation
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error("Columns returned from yfinance for %s: %s", ticker, list(df.columns))
        raise ValueError(f"Missing columns {missing} for {ticker}.")
    df.index = pd.to_datetime(df.index)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds RSI_14, SMA20, SMA50, MACD, cross_up/down flags, daily returns.
    """
    df = df.copy()
    # enforce numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # RSI
    df["RSI_14"] = ta.momentum.RSIIndicator(close=df["Close"], window=RSI_WINDOW).rsi()
    # SMAs
    df[f"SMA_{SMA_SHORT}"] = df["Close"].rolling(SMA_SHORT).mean()
    df[f"SMA_{SMA_LONG}"] = df["Close"].rolling(SMA_LONG).mean()
    # MACD for ML
    macd = ta.trend.MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    # SMA diff and cross flags
    df["sma_diff"] = df[f"SMA_{SMA_SHORT}"] - df[f"SMA_{SMA_LONG}"]
    df["short_gt_long"] = df[f"SMA_{SMA_SHORT}"] > df[f"SMA_{SMA_LONG}"]
    df["short_gt_long_prev"] = df["short_gt_long"].shift(1).fillna(False).astype(bool)
    df["cross_up"] = df["short_gt_long"] & (~df["short_gt_long_prev"])
    df["cross_down"] = (~df["short_gt_long"]) & (df["short_gt_long_prev"])
    # returns
    df["ret"] = df["Close"].pct_change()
    return df

# ---------------------------
# Backtester (assignment rules EXACT)
# ---------------------------
def backtest_df(df: pd.DataFrame, ticker: str) -> List[Dict[str, Any]]:
    """
    Uses assignment rules:
      BUY when (RSI < 30) AND (20-DMA just crossed above 50-DMA)
      Entry = next day's Open
      Exit = next day's Open when (20-DMA crosses below 50-DMA) OR (RSI > 60) OR end of data
    Returns list of trade dicts.
    """
    trades = []
    position = False
    entry_price = None
    entry_date = None

    # Buy signal exactly as assignment
    df["Buy_Signal"] = (df["RSI_14"] < 30) & (df["cross_up"] == True)

    # iterate by position so we can use next row's Open for entries/exits
    for i in range(len(df) - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]

        # ENTRY
        if not position and bool(row.get("Buy_Signal", False)):
            if not pd.isna(nxt["Open"]):
                position = True
                entry_price = float(nxt["Open"])
                entry_date = df.index[i + 1]
                # keep original message text exactly
                send_telegram_message(f"📈 BUY: {ticker} at {entry_price} on {entry_date.date()}")

        # EXIT
        elif position:
            exit_condition_rsi = row.get("RSI_14", 0) > 60
            exit_condition_crossdown = bool(row.get("cross_down", False))
            if exit_condition_rsi or exit_condition_crossdown:
                if not pd.isna(nxt["Open"]):
                    exit_price = float(nxt["Open"])
                    exit_date = df.index[i + 1]
                else:
                    exit_price = float(row["Close"])
                    exit_date = df.index[i]
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                trades.append({
                    "Ticker": ticker,
                    "Entry_Date": entry_date.date().isoformat() if entry_date is not None else None,
                    "Entry_Price": round(entry_price, 4),
                    "Exit_Date": exit_date.date().isoformat(),
                    "Exit_Price": round(exit_price, 4),
                    "PnL_%": round(pnl_pct, 4)
                })
                send_telegram_message(f"📉 SELL: {ticker} at {exit_price} on {exit_date.date()} | PnL: {pnl_pct:.2f}%")
                position = False
                entry_price = None
                entry_date = None

    # Force exit at end of data
    if position and entry_price is not None:
        last_close = float(df.iloc[-1]["Close"])
        last_date = df.index[-1]
        pnl_pct = ((last_close - entry_price) / entry_price) * 100.0
        trades.append({
            "Ticker": ticker,
            "Entry_Date": entry_date.date().isoformat(),
            "Entry_Price": round(entry_price, 4),
            "Exit_Date": last_date.date().isoformat(),
            "Exit_Price": round(last_close, 4),
            "PnL_%": round(pnl_pct, 4)
        })
        send_telegram_message(f"📉 SELL: {ticker} at {last_close} on {last_date.date()} | PnL: {pnl_pct:.2f}%")

    return trades

# ---------------------------
# Portfolio analytics
# ---------------------------
def summarize_trades(trades_df: pd.DataFrame) -> Dict[str, Any]:
    if trades_df.empty:
        return {"n_trades": 0, "n_wins": 0, "win_rate_pct": 0.0, "total_pnl_pct": 0.0, "avg_pnl_pct": 0.0, "max_drawdown_pct": 0.0}
    n_trades = len(trades_df)
    n_wins = int((trades_df["PnL_%"] > 0).sum())
    win_rate = round(n_wins / n_trades * 100.0, 2)
    total_pnl = round(float(trades_df["PnL_%"].sum()), 4)
    avg_pnl = round(float(trades_df["PnL_%"].mean()), 4)
    # approximate equity curve
    equity = [1.0]
    for pnl in trades_df["PnL_%"]:
        equity.append(equity[-1] * (1.0 + float(pnl) / 100.0))
    equity = np.array(equity[1:])
    if equity.size > 0:
        peak = np.maximum.accumulate(equity)
        drawdowns = (equity - peak) / peak
        max_dd = round(float(np.min(drawdowns) * 100.0), 4)
    else:
        max_dd = 0.0
    return {"n_trades": n_trades, "n_wins": n_wins, "win_rate_pct": win_rate, "total_pnl_pct": total_pnl, "avg_pnl_pct": avg_pnl, "max_drawdown_pct": max_dd}

# ---------------------------
# ML (Decision Tree)
# ---------------------------
def prepare_ml(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for ML:
    - RSI_14, MACD, MACD_signal, Volume (log), sma_diff, returns
    - Target: next-day up(1)/down(0)
    """
    df = df.copy()
    df["vol_log"] = np.log1p(df["Volume"].fillna(0))
    features = ["RSI_14", "MACD", "MACD_signal", "vol_log", "sma_diff", "ret"]
    feat = df[features + ["Close"]].dropna().copy()
    # target
    feat["target"] = (feat["Close"].shift(-1) > feat["Close"]).astype(int)
    feat = feat.dropna()
    return feat

def run_ml(df: pd.DataFrame) -> Dict[str, Any]:
    feat = prepare_ml(df)
    if feat.shape[0] < 30:
        logger.info("Not enough rows for ML training (need >=30). Skipping ML for this ticker.")
        return {"trained": False}
    X = feat[["RSI_14", "MACD", "MACD_signal", "vol_log", "sma_diff", "ret"]]
    y = feat["target"]
    split = int(0.7 * len(feat))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    report = classification_report(y_test, preds, output_dict=True)
    return {"trained": True, "accuracy": acc, "report": report, "model": clf}

# ---------------------------
# Main run
# ---------------------------
def main():
    logger.info("Starting backtest run for tickers: %s", TICKERS)
    all_trades_list = []
    ml_summary_rows = []

    sh = init_gsheets(CREDS) if CREDS else None
    trade_ws = ensure_worksheet(sh, "TradeLog") if sh else None
    pnl_ws = ensure_worksheet(sh, "P&L Summary") if sh else None
    win_ws = ensure_worksheet(sh, "WinRatio") if sh else None
    ml_ws = ensure_worksheet(sh, "ML Summary") if sh else None

    for ticker in TICKERS:
        try:
            df = fetch_data(ticker, period=PERIOD)
            df = add_indicators(df)
        except Exception as e:
            logger.exception("Failed to fetch/prepare data for %s: %s", ticker, e)
            send_telegram_message(f"❌ ERROR fetching/prepping {ticker}: {e}")
            continue

        # ML
        ml_result = run_ml(df)
        if ml_result.get("trained"):
            acc_pct = round(ml_result["accuracy"] * 100.0, 2)
            # Build a small prediction for next day using last row
            last_row = prepare_ml(df).iloc[-1:]
            X_last = last_row[["RSI_14", "MACD", "MACD_signal", "vol_log", "sma_diff", "ret"]].values
            try:
                pred = ml_result["model"].predict(X_last)[0]
                pred_text = "UP" if int(pred) == 1 else "DOWN"
            except Exception:
                pred_text = "N/A"
            ml_summary_rows.append({"Ticker": ticker, "ML_Accuracy%": acc_pct, "ML_Pred": pred_text})
            # send optional ML telegram (safe - wording you approved earlier was buy/sell only, this is additional)
            send_telegram_message(f"🤖 ML {ticker}: Accuracy {acc_pct:.2f}%  Prediction(next): {pred_text}")
        else:
            ml_summary_rows.append({"Ticker": ticker, "ML_Accuracy%": None, "ML_Pred": "N/A"})
            send_telegram_message(f"⚠️ ML not trained for {ticker} (insufficient data).")

        # Backtest
        trades = backtest_df(df, ticker)
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            logger.info("Trades found for %s: %d", ticker, len(trades_df))
        else:
            logger.info("No trades for %s in 6-month window.", ticker)
        trades_df.to_csv(os.path.join(OUTPUT_DIR, f"{ticker}_trades.csv"), index=False)
        df.to_csv(os.path.join(OUTPUT_DIR, f"{ticker}_with_indicators.csv"))

        # Plot chart with buy markers (if any)
        try:
            plt.figure(figsize=(10,4))
            plt.plot(df["Close"], label="Close")
            plt.plot(df[f"SMA_{SMA_SHORT}"], label=f"SMA{SMA_SHORT}")
            plt.plot(df[f"SMA_{SMA_LONG}"], label=f"SMA{SMA_LONG}")
            if not trades_df.empty:
                buy_dates = pd.to_datetime(trades_df["Entry_Date"]).intersection(df.index)
                if len(buy_dates):
                    plt.scatter(buy_dates, df.loc[buy_dates]["Close"], marker="^", s=80, label="Buys", zorder=5)
            plt.legend()
            plt.title(f"{ticker} — Close & SMA{SMA_SHORT}/{SMA_LONG}")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_chart.png"))
            plt.close()
        except Exception:
            logger.exception("Failed to plot chart for %s", ticker)

        all_trades_list.append(trades_df)

    # Combine and summarize
    combined = pd.concat(all_trades_list, ignore_index=True, sort=False) if all_trades_list else pd.DataFrame()
    combined.to_csv(os.path.join(OUTPUT_DIR, "Backtest_Results.csv"), index=False)
    summary = summarize_trades(combined)

    # Console output and Telegram summary
    logger.info("=== BACKTEST RESULTS ===")
    if combined.empty:
        logger.info("No trades found in 6-month window.")
    else:
        logger.info("\n%s", combined.to_string(index=False))

    logger.info("=== SUMMARY ===")
    logger.info(summary)

    send_telegram_message(f"✅ Backtest finished. Trades: {summary['n_trades']}  Win%: {summary['win_rate_pct']}  Total PnL%: {summary['total_pnl_pct']}")

    # Google Sheets updates
    if sh:
        try:
            # Append trades deduped
            if not combined.empty:
                appended = append_rows_dedup(trade_ws, combined, key_cols=["Ticker", "Entry_Date", "Entry_Price"])
                logger.info("Appended %d new trades to Google Sheet.", appended)
            # Append summary
            pnl_row = {
                "Trades": summary["n_trades"],
                "Wins": summary["n_wins"],
                "Win%": summary["win_rate_pct"],
                "Total_PnL%": summary["total_pnl_pct"],
                "Avg_PnL%": summary["avg_pnl_pct"],
                "Max_Drawdown%": summary["max_drawdown_pct"]
            }
            append_summary(pnl_ws, pnl_row)
            # Win ratio
            append_summary(win_ws, {"Win%": summary["win_rate_pct"], "Wins": summary["n_wins"], "Trades": summary["n_trades"]})
            # ML results
            if ml_summary_rows:
                ml_df = pd.DataFrame(ml_summary_rows)
                append_rows_dedup(ml_ws, ml_df, key_cols=["Ticker"])
            logger.info("Google Sheets updated.")
        except Exception as e:
            logger.exception("Failed to update Google Sheets: %s", e)
            send_telegram_message(f"❌ Google Sheets update failed: {e}")
    else:
        logger.warning("Skipping Google Sheets updates (no credentials).")

    logger.info("Run complete.")

if __name__ == "__main__":
    main()
