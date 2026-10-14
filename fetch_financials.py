"""
fetch_financials.py

Downloads Income Statement, Balance Sheet, and Cash Flow from Alpha Vantage.

Output structure (mirrors SEC filings folder layout):
    Financial data/
        NVDA/
            INCOME_STATEMENT.json
            BALANCE_SHEET.json
            CASH_FLOW.json
        TSM/
            ...
"""

import json
import time
import os
import requests

# ── Configuration ─────────────────────────────────────────────────────────────

API_KEY  = os.environ["ALPHAVANTAGE_API_KEY"]
BASE_URL = "https://www.alphavantage.co/query"
DATA_DIR = "Financial data"

TICKER_LIST = [
    "TSM", "INTC", "NVDA", "ASML", "AVGO", "TXN", "QCOM", "MU",
    "AMAT", "AMD", "ADI", "LRCX", "KLAC", "MCHP", "SWKS", "MRVL",
    "TER", "ON", "MPWR", "ENTG", "MKSI", "SLAB", "CRUS", "SMTC",
]

REPORT_TYPES = ["INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW"]


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_report(ticker, report_type):
    params = {
        "function": report_type,
        "symbol":   ticker,
        "apikey":   API_KEY,
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "Information" in data:
            print(f"  [RATE LIMIT] {data['Information'][:80]}")
            return None
        if "Error Message" in data:
            print(f"  [ERROR] {data['Error Message']}")
            return None
        if "annualReports" not in data:
            print(f"  [WARN] No annualReports in response: {list(data.keys())}")
            return None

        return data

    except Exception as e:
        print(f"  [EXCEPTION] {e}")
        return None


# ── Save / Load ───────────────────────────────────────────────────────────────

def _save(data, ticker, report_type):
    folder = os.path.join(DATA_DIR, ticker)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{report_type}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _exists(ticker, report_type):
    return os.path.exists(os.path.join(DATA_DIR, ticker, f"{report_type}.json"))


def load_report(ticker, report_type):
    path = os.path.join(DATA_DIR, ticker, f"{report_type}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


# ── Batch Download ────────────────────────────────────────────────────────────

def batch_download(ticker_list=TICKER_LIST, report_types=REPORT_TYPES, sleep_seconds=1):
    total   = len(ticker_list) * len(report_types)
    done    = 0
    skipped = 0
    failed  = []

    for ticker in ticker_list:
        print(f"\nProcessing: {ticker}")
        for report_type in report_types:
            if _exists(ticker, report_type):
                print(f"  [SKIP] {report_type} already exists")
                skipped += 1
                continue

            print(f"  Fetching {report_type} ...", end=" ")
            data = fetch_report(ticker, report_type)

            if data:
                _save(data, ticker, report_type)
                print(f"Saved  ({len(data.get('quarterlyReports', []))}Q / "
                      f"{len(data.get('annualReports', []))}A)")
                done += 1
            else:
                print("FAILED")
                failed.append((ticker, report_type))

            time.sleep(sleep_seconds)

    print(f"\n{'='*50}")
    print(f"Downloaded : {done}")
    print(f"Skipped    : {skipped} (already cached)")
    print(f"Failed     : {len(failed)}")
    if failed:
        print(f"  {failed}")


# ── Coverage Check ────────────────────────────────────────────────────────────

def check_coverage():
    short = {"INCOME_STATEMENT": "IS", "BALANCE_SHEET": "BS", "CASH_FLOW": "CF"}
    header = f"{'Ticker':<8}  {'IS':<5}  {'BS':<5}  {'CF':<5}"
    print(f"\n{header}")
    print("-" * 28)
    for ticker in TICKER_LIST:
        row = f"{ticker:<8}"
        for r in REPORT_TYPES:
            row += f"  {'✓' if _exists(ticker, r) else '✗':<5}"
        print(row)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_coverage()
    print("\nStarting download (skipping existing)...")
    batch_download()
    check_coverage()
