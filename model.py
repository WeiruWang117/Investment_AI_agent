import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import time
import csv
import os
from sec_edgar_downloader import Downloader
import json

# determine tickers pool

def check_stock_data_integrity(tickers, start_year=2010, end_year=2024):
    results = []
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
   
    for symbol in tickers:
        try:
            ticker_obj = yf.Ticker(symbol)
            
            data = ticker_obj.history(start=start_date, end=end_date)
            
            if data.empty:
                results.append({"Symbol": symbol, "Status": "No data", "Start": "N/A", "End": "N/A"})
                continue
            
            
            actual_start = data.index.min().strftime('%Y-%m-%d')
            actual_end = data.index.max().strftime('%Y-%m-%d')
            target_start = pd.Timestamp("2010-01-04")
            target_end = pd.Timestamp("2024-12-30")

            has_2010 = data.index.tz_localize(None).min() <= target_start
            has_2024 = data.index.tz_localize(None).max() >= target_end
            
            status = "Available" if (has_2010 and has_2024) else "Insufficient data"
            
            results.append({
                "Symbol": symbol,
                "Status": status,
                "Actual_Start": actual_start,
                "Actual_End": actual_end,
            })
        except Exception as e:
            results.append({"Symbol": symbol, "Status": f"error: {str(e)}", "Start": "N/A", "End": "N/A"})

    return pd.DataFrame(results)


tickers_for_check = ["TSM", "INTC", "NVDA", "ASML", "AVGO", "TXN", "QCOM", "MU", "AMAT", "AMD", "ADI", "LRCX", "NXPI", "KLAC", "MCHP", "SWKS", "MRVL", "QRVO", "TER", "ON", "MPWR", "ENTG", "MKSI", "SLAB", "CRUS", "CCMP", "SMTC"]
check_report = check_stock_data_integrity(tickers_for_check)
print(check_report)

ticker_pool = check_report[check_report["Status"] == "Available"]
ticker_list = ticker_pool["Symbol"].tolist()
print(ticker_list)

""" Tickers we choose: ['TSM', 'INTC', 'NVDA', 'ASML', 'AVGO', 'TXN', 
'QCOM', 'MU', 'AMAT', 'AMD', 'ADI', 'LRCX', 'NXPI', 'KLAC', 'MCHP', 'SWKS', 
'MRVL', 'TER', 'ON', 'MPWR', 'ENTG', 'MKSI', 'SLAB', 'CRUS', 'SMTC']"""

# fetch stock market data from yfinance

raw_market_data = yf.download(ticker_list, start="2010-01-04", end="2024-12-30")
raw_market_data.to_csv("semiconductor_tickers_2010_2024.csv")

# data cleaning

df_uncleaned = pd.read_csv("semiconductor_tickers_2010_2024.csv", index_col=0, parse_dates=True)
missing_summary = df_uncleaned.isnull().sum()
missing_summary.to_csv("missing_values_report.csv")

# we found that NPXI has lots of missing values, therefore, we are going to remove NPXI from our ticker list
# update ticker list

ticker_list =  ['TSM', 'INTC', 'NVDA', 'ASML', 'AVGO', 'TXN', 'QCOM', 'MU', 'AMAT', 'AMD', 'ADI', 'LRCX', 'KLAC', 'MCHP', 'SWKS', 'MRVL', 'TER', 'ON', 'MPWR', 'ENTG', 'MKSI', 'SLAB', 'CRUS', 'SMTC']
market_data = yf.download(ticker_list, start="2010-01-04", end="2024-12-30")
market_data.to_csv("semiconductor_tickers_cleaned_2010_2024.csv")

# no need to download the data from yfinance every time open the file

cleaned_ticker_price_df = pd.read_csv("semiconductor_tickers_cleaned_2010_2024.csv", skiprows=[3],header = [0,1], index_col=0)

# seperate different dataframes

close_df = cleaned_ticker_price_df['Close']
volume_df = cleaned_ticker_price_df['Volume']
high_df = cleaned_ticker_price_df['High']
low_df = cleaned_ticker_price_df['Low']
open_df = cleaned_ticker_price_df['Open']


#  Fetch news data from alpha vantage

def get_news_to_df(ticker, start_date, end_date, limit=50):

    def format_date(d_str):
        return d_str.replace("-", "") + "T0000"

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": format_date(start_date),
        "time_to": format_date(end_date),
        "limit": limit,
        "apikey": os.environ["ALPHAVANTAGE_API_KEY"]  
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "feed" not in data or not data["feed"]:
        return pd.DataFrame()

    df = pd.DataFrame(data["feed"])

    required_cols = ["time_published", "title", "source", "overall_sentiment_score", "overall_sentiment_label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  [WARN] Missing columns {missing}, got: {list(df.columns)}")
        return pd.DataFrame()

    df = df[required_cols]
    df["time_published"] = pd.to_datetime(df["time_published"])

    return df


def get_historical_segments(ticker, start_year=2010, end_year=2024, sleep_seconds=2,
                            quarterly_from=2020):
    """
    Fetch news data in batches to avoid hitting the 1000-article limit.
    - Yearly segments for start_year to quarterly_from-1 (low volume, fewer requests)
    - Quarterly segments from quarterly_from onward (high volume, avoid cap)
    Alpha Vantage max limit per request is 1000.
    """
    segments = []

    # Yearly for early years
    for year in range(start_year, min(quarterly_from, end_year + 1)):
        segments.append((f"{year}-01-01", f"{year}-12-31"))

    # Quarterly for recent high-volume years
    quarter_starts = [(1, 1), (4, 1), (7, 1), (10, 1)]
    quarter_ends   = [(3, 31), (6, 30), (9, 30), (12, 31)]
    for year in range(max(quarterly_from, start_year), end_year + 1):
        for (sm, sd), (em, ed) in zip(quarter_starts, quarter_ends):
            segments.append((f"{year}-{sm:02d}-{sd:02d}", f"{year}-{em:02d}-{ed:02d}"))

    combined_list = []
    for i, (start, end) in enumerate(segments):
        print(f"  [{i+1}/{len(segments)}] {start} → {end}", end=" ... ")

        df_segment = get_news_to_df(ticker, start, end, limit=1000)

        if not df_segment.empty:
            if len(df_segment) >= 1000:
                print(f"{len(df_segment)} records  ⚠ cap reached")
            else:
                print(f"{len(df_segment)} records")
            combined_list.append(df_segment)
        else:
            print("0 records")

        if i < len(segments) - 1:
            time.sleep(sleep_seconds)

    if combined_list:
        final_df = pd.concat(combined_list, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=["time_published", "title"])
        final_df = final_df.sort_values(by="time_published", ascending=False)
        return final_df
    else:
        print("No data retrieved.")
        return pd.DataFrame()


def clean_sentiment_data(df):
    """
    Clean the DataFrame by removing rows where overall_sentiment_score 
    is outside the range of [-1, 1].
    """

    # clean sentiment column

    df['overall_sentiment_score'] = pd.to_numeric(df['overall_sentiment_score'], errors='coerce')
    
    initial_count = len(df)
    df = df.dropna(subset=['overall_sentiment_score'])
    
    df = df[(df['overall_sentiment_score'] >= -1) & (df['overall_sentiment_score'] <= 1)]
    
    final_count = len(df)
    print(f"Cleaning complete: Removed {initial_count - final_count} invalid rows.")
    
    # clean time column

    df['time_published'] = pd.to_datetime(df['time_published'], errors='coerce')
    df = df.dropna(subset=['time_published'])
    
    max_date = pd.Timestamp.now()
    min_date = pd.Timestamp("2010-01-01")
    
    df = df[(df['time_published'] >= min_date) & (df['time_published'] <= max_date)]
    print(f"Final data count after time & sentiment cleaning: {len(df)}")
    
    return df

def batch_process_tickers(ticker_list):
    """
    Loop through a list of tickers, fetch, clean, and save each one.
    """

    folder_name = "News data"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for ticker in ticker_list:
        print(f"\n--- Starting process for {ticker} ---")
        
        raw_df = get_historical_segments(ticker)
        
        if not raw_df.empty:
            cleaned_df = clean_sentiment_data(raw_df)
            
            filename = f"{ticker}_Historical_News_Cleaned.csv"
            save_path = os.path.join(folder_name,filename)

            cleaned_df.to_csv(
                save_path, 
                index=False, 
                encoding="utf-8-sig",
                quoting=csv.QUOTE_ALL
            )
            print(f"Finished {ticker}: Saved {len(cleaned_df)} rows to {filename}")
        else:
            print(f"{ticker}: No data retrieved.")
            
        print(f"Cooling down after {ticker}...")
        time.sleep(5)

batch_process_tickers(ticker_list)


# Fetch 10K, 10Q, 8K

filing_types = ["10-K", "10-Q", "8-K"]
start_date = "2015-01-01"
end_date = "2026-01-01"

download_path = "SEC filings"
email_address = "ww569@cornell.edu" 
dl = Downloader("CornellResearchProject", email_address, download_path)


print(f"Starting batch download for {filing_types} filings...")
print(f"Target Period: {start_date} to {end_date}")
print("-" * 50)

for ticker in ticker_list:
    print(f"Processing Ticker: {ticker}")
    
    ticker_dir = os.path.join(download_path, ticker)
    
    for f_type in filing_types:
        try:
            print(f" -> Downloading {f_type} for {ticker}...")
            
            num_downloaded = dl.get(
                f_type, 
                ticker, 
                after=start_date, 
                before=end_date,
                download_details=True  
            )
            
            print(f"    - Success: {num_downloaded} files downloaded.")
            
        except Exception as e:
            print(f"    - Error downloading {f_type} for {ticker}: {e}")

print("-" * 50)
print("Batch download task completed.")
print(f"Storage Location: {os.path.abspath(download_path)}")

# Fetch Earning Calls

YEARS = range(2015, 2026)  
QUARTERS = [1, 2, 3, 4]

PARENT_DIR = "Earning calls"

def fetch_and_save_transcript(ticker, year, quarter):
    """
    Downloads transcript and saves it into a ticker-specific folder.
    """
    period = f"{year}Q{quarter}"
    
    # Define and create the ticker-specific sub-directory
    ticker_dir = os.path.join(PARENT_DIR, ticker)
    if not os.path.exists(ticker_dir):
        os.makedirs(ticker_dir)
    
    file_path = os.path.join(ticker_dir, f"{ticker}_{period}.json")

    if os.path.exists(file_path):
        return "Exists"

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "EARNINGS_CALL_TRANSCRIPT",
        "symbol": ticker,
        "quarter": period,
        "apikey": os.environ["ALPHAVANTAGE_API_KEY"]
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if "transcript" in data:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            return "Success"
        elif "Error Message" in data:
            return f"Error: {data['Error Message']}"
        else:
            return "No Data"

    except Exception as e:
        return f"Failed: {e}"


print(f"Starting organized download for {len(ticker_list)} tickers...")

for ticker in ticker_list:
    print(f"\nProcessing Folder: {PARENT_DIR}/{ticker}")
    for year in YEARS:
        for quarter in QUARTERS:
            result = fetch_and_save_transcript(ticker, year, quarter)
            
            if result == "Success":
                print(f"  [+] {year}Q{quarter}: Saved")
            elif result == "Exists":
                pass 
            else:
                print(f"  [-] {year}Q{quarter}: {result}")
            
            time.sleep(12) 

print("\n--- Download complete. Files are organized by ticker folders. ---")


# Fetch income statement, balance sheet, cash flow statement

FINANCIAL_DATA_DIR = "Financial data"
REPORT_TYPES = ["INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW"]


def fetch_report(ticker, report_type):
    params = {
        "function": report_type,
        "symbol":   ticker,
        "apikey":   os.environ["ALPHAVANTAGE_API_KEY"],
    }
    try:
        resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
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


def _financial_exists(ticker, report_type):
    return os.path.exists(os.path.join(FINANCIAL_DATA_DIR, ticker, f"{report_type}.json"))


def _save_financial(data, ticker, report_type):
    folder = os.path.join(FINANCIAL_DATA_DIR, ticker)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{report_type}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_financial_report(ticker, report_type):
    path = os.path.join(FINANCIAL_DATA_DIR, ticker, f"{report_type}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def batch_download_financials(sleep_seconds=1):
    done, skipped, failed = 0, 0, []

    for ticker in ticker_list:
        print(f"\nProcessing: {ticker}")
        for report_type in REPORT_TYPES:
            if _financial_exists(ticker, report_type):
                print(f"  [SKIP] {report_type} already exists")
                skipped += 1
                continue

            print(f"  Fetching {report_type} ...", end=" ")
            data = fetch_report(ticker, report_type)

            if data:
                _save_financial(data, ticker, report_type)
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


batch_download_financials()


# ── Fetch Macroeconomic Indicators ─────────────────────────────────────────────
#
# Indicators fetched (all via Alpha Vantage Economic Indicators API):
#
#   Core — directly affect semiconductor valuations:
#     TREASURY_YIELD_10Y   10-year Treasury yield (monthly)
#     TREASURY_YIELD_2Y    2-year Treasury yield  (monthly)
#     FEDERAL_FUNDS_RATE   Fed funds rate          (monthly)
#     CPI                  Consumer Price Index    (monthly)
#     INFLATION            Annual CPI inflation    (annual)
#
#   Supporting — demand-side signals for semiconductor end-markets:
#     REAL_GDP             Real GDP growth         (quarterly)
#     RETAIL_SALES         Retail & food sales     (monthly)
#     DURABLES             Durable goods orders    (monthly)
#     UNEMPLOYMENT         Unemployment rate       (monthly)
#     NONFARM_PAYROLL      Nonfarm payroll         (monthly)
#
# Output: Macro data/{INDICATOR_NAME}.json
#   Each file contains {"data": [{"date": "YYYY-MM-DD", "value": "..."}, ...]}
# ──────────────────────────────────────────────────────────────────────────────

MACRO_DATA_DIR = "Macro data"

MACRO_INDICATORS = {
    "TREASURY_YIELD_10Y": {
        "function": "TREASURY_YIELD",
        "interval": "monthly",
        "maturity": "10year",
    },
    "TREASURY_YIELD_2Y": {
        "function": "TREASURY_YIELD",
        "interval": "monthly",
        "maturity": "2year",
    },
    "FEDERAL_FUNDS_RATE": {
        "function": "FEDERAL_FUNDS_RATE",
        "interval": "monthly",
    },
    "CPI": {
        "function": "CPI",
        "interval": "monthly",
    },
    "INFLATION": {
        "function": "INFLATION",
    },
    "REAL_GDP": {
        "function": "REAL_GDP",
        "interval": "quarterly",
    },
    "RETAIL_SALES": {
        "function": "RETAIL_SALES",
    },
    "DURABLES": {
        "function": "DURABLES",
    },
    "UNEMPLOYMENT": {
        "function": "UNEMPLOYMENT",
    },
    "NONFARM_PAYROLL": {
        "function": "NONFARM_PAYROLL",
    },
}

AV_API_KEY = os.environ["ALPHAVANTAGE_API_KEY"]


def _macro_path(indicator_name: str) -> str:
    return os.path.join(MACRO_DATA_DIR, f"{indicator_name}.json")


def _macro_exists(indicator_name: str) -> bool:
    return os.path.exists(_macro_path(indicator_name))


def fetch_macro_indicator(params_extra: dict) -> dict | None:
    """
    Fetch a single macroeconomic indicator from Alpha Vantage.
    Returns the raw JSON dict, or None on failure.
    """
    params = {**params_extra, "apikey": AV_API_KEY}
    try:
        resp = requests.get(
            "https://www.alphavantage.co/query",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if "Information" in data:
            print(f"  [RATE LIMIT] {data['Information'][:80]}")
            return None
        if "Error Message" in data:
            print(f"  [ERROR] {data['Error Message']}")
            return None
        if "data" not in data:
            print(f"  [WARN] Unexpected response keys: {list(data.keys())}")
            return None

        return data

    except Exception as e:
        print(f"  [EXCEPTION] {e}")
        return None


def _save_macro(data: dict, indicator_name: str):
    os.makedirs(MACRO_DATA_DIR, exist_ok=True)
    with open(_macro_path(indicator_name), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_macro_indicator(indicator_name: str) -> pd.DataFrame | None:
    """
    Load a cached macro indicator and return as a DataFrame.

    Returns DataFrame with columns: date (datetime), value (float)
    Sorted ascending by date, filtered to 2010-01-01 onward.
    """
    path = _macro_path(indicator_name)
    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    records = raw.get("data", [])
    if not records:
        return None

    df = pd.DataFrame(records)
    df["date"]  = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"])
    df = df[df["date"] >= pd.Timestamp("2010-01-01")]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def batch_download_macro(force_refresh: bool = False, sleep_seconds: int = 13):
    """
    Download all macro indicators and save to Macro data/.

    Args:
        force_refresh  : Re-download even if cache exists.
        sleep_seconds  : Delay between calls (free tier: 5 calls/min → 13s safe).
    """
    os.makedirs(MACRO_DATA_DIR, exist_ok=True)
    done, skipped, failed = 0, 0, []
    total = len(MACRO_INDICATORS)

    print(f"\nFetching {total} macroeconomic indicators → {MACRO_DATA_DIR}/")
    print("-" * 50)

    for i, (name, params_extra) in enumerate(MACRO_INDICATORS.items(), 1):
        print(f"  [{i}/{total}] {name} ...", end=" ")

        if not force_refresh and _macro_exists(name):
            print("SKIP (cached)")
            skipped += 1
            continue

        data = fetch_macro_indicator(params_extra)

        if data:
            _save_macro(data, name)
            n_records = len(data.get("data", []))
            print(f"Saved  ({n_records} records)")
            done += 1
        else:
            print("FAILED")
            failed.append(name)

        if i < total:
            time.sleep(sleep_seconds)

    print("-" * 50)
    print(f"Downloaded : {done}")
    print(f"Skipped    : {skipped} (already cached)")
    print(f"Failed     : {len(failed)}")
    if failed:
        print(f"  {failed}")
    print(f"Files saved to: {os.path.abspath(MACRO_DATA_DIR)}/")


batch_download_macro()

# fetch vix index data
vix = yf.download("^VIX", start="2020-01-01")
vix[["Close"]].rename(columns={"Close": "vix_close"}).to_csv("Macro data/VIX.csv")