# Investment AI Agent

Python investment research pipeline for semiconductor stocks. It combines technical, fundamental, sentiment, SEC filing, bull/bear, macro, risk and decision agents, with portfolio recording and backtesting tools.

## Setup

Requires Python 3.10 or later.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:ANTHROPIC_API_KEY = "your-key"
$env:ALPHAVANTAGE_API_KEY = "your-key"
```

`.env.example` documents the required variables; the scripts read process environment variables and do not automatically load `.env` files.

## Run

```powershell
python pipeline.py 2021-03-01 NVDA AMD
```

The existing scripts use local paths under `D:/Investment_AI_agent`. Review these paths before running from another location. Downloaded market data, financial data, SEC filings and other input datasets must be prepared locally; they are excluded from this repository along with caches, generated reports and notebook execution outputs.

`model.py` contains data preparation routines; review it before execution because it performs downloads and writes local datasets. API usage can incur provider charges.

## Key files

- `pipeline.py`: orchestrates the analysis stages.
- `*_agent.py`: research and decision agents.
- `backtesting.py` / `backtest_runner.py`: backtesting.
- `generate_report.py`: report generation.
- `fetch_financials.py`, `parse_10k.py`, `parse_10q.py`, `parse_8k.py`: data preparation.

This project is for research and does not guarantee investment results.