"""
Fundamental Agent

Computes fundamental financial indicators for semiconductor stocks
from Alpha Vantage Income Statement, Balance Sheet, and Cash Flow data.
No LLM — pure quantitative signal generation.

Data source:
  Financial data/{TICKER}/INCOME_STATEMENT.json
  Financial data/{TICKER}/BALANCE_SHEET.json
  Financial data/{TICKER}/CASH_FLOW.json

Point-in-time safety:
  Quarterly reports become available fiscalDateEnding + 45 days
  Annual    reports become available fiscalDateEnding + 60 days

Output cache:
  fundamental_cache/{TICKER}/signal_{DATE}.json   (composite signal for Orchestrator)
"""

import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_DIR      = Path("d:/Investment_AI_agent")
FINANCIAL_DIR = BASE_DIR / "Financial data"
CACHE_DIR     = BASE_DIR / "fundamental_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Publication lag (conservative, point-in-time safe)
QUARTERLY_LAG_DAYS = 45
ANNUAL_LAG_DAYS    = 60

TICKER_LIST = [
    "TSM", "INTC", "NVDA", "ASML", "AVGO", "TXN", "QCOM", "MU",
    "AMAT", "AMD", "ADI", "LRCX", "KLAC", "MCHP", "SWKS", "MRVL",
    "TER", "ON", "MPWR", "ENTG", "MKSI", "SLAB", "CRUS", "SMTC",
]

SUBSECTORS = {
    "Equipment":   ["AMAT", "LRCX", "KLAC", "ENTG"],
    "Foundry":     ["TSM"],
    "Fabless":     ["NVDA", "AMD", "QCOM", "AVGO", "MRVL"],
    "Analog":      ["ADI", "TXN", "MCHP", "ON", "SWKS", "SLAB", "CRUS", "SMTC"],
    "Memory":      ["MU"],
    "Diversified": ["INTC", "MKSI", "MPWR", "TER", "ASML"],
}


# ── Cache Utilities ────────────────────────────────────────────────────────────

def _cache_dir(ticker: str) -> Path:
    d = CACHE_DIR / ticker
    d.mkdir(exist_ok=True)
    return d


def _load_json(path: Path) -> Optional[dict]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _save_json(data: dict, path: Path):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Data Loading ───────────────────────────────────────────────────────────────

def _load_raw(ticker: str, report_type: str) -> Optional[dict]:
    path = FINANCIAL_DIR / ticker / f"{report_type}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(val) -> Optional[float]:
    """Convert Alpha Vantage string value to float. Returns None for 'None' or errors."""
    if val is None or val == "None" or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _reports_to_df(reports: list[dict], lag_days: int) -> pd.DataFrame:
    """
    Convert list of report dicts to DataFrame.
    Adds 'available_date' = fiscalDateEnding + lag_days (point-in-time gate).
    Sorted descending by fiscalDateEnding (newest first).
    """
    rows = []
    for r in reports:
        fiscal_end = pd.Timestamp(r["fiscalDateEnding"])
        available  = fiscal_end + timedelta(days=lag_days)
        row = {"fiscalDateEnding": fiscal_end, "available_date": available}
        for k, v in r.items():
            if k not in ("fiscalDateEnding", "reportedCurrency"):
                row[k] = _to_float(v)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("fiscalDateEnding", ascending=False).reset_index(drop=True)
    return df


def _load_ticker_data(ticker: str) -> Optional[dict]:
    """Load and parse all three statements for a ticker. Returns dict of DataFrames."""
    inc = _load_raw(ticker, "INCOME_STATEMENT")
    bal = _load_raw(ticker, "BALANCE_SHEET")
    cf  = _load_raw(ticker, "CASH_FLOW")
    if not inc or not bal or not cf:
        return None
    return {
        "inc_q": _reports_to_df(inc["quarterlyReports"], QUARTERLY_LAG_DAYS),
        "inc_a": _reports_to_df(inc["annualReports"],    ANNUAL_LAG_DAYS),
        "bal_q": _reports_to_df(bal["quarterlyReports"], QUARTERLY_LAG_DAYS),
        "bal_a": _reports_to_df(bal["annualReports"],    ANNUAL_LAG_DAYS),
        "cf_q":  _reports_to_df(cf["quarterlyReports"],  QUARTERLY_LAG_DAYS),
        "cf_a":  _reports_to_df(cf["annualReports"],     ANNUAL_LAG_DAYS),
    }


# ── Point-in-Time Row Selection ────────────────────────────────────────────────

def _latest_available(df: pd.DataFrame, ref: pd.Timestamp) -> Optional[pd.Series]:
    """Return the most recent row whose available_date <= ref."""
    mask = df["available_date"] <= ref
    if not mask.any():
        return None
    return df[mask].iloc[0]  # already sorted descending


def _row_n_ago(df: pd.DataFrame, ref: pd.Timestamp, n: int) -> Optional[pd.Series]:
    """Return the nth-most-recent available row (0-indexed). Used for YoY / QoQ."""
    mask = df["available_date"] <= ref
    available = df[mask]
    if len(available) <= n:
        return None
    return available.iloc[n]


def _f(row: Optional[pd.Series], col: str) -> Optional[float]:
    if row is None:
        return None
    v = row.get(col, None)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)


# ── TTM (Trailing Twelve Months) Builder ───────────────────────────────────────

def _ttm(df_q: pd.DataFrame, ref: pd.Timestamp, col: str) -> Optional[float]:
    """Sum the 4 most recent available quarterly values for a flow statement column."""
    mask = df_q["available_date"] <= ref
    rows = df_q[mask].head(4)
    if len(rows) < 4:
        return None
    vals = [_to_float(rows.iloc[i].get(col)) for i in range(4)]
    if any(v is None for v in vals):
        return None
    return sum(vals)


# ── Indicator Calculations ─────────────────────────────────────────────────────

def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _yoy(curr: Optional[float], prior: Optional[float]) -> Optional[float]:
    """Year-over-year growth rate."""
    if curr is None or prior is None or prior == 0:
        return None
    return (curr - prior) / abs(prior)


def _qoq(curr: Optional[float], prior: Optional[float]) -> Optional[float]:
    """Quarter-over-quarter growth rate."""
    return _yoy(curr, prior)


def _trend_direction(values: list[Optional[float]]) -> str:
    """
    Given [newest, ..., oldest], classify trend as improving/deteriorating/stable.
    Uses simple linear slope over valid values.
    """
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(valid) < 2:
        return "unknown"
    xs = np.array([x[0] for x in valid], dtype=float)
    ys = np.array([x[1] for x in valid], dtype=float)
    slope = np.polyfit(xs, ys, 1)[0]
    # Negative index because newest=0, so a downward slope in index = improvement
    if slope < -0.005:
        return "improving"
    if slope > 0.005:
        return "deteriorating"
    return "stable"


# ── Composite Score ────────────────────────────────────────────────────────────

def _compute_fundamental_score(metrics: dict) -> float:
    """
    Composite fundamental score in [-1, +1].
    Weights:
      Profitability (gross_margin, operating_margin, roic) : 30%
      Growth (revenue_yoy, eps_yoy)                        : 25%
      FCF quality (fcf_margin, fcf_to_net_income)          : 25%
      Balance sheet (net_debt_to_ebitda, current_ratio)    : 20%
    """
    scores, weights = [], []

    def _push(value, benchmark_low, benchmark_high, weight):
        """Normalise value to [-1,+1] relative to a typical sector range."""
        if value is None:
            return
        mid   = (benchmark_low + benchmark_high) / 2
        scale = (benchmark_high - benchmark_low) / 2
        scores.append(float(np.tanh((value - mid) / (scale + 1e-9))))
        weights.append(weight)

    # Profitability
    _push(metrics.get("gross_margin"),     0.30, 0.70, 0.12)
    _push(metrics.get("operating_margin"), 0.10, 0.40, 0.10)
    _push(metrics.get("roic"),             0.05, 0.30, 0.08)

    # Growth
    _push(metrics.get("revenue_yoy"),  -0.10, 0.30, 0.13)
    _push(metrics.get("eps_yoy"),      -0.20, 0.40, 0.12)

    # FCF quality
    _push(metrics.get("fcf_margin"),        0.05, 0.35, 0.13)
    _push(metrics.get("fcf_to_net_income"), 0.50, 1.50, 0.12)

    # Balance sheet
    nd_ebitda = metrics.get("net_debt_to_ebitda")
    if nd_ebitda is not None:
        # Lower is better — invert
        scores.append(float(np.tanh(-(nd_ebitda - 1.0) / 2.0)))
        weights.append(0.10)
    _push(metrics.get("current_ratio"), 1.0, 3.0, 0.10)

    if not scores:
        return 0.0
    total_w = sum(weights)
    score   = sum(s * w for s, w in zip(scores, weights)) / total_w
    return float(np.clip(score, -1.0, 1.0))


# ── Signal Extraction ──────────────────────────────────────────────────────────

def _extract_signal(ticker: str, data: dict, ref: pd.Timestamp) -> dict:
    inc_q, inc_a = data["inc_q"], data["inc_a"]
    bal_q, bal_a = data["bal_q"], data["bal_a"]
    cf_q,  cf_a  = data["cf_q"],  data["cf_a"]

    # Most recent available rows
    inc  = _latest_available(inc_q, ref)
    bal  = _latest_available(bal_q, ref)
    cf   = _latest_available(cf_q,  ref)
    inc1 = _row_n_ago(inc_q, ref, 1)   # previous quarter
    inc4 = _row_n_ago(inc_q, ref, 4)   # same quarter last year
    cf4  = _row_n_ago(cf_q,  ref, 4)

    # Annual rows for YoY (more stable than quarterly for annual reports)
    inc_ay  = _latest_available(inc_a, ref)
    inc_ay1 = _row_n_ago(inc_a, ref, 1)
    bal_ay  = _latest_available(bal_a, ref)
    cf_ay   = _latest_available(cf_a,  ref)
    cf_ay1  = _row_n_ago(cf_a, ref, 1)

    fiscal_date = str(inc["fiscalDateEnding"].date()) if inc is not None else None

    # ── Revenue ───────────────────────────────────────────────────────────────
    revenue_q     = _f(inc,  "totalRevenue")
    revenue_q1    = _f(inc1, "totalRevenue")
    revenue_q_ly  = _f(inc4, "totalRevenue")
    revenue_a     = _f(inc_ay,  "totalRevenue")
    revenue_a1    = _f(inc_ay1, "totalRevenue")
    revenue_ttm   = _ttm(inc_q, ref, "totalRevenue")

    # ── Profitability ─────────────────────────────────────────────────────────
    gross_profit_q  = _f(inc,    "grossProfit")
    gross_profit_a  = _f(inc_ay, "grossProfit")
    gross_profit_a1 = _f(inc_ay1,"grossProfit")
    op_income_q     = _f(inc,    "operatingIncome")
    op_income_a     = _f(inc_ay, "operatingIncome")
    net_income_q    = _f(inc,    "netIncome")
    net_income_a    = _f(inc_ay, "netIncome")
    net_income_a1   = _f(inc_ay1,"netIncome")
    ebitda_a        = _f(inc_ay, "ebitda")
    rd_q            = _f(inc,    "researchAndDevelopment")
    rd_a            = _f(inc_ay, "researchAndDevelopment")

    gross_margin    = _safe_div(gross_profit_q, revenue_q)
    operating_margin= _safe_div(op_income_q,    revenue_q)
    net_margin      = _safe_div(net_income_q,   revenue_q)
    rd_intensity    = _safe_div(rd_q, revenue_q)

    # ── Cash Flow ─────────────────────────────────────────────────────────────
    op_cf_q    = _f(cf,    "operatingCashflow")
    op_cf_a    = _f(cf_ay, "operatingCashflow")
    op_cf_a1   = _f(cf_ay1,"operatingCashflow")
    capex_q    = _f(cf,    "capitalExpenditures")
    capex_a    = _f(cf_ay, "capitalExpenditures")

    # FCF = operating CF - capex (capex is negative in some APIs; take abs)
    def _fcf(op, cap):
        if op is None or cap is None:
            return None
        return op - abs(cap)

    fcf_q = _fcf(op_cf_q, capex_q)
    fcf_a = _fcf(op_cf_a, capex_a)
    fcf_a1= _fcf(op_cf_a1, _f(cf_ay1, "capitalExpenditures"))
    fcf_ttm = _ttm(cf_q, ref, "operatingCashflow")
    if fcf_ttm is not None and _ttm(cf_q, ref, "capitalExpenditures") is not None:
        fcf_ttm = fcf_ttm - abs(_ttm(cf_q, ref, "capitalExpenditures"))

    fcf_margin       = _safe_div(fcf_q, revenue_q)
    fcf_to_net_income= _safe_div(fcf_q, net_income_q)
    capex_intensity  = _safe_div(abs(capex_q) if capex_q else None, revenue_q)
    op_cf_margin     = _safe_div(op_cf_q, revenue_q)

    # ── Balance Sheet ─────────────────────────────────────────────────────────
    total_assets      = _f(bal, "totalAssets")
    current_assets    = _f(bal, "totalCurrentAssets")
    current_liab      = _f(bal, "totalCurrentLiabilities")
    total_liab        = _f(bal, "totalLiabilities")
    equity            = _f(bal, "totalShareholderEquity")
    long_term_debt    = _f(bal, "longTermDebt")
    short_term_debt   = _f(bal, "shortTermDebt")
    cash              = _f(bal, "cashAndShortTermInvestments")
    inventory         = _f(bal, "inventory")
    inventory_q1      = _f(_latest_available(bal_q, ref - timedelta(days=90)), "inventory") if bal_q is not None else None

    total_debt      = (long_term_debt or 0) + (short_term_debt or 0)
    net_debt        = total_debt - (cash or 0)
    current_ratio   = _safe_div(current_assets, current_liab)
    debt_to_equity  = _safe_div(total_debt, equity)
    net_debt_to_ebitda = _safe_div(net_debt, ebitda_a)

    # ROIC = NOPAT / Invested Capital
    nopat           = _safe_div(_safe_div(op_income_a, 1), 1)  # simplification without tax rate
    invested_capital= None
    if equity is not None and total_debt is not None:
        invested_capital = equity + total_debt - (cash or 0)
    roic = _safe_div(op_income_a, invested_capital)

    # Inventory days = inventory / (COGS / 90)
    cogs_q = _f(inc, "costOfRevenue")
    inventory_days = _safe_div(inventory, _safe_div(cogs_q, 90))

    # ── Growth Rates ──────────────────────────────────────────────────────────
    revenue_yoy   = _yoy(revenue_a,     revenue_a1)
    revenue_qoq   = _qoq(revenue_q,     revenue_q1)
    revenue_yoy_q = _yoy(revenue_q,     revenue_q_ly)   # same-quarter YoY
    gp_yoy        = _yoy(gross_profit_a, gross_profit_a1)
    ni_yoy        = _yoy(net_income_a,  net_income_a1)
    fcf_yoy       = _yoy(fcf_a,         fcf_a1)
    op_cf_yoy     = _yoy(op_cf_a,       op_cf_a1)

    # EPS proxy (net income per share not in Alpha Vantage basic — use net income YoY)
    eps_yoy = ni_yoy

    # ── Margin Trend (last 4 quarters) ────────────────────────────────────────
    gm_trend_vals = []
    om_trend_vals = []
    for i in range(4):
        r_i  = _row_n_ago(inc_q, ref, i)
        r_q  = _f(r_i, "totalRevenue")
        gm_trend_vals.append(_safe_div(_f(r_i, "grossProfit"), r_q))
        om_trend_vals.append(_safe_div(_f(r_i, "operatingIncome"), r_q))

    margin_trend = _trend_direction(gm_trend_vals)
    op_margin_trend = _trend_direction(om_trend_vals)

    # Revenue acceleration: compare current YoY vs prior-quarter YoY
    rev_q_prev_ly = _f(_row_n_ago(inc_q, ref, 5), "totalRevenue")
    rev_yoy_prior = _yoy(revenue_q1, rev_q_prev_ly)
    revenue_acceleration = (
        (revenue_yoy_q > rev_yoy_prior)
        if revenue_yoy_q is not None and rev_yoy_prior is not None
        else None
    )

    # ── Inventory signal (semiconductor cycle indicator) ───────────────────────
    inv_q1_row = _row_n_ago(bal_q, ref, 1)
    inventory_q1_val = _f(inv_q1_row, "inventory")
    inventory_yoy_q_row = _row_n_ago(bal_q, ref, 4)
    inventory_ly = _f(inventory_yoy_q_row, "inventory")
    inventory_yoy = _yoy(inventory, inventory_ly)

    # ── Composite Score ────────────────────────────────────────────────────────
    metrics = {
        "gross_margin":        gross_margin,
        "operating_margin":    operating_margin,
        "roic":                roic,
        "revenue_yoy":         revenue_yoy,
        "eps_yoy":             eps_yoy,
        "fcf_margin":          fcf_margin,
        "fcf_to_net_income":   fcf_to_net_income,
        "net_debt_to_ebitda":  net_debt_to_ebitda,
        "current_ratio":       current_ratio,
    }
    fundamental_score = _compute_fundamental_score(metrics)

    def _r(v, n=4):
        return round(v, n) if v is not None else None

    signal = {
        "ticker":           ticker,
        "reference_date":   str(ref.date()),
        "fiscal_date":      fiscal_date,
        "analysis_date":    datetime.now().strftime("%Y-%m-%d"),

        # ── Composite outputs (for Orchestrator) ──────────────────────────────
        "fundamental_score":     round(fundamental_score, 4),
        "margin_trend":          margin_trend,
        "op_margin_trend":       op_margin_trend,
        "revenue_acceleration":  revenue_acceleration,

        # ── Profitability ──────────────────────────────────────────────────────
        "gross_margin":          _r(gross_margin),
        "operating_margin":      _r(operating_margin),
        "net_margin":            _r(net_margin),
        "roic":                  _r(roic),
        "rd_intensity":          _r(rd_intensity),

        # ── Growth ─────────────────────────────────────────────────────────────
        "revenue_yoy":           _r(revenue_yoy),
        "revenue_qoq":           _r(revenue_qoq),
        "revenue_yoy_q":         _r(revenue_yoy_q),
        "eps_yoy":               _r(eps_yoy),
        "gp_yoy":                _r(gp_yoy),
        "fcf_yoy":               _r(fcf_yoy),
        "op_cf_yoy":             _r(op_cf_yoy),

        # ── Cash Flow Quality ──────────────────────────────────────────────────
        "fcf_margin":            _r(fcf_margin),
        "fcf_to_net_income":     _r(fcf_to_net_income),
        "op_cf_margin":          _r(op_cf_margin),
        "capex_intensity":       _r(capex_intensity),

        # ── Balance Sheet ──────────────────────────────────────────────────────
        "current_ratio":         _r(current_ratio),
        "debt_to_equity":        _r(debt_to_equity),
        "net_debt_to_ebitda":    _r(net_debt_to_ebitda),

        # ── Semiconductor cycle indicator ──────────────────────────────────────
        "inventory_days":        _r(inventory_days),
        "inventory_yoy":         _r(inventory_yoy),

        # ── TTM absolutes (for Orchestrator valuation) ─────────────────────────
        "revenue_ttm":           _r(revenue_ttm, 0),
        "fcf_ttm":               _r(fcf_ttm, 0),
    }

    return signal


# ── Cross-Sectional Quality Rank ───────────────────────────────────────────────

def _add_quality_rank(all_signals: dict[str, dict]) -> dict[str, dict]:
    """
    Add quality_rank (0–1 percentile within 24 stocks) based on fundamental_score.
    """
    scores = {
        t: s["fundamental_score"]
        for t, s in all_signals.items()
        if "error" not in s and s.get("fundamental_score") is not None
    }
    if not scores:
        return all_signals

    tickers = list(scores.keys())
    vals    = np.array([scores[t] for t in tickers])
    ranks   = (vals.argsort().argsort() + 1) / len(vals)   # percentile 0–1

    for t, rank in zip(tickers, ranks):
        all_signals[t]["quality_rank"] = round(float(rank), 4)

    return all_signals


# ── FundamentalAgent Class ─────────────────────────────────────────────────────

class FundamentalAgent:
    """
    Computes and caches fundamental signals for semiconductor stocks.

    Usage:
        agent = FundamentalAgent()
        signal = agent.get_signal("NVDA", reference_date="2024-06-30")
        all    = agent.get_all_signals(reference_date="2024-06-30")
    """

    def __init__(self):
        print("[FundamentalAgent] Loading financial data...")
        self._data = {}
        for ticker in TICKER_LIST:
            d = _load_ticker_data(ticker)
            if d:
                self._data[ticker] = d
            else:
                print(f"  [WARN] {ticker}: missing financial data, skipping.")
        print(f"[FundamentalAgent] Ready — {len(self._data)} tickers loaded.")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_signal(
        self,
        ticker: str,
        reference_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> dict:
        """
        Return the fundamental signal dict for a single ticker.
        Cached per (ticker, date).
        """
        ref_label  = reference_date or "latest"
        cache_path = _cache_dir(ticker) / f"signal_{ref_label}.json"

        if not force_refresh:
            cached = _load_json(cache_path)
            if cached:
                return cached

        if ticker not in self._data:
            raise ValueError(f"{ticker}: financial data not available.")

        if reference_date is None:
            ref = pd.Timestamp.today().normalize()
        else:
            ref = pd.Timestamp(reference_date)

        signal = _extract_signal(ticker, self._data[ticker], ref)
        _save_json(signal, cache_path)
        return signal

    def get_all_signals(
        self,
        ticker_list: Optional[list[str]] = None,
        reference_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> dict[str, dict]:
        """Return signals for all tickers, with cross-sectional quality_rank added."""
        tickers = ticker_list or TICKER_LIST
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.get_signal(ticker, reference_date, force_refresh)
            except Exception as e:
                print(f"[WARNING] {ticker}: {e}")
                results[ticker] = {"ticker": ticker, "error": str(e)}

        results = _add_quality_rank(results)
        return results

    def print_summary(
        self,
        reference_date: Optional[str] = None,
        top_n: int = 10,
    ) -> None:
        """Print a ranked summary table sorted by fundamental_score."""
        signals = self.get_all_signals(reference_date=reference_date)
        rows = [
            {
                "ticker":       s["ticker"],
                "score":        s.get("fundamental_score"),
                "rank":         s.get("quality_rank"),
                "gm":           s.get("gross_margin"),
                "op_margin":    s.get("operating_margin"),
                "rev_yoy":      s.get("revenue_yoy"),
                "fcf_margin":   s.get("fcf_margin"),
                "inv_yoy":      s.get("inventory_yoy"),
            }
            for s in signals.values() if "error" not in s
        ]
        df = pd.DataFrame(rows).sort_values("score", ascending=False)
        ref = reference_date or "latest"
        print(f"\n{'='*80}")
        print(f"  Fundamental Signals — {ref}  (top {top_n} by fundamental_score)")
        print(f"{'='*80}")
        print(df.head(top_n).to_string(index=False, float_format="{:.3f}".format))

    def export_signals_csv(
        self,
        reference_date: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """Export all signals to CSV. Returns the file path."""
        signals  = self.get_all_signals(reference_date=reference_date)
        rows     = [s for s in signals.values() if "error" not in s]
        df       = pd.DataFrame(rows)
        ref      = (reference_date or datetime.now().strftime("%Y-%m-%d")).replace("-", "")
        out_path = output_path or str(BASE_DIR / f"fundamental_signals_{ref}.csv")
        df.to_csv(out_path, index=False)
        print(f"[FundamentalAgent] Exported {len(df)} signals → {out_path}")
        return out_path


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = FundamentalAgent()

    # Single ticker
    sig = agent.get_signal("NVDA", reference_date="2024-06-30")
    print(f"\nNVDA fundamental signal ({sig['reference_date']}):")
    print(f"  fiscal_date       : {sig['fiscal_date']}")
    print(f"  fundamental_score : {sig['fundamental_score']}")
    print(f"  gross_margin      : {sig['gross_margin']}")
    print(f"  operating_margin  : {sig['operating_margin']}")
    print(f"  revenue_yoy       : {sig['revenue_yoy']}")
    print(f"  fcf_margin        : {sig['fcf_margin']}")
    print(f"  inventory_yoy     : {sig['inventory_yoy']}")
    print(f"  margin_trend      : {sig['margin_trend']}")

    # Summary table
    agent.print_summary(reference_date="2024-06-30")
