"""
Technical Agent

Computes price/volume technical indicators for semiconductor stocks from local OHLCV data.
No LLM — pure quantitative signal generation.

Data source:
  semiconductor_tickers_cleaned_2010_2024.csv  (wide format, MultiIndex columns)

Output cache:
  technical_cache/{TICKER}/indicators_{DATE}.json   (full indicator snapshot)
  technical_cache/{TICKER}/signal_{DATE}.json        (composite signal for Orchestrator)
"""

import json
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_DIR    = Path("d:/Investment_AI_agent")
PRICE_CSV   = BASE_DIR / "semiconductor_tickers_cleaned_2010_2024.csv"
CACHE_DIR   = BASE_DIR / "technical_cache"
CACHE_DIR.mkdir(exist_ok=True)

TICKER_LIST = [
    "TSM", "INTC", "NVDA", "ASML", "AVGO", "TXN", "QCOM", "MU",
    "AMAT", "AMD", "ADI", "LRCX", "KLAC", "MCHP", "SWKS", "MRVL",
    "TER", "ON", "MPWR", "ENTG", "MKSI", "SLAB", "CRUS", "SMTC",
]

# Sub-sector mapping
SUBSECTORS = {
    "Equipment":  ["AMAT", "LRCX", "KLAC", "ENTG"],
    "Foundry":    ["TSM"],
    "Fabless":    ["NVDA", "AMD", "QCOM", "AVGO", "MRVL"],
    "Analog":     ["ADI", "TXN", "MCHP", "ON", "SWKS", "SLAB", "CRUS", "SMTC"],
    "Memory":     ["MU"],
    "Diversified":["INTC", "MKSI", "MPWR", "TER", "ASML"],
}

# Indicator parameters
RSI_PERIOD      = 14
ATR_PERIOD      = 14
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
BB_PERIOD       = 20
BB_STD          = 2.0
VOL_RATIO_WIN   = 20   # days for volume baseline
HV_WINDOWS      = [20, 60]          # historical volatility windows (days)
ROC_WINDOWS     = [20, 60, 120, 252] # rate of change windows (days)
RS_WINDOW       = 60                # relative strength window (days)


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

def _load_price_data() -> dict[str, pd.DataFrame]:
    """
    Load the wide-format CSV and return a dict of {ticker: DataFrame}.
    Each DataFrame has columns [open, high, low, close, volume] indexed by date.
    """
    raw = pd.read_csv(PRICE_CSV, header=[0, 1])

    # Row 0 is a stray header artefact — drop it
    raw = raw.iloc[1:].reset_index(drop=True)

    # Date column sits at ('Price', 'Ticker')
    dates = pd.to_datetime(raw[("Price", "Ticker")], errors="coerce")
    raw = raw[dates.notna()].copy()
    dates = dates[dates.notna()].reset_index(drop=True)
    raw = raw.reset_index(drop=True)

    price_data = {}
    for ticker in TICKER_LIST:
        try:
            df = pd.DataFrame({
                "open":   pd.to_numeric(raw[("Open",   ticker)], errors="coerce").values,
                "high":   pd.to_numeric(raw[("High",   ticker)], errors="coerce").values,
                "low":    pd.to_numeric(raw[("Low",    ticker)], errors="coerce").values,
                "close":  pd.to_numeric(raw[("Close",  ticker)], errors="coerce").values,
                "volume": pd.to_numeric(raw[("Volume", ticker)], errors="coerce").values,
            }, index=dates.values)
            df.index.name = "date"
            df = df.dropna(subset=["close"]).sort_index()
            price_data[ticker] = df
        except KeyError:
            print(f"[WARNING] {ticker} not found in CSV, skipping.")
    return price_data


# ── Technical Indicator Calculations ──────────────────────────────────────────

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    fast   = _ema(close, MACD_FAST)
    slow   = _ema(close, MACD_SLOW)
    macd   = fast - slow
    signal = _ema(macd, MACD_SIGNAL)
    hist   = macd - signal
    return macd, signal, hist


def _bollinger(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, middle, lower)."""
    mid   = _sma(close, BB_PERIOD)
    std   = close.rolling(BB_PERIOD, min_periods=BB_PERIOD).std()
    upper = mid + BB_STD * std
    lower = mid - BB_STD * std
    return upper, mid, lower


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = ATR_PERIOD) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _historical_volatility(close: pd.Series, window: int) -> pd.Series:
    """Annualised historical volatility from log returns."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window, min_periods=window).std() * np.sqrt(252)


def _roc(close: pd.Series, window: int) -> pd.Series:
    return (close / close.shift(window) - 1)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators for a single ticker DataFrame. Returns augmented DataFrame."""
    out = df.copy()

    # Moving averages
    out["sma50"]  = _sma(df["close"], 50)
    out["sma200"] = _sma(df["close"], 200)
    out["ema20"]  = _ema(df["close"], 20)

    # Price vs MA ratios
    out["price_vs_sma50"]  = df["close"] / out["sma50"]
    out["price_vs_sma200"] = df["close"] / out["sma200"]

    # Trend state: golden/death cross
    out["golden_cross"] = (out["sma50"] > out["sma200"]).astype(int)
    out["cross_signal"] = out["golden_cross"].diff()  # +1 golden, -1 death

    # RSI
    out["rsi14"] = _rsi(df["close"])

    # MACD
    out["macd"], out["macd_signal"], out["macd_hist"] = _macd(df["close"])

    # Bollinger Bands
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = _bollinger(df["close"])
    bb_range = out["bb_upper"] - out["bb_lower"]
    out["bb_pct"]   = (df["close"] - out["bb_lower"]) / bb_range.replace(0, np.nan)
    out["bb_width"] = bb_range / out["bb_mid"]

    # ATR (normalised)
    out["atr14"]     = _atr(df["high"], df["low"], df["close"])
    out["atr_pct"]   = out["atr14"] / df["close"]

    # Historical volatility
    for w in HV_WINDOWS:
        out[f"hv_{w}d"] = _historical_volatility(df["close"], w)

    # Rate of Change
    for w in ROC_WINDOWS:
        out[f"roc_{w}d"] = _roc(df["close"], w)

    # Volume
    out["vol_sma20"]   = _sma(df["volume"], VOL_RATIO_WIN)
    out["vol_ratio"]   = df["volume"] / out["vol_sma20"]

    # OBV slope (linear regression over last 20 days)
    obv = _obv(df["close"], df["volume"])
    out["obv"] = obv
    out["obv_slope"] = (
        obv.rolling(VOL_RATIO_WIN, min_periods=VOL_RATIO_WIN)
        .apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
    )

    return out


# ── Cross-Sectional Relative Strength ─────────────────────────────────────────

def _add_relative_strength(
    all_indicators: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Adds to each ticker's DataFrame:
      rs_sector_{w}d   : return vs equal-weight sector average over window w
      rs_subsector_{w}d: return vs equal-weight sub-sector average
      rank_{w}d        : percentile rank (0–1) within 24 stocks by ROC
    """
    # Build a close matrix: index=date, columns=ticker
    close_matrix = pd.DataFrame({
        t: df["close"] for t, df in all_indicators.items()
    }).sort_index()

    # Sector-wide equal-weight return
    sector_returns = close_matrix.pct_change(RS_WINDOW, fill_method=None)
    sector_avg     = sector_returns.mean(axis=1)

    # Sub-sector averages
    subsector_avg = {}
    for subsector, members in SUBSECTORS.items():
        cols = [m for m in members if m in close_matrix.columns]
        if cols:
            subsector_avg[subsector] = sector_returns[cols].mean(axis=1)

    # Assign per ticker
    ticker_to_subsector = {
        t: s for s, members in SUBSECTORS.items() for t in members
    }

    for ticker, df in all_indicators.items():
        ticker_ret = close_matrix[ticker].pct_change(RS_WINDOW, fill_method=None)

        df[f"rs_sector_{RS_WINDOW}d"] = ticker_ret - sector_avg

        sub = ticker_to_subsector.get(ticker)
        if sub and sub in subsector_avg:
            df[f"rs_subsector_{RS_WINDOW}d"] = ticker_ret - subsector_avg[sub]
        else:
            df[f"rs_subsector_{RS_WINDOW}d"] = np.nan

        # Percentile rank across 24 stocks for each ROC window
        for w in [20, 60, 252]:
            roc_col = f"roc_{w}d"
            if roc_col in df.columns:
                rank_series = sector_returns.rank(axis=1, pct=True)[ticker] if w == RS_WINDOW else (
                    close_matrix.pct_change(w, fill_method=None).rank(axis=1, pct=True)[ticker]
                )
                df[f"rank_{w}d"] = rank_series

        all_indicators[ticker] = df

    return all_indicators


# ── Signal Extraction ──────────────────────────────────────────────────────────

def _trend_state(row: pd.Series) -> str:
    """Classify trend as uptrend / downtrend / sideways from a single row."""
    above_200 = row.get("price_vs_sma200", np.nan)
    golden    = row.get("golden_cross", np.nan)
    roc_60    = row.get("roc_60d", np.nan)

    if pd.isna(above_200) or pd.isna(golden) or pd.isna(roc_60):
        return "unknown"
    if above_200 > 1.02 and golden == 1 and roc_60 > 0.05:
        return "uptrend"
    if above_200 < 0.98 and golden == 0 and roc_60 < -0.05:
        return "downtrend"
    return "sideways"


def _compute_momentum_score(row: pd.Series) -> float:
    """
    Composite momentum score in [-1, +1] from:
      ROC multi-window (40%), RSI (20%), MACD direction (20%), Sector Rank (20%)
    """
    scores = []
    weights = []

    # ROC multi-window: normalise each window to [-1,+1] with soft tanh clamp
    roc_windows_weights = [(20, 0.10), (60, 0.15), (120, 0.10), (252, 0.15)]
    for w, wt in roc_windows_weights:
        v = row.get(f"roc_{w}d", np.nan)
        if not pd.isna(v):
            # tanh maps unbounded return → (-1, +1); scale by typical semi vol
            scores.append(float(np.tanh(v / 0.30)))
            weights.append(wt)

    # RSI: 50 = neutral, map to [-1, +1]
    rsi = row.get("rsi14", np.nan)
    if not pd.isna(rsi):
        scores.append((rsi - 50) / 50)
        weights.append(0.20)

    # MACD histogram direction
    macd_hist = row.get("macd_hist", np.nan)
    macd_val  = row.get("macd", np.nan)
    if not pd.isna(macd_hist) and not pd.isna(macd_val) and macd_val != 0:
        scores.append(float(np.tanh(macd_hist / abs(macd_val + 1e-9))))
        weights.append(0.20)

    # Sector rank percentile: 1.0 → +1, 0.0 → -1
    rank = row.get("rank_60d", np.nan)
    if not pd.isna(rank):
        scores.append(rank * 2 - 1)
        weights.append(0.20)

    if not scores:
        return 0.0

    total_w = sum(weights)
    score   = sum(s * w for s, w in zip(scores, weights)) / total_w
    return float(np.clip(score, -1.0, 1.0))


def _extract_signal(ticker: str, df: pd.DataFrame,
                    reference_date: Optional[str] = None) -> dict:
    """Extract a single-row signal dict for a given reference date."""
    if reference_date is None:
        ref = df.index[-1]
    else:
        ref = pd.Timestamp(reference_date)
        # Use latest available date on or before reference_date
        valid = df.index[df.index <= ref]
        if valid.empty:
            raise ValueError(f"{ticker}: no data on or before {reference_date}")
        ref = valid[-1]

    row = df.loc[ref]

    def _f(col: str) -> Optional[float]:
        v = row.get(col, np.nan)
        return None if pd.isna(v) else round(float(v), 6)

    momentum_score = _compute_momentum_score(row)
    trend          = _trend_state(row)

    signal = {
        "ticker":         ticker,
        "reference_date": str(ref.date()),
        "analysis_date":  datetime.now().strftime("%Y-%m-%d"),

        # ── Composite outputs (for Orchestrator) ──────────────────────────────
        "momentum_score":        round(momentum_score, 4),
        "sector_rank_60d":       _f("rank_60d"),
        "trend_state":           trend,

        # ── Trend indicators ──────────────────────────────────────────────────
        "sma50":                 _f("sma50"),
        "sma200":                _f("sma200"),
        "ema20":                 _f("ema20"),
        "price_vs_sma50":        _f("price_vs_sma50"),
        "price_vs_sma200":       _f("price_vs_sma200"),
        "golden_cross":          int(row.get("golden_cross", 0)),

        # ── Momentum ──────────────────────────────────────────────────────────
        "rsi14":                 _f("rsi14"),
        "macd":                  _f("macd"),
        "macd_signal":           _f("macd_signal"),
        "macd_hist":             _f("macd_hist"),
        "roc_20d":               _f("roc_20d"),
        "roc_60d":               _f("roc_60d"),
        "roc_120d":              _f("roc_120d"),
        "roc_252d":              _f("roc_252d"),

        # ── Overbought / oversold ─────────────────────────────────────────────
        "bb_pct":                _f("bb_pct"),
        "bb_width":              _f("bb_width"),

        # ── Volatility ────────────────────────────────────────────────────────
        "atr_pct":               _f("atr_pct"),
        "hv_20d":                _f("hv_20d"),
        "hv_60d":                _f("hv_60d"),

        # ── Volume ────────────────────────────────────────────────────────────
        "vol_ratio":             _f("vol_ratio"),
        "obv_slope_sign":        int(np.sign(_f("obv_slope") or 0)),

        # ── Cross-sectional relative strength ────────────────────────────────
        "rs_sector_60d":         _f(f"rs_sector_{RS_WINDOW}d"),
        "rs_subsector_60d":      _f(f"rs_subsector_{RS_WINDOW}d"),
        "rank_20d":              _f("rank_20d"),
        "rank_60d":              _f("rank_60d"),
        "rank_252d":             _f("rank_252d"),
    }

    return signal


# ── TechnicalAgent Class ───────────────────────────────────────────────────────

class TechnicalAgent:
    """
    Computes and caches technical indicators for semiconductor stocks.

    Usage:
        agent = TechnicalAgent()
        signal = agent.get_signal("NVDA", reference_date="2024-06-30")
        all    = agent.get_all_signals(reference_date="2024-06-30")
    """

    def __init__(self):
        print("[TechnicalAgent] Loading price data...")
        raw_data       = _load_price_data()
        print(f"[TechnicalAgent] Computing indicators for {len(raw_data)} tickers...")
        indicators     = {t: _compute_all_indicators(df) for t, df in raw_data.items()}
        print("[TechnicalAgent] Adding cross-sectional relative strength...")
        self._indicators = _add_relative_strength(indicators)
        print("[TechnicalAgent] Ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_signal(
        self,
        ticker: str,
        reference_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> dict:
        """
        Return the technical signal dict for a single ticker.
        Cached per (ticker, date) — recomputation is fast but avoids redundant work.
        """
        ref_label = reference_date or "latest"
        cache_path = _cache_dir(ticker) / f"signal_{ref_label}.json"

        if not force_refresh:
            cached = _load_json(cache_path)
            if cached:
                return cached

        if ticker not in self._indicators:
            raise ValueError(f"Ticker {ticker} not in dataset.")

        signal = _extract_signal(ticker, self._indicators[ticker], reference_date)
        _save_json(signal, cache_path)
        return signal

    def get_all_signals(
        self,
        ticker_list: Optional[list[str]] = None,
        reference_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> dict[str, dict]:
        """Return signals for all tickers. Dict keyed by ticker."""
        tickers = ticker_list or TICKER_LIST
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.get_signal(ticker, reference_date, force_refresh)
            except Exception as e:
                print(f"[WARNING] {ticker}: {e}")
                results[ticker] = {"ticker": ticker, "error": str(e)}
        return results

    def print_summary(
        self,
        reference_date: Optional[str] = None,
        top_n: int = 10,
    ) -> None:
        """Print a ranked summary table sorted by momentum_score."""
        signals = self.get_all_signals(reference_date=reference_date)
        rows = [
            {
                "ticker":      s["ticker"],
                "trend":       s.get("trend_state", "?"),
                "momentum":    s.get("momentum_score"),
                "rank_60d":    s.get("rank_60d"),
                "roc_252d":    s.get("roc_252d"),
                "rsi14":       s.get("rsi14"),
                "rs_sector":   s.get("rs_sector_60d"),
            }
            for s in signals.values() if "error" not in s
        ]
        df = pd.DataFrame(rows).sort_values("momentum", ascending=False)
        ref = reference_date or "latest"
        print(f"\n{'='*72}")
        print(f"  Technical Signals — {ref}  (top {top_n} by momentum)")
        print(f"{'='*72}")
        print(df.head(top_n).to_string(index=False, float_format="{:.3f}".format))

    def export_signals_csv(
        self,
        reference_date: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """Export all signals to a CSV file. Returns the file path."""
        signals  = self.get_all_signals(reference_date=reference_date)
        rows     = [s for s in signals.values() if "error" not in s]
        df       = pd.DataFrame(rows)
        ref      = (reference_date or datetime.now().strftime("%Y-%m-%d")).replace("-", "")
        out_path = output_path or str(BASE_DIR / f"technical_signals_{ref}.csv")
        df.to_csv(out_path, index=False)
        print(f"[TechnicalAgent] Exported {len(df)} signals → {out_path}")
        return out_path


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = TechnicalAgent()

    # Single ticker
    sig = agent.get_signal("NVDA", reference_date="2024-06-28")
    print(f"\nNVDA signal ({sig['reference_date']}):")
    print(f"  momentum_score  : {sig['momentum_score']}")
    print(f"  trend_state     : {sig['trend_state']}")
    print(f"  rsi14           : {sig['rsi14']}")
    print(f"  roc_252d        : {sig['roc_252d']}")
    print(f"  rank_60d        : {sig['rank_60d']}")
    print(f"  rs_sector_60d   : {sig['rs_sector_60d']}")

    # Summary table
    agent.print_summary(reference_date="2024-06-28")
