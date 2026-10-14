"""
MacroAgent

Applies macroeconomic overlay to ArbiterAgent's verdict, producing a
macro-adjusted score and regime classification for each semiconductor stock.

Called once per ticker per trading day. Macro data (monthly/quarterly) is
pre-fetched to 'Macro data/' by model.py — this agent reads from disk only,
no live API calls.

Architecture:
  1. Load all 10 macro indicator JSON files (newest-first arrays)
  2. Compute quantitative signals — yield curve, rate trend, inflation, GDP, demand
  3. Identify ticker's semiconductor subsector
  4. Call LLM with computed signals + arbiter verdict → structured output

Inputs (per-ticker):
  arbiter_verdict : dict from ArbiterAgent.get_verdict()
      Required keys: arbiter_score, verdict, confidence, key_risk

Output schema:
  ticker                  str
  date                    str   YYYY-MM-DD
  subsector               str   "ai_datacenter" | "equipment" | "analog" | "memory" | "foundry"
  macro_regime            str   "expansion" | "peak" | "contraction" | "trough"
  yield_curve_spread      float  10Y - 2Y in pct pts
  rate_environment        float  [-1, +1]  (-1=restrictive, +1=accommodative)
  inflation_trend         float  [-1, +1]  (-1=rising inflation, +1=declining)
  gdp_momentum            float  [-1, +1]
  demand_signal           float  [-1, +1]  composite of durables + retail
  regime_composite_score  float  weighted aggregate of above signals
  subsector_adjustment    float  [-0.3, +0.3] macro tailwind/headwind for subsector
  macro_adjusted_score    float  [-1, +1]  arbiter_score + subsector_adjustment, clamped
  override_flags          list[str]  e.g. ["yield_curve_inverted", "rates_restrictive"]
  macro_summary           str   2-3 sentence macro narrative
  data_as_of              str   date of most recent macro data used
  arbiter_score_input     float passed-through from arbiter
  arbiter_verdict_input   str   passed-through from arbiter

Cache:
  macro_cache/{TICKER}/macro_{DATE}.json
"""

import os
import json
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

import anthropic

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_DIR  = Path("d:/Investment_AI_agent")
MACRO_DIR = BASE_DIR / "Macro data"
CACHE_DIR = BASE_DIR / "macro_cache"
CACHE_DIR.mkdir(exist_ok=True)

MODEL = "claude-sonnet-4-6"

TICKER_LIST = [
    "TSM", "INTC", "NVDA", "ASML", "AVGO", "TXN", "QCOM", "MU",
    "AMAT", "AMD", "ADI", "LRCX", "KLAC", "MCHP", "SWKS", "MRVL",
    "TER", "ON", "MPWR", "ENTG", "MKSI", "SLAB", "CRUS", "SMTC",
]

# Semiconductor subsector classification
# ai_datacenter : AI inference/training chips, networking ASICs
# equipment     : wafer fab equipment and materials
# analog        : industrial, auto, consumer analog/mixed-signal
# memory        : DRAM / NAND
# foundry       : contract manufacturing, EDA, IP
SUBSECTOR_MAP = {
    "NVDA": "ai_datacenter", "AMD": "ai_datacenter",
    "MRVL": "ai_datacenter", "AVGO": "ai_datacenter",
    "AMAT": "equipment",     "LRCX": "equipment",
    "KLAC": "equipment",     "TER":  "equipment",
    "ENTG": "equipment",     "MKSI": "equipment",
    "TXN":  "analog",        "ADI":  "analog",
    "MCHP": "analog",        "ON":   "analog",
    "SWKS": "analog",        "MPWR": "analog",
    "SLAB": "analog",        "CRUS": "analog",
    "SMTC": "analog",
    "MU":   "memory",
    "TSM":  "foundry",       "INTC": "foundry",
    "ASML": "foundry",       "QCOM": "foundry",
}

client = anthropic.Anthropic()


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


# ── Macro Data Helpers ─────────────────────────────────────────────────────────

def _load_series(indicator: str, n: int = 14) -> list[dict]:
    """Load last n valid data points from a macro indicator JSON (newest first)."""
    path = MACRO_DIR / f"{indicator}.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    data = d.get("data", [])
    valid = [x for x in data if str(x.get("value", ".")).strip() not in (".", "", "None")]
    return valid[:n]


def _f(series: list[dict], idx: int) -> Optional[float]:
    """Safe float extraction at index."""
    try:
        return float(series[idx]["value"])
    except (IndexError, ValueError, KeyError, TypeError):
        return None


# ── Quantitative Signal Computation ───────────────────────────────────────────

def _compute_macro_signals() -> dict:
    """
    Pure Python — no LLM calls.
    Returns a dict of all quantitative macro signals for use in the LLM prompt.
    """

    # ── Treasury yields ──────────────────────────────────────────────────────
    y10 = _load_series("TREASURY_YIELD_10Y", 14)
    y2  = _load_series("TREASURY_YIELD_2Y",  14)

    y10_latest = _f(y10, 0)
    y2_latest  = _f(y2,  0)
    y10_3m     = _f(y10, 3)
    y2_3m      = _f(y2,  3)

    yield_spread      = round(y10_latest - y2_latest, 3)       if (y10_latest and y2_latest) else None
    yield_spread_3m   = round(_f(y10, 3) - _f(y2, 3), 3)      if (_f(y10, 3) and _f(y2, 3)) else None
    yield_spread_trend = round(yield_spread - yield_spread_3m, 3) if (yield_spread is not None and yield_spread_3m is not None) else None

    # ── Federal Funds Rate ────────────────────────────────────────────────────
    ffr = _load_series("FEDERAL_FUNDS_RATE", 14)
    ffr_latest  = _f(ffr, 0)
    ffr_3m_ago  = _f(ffr, 3)
    ffr_12m_ago = _f(ffr, 12)

    ffr_3m_delta  = round(ffr_latest - ffr_3m_ago,  3) if (ffr_latest and ffr_3m_ago)  else None
    ffr_12m_delta = round(ffr_latest - ffr_12m_ago, 3) if (ffr_latest and ffr_12m_ago) else None

    # Rate environment score: -1=very restrictive, +1=accommodative
    rate_env = 0.0
    if ffr_latest is not None:
        if   ffr_latest > 5.0:  base = -0.8
        elif ffr_latest > 4.0:  base = -0.5
        elif ffr_latest > 2.5:  base =  0.0
        else:                   base =  0.7
        # Adjust for direction: falling rates are bullish even if still high
        direction_adj = 0.0
        if ffr_3m_delta is not None:
            if   ffr_3m_delta < -0.5:  direction_adj =  0.3   # aggressive easing
            elif ffr_3m_delta < 0:     direction_adj =  0.15  # mild easing
            elif ffr_3m_delta > 0.5:   direction_adj = -0.2   # aggressive hiking
            elif ffr_3m_delta > 0:     direction_adj = -0.1   # mild hiking
        rate_env = max(-1.0, min(1.0, base + direction_adj))

    # ── CPI / Inflation ───────────────────────────────────────────────────────
    cpi = _load_series("CPI", 14)

    cpi_latest   = _f(cpi, 0)
    cpi_3m_ago   = _f(cpi, 3)
    cpi_12m_ago  = _f(cpi, 12)

    cpi_yoy = round((cpi_latest - cpi_12m_ago) / cpi_12m_ago * 100, 2) \
              if (cpi_latest and cpi_12m_ago) else None
    cpi_3m_ann = round((cpi_latest - cpi_3m_ago) / cpi_3m_ago * 400, 2) \
                 if (cpi_latest and cpi_3m_ago) else None

    # Inflation trend score: +1=below target / declining (good), -1=high/rising
    infl_trend = 0.0
    if cpi_yoy is not None:
        if   cpi_yoy < 2.0:  base = 0.8
        elif cpi_yoy < 3.0:  base = 0.3
        elif cpi_yoy < 4.0:  base = -0.3
        else:                base = -0.8
        # Adjust for direction
        direction_adj = 0.0
        if cpi_3m_ann is not None:
            gap = cpi_3m_ann - cpi_yoy
            if   gap < -1.0: direction_adj =  0.2   # deceleration
            elif gap < 0:    direction_adj =  0.1
            elif gap > 1.0:  direction_adj = -0.2   # re-acceleration
            elif gap > 0:    direction_adj = -0.1
        infl_trend = max(-1.0, min(1.0, base + direction_adj))

    # ── Real GDP ──────────────────────────────────────────────────────────────
    gdp = _load_series("REAL_GDP", 8)

    gdp_latest = _f(gdp, 0)
    gdp_prev   = _f(gdp, 1)
    gdp_2q_ago = _f(gdp, 2)

    gdp_qoq_ann = round((gdp_latest - gdp_prev) / gdp_prev * 400, 2) \
                  if (gdp_latest and gdp_prev) else None
    gdp_prev_qoq = round((gdp_prev - gdp_2q_ago) / gdp_2q_ago * 400, 2) \
                   if (gdp_prev and gdp_2q_ago) else None
    gdp_accel = round(gdp_qoq_ann - gdp_prev_qoq, 2) \
                if (gdp_qoq_ann is not None and gdp_prev_qoq is not None) else None

    gdp_momentum = 0.0
    if gdp_qoq_ann is not None:
        if   gdp_qoq_ann > 3.0:  gdp_momentum = 0.8
        elif gdp_qoq_ann > 1.5:  gdp_momentum = 0.4
        elif gdp_qoq_ann > 0:    gdp_momentum = 0.0
        elif gdp_qoq_ann > -2.0: gdp_momentum = -0.5
        else:                    gdp_momentum = -1.0
        # Acceleration bonus
        if gdp_accel is not None:
            gdp_momentum = max(-1.0, min(1.0, gdp_momentum + (0.1 if gdp_accel > 0 else -0.1)))

    # ── Durable Goods ─────────────────────────────────────────────────────────
    dur = _load_series("DURABLES", 6)
    dur_latest = _f(dur, 0)
    dur_3m_ago = _f(dur, 3)
    durables_3m_pct = round((dur_latest - dur_3m_ago) / dur_3m_ago * 100, 2) \
                      if (dur_latest and dur_3m_ago) else None

    # ── Retail Sales ──────────────────────────────────────────────────────────
    ret = _load_series("RETAIL_SALES", 6)
    ret_latest = _f(ret, 0)
    ret_3m_ago = _f(ret, 3)
    retail_3m_pct = round((ret_latest - ret_3m_ago) / ret_3m_ago * 100, 2) \
                    if (ret_latest and ret_3m_ago) else None

    # Composite demand signal [-1, +1]
    demand_signal = 0.0
    count = 0
    if durables_3m_pct is not None:
        demand_signal += max(-1, min(1, durables_3m_pct / 5.0))
        count += 1
    if retail_3m_pct is not None:
        demand_signal += max(-1, min(1, retail_3m_pct / 5.0))
        count += 1
    if count > 0:
        demand_signal = round(demand_signal / count, 3)

    # ── Unemployment ──────────────────────────────────────────────────────────
    unemp = _load_series("UNEMPLOYMENT", 6)
    unemp_latest = _f(unemp, 0)
    unemp_3m_ago = _f(unemp, 3)
    unemp_trend  = round(unemp_latest - unemp_3m_ago, 2) \
                   if (unemp_latest and unemp_3m_ago) else None

    # Nonfarm payrolls (level change, thousands)
    nfp = _load_series("NONFARM_PAYROLL", 4)
    nfp_latest = _f(nfp, 0)
    nfp_prev   = _f(nfp, 1)

    # ── Regime composite score ────────────────────────────────────────────────
    # Weighted average of normalized signals → [-1, +1]
    regime_score = (
        gdp_momentum  * 0.35 +
        infl_trend    * 0.25 +
        rate_env      * 0.25 +
        demand_signal * 0.15
    )
    # Yield curve inversion penalty
    if yield_spread is not None and yield_spread < -0.2:
        regime_score -= 0.15
    regime_score = max(-1.0, min(1.0, round(regime_score, 3)))

    # ── Macro regime classification ───────────────────────────────────────────
    if regime_score > 0.3:
        macro_regime = "expansion"
    elif regime_score > 0.0:
        macro_regime = "peak"
    elif regime_score > -0.3:
        macro_regime = "contraction"
    else:
        macro_regime = "trough"

    # ── Override flags ────────────────────────────────────────────────────────
    override_flags = []
    if yield_spread is not None and yield_spread < -0.3:
        override_flags.append("yield_curve_inverted")
    if ffr_latest is not None and ffr_latest > 5.0:
        override_flags.append("rates_restrictive")
    if cpi_yoy is not None and cpi_yoy > 4.0:
        override_flags.append("inflation_elevated")
    if gdp_qoq_ann is not None and gdp_qoq_ann < 0:
        override_flags.append("gdp_negative")
    if unemp_trend is not None and unemp_trend > 0.4:
        override_flags.append("unemployment_rising")
    if ffr_3m_delta is not None and ffr_3m_delta < -0.25:
        override_flags.append("fed_easing")
    if ffr_3m_delta is not None and ffr_3m_delta > 0.25:
        override_flags.append("fed_hiking")

    # ── data_as_of ────────────────────────────────────────────────────────────
    data_as_of = y10[0]["date"] if y10 else "unknown"

    return {
        # Raw values
        "y10_rate":               y10_latest,
        "y2_rate":                y2_latest,
        "yield_curve_spread":     yield_spread,
        "yield_curve_trend_3m":   yield_spread_trend,
        "fed_funds_rate":         ffr_latest,
        "fed_funds_3m_delta":     ffr_3m_delta,
        "fed_funds_12m_delta":    ffr_12m_delta,
        "cpi_yoy_pct":            cpi_yoy,
        "cpi_3m_annualized_pct":  cpi_3m_ann,
        "gdp_qoq_annualized_pct": gdp_qoq_ann,
        "gdp_prev_qoq_ann_pct":   gdp_prev_qoq,
        "gdp_acceleration":       gdp_accel,
        "durables_3m_pct":        durables_3m_pct,
        "retail_3m_pct":          retail_3m_pct,
        "unemployment_latest":    unemp_latest,
        "unemployment_3m_delta":  unemp_trend,
        "nfp_latest_k":           nfp_latest,
        "nfp_prev_k":             nfp_prev,
        # Derived scores
        "rate_environment":       round(rate_env, 3),
        "inflation_trend":        round(infl_trend, 3),
        "gdp_momentum":           round(gdp_momentum, 3),
        "demand_signal":          demand_signal,
        # Regime
        "regime_composite_score": regime_score,
        "macro_regime":           macro_regime,
        "override_flags":         override_flags,
        "data_as_of":             data_as_of,
    }


# ── System Prompt ──────────────────────────────────────────────────────────────

SYSTEM_MACRO = (
    "You are a senior macro strategist specializing in semiconductor sector investing. "
    "You receive pre-computed quantitative macroeconomic signals and an ArbiterAgent verdict "
    "for a specific semiconductor ticker. "
    "Your job is to synthesize the macro signals into a regime assessment, determine how the "
    "current macro environment specifically affects this ticker's subsector, and produce a "
    "macro-adjusted conviction score. "
    "Semiconductor subsectors respond very differently to macro regimes:\n"
    "  - ai_datacenter: driven by hyperscaler AI capex, less cyclical than other semis. "
    "    Resilient in mild contractions if AI capex continues; very sensitive to rate hikes "
    "    via high PE multiples.\n"
    "  - equipment: most cyclically sensitive, leads the semiconductor cycle. "
    "    Suffers first in downturns (capex cuts), recovers first when fabs re-invest.\n"
    "  - analog: tied to industrial and automotive end-markets. Longer product cycles, "
    "    less volatile than digital semis but still cyclical.\n"
    "  - memory: most volatile. Prices collapse during oversupply/contraction, "
    "    spike during recovery/expansion. Very sensitive to demand signals.\n"
    "  - foundry: utilization-driven. Moderately cyclical, follows overall chip demand.\n"
    "Output ONLY valid JSON. No markdown, no code fences, no commentary outside the JSON."
)


# ── LLM Call ──────────────────────────────────────────────────────────────────

def _call_llm(prompt: str, max_tokens: int = 4000, retries: int = 5) -> str:
    for attempt in range(retries + 1):
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=max_tokens,
                thinking={"type": "adaptive"},
                system=SYSTEM_MACRO,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                message = stream.get_final_message()
                for block in reversed(message.content):
                    if block.type == "text":
                        return block.text
                raise ValueError("No text block in LLM response.")
        except Exception as e:
            if attempt < retries:
                print(f"    [RETRY {attempt + 1}] {e}")
                wait = 2 ** (attempt + 1); time.sleep(wait)
            else:
                raise


# ── JSON Repair ────────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict:
    """Extract and parse JSON from LLM output, handling markdown fences."""
    text = raw.strip()
    # Strip markdown code fences if present
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            break
    # Find first '{' and last '}'
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM response: {raw[:200]}")
    return json.loads(text[start:end + 1])


# ── Prompt Builder ─────────────────────────────────────────────────────────────

def _build_prompt(ticker: str, subsector: str, signals: dict, arbiter_verdict: dict, date_str: str = "") -> str:
    arbiter_score   = arbiter_verdict.get("arbiter_score", 0.0)
    verdict_str     = arbiter_verdict.get("verdict", "neutral")
    confidence      = arbiter_verdict.get("confidence", 0.5)
    key_risk        = arbiter_verdict.get("key_risk", "Not specified")
    resolution_sum  = arbiter_verdict.get("resolution_summary", "")
    bull_wins       = arbiter_verdict.get("bull_wins_on", [])
    bear_wins       = arbiter_verdict.get("bear_wins_on", [])

    regime          = signals["macro_regime"]
    override_flags  = signals["override_flags"]
    data_as_of      = signals["data_as_of"]

    prompt = f"""
MACRO ANALYSIS REQUEST
======================
Ticker   : {ticker}
Subsector: {subsector}
Date     : {date_str}
Data as of: {data_as_of}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARBITER AGENT VERDICT (pre-macro)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
arbiter_score  : {arbiter_score:+.3f}  (range -1 to +1)
verdict        : {verdict_str}
confidence     : {confidence:.2f}
key_risk       : {key_risk}
resolution     : {resolution_sum}
bull_wins_on   : {bull_wins}
bear_wins_on   : {bear_wins}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUANTITATIVE MACRO SIGNALS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Preliminary regime classification: {regime}
Override flags: {override_flags if override_flags else 'none'}

INTEREST RATES
  10Y Treasury yield    : {signals['y10_rate']}%
  2Y Treasury yield     : {signals['y2_rate']}%
  Yield curve spread    : {signals['yield_curve_spread']} pct pts  (3-month change: {signals['yield_curve_trend_3m']})
  Federal Funds Rate    : {signals['fed_funds_rate']}%
  FFR 3-month change    : {signals['fed_funds_3m_delta']}%  (12-month change: {signals['fed_funds_12m_delta']}%)
  Rate environment score: {signals['rate_environment']} (−1=restrictive, +1=accommodative)

INFLATION
  CPI year-over-year    : {signals['cpi_yoy_pct']}%
  CPI 3-month annualized: {signals['cpi_3m_annualized_pct']}%
  Inflation trend score : {signals['inflation_trend']} (−1=rising, +1=declining)

GROWTH
  Real GDP QoQ ann.     : {signals['gdp_qoq_annualized_pct']}%  (prior quarter: {signals['gdp_prev_qoq_ann_pct']}%)
  GDP acceleration      : {signals['gdp_acceleration']}%  (positive = re-accelerating)
  GDP momentum score    : {signals['gdp_momentum']} (−1=recession, +1=strong growth)

DEMAND INDICATORS
  Durable goods (3-month): {signals['durables_3m_pct']}%
  Retail sales  (3-month): {signals['retail_3m_pct']}%
  Composite demand score : {signals['demand_signal']} (−1=weak, +1=strong)

LABOR MARKET
  Unemployment rate     : {signals['unemployment_latest']}%  (3-month change: {signals['unemployment_3m_delta']}%)
  Nonfarm payrolls      : {signals['nfp_latest_k']}K  (prior: {signals['nfp_prev_k']}K)

COMPOSITE
  Regime composite score: {signals['regime_composite_score']} (−1=contraction, +1=expansion)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Confirm or override the preliminary macro_regime based on your synthesis.
2. Assess how this macro regime specifically affects the '{subsector}' subsector — focus
   on this ticker's end markets, capex cycles, and rate sensitivity.
3. Set subsector_adjustment in [-0.3, +0.3]. This is the incremental macro tailwind (+)
   or headwind (−) beyond what the arbiter already captured in its stock-specific analysis.
   Use 0.0 if macro is neutral for this subsector right now.
4. Compute macro_adjusted_score = arbiter_score + subsector_adjustment, clamped to [-1, +1].
5. Write a 2-3 sentence macro_summary covering: regime, key driver, and implication for {ticker}.

Return EXACTLY this JSON (all fields required, no extras):
{{
  "macro_regime": "expansion|peak|contraction|trough",
  "subsector_adjustment": <float -0.3 to +0.3>,
  "macro_adjusted_score": <float -1.0 to +1.0>,
  "override_flags": [<list of strings, or empty>],
  "macro_summary": "<2-3 sentence string>",
  "subsector_reasoning": "<1-2 sentence explanation of the subsector_adjustment>"
}}
""".strip()
    return prompt


# ── MacroAgent Class ───────────────────────────────────────────────────────────

class MacroAgent:
    """
    MacroAgent: applies macro overlay to ArbiterAgent verdict.

    Usage:
        agent = MacroAgent()
        result = agent.get_signal("NVDA", arbiter_verdict_dict, date_str="2026-03-23")
    """

    def __init__(self):
        # Compute macro signals once per agent instance (shared across all tickers today)
        self._signals: Optional[dict] = None

    def _get_signals(self) -> dict:
        if self._signals is None:
            self._signals = _compute_macro_signals()
        return self._signals

    def get_signal(
        self,
        ticker: str,
        arbiter_verdict: dict,
        date_str: Optional[str] = None,
        force_refresh: bool = False,
    ) -> dict:
        """
        Compute macro-adjusted signal for one ticker.

        Args:
            ticker          : e.g. "NVDA"
            arbiter_verdict : dict from ArbiterAgent.get_verdict()
            date_str        : YYYY-MM-DD (defaults to today)
            force_refresh   : if True, ignore cache

        Returns:
            Full output dict (see module docstring for schema).
        """
        today = date_str or datetime.now().strftime("%Y-%m-%d")
        cache_path = _cache_dir(ticker) / f"macro_{today}.json"

        if not force_refresh:
            cached = _load_json(cache_path)
            if cached:
                print(f"  [MacroAgent] {ticker}: loaded from cache ({today})")
                return cached

        ticker = ticker.upper()
        subsector = SUBSECTOR_MAP.get(ticker, "unknown")

        print(f"  [MacroAgent] {ticker} ({subsector}): computing macro signals...")
        signals = self._get_signals()

        print(f"  [MacroAgent] {ticker}: calling LLM (regime={signals['macro_regime']})...")
        prompt   = _build_prompt(ticker, subsector, signals, arbiter_verdict, date_str=today)
        raw      = _call_llm(prompt)
        llm_out  = _extract_json(raw)

        # Clamp macro_adjusted_score
        raw_adj  = arbiter_verdict.get("arbiter_score", 0.0) + llm_out.get("subsector_adjustment", 0.0)
        adj_score = max(-1.0, min(1.0, round(raw_adj, 4)))

        result = {
            "ticker":                  ticker,
            "date":                    today,
            "subsector":               subsector,
            # LLM outputs
            "macro_regime":            llm_out.get("macro_regime", signals["macro_regime"]),
            "subsector_adjustment":    round(float(llm_out.get("subsector_adjustment", 0.0)), 4),
            "macro_adjusted_score":    adj_score,
            "override_flags":          llm_out.get("override_flags", signals["override_flags"]),
            "macro_summary":           llm_out.get("macro_summary", ""),
            "subsector_reasoning":     llm_out.get("subsector_reasoning", ""),
            # Quantitative signals (pass-through for downstream agents)
            "yield_curve_spread":      signals["yield_curve_spread"],
            "rate_environment":        signals["rate_environment"],
            "inflation_trend":         signals["inflation_trend"],
            "gdp_momentum":            signals["gdp_momentum"],
            "demand_signal":           signals["demand_signal"],
            "regime_composite_score":  signals["regime_composite_score"],
            "fed_funds_rate":          signals["fed_funds_rate"],
            "cpi_yoy_pct":             signals["cpi_yoy_pct"],
            "gdp_qoq_annualized_pct":  signals["gdp_qoq_annualized_pct"],
            "data_as_of":              signals["data_as_of"],
            # Pass-through arbiter inputs
            "arbiter_score_input":     arbiter_verdict.get("arbiter_score"),
            "arbiter_verdict_input":   arbiter_verdict.get("verdict"),
            "arbiter_confidence":      arbiter_verdict.get("confidence"),
        }

        _save_json(result, cache_path)
        print(f"  [MacroAgent] {ticker}: done — regime={result['macro_regime']}, "
              f"adj={result['subsector_adjustment']:+.3f}, "
              f"final={result['macro_adjusted_score']:+.4f}")
        return result


# ── CLI Entry Point ────────────────────────────────────────────────────────────

def run_all(date_str: Optional[str] = None, force_refresh: bool = False):
    """
    Run MacroAgent for all tickers that have an arbiter cache for today.
    This is the daily batch entry point.
    """
    today = date_str or datetime.now().strftime("%Y-%m-%d")
    arbiter_cache = BASE_DIR / "arbiter_cache"

    agent   = MacroAgent()
    results = {}
    skipped = []

    print(f"\n{'='*60}")
    print(f"MacroAgent — daily run for {today}")
    print(f"{'='*60}")

    # Pre-compute signals once and print summary
    signals = agent._get_signals()
    print(f"\nMacro signals computed:")
    print(f"  Regime        : {signals['macro_regime']}  (composite={signals['regime_composite_score']:+.3f})")
    print(f"  Yield spread  : {signals['yield_curve_spread']} pct pts")
    print(f"  FFR           : {signals['fed_funds_rate']}%  (3m delta: {signals['fed_funds_3m_delta']}%)")
    print(f"  CPI YoY       : {signals['cpi_yoy_pct']}%")
    print(f"  GDP QoQ ann   : {signals['gdp_qoq_annualized_pct']}%")
    print(f"  Override flags: {signals['override_flags'] or 'none'}")
    print(f"  Data as of    : {signals['data_as_of']}\n")

    for ticker in TICKER_LIST:
        arbiter_path = arbiter_cache / ticker / f"arbiter_{today}.json"
        if not arbiter_path.exists():
            skipped.append(ticker)
            continue

        arbiter_verdict = _load_json(arbiter_path)
        if not arbiter_verdict:
            skipped.append(ticker)
            continue

        try:
            result = agent.get_signal(ticker, arbiter_verdict, date_str=today, force_refresh=force_refresh)
            results[ticker] = result
        except Exception as e:
            print(f"  [MacroAgent] {ticker}: ERROR — {e}")

    # Summary table
    if results:
        print(f"\n{'─'*70}")
        print(f"{'Ticker':<8} {'Subsector':<16} {'Regime':<14} {'Arbiter':>9} {'Adj':>6} {'Final':>9}")
        print(f"{'─'*70}")
        for t, r in sorted(results.items()):
            print(
                f"{t:<8} {r['subsector']:<16} {r['macro_regime']:<14} "
                f"{r['arbiter_score_input']:>+9.4f} {r['subsector_adjustment']:>+6.3f} "
                f"{r['macro_adjusted_score']:>+9.4f}"
            )
        print(f"{'─'*70}")

    if skipped:
        print(f"\nSkipped (no arbiter cache for {today}): {skipped}")

    print(f"\nMacroAgent complete: {len(results)} tickers processed.\n")
    return results


if __name__ == "__main__":
    import sys
    import re as _re

    # Only accept args that look like YYYY-MM-DD (ignore Jupyter kernel args like --f=...)
    _date_pattern = _re.compile(r"^\d{4}-\d{2}-\d{2}$")
    date_arg  = next((a for a in sys.argv[1:] if _date_pattern.match(a)), None)
    force_arg = "--force" in sys.argv

    run_all(date_str=date_arg, force_refresh=force_arg)
