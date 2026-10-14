"""
Bull Agent 

Synthesizes signals from 4 upstream agents and constructs a structured bullish
investment thesis for each semiconductor stock.

Upstream inputs (all as dicts from their respective agents):
  - technical  : TechnicalAgent.get_signal()
  - fundamental: FundamentalAgent.get_signal()
  - sec        : SECFilingAgent daily signal or latest filing analysis
  - sentiment  : SentimentAgent composite signal

Output cache:
  bull_cache/{TICKER}/bull_{DATE}.json

Output schema (consumed by debate layer and downstream Orchestrator):
  bullish_score       float [-1, +1]   overall conviction in bull case
  conviction_level    str              "high" | "medium" | "low" | "none"
  top_arguments       list[dict]       3-5 specific, quantified bull arguments
  bull_catalysts      list[str]        near-term potential catalysts
  acknowledged_risks  list[str]        bear risks the bull must concede
  debate_summary      str              2-3 sentence summary for debate use
  supporting_metrics  dict             key numbers from upstream agents
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
CACHE_DIR = BASE_DIR / "bull_cache"
CACHE_DIR.mkdir(exist_ok=True)

MODEL = "claude-sonnet-4-6"

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


# ── LLM Utilities ──────────────────────────────────────────────────────────────

SYSTEM_BULL = (
    "You are a seasoned long-only equity analyst specializing in semiconductor stocks. "
    "Your job is to construct the strongest possible bullish investment thesis "
    "using data provided from technical, fundamental, SEC filing, and sentiment signals. "
    "Be rigorous: only cite evidence that is actually present in the data. "
    "Acknowledge genuine risks honestly — a credible bull case does not ignore bears. "
    "Always respond with a single valid JSON object exactly matching the specified schema. "
    "Do not include any text outside the JSON object."
)


def _extract_json(text: str) -> dict:
    if "```" in text:
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines)
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found in LLM response:\n{text[:300]}")
    return json.loads(text[start:end])


def _call_llm(prompt: str, max_tokens: int = 6000, retries: int = 5) -> str:
    for attempt in range(retries + 1):
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=max_tokens,
                thinking={"type": "adaptive"},
                system=SYSTEM_BULL,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                message = stream.get_final_message()
                for block in reversed(message.content):
                    if block.type == "text":
                        return block.text
                raise ValueError("No text block in LLM response.")
        except Exception as e:
            if attempt < retries:
                print(f"    [RETRY {attempt+1}] {e}")
                wait = 2 ** (attempt + 1); time.sleep(wait)
            else:
                raise


# ── Input Packaging ────────────────────────────────────────────────────────────

def _safe(d: Optional[dict], key: str, default=None):
    """Safely extract a value from a signal dict."""
    if d is None:
        return default
    return d.get(key, default)


def _build_supporting_metrics(
    tech: Optional[dict],
    fund: Optional[dict],
    sec:  Optional[dict],
    sent: Optional[dict],
) -> dict:
    """Flatten key metrics from all 4 agents into a single summary dict."""
    return {
        # Technical
        "momentum_score":    _safe(tech, "momentum_score"),
        "trend_state":       _safe(tech, "trend_state"),
        "rsi14":             _safe(tech, "rsi14"),
        "roc_252d":          _safe(tech, "roc_252d"),
        "roc_60d":           _safe(tech, "roc_60d"),
        "rank_60d":          _safe(tech, "rank_60d"),
        "rs_sector_60d":     _safe(tech, "rs_sector_60d"),
        "golden_cross":      _safe(tech, "golden_cross"),
        "vol_ratio":         _safe(tech, "vol_ratio"),
        "bb_pct":            _safe(tech, "bb_pct"),

        # Fundamental
        "fundamental_score": _safe(fund, "fundamental_score"),
        "quality_rank":      _safe(fund, "quality_rank"),
        "gross_margin":      _safe(fund, "gross_margin"),
        "operating_margin":  _safe(fund, "operating_margin"),
        "revenue_yoy":       _safe(fund, "revenue_yoy"),
        "revenue_yoy_q":     _safe(fund, "revenue_yoy_q"),
        "revenue_acceleration": _safe(fund, "revenue_acceleration"),
        "fcf_margin":        _safe(fund, "fcf_margin"),
        "fcf_to_net_income": _safe(fund, "fcf_to_net_income"),
        "roic":              _safe(fund, "roic"),
        "rd_intensity":      _safe(fund, "rd_intensity"),
        "net_debt_to_ebitda":_safe(fund, "net_debt_to_ebitda"),
        "current_ratio":     _safe(fund, "current_ratio"),
        "inventory_yoy":     _safe(fund, "inventory_yoy"),
        "margin_trend":      _safe(fund, "margin_trend"),

        # SEC Filing
        "sec_tone_score":        _safe(sec, "tone_score"),
        "sec_business_momentum": _safe(sec, "business_momentum"),
        "sec_forward_guidance":  _safe(sec, "forward_guidance"),
        "sec_mgmt_confidence":   _safe(sec, "management_confidence"),
        "sec_risk_score":        _safe(sec, "risk_score"),
        "sec_geopolitical_risk": _safe(sec, "geopolitical_risk_present"),
        "sec_top_risks":         _safe(sec, "top_risks", []),
        "sec_key_themes":        _safe(sec, "key_themes", []),

        # Sentiment
        "news_score":        _safe(sent, "news_score"),
        "news_momentum":     _safe(sent, "momentum"),
        "news_z_score":      _safe(sent, "z_score"),
        "extreme_positive":  _safe(sent, "extreme_positive"),
        "call_tone":         _safe(sent, "call_tone_score"),
        "call_guidance":     _safe(sent, "call_forward_guidance"),
        "call_mgmt_confidence": _safe(sent, "call_management_confidence"),
    }


# ── Prompt Builder ─────────────────────────────────────────────────────────────

def _build_prompt(
    ticker: str,
    reference_date: str,
    subsector: str,
    metrics: dict,
    tech: Optional[dict],
    fund: Optional[dict],
    sec:  Optional[dict],
    sent: Optional[dict],
) -> str:

    def _fmt(v) -> str:
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    prompt = f"""You are constructing a bullish investment thesis for {ticker} ({subsector} sub-sector) as of {reference_date}.

Below is a structured summary of signals from four upstream analysis agents.
Use ONLY the data provided — do not invent or assume metrics not listed here.

══════════════════════════════════════════════════════
TECHNICAL SIGNALS
══════════════════════════════════════════════════════
momentum_score (composite, -1 to +1) : {_fmt(metrics['momentum_score'])}
trend_state                           : {_fmt(metrics['trend_state'])}
rsi14                                 : {_fmt(metrics['rsi14'])}
roc_60d  (2-month price return)       : {_fmt(metrics['roc_60d'])}
roc_252d (12-month price return)      : {_fmt(metrics['roc_252d'])}
rank_60d (sector percentile 0-1)      : {_fmt(metrics['rank_60d'])}
rs_sector_60d (excess vs sector)      : {_fmt(metrics['rs_sector_60d'])}
golden_cross (SMA50 > SMA200)         : {_fmt(metrics['golden_cross'])}
vol_ratio (today vs 20d avg volume)   : {_fmt(metrics['vol_ratio'])}
bb_pct (Bollinger position 0-1)       : {_fmt(metrics['bb_pct'])}

══════════════════════════════════════════════════════
FUNDAMENTAL SIGNALS
══════════════════════════════════════════════════════
fundamental_score (composite, -1 to +1): {_fmt(metrics['fundamental_score'])}
quality_rank (sector percentile 0-1)   : {_fmt(metrics['quality_rank'])}
gross_margin                           : {_fmt(metrics['gross_margin'])}
operating_margin                       : {_fmt(metrics['operating_margin'])}
roic                                   : {_fmt(metrics['roic'])}
rd_intensity (R&D / Revenue)           : {_fmt(metrics['rd_intensity'])}
revenue_yoy  (annual YoY growth)       : {_fmt(metrics['revenue_yoy'])}
revenue_yoy_q (same-quarter YoY)       : {_fmt(metrics['revenue_yoy_q'])}
revenue_acceleration                   : {_fmt(metrics['revenue_acceleration'])}
fcf_margin                             : {_fmt(metrics['fcf_margin'])}
fcf_to_net_income (quality proxy)      : {_fmt(metrics['fcf_to_net_income'])}
net_debt_to_ebitda                     : {_fmt(metrics['net_debt_to_ebitda'])}
current_ratio                          : {_fmt(metrics['current_ratio'])}
inventory_yoy                          : {_fmt(metrics['inventory_yoy'])}
margin_trend                           : {_fmt(metrics['margin_trend'])}

══════════════════════════════════════════════════════
SEC FILING SIGNALS
══════════════════════════════════════════════════════
tone_score (-1 to +1)      : {_fmt(metrics['sec_tone_score'])}
business_momentum          : {_fmt(metrics['sec_business_momentum'])}
forward_guidance           : {_fmt(metrics['sec_forward_guidance'])}
management_confidence(0-1) : {_fmt(metrics['sec_mgmt_confidence'])}
risk_score (0-1)           : {_fmt(metrics['sec_risk_score'])}
geopolitical_risk_present  : {_fmt(metrics['sec_geopolitical_risk'])}
key_themes                 : {json.dumps(metrics['sec_key_themes'])}
top_risks                  : {json.dumps(metrics['sec_top_risks'][:3])}

══════════════════════════════════════════════════════
SENTIMENT SIGNALS
══════════════════════════════════════════════════════
news_score (-1 to +1)           : {_fmt(metrics['news_score'])}
news_momentum (7d - 90d delta)  : {_fmt(metrics['news_momentum'])}
news_z_score                    : {_fmt(metrics['news_z_score'])}
extreme_positive_sentiment      : {_fmt(metrics['extreme_positive'])}
earnings_call_tone_score        : {_fmt(metrics['call_tone'])}
earnings_call_guidance          : {_fmt(metrics['call_guidance'])}
earnings_call_mgmt_confidence   : {_fmt(metrics['call_mgmt_confidence'])}

══════════════════════════════════════════════════════
TASK
══════════════════════════════════════════════════════
Construct the strongest credible bullish thesis for {ticker} using only the data above.

Return a JSON object with EXACTLY these fields:
{{
  "bullish_score": <float -1.0 to 1.0, overall conviction in the bull case>,
  "conviction_level": <"high" | "medium" | "low" | "none">,
  "top_arguments": [
    {{
      "category": <"technical" | "fundamental" | "sec_filing" | "sentiment" | "cross_signal">,
      "argument": <1-2 sentence specific bullish argument, cite exact numbers>,
      "evidence_keys": [<list of metric names from above that support this argument>],
      "strength": <float 0.0 to 1.0, how strong this specific argument is>
    }}
  ],
  "bull_catalysts": [<2-4 specific near-term catalysts that could drive upside>],
  "acknowledged_risks": [<2-3 genuine risks the bull must concede, based on the data>],
  "debate_summary": <2-3 sentence summary of the bull case for use in a bull-bear debate>,
  "data_quality_note": <brief note on any important data gaps that limit confidence>
}}

Rules:
- top_arguments must have 3 to 5 items, ordered by strength descending
- If a signal is N/A, do not cite it as evidence
- bullish_score should reflect actual signal strength, not optimism bias
- If most signals are negative, bullish_score should be near 0 or negative
"""
    return prompt


# ── Core Analysis ──────────────────────────────────────────────────────────────

def _get_subsector(ticker: str) -> str:
    for sub, members in SUBSECTORS.items():
        if ticker in members:
            return sub
    return "Unknown"


def analyze_bull_case(
    ticker: str,
    reference_date: str,
    tech:  Optional[dict],
    fund:  Optional[dict],
    sec:   Optional[dict],
    sent:  Optional[dict],
) -> dict:
    """
    Run the LLM bull analysis for a single ticker.
    Returns the structured bull thesis dict.
    """
    subsector = _get_subsector(ticker)
    metrics   = _build_supporting_metrics(tech, fund, sec, sent)
    prompt    = _build_prompt(ticker, reference_date, subsector, metrics,
                               tech, fund, sec, sent)

    print(f"  [BullAgent] Calling LLM for {ticker} ...", end=" ", flush=True)
    raw      = _call_llm(prompt)
    result   = _extract_json(raw)
    print(f"score={result.get('bullish_score', '?'):.3f}  "
          f"conviction={result.get('conviction_level', '?')}")

    # Stamp metadata
    result["ticker"]          = ticker
    result["reference_date"]  = reference_date
    result["analysis_date"]   = datetime.now().strftime("%Y-%m-%d")
    result["subsector"]       = subsector
    result["supporting_metrics"] = metrics

    return result


# ── BullAgent Class ────────────────────────────────────────────────────────────

class BullAgent:
    """
    Constructs bullish investment theses for semiconductor stocks by synthesizing
    signals from Technical, Fundamental, SEC Filing, and Sentiment agents.

    Usage:
        agent = BullAgent()
        thesis = agent.get_thesis(
            ticker="NVDA",
            reference_date="2024-06-30",
            technical_signal=tech_sig,
            fundamental_signal=fund_sig,
            sec_signal=sec_sig,
            sentiment_signal=sent_sig,
        )
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def get_thesis(
        self,
        ticker: str,
        reference_date: str,
        technical_signal:   Optional[dict] = None,
        fundamental_signal: Optional[dict] = None,
        sec_signal:         Optional[dict] = None,
        sentiment_signal:   Optional[dict] = None,
        force_refresh:      bool = False,
    ) -> dict:
        """
        Return the bull thesis dict for a single ticker.
        Cached per (ticker, reference_date).

        Args:
            ticker            : Stock ticker (e.g. "NVDA")
            reference_date    : ISO date string (e.g. "2024-06-30")
            technical_signal  : Output of TechnicalAgent.get_signal()
            fundamental_signal: Output of FundamentalAgent.get_signal()
            sec_signal        : Output of SECFilingAgent daily signal
            sentiment_signal  : Output of SentimentAgent composite signal
            force_refresh     : Re-run LLM even if cache exists
        """
        cache_path = _cache_dir(ticker) / f"bull_{reference_date}.json"

        if not force_refresh:
            cached = _load_json(cache_path)
            if cached:
                return cached

        result = analyze_bull_case(
            ticker=ticker,
            reference_date=reference_date,
            tech=technical_signal,
            fund=fundamental_signal,
            sec=sec_signal,
            sent=sentiment_signal,
        )
        _save_json(result, cache_path)
        return result

    def get_all_theses(
        self,
        reference_date: str,
        technical_signals:   Optional[dict[str, dict]] = None,
        fundamental_signals: Optional[dict[str, dict]] = None,
        sec_signals:         Optional[dict[str, dict]] = None,
        sentiment_signals:   Optional[dict[str, dict]] = None,
        ticker_list:         Optional[list[str]] = None,
        force_refresh:       bool = False,
    ) -> dict[str, dict]:
        """
        Run bull analysis for all tickers.

        Args:
            reference_date       : ISO date string
            technical_signals    : {ticker: signal} from TechnicalAgent.get_all_signals()
            fundamental_signals  : {ticker: signal} from FundamentalAgent.get_all_signals()
            sec_signals          : {ticker: signal} from SECFilingAgent
            sentiment_signals    : {ticker: signal} from SentimentAgent
        """
        tickers = ticker_list or TICKER_LIST
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.get_thesis(
                    ticker=ticker,
                    reference_date=reference_date,
                    technical_signal=   (technical_signals   or {}).get(ticker),
                    fundamental_signal= (fundamental_signals or {}).get(ticker),
                    sec_signal=         (sec_signals         or {}).get(ticker),
                    sentiment_signal=   (sentiment_signals   or {}).get(ticker),
                    force_refresh=force_refresh,
                )
            except Exception as e:
                print(f"[WARNING] {ticker}: {e}")
                results[ticker] = {"ticker": ticker, "error": str(e)}
        return results

    def print_summary(self, theses: dict[str, dict], top_n: int = 10) -> None:
        """Print a ranked table of bull theses sorted by bullish_score."""
        rows = []
        for ticker, t in theses.items():
            if "error" in t:
                continue
            rows.append({
                "ticker":     ticker,
                "score":      t.get("bullish_score"),
                "conviction": t.get("conviction_level"),
                "top_arg":    t.get("top_arguments", [{}])[0].get("category", ""),
            })
        rows.sort(key=lambda x: x["score"] or -99, reverse=True)

        ref = rows[0].get("reference_date", "") if rows else ""
        print(f"\n{'='*65}")
        print(f"  Bull Theses — top {top_n} by bullish_score")
        print(f"{'='*65}")
        print(f"{'Ticker':<8}  {'Score':>6}  {'Conviction':<10}  {'Top Category'}")
        print("-" * 65)
        for r in rows[:top_n]:
            print(f"{r['ticker']:<8}  {r['score']:>6.3f}  {r['conviction']:<10}  {r['top_arg']}")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from technical_agent   import TechnicalAgent
    from fundamental_agent import FundamentalAgent

    REF_DATE = "2024-06-28"

    print("Loading upstream agents...")
    tech_agent = TechnicalAgent()
    fund_agent = FundamentalAgent()

    # Get upstream signals for one ticker as a demo
    ticker = "NVDA"
    tech_sig = tech_agent.get_signal(ticker, reference_date=REF_DATE)
    fund_sig = fund_agent.get_signal(ticker, reference_date=REF_DATE)

    # SEC and Sentiment signals can be None if not yet available
    # (BullAgent gracefully handles missing upstream signals)
    sec_sig  = None
    sent_sig = None

    bull = BullAgent()
    thesis = bull.get_thesis(
        ticker=ticker,
        reference_date=REF_DATE,
        technical_signal=tech_sig,
        fundamental_signal=fund_sig,
        sec_signal=sec_sig,
        sentiment_signal=sent_sig,
    )

    print(f"\n{'='*65}")
    print(f"  Bull Thesis — {ticker}  ({REF_DATE})")
    print(f"{'='*65}")
    print(f"  bullish_score    : {thesis['bullish_score']}")
    print(f"  conviction_level : {thesis['conviction_level']}")
    print(f"\n  Top Arguments:")
    for i, arg in enumerate(thesis.get("top_arguments", []), 1):
        print(f"  {i}. [{arg['category']}] strength={arg['strength']:.2f}")
        print(f"     {arg['argument']}")
    print(f"\n  Catalysts:")
    for c in thesis.get("bull_catalysts", []):
        print(f"  • {c}")
    print(f"\n  Acknowledged Risks:")
    for r in thesis.get("acknowledged_risks", []):
        print(f"  ⚠ {r}")
    print(f"\n  Debate Summary:")
    print(f"  {thesis.get('debate_summary', '')}")
