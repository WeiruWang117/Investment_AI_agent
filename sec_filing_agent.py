"""
SEC Filing Analysis Agent

Analyzes 10-K and 10-Q filings (MDA + Risk sections) for semiconductor stocks.
Produces structured investment signals for downstream portfolio construction.

Data layout:
  parsed_10k/{TICKER}/mda/{TICKER}_{YEAR}_mda.txt
  parsed_10k/{TICKER}/risk/{TICKER}_{YEAR}_risk.txt
  parsed_10q/{TICKER}/mda/{TICKER}_{YEAR}Q{Q}_mda.txt
  parsed_10q/{TICKER}/risk/{TICKER}_{YEAR}Q{Q}_risk.txt

Output cache:
  sec_analysis/{TICKER}/{10K|10Q}_{PERIOD}.json
"""

import os
import json
import time
from pathlib import Path
from typing import Optional
from datetime import datetime, date

import anthropic

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_DIR          = Path("d:/Investment_AI_agent")
PARSED_10K_DIR    = BASE_DIR / "parsed_10k"
PARSED_10Q_DIR    = BASE_DIR / "parsed_10q"
PARSED_8K_DIR     = BASE_DIR / "parsed_8k"
OUTPUT_DIR        = BASE_DIR / "sec_analysis"      # filing-level cache
DAILY_SIGNALS_DIR = BASE_DIR / "daily_signals"     # one file per trading date
OUTPUT_DIR.mkdir(exist_ok=True)
DAILY_SIGNALS_DIR.mkdir(exist_ok=True)

MODEL          = "claude-sonnet-4-6"

# Text budget controls (key tradeoff: coverage vs. API cost)
MAX_MDA_CHARS  = 15_000   # Truncate MDA — first 15K covers strategic overview
MAX_RISK_CHUNK = 10_000   # Chars per risk chunk
MAX_RISK_CHUNKS = 8       # Max chunks (covers ~80K of typical 160K risk section)

TICKER_LIST = [
    "TSM", "INTC", "NVDA", "ASML", "AVGO", "TXN", "QCOM", "MU",
    "AMAT", "AMD", "ADI", "LRCX", "KLAC", "MCHP", "SWKS", "MRVL",
    "TER", "ON", "MPWR", "ENTG", "MKSI", "SLAB", "CRUS", "SMTC",
]

client = anthropic.Anthropic()


# ── Data Loading ───────────────────────────────────────────────────────────────

def _base_dir(filing_type: str) -> Path:
    return PARSED_10K_DIR if filing_type == "10-K" else PARSED_10Q_DIR


def list_periods(ticker: str, filing_type: str = "10-K") -> list[str]:
    """Return sorted list of available period labels for a ticker."""
    mda_dir = _base_dir(filing_type) / ticker / "mda"
    if not mda_dir.exists():
        return []
    files = sorted(mda_dir.glob(f"{ticker}_*_mda.txt"))
    return [f.stem.replace(f"{ticker}_", "").replace("_mda", "") for f in files]


def load_section(
    ticker: str, period: str, section: str, filing_type: str = "10-K"
) -> Optional[str]:
    """Load a parsed section text file. Returns None if not found."""
    path = _base_dir(filing_type) / ticker / section / f"{ticker}_{period}_{section}.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


# ── 8-K Data Loading ───────────────────────────────────────────────────────────

def list_8k_dates(ticker: str) -> list[str]:
    """Return sorted list of 8-K filing dates (YYYYMMDD) for a ticker."""
    d = PARSED_8K_DIR / ticker
    if not d.exists():
        return []
    files = sorted(d.glob(f"{ticker}_*.txt"))
    return [f.stem.replace(f"{ticker}_", "") for f in files]


def load_8k(ticker: str, filing_date: str) -> Optional[str]:
    """Load a parsed 8-K text file by date. Returns None if not found."""
    path = PARSED_8K_DIR / ticker / f"{ticker}_{filing_date}.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def get_available_8k_dates(ticker: str, as_of: date) -> list[str]:
    """Return 8-K filing dates available on or before as_of."""
    return [d for d in list_8k_dates(ticker) if d <= as_of.strftime("%Y%m%d")]


# ── Cache Helpers ──────────────────────────────────────────────────────────────

def _cache_path(ticker: str, filing_type: str, period: str) -> Path:
    ticker_dir = OUTPUT_DIR / ticker
    ticker_dir.mkdir(exist_ok=True)
    return ticker_dir / f"{filing_type.replace('-', '')}_{period}.json"


def load_cached(ticker: str, filing_type: str, period: str) -> Optional[dict]:
    path = _cache_path(ticker, filing_type, period)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_cache(result: dict, ticker: str, filing_type: str, period: str):
    path = _cache_path(ticker, filing_type, period)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Text Utilities ─────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int) -> list[str]:
    """
    Split text into chunks of roughly chunk_size chars.

    Strategy (in priority order):
      1. Split on double-newlines (paragraphs) — ideal for parsed HTML with formatting
      2. Split on sentence endings ('. ') — fallback for single-blob text
      3. Hard split at chunk_size — last resort
    """
    if len(text) <= chunk_size:
        return [text]

    # Determine best separator
    if text.count("\n\n") > 5:
        units = text.split("\n\n")
        sep = "\n\n"
    elif text.count(". ") > 20:
        # Split on sentence boundaries; keep the period by re-appending it
        raw_units = text.split(". ")
        units = [u + ". " for u in raw_units[:-1]] + [raw_units[-1]]
        sep = ""
    else:
        # Hard split: break at word boundaries near chunk_size
        units, pos = [], 0
        while pos < len(text):
            end = min(pos + chunk_size, len(text))
            # Try to break at a space
            if end < len(text):
                space = text.rfind(" ", pos, end)
                if space > pos:
                    end = space
            units.append(text[pos:end])
            pos = end
        return units

    chunks, current, current_len = [], [], 0
    for unit in units:
        unit_len = len(unit) + len(sep)
        if current_len + unit_len > chunk_size and current:
            chunks.append(sep.join(current))
            current, current_len = [], 0
        current.append(unit)
        current_len += unit_len
    if current:
        chunks.append(sep.join(current))
    return chunks


def extract_json(text: str) -> dict:
    """Extract the first JSON object from an LLM response string."""
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response:\n{text[:300]}")
    return json.loads(text[start:end])


# ── LLM Call ──────────────────────────────────────────────────────────────────

SYSTEM_ANALYST = (
    "You are a senior equity analyst specializing in semiconductor stocks. "
    "You analyze SEC filings objectively to extract investment-relevant signals. "
    "Always respond with a single valid JSON object exactly matching the specified schema. "
    "Do not include any text outside the JSON object."
)


def call_llm(
    prompt: str,
    system: str = SYSTEM_ANALYST,
    max_tokens: int = 2048,
    retries: int = 5,
) -> str:
    """
    Streaming LLM call with adaptive thinking. Returns the text response.
    Retries on JSON-unrecoverable failures.
    """
    for attempt in range(retries + 1):
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                message = stream.get_final_message()
                # With adaptive thinking, response may start with ThinkingBlock(s)
                for block in reversed(message.content):
                    if block.type == "text":
                        return block.text
                raise ValueError("LLM response contained no text block.")
        except Exception as e:
            if attempt < retries:
                print(f"    [RETRY {attempt+1}] LLM call failed: {e}")
                wait = 2 ** (attempt + 1); time.sleep(wait)
            else:
                raise


# ── Prompt Templates & Analysis Functions ─────────────────────────────────────

def analyze_mda(ticker: str, period: str, mda_text: str) -> dict:
    """Analyze the MDA section. Truncates input to MAX_MDA_CHARS."""
    truncated = mda_text[:MAX_MDA_CHARS]

    prompt = f"""Analyze this MD&A section from {ticker}'s {period} SEC filing.

MD&A TEXT:
{truncated}

Return a JSON object with EXACTLY these fields:
{{
  "tone_score": <float -1.0 (very negative) to 1.0 (very positive)>,
  "business_momentum": <"accelerating" | "stable" | "decelerating" | "declining">,
  "forward_guidance": <"positive" | "neutral" | "negative" | "mixed">,
  "revenue_tone": <"growth" | "stable" | "declining" | "uncertain">,
  "margin_tone": <"expanding" | "stable" | "contracting" | "uncertain">,
  "management_confidence": <float 0.0 to 1.0>,
  "key_themes": [<3-5 most important business themes>],
  "forward_looking_statements": [<2-3 key forward-looking statements, quoted briefly>],
  "reasoning": <2-3 sentence explanation of your tone_score>
}}"""

    return extract_json(call_llm(prompt))


def analyze_risk_chunk(ticker: str, period: str, chunk: str, chunk_idx: int) -> dict:
    """Analyze one chunk of the risk factors section."""
    prompt = f"""Analyze this excerpt (chunk {chunk_idx}) from {ticker}'s {period} Risk Factors.

RISK FACTORS EXCERPT:
{chunk}

Return a JSON object with EXACTLY these fields:
{{
  "risk_categories": [<risk category names found in this excerpt>],
  "high_severity_risks": [<2-3 most severe risks, brief description>],
  "semiconductor_specific_risks": [<risks specific to semiconductor industry>],
  "geopolitical_risks": [<China / Taiwan / export control / trade risks, empty list if none>],
  "chunk_risk_level": <"low" | "moderate" | "elevated" | "high">
}}"""

    return extract_json(call_llm(prompt, max_tokens=2048))


def synthesize_risk(ticker: str, period: str, chunk_analyses: list[dict]) -> dict:
    """Map-reduce step: synthesize chunk analyses into one risk assessment."""
    prompt = f"""You analyzed {len(chunk_analyses)} chunks of {ticker}'s {period} Risk Factors.

CHUNK ANALYSES:
{json.dumps(chunk_analyses, indent=2)}

Synthesize into a final risk assessment. Return a JSON object with EXACTLY these fields:
{{
  "risk_score": <float 0.0 (minimal risk) to 1.0 (extreme risk)>,
  "overall_risk_level": <"low" | "moderate" | "elevated" | "high">,
  "top_risks": [<5 most important risks overall, brief description each>],
  "risk_categories": [<all unique risk categories observed>],
  "geopolitical_risk_present": <true | false>,
  "china_revenue_risk": <"high" | "medium" | "low" | "not_mentioned">,
  "supply_chain_risk": <"high" | "medium" | "low" | "not_mentioned">,
  "competition_risk": <"high" | "medium" | "low" | "not_mentioned">,
  "reasoning": <2-3 sentence explanation of the risk_score>
}}"""

    return extract_json(call_llm(prompt))


def analyze_risk(ticker: str, period: str, risk_text: str) -> dict:
    """Full risk section analysis using map-reduce for long texts."""
    chunks = chunk_text(risk_text, MAX_RISK_CHUNK)[:MAX_RISK_CHUNKS]
    print(f"    Risk section: {len(risk_text):,} chars → analyzing {len(chunks)} chunks")

    chunk_analyses = []
    for i, chunk in enumerate(chunks, start=1):
        result = analyze_risk_chunk(ticker, period, chunk, i)
        chunk_analyses.append(result)

    # Single chunk: convert format directly without an extra LLM call
    if len(chunk_analyses) == 1:
        c = chunk_analyses[0]
        level_map = {"low": 0.2, "moderate": 0.4, "elevated": 0.65, "high": 0.85}
        return {
            "risk_score":               level_map.get(c.get("chunk_risk_level", "moderate"), 0.4),
            "overall_risk_level":       c.get("chunk_risk_level", "moderate"),
            "top_risks":                c.get("high_severity_risks", []),
            "risk_categories":          c.get("risk_categories", []),
            "geopolitical_risk_present": bool(c.get("geopolitical_risks")),
            "china_revenue_risk":       "not_mentioned",
            "supply_chain_risk":        "not_mentioned",
            "competition_risk":         "not_mentioned",
            "reasoning":                "Single-chunk analysis.",
        }

    return synthesize_risk(ticker, period, chunk_analyses)


def compute_delta(
    ticker: str,
    current_period: str, current: dict,
    prior_period: str,   prior: dict,
) -> dict:
    """Compare two filing analyses and return a delta assessment."""
    prompt = f"""Compare {ticker}'s SEC filings: {current_period} (current) vs {prior_period} (prior).

CURRENT ({current_period}) — MDA:
{json.dumps(current.get("mda", {}), indent=2)}

CURRENT ({current_period}) — RISK:
{json.dumps(current.get("risk", {}), indent=2)}

PRIOR ({prior_period}) — MDA:
{json.dumps(prior.get("mda", {}), indent=2)}

PRIOR ({prior_period}) — RISK:
{json.dumps(prior.get("risk", {}), indent=2)}

Return a JSON object with EXACTLY these fields:
{{
  "mda_direction":  <"improving" | "stable" | "deteriorating">,
  "risk_direction": <"improving" | "stable" | "deteriorating">,
  "tone_change":    <float, positive = more positive vs prior>,
  "risk_change":    <float, positive = more risk vs prior>,
  "new_themes":     [<themes in current not in prior>],
  "dropped_themes": [<themes in prior not in current>],
  "key_changes":    [<3-5 most important changes between filings>],
  "overall_delta":  <"significantly_improved" | "improved" | "stable" | "deteriorated" | "significantly_deteriorated">,
  "delta_summary":  <2-3 sentence summary of what changed and why it matters>
}}"""

    result = extract_json(call_llm(prompt))
    result["vs_prior_period"] = prior_period
    return result


def composite_score(mda: dict, risk: dict) -> float:
    """
    Aggregate MDA and Risk into a single score from -1 (bearish) to +1 (bullish).

    Formula:
      - MDA tone contributes 50%  (tone_score directly)
      - Risk penalty contributes 30%  (risk_score above 0.5 penalizes)
      - Management confidence contributes 20%
    """
    tone       = mda.get("tone_score", 0.0)
    risk_score = risk.get("risk_score", 0.5)
    confidence = mda.get("management_confidence", 0.5)

    score = (
        tone * 0.50
        + (0.5 - risk_score) * 0.30       # high risk → negative contribution
        + (confidence - 0.5) * 0.20
    )
    return round(max(-1.0, min(1.0, score)), 3)


# ── 8-K Analysis ──────────────────────────────────────────────────────────────

MAX_8K_CHARS = 12_000  # 8-Ks are shorter than 10-K/Q; 12K covers most fully


def analyze_8k(ticker: str, filing_date: str, text: str) -> dict:
    """Analyze a single 8-K filing. Returns structured signal dict."""
    truncated = text[:MAX_8K_CHARS]

    prompt = f"""Analyze this 8-K filing from {ticker} filed on {filing_date}.

8-K TEXT:
{truncated}

Return a JSON object with EXACTLY these fields:
{{
  "event_type": <"earnings" | "guidance" | "management_change" | "material_agreement" | "other">,
  "tone_score": <float -1.0 (very negative) to 1.0 (very positive)>,
  "revenue_surprise": <"beat" | "miss" | "in-line" | "not_reported">,
  "guidance_direction": <"raised" | "lowered" | "maintained" | "not_provided">,
  "management_confidence": <float 0.0 to 1.0>,
  "key_facts": [<3-5 most important factual disclosures>],
  "market_impact": <"positive" | "negative" | "neutral" | "uncertain">,
  "reasoning": <2 sentence explanation of tone_score and market_impact>
}}"""

    return extract_json(call_llm(prompt, max_tokens=1024))


# ── Agent Class ────────────────────────────────────────────────────────────────

class SECFilingAgent:
    """
    Analyzes SEC 10-K and 10-Q filings for investment signals.

    Results are cached to disk (sec_analysis/).
    Re-running is fast — cached analyses skip LLM calls entirely.

    Typical workflow:
        agent = SECFilingAgent()

        # Analyze all 10-K filings for one ticker (with delta between years)
        results = agent.analyze_ticker("NVDA", filing_type="10-K")

        # Get latest signal for all tickers (for Orchestrator)
        signals = agent.get_all_signals(TICKER_LIST)
    """

    # ── Single Filing ──────────────────────────────────────────────────────────

    def analyze_filing(
        self,
        ticker: str,
        period: str,
        filing_type: str = "10-K",
        force_refresh: bool = False,
    ) -> Optional[dict]:
        """
        Analyze one filing (MDA + Risk). Returns structured result dict.
        Loads from disk cache unless force_refresh=True.
        """
        if not force_refresh:
            cached = load_cached(ticker, filing_type, period)
            if cached:
                print(f"  [CACHE]   {ticker} {filing_type} {period}")
                return cached

        mda_text  = load_section(ticker, period, "mda",  filing_type)
        risk_text = load_section(ticker, period, "risk", filing_type)

        if not mda_text and not risk_text:
            print(f"  [SKIP]    {ticker} {filing_type} {period} — no text files found")
            return None

        print(f"  [ANALYZE] {ticker} {filing_type} {period}")

        result: dict = {
            "ticker":          ticker,
            "period":          period,
            "filing_type":     filing_type,
            "analysis_date":   datetime.now().strftime("%Y-%m-%d"),
            "mda":             {},
            "risk":            {},
            "delta":           None,
            "composite_score": 0.0,
            "confidence":      0.0,
        }

        if mda_text:
            print(f"    MDA: {len(mda_text):,} chars (truncated to {MAX_MDA_CHARS:,})")
            result["mda"] = analyze_mda(ticker, period, mda_text)

        if risk_text:
            result["risk"] = analyze_risk(ticker, period, risk_text)

        result["composite_score"] = composite_score(result["mda"], result["risk"])
        result["confidence"]      = 0.85 if (mda_text and risk_text) else 0.5

        save_cache(result, ticker, filing_type, period)
        return result

    # ── Full Ticker (all periods + delta) ─────────────────────────────────────

    def analyze_ticker(
        self,
        ticker: str,
        filing_type: str = "10-K",
        force_refresh: bool = False,
    ) -> list[dict]:
        """
        Analyze all available filings for a ticker.
        Adds year-over-year delta analysis between consecutive filings.
        Returns list sorted chronologically.
        """
        periods = list_periods(ticker, filing_type)
        if not periods:
            print(f"[WARNING] No {filing_type} data found for {ticker}")
            return []

        print(f"\n{'='*55}")
        print(f"  {ticker} ({filing_type})  —  {len(periods)} filings: {periods[0]}…{periods[-1]}")
        print(f"{'='*55}")

        # Step 1: analyze each filing individually
        results = []
        for period in periods:
            r = self.analyze_filing(ticker, period, filing_type, force_refresh)
            if r:
                results.append(r)

        # Step 2: delta analysis (compare consecutive filings)
        print(f"\n  Computing delta ({len(results)-1} comparisons)…")
        for i in range(1, len(results)):
            current = results[i]
            prior   = results[i - 1]

            # Skip if delta already in cache
            if current.get("delta") is not None:
                continue

            try:
                delta = compute_delta(
                    ticker,
                    current["period"], current,
                    prior["period"],   prior,
                )
                results[i]["delta"] = delta
                save_cache(results[i], ticker, filing_type, current["period"])
                print(f"    {current['period']} vs {prior['period']}: "
                      f"{delta.get('overall_delta', '?')}")
            except Exception as e:
                print(f"    [ERROR] Delta {current['period']}: {e}")

        return results

    # ── Convenience Methods (for Orchestrator) ────────────────────────────────

    def get_latest_signal(
        self,
        ticker: str,
        filing_type: str = "10-K",
        reference_date: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Return the most recent filing analysis for a ticker that was
        publicly available as of reference_date (no look-ahead bias).

        Filing availability heuristics (conservative):
          10-K "YYYY"   → assume available after April 1 of YYYY
                          (fiscal year ends ~Jan/Feb; 10-K filed within 60 days)
          10-Q "YYYYQN" → assume available 45 days after quarter end:
                          Q1 → after May 15;  Q2 → after Aug 14
                          Q3 → after Nov 14;  Q4 → after Feb 14 next year
        If reference_date is None, returns the absolute latest period.
        """
        from datetime import date as _date

        periods = list_periods(ticker, filing_type)
        if not periods:
            return None

        if reference_date is not None:
            ref = _date.fromisoformat(reference_date)

            def _available(period: str) -> bool:
                if "Q" in period:  # 10-Q: e.g. "2020Q3"
                    year, q = int(period[:4]), int(period[5])
                    q_ends = {1: _date(year, 3, 31), 2: _date(year, 6, 30),
                              3: _date(year, 9, 30), 4: _date(year, 12, 31)}
                    from datetime import timedelta
                    avail = q_ends[q] + timedelta(days=45)
                else:  # 10-K: e.g. "2020"
                    avail = _date(int(period), 4, 1)
                return avail <= ref

            eligible = [p for p in periods if _available(p)]
            if not eligible:
                return None
            latest = eligible[-1]
        else:
            latest = periods[-1]

        cached = load_cached(ticker, filing_type, latest)
        if cached:
            return cached
        return self.analyze_filing(ticker, latest, filing_type)

    def get_all_signals(
        self,
        ticker_list: list[str] = TICKER_LIST,
        filing_type: str = "10-K",
    ) -> dict[str, dict]:
        """
        Get latest filing signal for every ticker.
        Returns { ticker: result_dict }.
        Used by Orchestrator to feed downstream agents.
        """
        signals = {}
        for ticker in ticker_list:
            signal = self.get_latest_signal(ticker, filing_type)
            if signal:
                signals[ticker] = signal
        return signals

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_summary(self, results: list[dict]):
        """Print a human-readable summary table."""
        if not results:
            return
        ticker = results[0]["ticker"]
        ftype  = results[0]["filing_type"]
        print(f"\n{'='*60}")
        print(f"  SUMMARY  {ticker}  ({ftype})")
        print(f"{'='*60}")
        print(f"  {'Period':<10} {'Score':>6}  {'MDA Tone':>8}  {'Momentum':<13}  "
              f"{'Risk':>8}  {'vs Prior'}")
        print(f"  {'-'*10} {'-'*6}  {'-'*8}  {'-'*13}  {'-'*8}  {'-'*22}")
        for r in results:
            delta_str = ""
            if r.get("delta"):
                delta_str = r["delta"].get("overall_delta", "")
            print(
                f"  {r['period']:<10} "
                f"{r['composite_score']:>+6.2f}  "
                f"{r['mda'].get('tone_score', 0):>+8.2f}  "
                f"{r['mda'].get('business_momentum', 'N/A'):<13}  "
                f"{r['risk'].get('overall_risk_level', 'N/A'):>8}  "
                f"{delta_str}"
            )

    def export_signals_csv(
        self,
        ticker_list: list[str] = TICKER_LIST,
        filing_type: str = "10-K",
        output_path: str = "sec_signals.csv",
    ):
        """
        Export latest signals for all tickers to CSV.
        Useful for feeding into Portfolio Agent.
        """
        import csv
        signals = self.get_all_signals(ticker_list, filing_type)
        rows = []
        for ticker, s in signals.items():
            rows.append({
                "ticker":            ticker,
                "period":            s.get("period"),
                "composite_score":   s.get("composite_score"),
                "mda_tone":          s.get("mda", {}).get("tone_score"),
                "mda_momentum":      s.get("mda", {}).get("business_momentum"),
                "mda_guidance":      s.get("mda", {}).get("forward_guidance"),
                "risk_score":        s.get("risk", {}).get("risk_score"),
                "risk_level":        s.get("risk", {}).get("overall_risk_level"),
                "geo_risk":          s.get("risk", {}).get("geopolitical_risk_present"),
                "china_risk":        s.get("risk", {}).get("china_revenue_risk"),
                "delta_direction":   s.get("delta", {}).get("overall_delta") if s.get("delta") else None,
                "analysis_date":     s.get("analysis_date"),
            })

        output_file = BASE_DIR / output_path
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nExported {len(rows)} signals → {output_file}")

    # ── Unified Signal Interface ───────────────────────────────────────────────

    def get_combined_signal(
        self,
        ticker: str,
        as_of: date,
    ) -> dict:
        """
        Return a single unified signal dict combining 10-K, 10-Q, and 8-K
        for a given ticker as of a specific date.

        Structure:
        {
          "ticker":     str,
          "as_of":      str,          # YYYY-MM-DD
          "annual":     dict | None,  # latest 10-K available
          "quarterly":  dict | None,  # latest 10-Q available
          "event":      dict | None,  # latest 8-K available
          "summary": {
            "annual_score":    float | None,
            "quarterly_score": float | None,
            "event_tone":      float | None,
            "event_type":      str | None,
            "revenue_surprise": str | None,
            "guidance":        str | None,
          }
        }
        """
        # 10-K
        annual_periods = get_available_periods(ticker, "10-K", as_of)
        annual = self.analyze_filing(ticker, annual_periods[-1], "10-K") if annual_periods else None

        # 10-Q
        quarterly_periods = get_available_periods(ticker, "10-Q", as_of)
        quarterly = self.analyze_filing(ticker, quarterly_periods[-1], "10-Q") if quarterly_periods else None

        # 8-K
        dates_8k = get_available_8k_dates(ticker, as_of)
        event = None
        if dates_8k:
            latest_date = dates_8k[-1]
            cache_path  = OUTPUT_DIR / ticker / f"8K_{latest_date}.json"
            if cache_path.exists():
                event = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                text_8k = load_8k(ticker, latest_date)
                if text_8k:
                    event = analyze_8k(ticker, latest_date, text_8k)
                    event["ticker"]     = ticker
                    event["filed_date"] = latest_date
                    cache_path.parent.mkdir(exist_ok=True)
                    cache_path.write_text(json.dumps(event, indent=2, ensure_ascii=False), encoding="utf-8")

        return {
            "ticker":    ticker,
            "as_of":     as_of.strftime("%Y-%m-%d"),
            "annual":    annual,
            "quarterly": quarterly,
            "event":     event,
            "summary": {
                "annual_score":     annual.get("composite_score")    if annual    else None,
                "quarterly_score":  quarterly.get("composite_score") if quarterly else None,
                "event_tone":       event.get("tone_score")          if event     else None,
                "event_type":       event.get("event_type")          if event     else None,
                "revenue_surprise": event.get("revenue_surprise")    if event     else None,
                "guidance":         event.get("guidance_direction")  if event     else None,
            },
        }

    def get_all_combined_signals(
        self,
        ticker_list: list[str],
        as_of: date,
    ) -> dict[str, dict]:
        """
        Get combined signals for all tickers as of a given date.
        Returns { ticker: combined_signal_dict }.
        Entry point for downstream agents (Portfolio, Risk, etc.).
        """
        return {t: self.get_combined_signal(t, as_of) for t in ticker_list}

    def generate_daily_snapshot(
        self,
        as_of: date,
        ticker_list: list[str] = TICKER_LIST,
        force_refresh: bool = False,
    ) -> dict:
        """
        Generate (or load) the daily signals snapshot for a given date.

        Output file: daily_signals/{YYYY-MM-DD}.json

        Schema (designed for multi-agent extension):
        {
          "date":         "YYYY-MM-DD",
          "generated_at": "...",
          "tickers":      [...],
          "signals": {
            "NVDA": {
              "sec_filing": {          # ← this agent's namespace
                "annual_period":    str,
                "quarterly_period": str,
                "event_date":       str,
                "annual_score":     float,
                "quarterly_score":  float,
                "risk_level":       str,
                "momentum":         str,
                "event_tone":       float,
                "event_type":       str,
                "revenue_surprise": str,
                "guidance":         str,
              },
              # future agents add their own keys here:
              # "sentiment":  { ... },
              # "technical":  { ... },
              # "macro":      { ... },
            },
            ...
          }
        }
        """
        snapshot_path = DAILY_SIGNALS_DIR / f"{as_of}.json"

        # Load existing snapshot if present and not forcing refresh
        existing = {}
        if snapshot_path.exists() and not force_refresh:
            existing = json.loads(snapshot_path.read_text(encoding="utf-8"))
            # If sec_filing already populated for all tickers, return as-is
            sigs = existing.get("signals", {})
            if all(t in sigs and "sec_filing" in sigs[t] for t in ticker_list):
                print(f"[SNAPSHOT] Loaded from cache: {snapshot_path.name}")
                return existing

        print(f"[SNAPSHOT] Generating sec_filing signals for {len(ticker_list)} tickers — {as_of}")
        combined = self.get_all_combined_signals(ticker_list, as_of)

        signals = existing.get("signals", {})
        for ticker, c in combined.items():
            if ticker not in signals:
                signals[ticker] = {}
            ann = c.get("annual")   or {}
            qty = c.get("quarterly") or {}
            evt = c.get("event")    or {}
            signals[ticker]["sec_filing"] = {
                "annual_period":    ann.get("period"),
                "quarterly_period": qty.get("period"),
                "event_date":       evt.get("filed_date"),
                "annual_score":     ann.get("composite_score"),
                "quarterly_score":  qty.get("composite_score"),
                "risk_level":       ann.get("risk", {}).get("overall_risk_level"),
                "momentum":         ann.get("mda",  {}).get("business_momentum"),
                "event_tone":       evt.get("tone_score"),
                "event_type":       evt.get("event_type"),
                "revenue_surprise": evt.get("revenue_surprise"),
                "guidance":         evt.get("guidance_direction"),
            }

        snapshot = {
            "date":         str(as_of),
            "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "tickers":      ticker_list,
            "signals":      signals,
        }
        snapshot_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[SNAPSHOT] Saved → {snapshot_path}")
        return snapshot


# ── Date-aware Filing Lookup ───────────────────────────────────────────────────

def period_available_date(period: str, filing_type: str) -> date:
    """
    Returns the approximate date when a filing became publicly available.
    Based on SEC deadlines for large accelerated filers (NVDA's category).

    10-K: due 60 days after fiscal year end (~Jan 26) → available ~late March
    10-Q periods in data (NVDA fiscal calendar, Feb-Jan FY):
      Q2 (quarter ending ~Jul) → available ~Sep
      Q3 (quarter ending ~Oct) → available ~Dec
      Q4 (quarter ending ~Jan) → available ~Mar of following year
    """
    if filing_type == "10-K":
        year = int(period)
        return date(year, 3, 31)
    else:
        year = int(period[:4])
        q    = int(period[5])
        release = {2: date(year, 9, 10), 3: date(year, 12, 10), 4: date(year + 1, 3, 31)}
        return release.get(q, date(year, 12, 31))


def get_available_periods(ticker: str, filing_type: str, as_of: date) -> list[str]:
    """Return all periods whose filing was publicly available on or before as_of."""
    return [
        p for p in list_periods(ticker, filing_type)
        if period_available_date(p, filing_type) <= as_of
    ]


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Set your trading date here ─────────────────────────────────────────────
    AS_OF        = date(2025, 1, 1)   # ← change to your target date
    TICKER_LIST  = ["NVDA"]           # ← set tickers to run (or use full TICKER_LIST)
    # ──────────────────────────────────────────────────────────────────────────

    agent    = SECFilingAgent()
    snapshot = agent.generate_daily_snapshot(AS_OF, TICKER_LIST)

    # Print summary for each ticker
    print(f"\n=== Daily Snapshot  |  {AS_OF} ===\n")
    for ticker, sigs in snapshot["signals"].items():
        sf = sigs.get("sec_filing", {})
        print(f"  {ticker}")
        print(f"    Annual  ({sf.get('annual_period')}):    score={sf.get('annual_score'):+.2f}  "
              f"momentum={sf.get('momentum')}  risk={sf.get('risk_level')}")
        print(f"    Quarterly ({sf.get('quarterly_period')}): score={sf.get('quarterly_score'):+.2f}")
        if sf.get("event_date"):
            print(f"    Event   ({sf.get('event_date')}): type={sf.get('event_type')}  "
                  f"tone={sf.get('event_tone'):+.2f}  surprise={sf.get('revenue_surprise')}  "
                  f"guidance={sf.get('guidance')}")
        print()
