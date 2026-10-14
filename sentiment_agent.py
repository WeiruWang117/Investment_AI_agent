"""
Sentiment Agent

Two independent components:

1. NewsSentimentAnalyzer  (no LLM)
   - Input : {TICKER}_Historical_News_Cleaned.csv
   - Columns: time_published, title, source, overall_sentiment_score, overall_sentiment_label
   - Output : rolling-window stats + momentum + z-score for any reference date

2. EarningsCallAnalyzer  (LLM — claude-opus-4-6)
   - Input : Earning calls/{TICKER}/{TICKER}_{YEAR}Q{Q}.json
   - Structure: {"symbol": ..., "quarter": ..., "transcript": [{"speaker", "title", "content"}]}
   - Output : tone, guidance, themes, Q&A dynamics, delta vs prior quarter

Output cache:
  sentiment_cache/{TICKER}/news_{DATE}.json          (news snapshot)
  sentiment_cache/{TICKER}/call_{QUARTER}.json       (earnings call analysis)
  sentiment_cache/{TICKER}/composite_{DATE}.json     (merged signal)
"""

import os
import json
import glob
import time
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import anthropic

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_DIR      = Path("d:/Investment_AI_agent")
NEWS_DIR      = BASE_DIR / "News data"
CALLS_DIR     = BASE_DIR / "Earning calls"
CACHE_DIR     = BASE_DIR / "sentiment_cache"
CACHE_DIR.mkdir(exist_ok=True)

MODEL = "claude-sonnet-4-6"

# News windows (days)
WINDOWS = {"7d": 7, "30d": 30, "90d": 90}

# Label → numeric mapping (for display / sanity checks)
LABEL_SCORES = {
    "Bullish":          0.75,
    "Somewhat-Bullish": 0.35,
    "Neutral":          0.00,
    "Somewhat-Bearish": -0.35,
    "Bearish":          -0.75,
}

TICKER_LIST = [
    "TSM", "INTC", "NVDA", "ASML", "AVGO", "TXN", "QCOM", "MU",
    "AMAT", "AMD", "ADI", "LRCX", "KLAC", "MCHP", "SWKS", "MRVL",
    "TER", "ON", "MPWR", "ENTG", "MKSI", "SLAB", "CRUS", "SMTC",
]

client = anthropic.Anthropic()


# ── Shared Utilities ───────────────────────────────────────────────────────────

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


def extract_json(text: str) -> dict:
    # Strip markdown code fences if present
    if "```" in text:
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON in LLM response:\n{text[:300]}")
    return json.loads(text[start:end])


def call_llm(prompt: str, system: str, max_tokens: int = 4096, retries: int = 5) -> str:
    for attempt in range(retries + 1):
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                message = stream.get_final_message()
                for block in reversed(message.content):
                    if block.type == "text":
                        return block.text
                raise ValueError("No text block in LLM response.")
        except Exception as e:
            if attempt < retries:
                wait = 2 ** (attempt + 1)   # 2s, 4s, 8s, 16s, 32s
                print(f"    [RETRY {attempt+1}] {e}  (wait {wait}s)")
                time.sleep(wait)
            else:
                raise


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — News Sentiment Analyzer (no LLM)
# ══════════════════════════════════════════════════════════════════════════════

class NewsSentimentAnalyzer:
    """
    Aggregates pre-computed news sentiment scores from CSV files.
    Computes rolling windows, momentum, and z-scores.
    No LLM calls — pure pandas analytics.

    Usage:
        nsa = NewsSentimentAnalyzer()
        signal = nsa.get_signal("NVDA", reference_date="2024-06-30")
    """

    def __init__(self):
        self._cache: dict[str, pd.DataFrame] = {}   # ticker → DataFrame

    # ── Data Loading ───────────────────────────────────────────────────────────

    def _load(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load and parse CSV for a ticker. Cached in memory."""
        if ticker in self._cache:
            return self._cache[ticker]

        csv_path = NEWS_DIR / f"{ticker}_Historical_News_Cleaned.csv"
        if not csv_path.exists():
            return None

        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["time_published"], errors="coerce")
        df = df.dropna(subset=["date", "overall_sentiment_score"])
        df["score"] = df["overall_sentiment_score"].astype(float)
        df = df.sort_values("date").reset_index(drop=True)

        self._cache[ticker] = df
        return df

    # ── Signal Computation ─────────────────────────────────────────────────────

    def get_signal(
        self,
        ticker: str,
        reference_date: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Compute news sentiment signal for a ticker as of reference_date.

        Args:
            ticker: Stock ticker symbol.
            reference_date: ISO date string (e.g. "2024-06-30").
                            Defaults to the latest date in the data.

        Returns:
            Dict with rolling stats, momentum, z-score, and extreme flags.
            None if no data available.
        """
        df = self._load(ticker)
        if df is None or df.empty:
            return None

        ref_dt = pd.Timestamp(reference_date) if reference_date else df["date"].max()

        # Slice up to reference date
        past = df[df["date"] <= ref_dt]
        if past.empty:
            return None

        # Rolling window stats
        window_stats: dict = {}
        for label, days in WINDOWS.items():
            cutoff = ref_dt - timedelta(days=days)
            window_df = past[past["date"] >= cutoff]
            window_stats[f"score_{label}"] = (
                round(float(window_df["score"].mean()), 4)
                if not window_df.empty else None
            )
            window_stats[f"count_{label}"] = int(len(window_df))

        # Momentum: short-term vs long-term
        s7  = window_stats.get("score_7d")
        s90 = window_stats.get("score_90d")
        momentum = round(float(s7 - s90), 4) if (s7 is not None and s90 is not None) else None

        # Z-score: how unusual is current 30d sentiment vs full history baseline
        # (use 30d rolling mean vs the historical distribution of 30d means)
        score_30d = window_stats.get("score_30d")
        hist_mean = float(past["score"].mean())
        hist_std  = float(past["score"].std())
        z_score = (
            round((score_30d - hist_mean) / hist_std, 3)
            if (score_30d is not None and hist_std > 0)
            else None
        )

        # Extreme sentiment flags
        extreme_positive = z_score is not None and z_score > 2.0
        extreme_negative = z_score is not None and z_score < -2.0

        # Label distribution in last 30 days
        cutoff_30 = ref_dt - timedelta(days=30)
        recent_df = past[past["date"] >= cutoff_30]
        label_dist = (
            recent_df["overall_sentiment_label"].value_counts().to_dict()
            if not recent_df.empty else {}
        )

        # news_score: best available window (30d → 90d → all-time mean)
        # Avoids defaulting to 0.0 when recent window is empty but older data exists
        best_score = (
            window_stats.get("score_30d")
            or window_stats.get("score_90d")
            or (round(hist_mean, 4) if len(past) > 0 else None)
        )
        news_score = (
            round(max(-1.0, min(1.0, best_score)), 3)
            if best_score is not None else 0.0
        )
        news_score_source = (
            "30d" if window_stats.get("score_30d") is not None
            else "90d" if window_stats.get("score_90d") is not None
            else "all_time" if len(past) > 0
            else "none"
        )

        # news_freshness: 1.0 if articles in last 30d, decays to 0 at 90 days since last article
        days_since_last = (ref_dt - past["date"].max()).days
        news_freshness = round(max(0.0, 1.0 - days_since_last / 90.0), 3)

        return {
            "ticker":             ticker,
            "reference_date":     str(ref_dt.date()),
            "data_available":     True,
            # Rolling window stats
            **window_stats,
            # Derived signals
            "momentum":           momentum,
            "z_score":            z_score,
            "extreme_positive":   extreme_positive,
            "extreme_negative":   extreme_negative,
            "label_dist_30d":     label_dist,
            # Investment signal (for Orchestrator)
            "news_score":         news_score,
            "news_score_source":  news_score_source,
            # Data quality
            "news_freshness":     news_freshness,
            "days_since_last":    days_since_last,
            "total_articles":     int(len(past)),
            "date_range":         f"{str(past['date'].min().date())} → {str(ref_dt.date())}",
        }

    def get_all_signals(
        self,
        ticker_list: list[str] = TICKER_LIST,
        reference_date: Optional[str] = None,
    ) -> dict[str, dict]:
        """Get news signals for all tickers. Returns {ticker: signal}."""
        signals = {}
        for ticker in ticker_list:
            sig = self.get_signal(ticker, reference_date)
            if sig:
                signals[ticker] = sig
        return signals


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Earnings Call Analyzer (LLM)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_ANALYST = (
    "You are a senior equity analyst specializing in semiconductor stocks. "
    "You analyze earnings call transcripts to extract investment-relevant signals. "
    "Always respond with a single valid JSON object exactly matching the specified schema. "
    "Do not include any text outside the JSON object."
)

# Max chars to include per transcript section
MAX_PREPARED_CHARS = 8_000
MAX_QA_CHARS       = 8_000


class EarningsCallAnalyzer:
    """
    Analyzes earnings call transcripts using LLM.
    Results are cached to disk — re-running is fast and cheap.

    Only tickers with data in Earning calls/ directory are supported.
    Other tickers return None gracefully.

    Usage:
        eca = EarningsCallAnalyzer()
        result = eca.analyze_call("TSM", "2021Q1")
        all_results = eca.analyze_ticker("TSM")
    """

    # ── Data Loading ───────────────────────────────────────────────────────────

    def list_quarters(self, ticker: str) -> list[str]:
        """Return sorted list of available quarters for a ticker."""
        pattern = str(CALLS_DIR / ticker / f"{ticker}_*.json")
        files   = sorted(glob.glob(pattern))
        quarters = []
        for f in files:
            stem = Path(f).stem                    # e.g. "TSM_2021Q1"
            quarter = stem.replace(f"{ticker}_", "") # e.g. "2021Q1"
            # Only include if transcript is non-empty
            data = json.loads(Path(f).read_text(encoding="utf-8"))
            if data.get("transcript"):
                quarters.append(quarter)
        return quarters

    def _load_transcript(self, ticker: str, quarter: str) -> Optional[list[dict]]:
        path = CALLS_DIR / ticker / f"{ticker}_{quarter}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("transcript") or None

    # ── Transcript Parsing ─────────────────────────────────────────────────────

    @staticmethod
    def _is_management(title: str) -> bool:
        """Classify a speaker as management (vs analyst or operator)."""
        management_keywords = [
            "ceo", "cfo", "coo", "cto", "president", "chairman",
            "director", "officer", "vice president", "vp ",
            "head of", "general manager", "investor relation",
        ]
        title_lower = title.lower()
        analyst_keywords = ["analyst", "research"]
        if any(k in title_lower for k in analyst_keywords):
            return False
        return any(k in title_lower for k in management_keywords)

    def _parse_sections(self, transcript: list[dict]) -> tuple[str, str, str]:
        """
        Split transcript into three sections:
          1. Prepared remarks  (management only, before Q&A)
          2. Q&A — analyst questions
          3. Q&A — management answers

        Returns (prepared_text, analyst_qa_text, mgmt_qa_text).
        """
        # Detect Q&A start: first analyst appearance after index > 5
        qa_start = len(transcript)
        for i, entry in enumerate(transcript):
            if i < 5:
                continue
            if not self._is_management(entry.get("title", "")):
                qa_start = i
                break

        prepared_entries = transcript[:qa_start]
        qa_entries       = transcript[qa_start:]

        def entries_to_text(entries: list[dict], label: bool = True) -> str:
            lines = []
            for e in entries:
                speaker = e.get("speaker", "")
                content = e.get("content", "").strip()
                if content:
                    prefix = f"[{speaker}]: " if label else ""
                    lines.append(f"{prefix}{content}")
            return "\n\n".join(lines)

        prepared_text = entries_to_text(
            [e for e in prepared_entries if self._is_management(e.get("title", ""))],
            label=False
        )
        analyst_qa = entries_to_text(
            [e for e in qa_entries if not self._is_management(e.get("title", ""))],
            label=True
        )
        mgmt_qa = entries_to_text(
            [e for e in qa_entries if self._is_management(e.get("title", ""))],
            label=True
        )

        # Truncate to budget
        prepared_text = prepared_text[:MAX_PREPARED_CHARS]
        analyst_qa    = analyst_qa[:MAX_QA_CHARS // 2]
        mgmt_qa       = mgmt_qa[:MAX_QA_CHARS // 2]

        return prepared_text, analyst_qa, mgmt_qa

    # ── LLM Analysis ──────────────────────────────────────────────────────────

    def _analyze_prepared_remarks(self, ticker: str, quarter: str, text: str) -> dict:
        prompt = f"""Analyze management's prepared remarks from {ticker}'s {quarter} earnings call.

PREPARED REMARKS:
{text}

Return a JSON object with EXACTLY these fields:
{{
  "tone_score": <float -1.0 (very negative) to 1.0 (very positive)>,
  "business_momentum": <"accelerating" | "stable" | "decelerating" | "declining">,
  "guidance_tone": <"positive" | "neutral" | "negative" | "mixed">,
  "revenue_tone": <"growth" | "stable" | "declining" | "uncertain">,
  "margin_tone": <"expanding" | "stable" | "contracting" | "uncertain">,
  "management_confidence": <float 0.0 to 1.0>,
  "key_themes": [<3-5 most important themes management emphasized>],
  "forward_looking_statements": [<2-3 key forward-looking statements, brief>],
  "capex_signals": <"increasing" | "stable" | "decreasing" | "not_mentioned">,
  "reasoning": <2-3 sentence explanation of tone_score>
}}"""
        return extract_json(call_llm(prompt, SYSTEM_ANALYST))

    def _analyze_qa(self, ticker: str, quarter: str, analyst_text: str, mgmt_text: str) -> dict:
        prompt = f"""Analyze the Q&A section from {ticker}'s {quarter} earnings call.

ANALYST QUESTIONS:
{analyst_text}

MANAGEMENT ANSWERS:
{mgmt_text}

Return a JSON object with EXACTLY these fields:
{{
  "analyst_aggressiveness": <float 0.0 (gentle/supportive) to 1.0 (very aggressive/skeptical)>,
  "qa_dynamics": <"supportive" | "neutral" | "skeptical" | "hostile">,
  "mgmt_answer_quality": <"evasive" | "partial" | "direct" | "very_direct">,
  "hot_topics": [<2-4 topics analysts pushed hardest on>],
  "mgmt_defense_strength": <float 0.0 to 1.0>,
  "analyst_concern_areas": [<2-3 areas of analyst concern or skepticism>],
  "positive_signals_from_qa": [<1-2 positive datapoints management revealed in Q&A>]
}}"""
        return extract_json(call_llm(prompt, SYSTEM_ANALYST, max_tokens=2048))

    def _compute_call_score(self, prepared: dict, qa: dict) -> float:
        """
        Aggregate prepared remarks + Q&A into a single call score [-1, +1].
        """
        tone       = prepared.get("tone_score", 0.0)
        confidence = prepared.get("management_confidence", 0.5)
        aggression = qa.get("analyst_aggressiveness", 0.5)
        defense    = qa.get("mgmt_defense_strength", 0.5)

        score = (
            tone       * 0.50     # Management tone dominates
            + (confidence - 0.5) * 0.20
            + (defense - 0.5) * 0.15
            + (0.5 - aggression) * 0.15   # High analyst aggression → negative signal
        )
        return round(max(-1.0, min(1.0, score)), 3)

    def _compute_delta(
        self,
        ticker: str,
        current_q: str, current: dict,
        prior_q: str,   prior: dict,
    ) -> dict:
        prompt = f"""Compare {ticker}'s earnings calls: {current_q} (current) vs {prior_q} (prior).

CURRENT ({current_q}):
Prepared tone: {current['prepared'].get('tone_score')}, momentum: {current['prepared'].get('business_momentum')}
Guidance: {current['prepared'].get('guidance_tone')}, themes: {current['prepared'].get('key_themes')}
Q&A dynamics: {current['qa'].get('qa_dynamics')}, analyst concerns: {current['qa'].get('analyst_concern_areas')}

PRIOR ({prior_q}):
Prepared tone: {prior['prepared'].get('tone_score')}, momentum: {prior['prepared'].get('business_momentum')}
Guidance: {prior['prepared'].get('guidance_tone')}, themes: {prior['prepared'].get('key_themes')}
Q&A dynamics: {prior['qa'].get('qa_dynamics')}, analyst concerns: {prior['qa'].get('analyst_concern_areas')}

Return a JSON object with EXACTLY these fields:
{{
  "tone_direction": <"improving" | "stable" | "deteriorating">,
  "confidence_direction": <"improving" | "stable" | "deteriorating">,
  "new_themes": [<themes in current not in prior>],
  "dropped_themes": [<themes in prior not in current>],
  "key_changes": [<2-4 most important changes between calls>],
  "overall_delta": <"significantly_improved" | "improved" | "stable" | "deteriorated" | "significantly_deteriorated">,
  "delta_summary": <2 sentence summary of what changed>
}}"""
        result = extract_json(call_llm(prompt, SYSTEM_ANALYST, max_tokens=2048))
        result["vs_prior_quarter"] = prior_q
        return result

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze_call(
        self,
        ticker: str,
        quarter: str,
        force_refresh: bool = False,
    ) -> Optional[dict]:
        """
        Analyze a single earnings call. Returns structured result.
        Loads from disk cache unless force_refresh=True.
        """
        cache_path = _cache_dir(ticker) / f"call_{quarter}.json"

        if not force_refresh:
            cached = _load_json(cache_path)
            if cached:
                print(f"  [CACHE]   {ticker} {quarter}")
                return cached

        transcript = self._load_transcript(ticker, quarter)
        if not transcript:
            print(f"  [SKIP]    {ticker} {quarter} — no transcript")
            return None

        print(f"  [ANALYZE] {ticker} {quarter} ({len(transcript)} transcript entries)")

        prepared_text, analyst_qa, mgmt_qa = self._parse_sections(transcript)

        print(f"    Prepared remarks: {len(prepared_text):,} chars")
        prepared = self._analyze_prepared_remarks(ticker, quarter, prepared_text)

        print(f"    Q&A: analyst {len(analyst_qa):,} + mgmt {len(mgmt_qa):,} chars")
        qa = self._analyze_qa(ticker, quarter, analyst_qa, mgmt_qa)

        call_score = self._compute_call_score(prepared, qa)

        result = {
            "ticker":       ticker,
            "quarter":      quarter,
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "call_score":   call_score,
            "prepared":     prepared,
            "qa":           qa,
            "delta":        None,
            "confidence":   0.85,
        }

        _save_json(result, cache_path)
        return result

    def analyze_ticker(
        self,
        ticker: str,
        force_refresh: bool = False,
    ) -> list[dict]:
        """
        Analyze all available earnings calls for a ticker.
        Adds quarter-over-quarter delta analysis.
        Returns list sorted chronologically.
        """
        quarters = self.list_quarters(ticker)
        if not quarters:
            print(f"[WARNING] No earnings call data for {ticker}")
            return []

        print(f"\n{'='*55}")
        print(f"  {ticker} Earnings Calls — {len(quarters)} quarters")
        print(f"{'='*55}")

        results = []
        for q in quarters:
            r = self.analyze_call(ticker, q, force_refresh)
            if r:
                results.append(r)

        # Quarter-over-quarter delta
        print(f"\n  Computing QoQ delta ({len(results)-1} comparisons)…")
        for i in range(1, len(results)):
            cur  = results[i]
            prev = results[i - 1]
            if cur.get("delta") is not None:
                continue
            try:
                delta = self._compute_delta(ticker, cur["quarter"], cur, prev["quarter"], prev)
                results[i]["delta"] = delta
                _save_json(results[i], _cache_dir(ticker) / f"call_{cur['quarter']}.json")
                print(f"    {cur['quarter']} vs {prev['quarter']}: {delta.get('overall_delta','?')}")
            except Exception as e:
                print(f"    [ERROR] Delta {cur['quarter']}: {e}")

        return results

    @staticmethod
    def _call_available_by(quarter: str) -> datetime:
        """
        Earnings calls are typically held 4-6 weeks after quarter end.
        Returns the date by which this quarter's call is expected to be available.
        Uses a conservative 45-day lag from quarter end.
        """
        year  = int(quarter[:4])
        q_num = int(quarter[5])
        quarter_end_month = {1: 3, 2: 6, 3: 9, 4: 12}[q_num]
        quarter_end_day   = {1: 31, 2: 30, 3: 30, 4: 31}[q_num]
        return datetime(year, quarter_end_month, quarter_end_day) + timedelta(days=45)

    def get_latest_call(self, ticker: str, reference_date: Optional[str] = None) -> Optional[dict]:
        """
        Get the most recent earnings call analysis as of reference_date.
        Also ensures the prior quarter is analyzed so delta is available.

        reference_date: ISO date string (e.g. "2020-01-01"). Uses a 45-day
                        lag from quarter end to determine availability, avoiding
                        look-ahead bias. Defaults to latest available.
        """
        quarters = self.list_quarters(ticker)
        if not quarters:
            return None

        if reference_date:
            ref_dt   = datetime.strptime(reference_date, "%Y-%m-%d")
            eligible = [q for q in quarters if self._call_available_by(q) <= ref_dt]
        else:
            eligible = quarters

        if not eligible:
            return None

        latest = eligible[-1]

        # Ensure prior quarter is analyzed first (needed for delta)
        if len(eligible) >= 2:
            prior = eligible[-2]
            prior_cached = _load_json(_cache_dir(ticker) / f"call_{prior}.json")
            if not prior_cached:
                self.analyze_call(ticker, prior)

        # Analyze latest quarter
        result = self.analyze_call(ticker, latest)
        if result is None:
            return None

        # Compute delta if not yet present
        if result.get("delta") is None and len(eligible) >= 2:
            prior = eligible[-2]
            prior_result = _load_json(_cache_dir(ticker) / f"call_{prior}.json")
            if prior_result:
                try:
                    delta = self._compute_delta(ticker, latest, result, prior, prior_result)
                    result["delta"] = delta
                    _save_json(result, _cache_dir(ticker) / f"call_{latest}.json")
                except Exception as e:
                    print(f"  [ERROR] Delta {latest}: {e}")

        return result


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Composite Sentiment Agent
# ══════════════════════════════════════════════════════════════════════════════

class SentimentAgent:
    """
    Combines NewsSentimentAnalyzer and EarningsCallAnalyzer into one agent.

    Weights:
      - If earnings call data available: 50% news + 50% call
      - If no call data:                 100% news

    Usage:
        agent = SentimentAgent()
        signal = agent.get_signal("NVDA", reference_date="2024-06-30")
        all_signals = agent.get_all_signals()
    """

    def __init__(self):
        self.news = NewsSentimentAnalyzer()
        self.calls = EarningsCallAnalyzer()

    def get_signal(
        self,
        ticker: str,
        reference_date: Optional[str] = None,
    ) -> dict:
        """
        Get composite sentiment signal for a ticker.

        reference_date: used for news window endpoint and earnings call
                        date filter (ISO date string).
        """
        news_sig = self.news.get_signal(ticker, reference_date)
        call_sig = self.calls.get_latest_call(ticker, reference_date)

        news_score    = news_sig["news_score"]    if news_sig else 0.0
        news_freshness = news_sig["news_freshness"] if news_sig else 0.0
        call_score    = call_sig["call_score"]    if call_sig else None

        # Dynamic weights: as news gets staler, call gets more weight
        # news_freshness=1.0 → 50/50; news_freshness=0.0 → 0/100
        if call_score is not None:
            call_weight = round(0.5 + 0.5 * (1.0 - news_freshness), 3)
            news_weight = round(1.0 - call_weight, 3)
            composite   = round(news_weight * news_score + call_weight * call_score, 3)
        else:
            news_weight = 1.0
            call_weight = 0.0
            composite   = news_score

        # Composite label
        if composite >= 0.3:
            label = "bullish"
        elif composite >= 0.1:
            label = "somewhat_bullish"
        elif composite >= -0.1:
            label = "neutral"
        elif composite >= -0.3:
            label = "somewhat_bearish"
        else:
            label = "bearish"

        return {
            "ticker":                  ticker,
            "reference_date":          reference_date or "latest",
            # Composite signal (for Orchestrator)
            "sentiment_score":         composite,
            "sentiment_label":         label,
            "news_weight":             news_weight,
            "call_weight":             call_weight,
            # News component
            "news_score":              news_score,
            "news_freshness":          news_freshness,
            "news_score_source":       news_sig.get("news_score_source") if news_sig else None,
            "news_momentum":           news_sig.get("momentum") if news_sig else None,
            "news_z_score":            news_sig.get("z_score") if news_sig else None,
            "news_extreme_positive":   news_sig.get("extreme_positive", False) if news_sig else False,
            "news_extreme_negative":   news_sig.get("extreme_negative", False) if news_sig else False,
            "news_count_30d":          news_sig.get("count_30d", 0) if news_sig else 0,
            "days_since_last_news":    news_sig.get("days_since_last") if news_sig else None,
            # Earnings call component
            "call_score":              call_score,
            "call_quarter":            call_sig["quarter"] if call_sig else None,
            "call_guidance_tone":      call_sig["prepared"].get("guidance_tone") if call_sig else None,
            "call_mgmt_confidence":    call_sig["prepared"].get("management_confidence") if call_sig else None,
            "call_key_themes":         call_sig["prepared"].get("key_themes", []) if call_sig else [],
            "call_delta":              call_sig["delta"].get("overall_delta") if (call_sig and call_sig.get("delta")) else None,
            # Data coverage
            "has_news":                news_sig is not None,
            "has_call":                call_sig is not None,
        }

    def get_all_signals(
        self,
        ticker_list: list[str] = TICKER_LIST,
        reference_date: Optional[str] = None,
    ) -> dict[str, dict]:
        """Get composite signals for all tickers. Returns {ticker: signal}."""
        signals = {}
        print(f"\nCollecting sentiment signals for {len(ticker_list)} tickers…")
        for ticker in ticker_list:
            sig = self.get_signal(ticker, reference_date)
            signals[ticker] = sig
            print(f"  {ticker:<6} score={sig['sentiment_score']:+.3f}  "
                  f"({sig['sentiment_label']})  "
                  f"news={'✓' if sig['has_news'] else '✗'}  "
                  f"call={'✓' if sig['has_call'] else '✗'}")
        return signals

    def print_summary(self, signals: dict[str, dict]):
        """Print ranked summary table."""
        rows = sorted(signals.values(), key=lambda x: x["sentiment_score"], reverse=True)
        if not rows:
            print("  [No signals to display]")
            return
        print(f"\n{'='*70}")
        print(f"  SENTIMENT SIGNALS  —  {rows[0].get('reference_date','')}")
        print(f"{'='*70}")
        print(f"  {'Ticker':<7} {'Score':>6}  {'Label':<18} {'Momentum':>9} "
              f"{'Call Score':>10}  {'Call Guidance'}")
        print(f"  {'-'*7} {'-'*6}  {'-'*18} {'-'*9} {'-'*10}  {'-'*16}")
        for r in rows:
            mom = f"{r['news_momentum']:+.3f}" if r.get("news_momentum") is not None else "  N/A "
            cs  = f"{r['call_score']:+.2f}" if r.get("call_score") is not None else "    N/A"
            cg  = r.get("call_guidance_tone", "N/A") or "N/A"
            print(f"  {r['ticker']:<7} {r['sentiment_score']:>+6.3f}  "
                  f"{r['sentiment_label']:<18} {mom:>9} {cs:>10}  {cg}")

    def export_signals_csv(
        self,
        ticker_list: list[str] = TICKER_LIST,
        reference_date: Optional[str] = None,
        output_path: str = "sentiment_signals.csv",
    ):
        """Export signals to CSV for use by Portfolio Agent."""
        import csv
        signals = self.get_all_signals(ticker_list, reference_date)
        rows = list(signals.values())
        if not rows:
            print("  [No signals to export]")
            return
        out = BASE_DIR / output_path
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nExported {len(rows)} signals → {out}")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    TICKER = "NVDA"
    REF_DATE = "2023-01-01"

    agent = SentimentAgent()
    print(f"=== Sentiment Agent — NVDA @ {REF_DATE} ===\n")

    # 1. News signal
    print("--- News Sentiment ---")
    news_sig = agent.news.get_signal(TICKER, reference_date=REF_DATE)
    if news_sig:
        print(f"  score_7d  : {news_sig.get('score_7d')}")
        print(f"  score_30d : {news_sig.get('score_30d')}")
        print(f"  score_90d : {news_sig.get('score_90d')}")
        print(f"  momentum  : {news_sig.get('momentum')}")
        print(f"  z_score   : {news_sig.get('z_score')}")
        print(f"  extreme+  : {news_sig.get('extreme_positive')}  extreme-: {news_sig.get('extreme_negative')}")
        print(f"  news_score: {news_sig.get('news_score')}")
        print(f"  articles  : {news_sig.get('count_30d')} (30d)  /  {news_sig.get('total_articles')} total")
    else:
        print("  No news data.")

    # 2. Earnings call — latest quarter available as of REF_DATE (45-day lag)
    print(f"\n--- Earnings Call (latest available as of {REF_DATE}) ---")
    all_quarters = agent.calls.list_quarters(TICKER)
    eligible = [q for q in all_quarters if agent.calls._call_available_by(q) <= datetime.strptime(REF_DATE, "%Y-%m-%d")]
    print(f"  Available quarters : {all_quarters}")
    print(f"  Eligible (45d lag) : {eligible}")

    if eligible:
        target_q = eligible[-1]
        call_result = agent.calls.analyze_call(TICKER, target_q)
        if call_result:
            print(f"\n  {TICKER} {call_result['quarter']}:")
            print(f"    call_score      : {call_result['call_score']:+.3f}")
            print(f"    guidance_tone   : {call_result['prepared'].get('guidance_tone')}")
            print(f"    mgmt_confidence : {call_result['prepared'].get('management_confidence'):.2f}")
            print(f"    key_themes      : {call_result['prepared'].get('key_themes', [])}")
            print(f"    qa_dynamics     : {call_result['qa'].get('qa_dynamics')}")
            if call_result.get("delta"):
                print(f"    delta           : {call_result['delta'].get('overall_delta')}")
                print(f"    delta_summary   : {call_result['delta'].get('delta_summary')}")
    else:
        print("  No eligible earnings call data.")

    # 3. Composite signal
    print("\n--- Composite Signal ---")
    sig = agent.get_signal(TICKER, reference_date=REF_DATE)
    print(json.dumps(sig, indent=2, ensure_ascii=False))
