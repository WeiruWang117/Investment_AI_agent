"""
RiskControlAgent

Evaluates risk and produces a final risk-adjusted trading decision for each
semiconductor stock. This agent does NOT generate new trading signals — it
takes the macro-adjusted score from MacroAgent and applies a risk overlay.

Position in pipeline:
  MacroAgent → macro_adjusted_score
      ↓
  RiskControlAgent → final_decision, position_size, risk_score
      ↓
  DecisionAgent (planned)

Inputs (all read from cache files):
  macro_cache/{TICKER}/macro_{DATE}.json         — MacroAgent output
  technical_cache/{TICKER}/signal_{DATE}.json    — volatility, volume, momentum
  fundamental_cache/{TICKER}/signal_{DATE}.json  — financial health metrics
  arbiter_cache/{TICKER}/arbiter_{DATE}.json     — debate flags, agreement level
  Macro data/VIX.csv                             — daily VIX (market fear gauge)

Risk dimensions evaluated:
  1. Volatility Risk      — hv_20d, atr_pct, bb_width
  2. Drawdown Risk        — roc_20d, roc_60d (recent price decline)
  3. Tail Risk            — VIX spike, override_flags, arbiter flags
  4. Liquidity Risk       — vol_ratio (trading volume vs average)
  5. Market Risk          — VIX level, macro_regime, override_flags
  6. Sector Concentration — always elevated (semi-only portfolio)
  7. Signal Confidence    — arbiter_confidence, agreement_level

Output schema:
  ticker                  str
  date                    str
  final_decision          str    "buy" | "sell" | "hold"
  position_size           float  [0.0, 1.0]  fraction of max allocation
  risk_score              float  [0.0, 1.0]  composite risk (higher = riskier)
  volatility_risk         float  [0.0, 1.0]
  drawdown_risk           float  [0.0, 1.0]
  tail_risk               float  [0.0, 1.0]
  liquidity_risk          float  [0.0, 1.0]
  market_risk             float  [0.0, 1.0]
  signal_confidence_risk  float  [0.0, 1.0]
  risk_adjustments        list[str]   specific adjustments applied
  summary                 str    concise explanation
  macro_adjusted_score    float  passed through from MacroAgent
  vix_latest              float  VIX value used
  data_date               str    date of VIX/technical data used

Cache:
  risk_cache/{TICKER}/risk_{DATE}.json
"""

import json
import time
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

import anthropic

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_DIR   = Path("d:/Investment_AI_agent")
CACHE_DIR  = BASE_DIR / "risk_cache"
CACHE_DIR.mkdir(exist_ok=True)

MODEL = "claude-sonnet-4-6"

TICKER_LIST = [
    "TSM", "INTC", "NVDA", "ASML", "AVGO", "TXN", "QCOM", "MU",
    "AMAT", "AMD", "ADI", "LRCX", "KLAC", "MCHP", "SWKS", "MRVL",
    "TER", "ON", "MPWR", "ENTG", "MKSI", "SLAB", "CRUS", "SMTC",
]

client = anthropic.Anthropic()


# ── Cache / IO Utilities ───────────────────────────────────────────────────────

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


# ── VIX Loader ─────────────────────────────────────────────────────────────────

def _load_vix(as_of_date: str) -> dict:
    """
    Load VIX data from Macro data/VIX.csv.
    Returns dict with latest value, 20-day average, and regime classification.
    VIX.csv has two extra yfinance header rows — skip them.
    """
    vix_path = BASE_DIR / "Macro data" / "VIX.csv"
    if not vix_path.exists():
        return {"vix_latest": None, "vix_20d_avg": None, "vix_regime": "unknown"}

    try:
        import pandas as pd
        # Skip rows 1 and 2 (yfinance metadata: "Ticker,^VIX" and "Date,")
        df = pd.read_csv(vix_path, skiprows=[1, 2], index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.dropna()
        df = df[df.index <= pd.Timestamp(as_of_date)]
        df = df.sort_index()

        if df.empty:
            return {"vix_latest": None, "vix_20d_avg": None, "vix_regime": "unknown"}

        col = df.columns[0]  # "vix_close"
        vix_latest  = round(float(df[col].iloc[-1]), 2)
        vix_20d_avg = round(float(df[col].tail(20).mean()), 2)
        vix_52w_pct = round(float(
            (vix_latest - df[col].tail(252).min()) /
            (df[col].tail(252).max() - df[col].tail(252).min() + 1e-9)
        ), 3)

        # VIX regime classification
        if vix_latest < 15:
            vix_regime = "low"          # complacency / risk-on
        elif vix_latest < 20:
            vix_regime = "normal"
        elif vix_latest < 30:
            vix_regime = "elevated"     # caution
        elif vix_latest < 40:
            vix_regime = "high"         # fear / risk-off
        else:
            vix_regime = "extreme"      # crisis

        vix_spike = vix_latest > vix_20d_avg * 1.3  # >30% above 20-day avg

        return {
            "vix_latest":    vix_latest,
            "vix_20d_avg":   vix_20d_avg,
            "vix_52w_pct":   vix_52w_pct,   # 0=52w low, 1=52w high
            "vix_regime":    vix_regime,
            "vix_spike":     vix_spike,
        }
    except Exception as e:
        print(f"  [RiskControl] VIX load error: {e}")
        return {"vix_latest": None, "vix_20d_avg": None, "vix_regime": "unknown", "vix_spike": False}


# ── Quantitative Risk Scoring ──────────────────────────────────────────────────

def _score(value: Optional[float], low: float, high: float) -> float:
    """
    Linearly map value from [low, high] → [0, 1].
    Values below low → 0.0 (no risk), above high → 1.0 (max risk).
    Returns 0.5 if value is None (unknown = moderate risk).
    """
    if value is None:
        return 0.5
    return max(0.0, min(1.0, (value - low) / (high - low + 1e-9)))


def _compute_risk_scores(
    technical: dict,
    fundamental: dict,
    macro: dict,
    arbiter: dict,
    vix: dict,
) -> dict:
    """
    Pure Python — no LLM.
    Computes 6 dimensional risk scores, each in [0, 1].
    """

    # ── 1. Volatility Risk ────────────────────────────────────────────────────
    hv_20d    = technical.get("hv_20d")         # annualized HV, e.g. 0.45 = 45%
    hv_60d    = technical.get("hv_60d")
    atr_pct   = technical.get("atr_pct")        # ATR as % of price, e.g. 0.03 = 3%
    bb_width  = technical.get("bb_width")       # Bollinger Band width

    # Semiconductor average HV ~35-40%; >60% = high risk, <25% = low risk
    hv_score   = _score(hv_20d,   0.25, 0.65)
    atr_score  = _score(atr_pct,  0.02, 0.06)
    bb_score   = _score(bb_width, 0.05, 0.20)
    vol_risk   = round(hv_score * 0.5 + atr_score * 0.3 + bb_score * 0.2, 3)

    # ── 2. Drawdown Risk ──────────────────────────────────────────────────────
    roc_20d  = technical.get("roc_20d")         # e.g. -0.15 = down 15% in 20 days
    roc_60d  = technical.get("roc_60d")
    roc_252d = technical.get("roc_252d")

    # Convert decline into risk (negative return = higher risk)
    dd_20_score  = _score(-(roc_20d  or 0), -0.05, 0.20)  # >20% drop → max risk
    dd_60_score  = _score(-(roc_60d  or 0), -0.05, 0.30)
    dd_252_score = _score(-(roc_252d or 0), -0.10, 0.50)
    dd_risk = round(dd_20_score * 0.50 + dd_60_score * 0.30 + dd_252_score * 0.20, 3)

    # ── 3. Tail Risk ──────────────────────────────────────────────────────────
    vix_spike     = vix.get("vix_spike", False)
    override_flags = macro.get("override_flags", [])
    arbiter_flags  = arbiter.get("flags", [])

    dangerous_flags = {
        "yield_curve_inverted", "inflation_elevated", "gdp_negative",
        "unemployment_rising", "rates_restrictive",
    }
    macro_flag_count  = sum(1 for f in override_flags if f in dangerous_flags)
    arbiter_flag_risk = 1 if "high_disagreement" in arbiter_flags else 0
    arbiter_flag_risk += 1 if "low_confidence" in arbiter_flags else 0

    tail_risk = round(min(1.0,
        (0.4 if vix_spike else 0.0) +
        macro_flag_count * 0.15 +
        arbiter_flag_risk * 0.15
    ), 3)

    # ── 4. Liquidity Risk ─────────────────────────────────────────────────────
    vol_ratio = technical.get("vol_ratio")  # current vol / 20d avg
    # Low volume ratio (<0.5) = thin market = harder to exit
    # High ratio (>3) = panic volume, also risky
    if vol_ratio is None:
        liq_risk = 0.5
    elif vol_ratio < 0.5:
        liq_risk = 0.7   # unusually thin
    elif vol_ratio > 3.0:
        liq_risk = 0.6   # panic/surge volume
    else:
        liq_risk = round(_score(1.0 / vol_ratio, 0.3, 2.0) * 0.5, 3)  # thin = risky

    # ── 5. Market Risk ────────────────────────────────────────────────────────
    vix_regime = vix.get("vix_regime", "unknown")
    macro_regime = macro.get("macro_regime", "unknown")

    vix_market_score = {
        "low": 0.1, "normal": 0.25, "elevated": 0.5,
        "high": 0.75, "extreme": 1.0, "unknown": 0.5,
    }.get(vix_regime, 0.5)

    regime_market_score = {
        "expansion": 0.2, "peak": 0.45, "contraction": 0.7, "trough": 0.55, "unknown": 0.5,
    }.get(macro_regime, 0.5)

    market_risk = round(vix_market_score * 0.55 + regime_market_score * 0.45, 3)

    # ── 6. Signal Confidence Risk ─────────────────────────────────────────────
    confidence       = arbiter.get("confidence", 0.5)      # [0,1] higher = more certain
    agreement_level  = arbiter.get("agreement_level", "mixed")
    macro_score_abs  = abs(macro.get("macro_adjusted_score", 0.0))

    agreement_penalty = {"aligned": 0.0, "mixed": 0.2, "divergent": 0.4}.get(agreement_level, 0.2)
    conf_risk = round(
        (1.0 - confidence) * 0.6 +
        agreement_penalty * 0.3 +
        (1.0 - min(1.0, macro_score_abs * 2)) * 0.1,  # near-zero score = uncertain
        3
    )

    # ── Composite Risk Score (weighted) ──────────────────────────────────────
    composite = round(
        vol_risk  * 0.20 +
        dd_risk   * 0.20 +
        tail_risk * 0.15 +
        liq_risk  * 0.10 +
        market_risk * 0.25 +
        conf_risk * 0.10,
        3
    )

    # ── Preliminary position size (before LLM) ────────────────────────────────
    # Base: |macro_adjusted_score|, scaled down by risk
    signal_strength = abs(macro.get("macro_adjusted_score", 0.0))
    risk_multiplier = max(0.1, 1.0 - composite * 1.2)  # high risk → shrink aggressively
    prelim_size = round(min(1.0, signal_strength * risk_multiplier), 3)

    return {
        "volatility_risk":        vol_risk,
        "drawdown_risk":          dd_risk,
        "tail_risk":              tail_risk,
        "liquidity_risk":         liq_risk,
        "market_risk":            market_risk,
        "signal_confidence_risk": conf_risk,
        "composite_risk_score":   composite,
        "preliminary_position_size": prelim_size,
        # Supporting raw values for LLM prompt
        "hv_20d":         hv_20d,
        "atr_pct":        atr_pct,
        "roc_20d":        roc_20d,
        "roc_60d":        roc_60d,
        "vol_ratio":      vol_ratio,
        "vix_regime":     vix_regime,
        "vix_spike":      vix.get("vix_spike", False),
        "macro_flag_count": macro_flag_count,
        "arbiter_flags":  arbiter_flags,
        "agreement_level": agreement_level,
        "confidence":     confidence,
    }


# ── System Prompt ──────────────────────────────────────────────────────────────

SYSTEM_RISK = """You are a Risk Control Agent in a multi-agent semiconductor investment system.

Your role is NOT to generate new trading signals, but to evaluate and adjust the final
decision based on risk considerations.

You receive:
1. A preliminary macro-adjusted score and direction signal (buy/sell/hold candidate)
2. Pre-computed quantitative risk scores across 6 dimensions
3. Bull and Bear analysis summaries
4. Macro environment context
5. Stock-level volatility and fundamental metrics

Your task:
- Assess downside risk holistically across all dimensions
- Set final_decision: "buy" / "sell" / "hold"
- Set position_size [0.0, 1.0] — fraction of maximum allowed allocation
- Set risk_score [0.0, 1.0] — your overall risk assessment
- List 3-5 specific risk_adjustments you applied
- Write a concise summary

RISK EVALUATION FRAMEWORK:

1. Volatility Risk — high HV/ATR → reduce position
2. Drawdown Risk — recent sharp declines → reduce or hold
3. Tail Risk — VIX spike, macro flags, earnings proximity → hard reduction
4. Liquidity Risk — low vol_ratio → harder to exit, reduce
5. Market Risk — VIX regime + macro regime combined
6. Sector Concentration — all stocks are semiconductors: NEVER allow full position
   in multiple high-risk semis simultaneously. Max sector exposure applies.
7. Signal Confidence — low confidence or divergent bull/bear → reduce

POSITION SIZING RULES:
- composite_risk_score < 0.25 → position_size up to 0.9
- composite_risk_score 0.25–0.45 → position_size 0.5–0.75
- composite_risk_score 0.45–0.65 → position_size 0.2–0.5
- composite_risk_score > 0.65 → position_size ≤ 0.2, consider hold

DECISION RULES:
- macro_adjusted_score > +0.20 AND risk allows → "buy"
- macro_adjusted_score < -0.20 AND risk allows → "sell"
- |macro_adjusted_score| < 0.20 → lean toward "hold"
- ANY of: extreme VIX, gdp_negative + yield_curve_inverted together,
  arbiter high_disagreement + low confidence → override to "hold" if uncertain

IMPORTANT:
- Do NOT ignore risk even if signals are strong
- Focus on downside protection over upside potential
- Be conservative under uncertainty (semiconductor sector is inherently volatile)
- Sector concentration is always a risk — never treat any semi as low-risk
- Output ONLY valid JSON. No markdown, no commentary outside the JSON."""


# ── LLM Call ──────────────────────────────────────────────────────────────────

def _call_llm(prompt: str, retries: int = 5) -> str:
    for attempt in range(retries + 1):
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=4000,
                thinking={"type": "adaptive"},
                system=SYSTEM_RISK,
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


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            break
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON in LLM response: {raw[:200]}")
    return json.loads(text[start:end + 1])


# ── Prompt Builder ─────────────────────────────────────────────────────────────

def _build_prompt(
    ticker: str,
    macro: dict,
    technical: dict,
    fundamental: dict,
    arbiter: dict,
    vix: dict,
    risk_scores: dict,
    date_str: str = "",
) -> str:

    macro_score    = macro.get("macro_adjusted_score", 0.0)
    macro_regime   = macro.get("macro_regime", "unknown")
    subsector      = macro.get("subsector", "unknown")
    override_flags = macro.get("override_flags", [])
    macro_summary  = macro.get("macro_summary", "")

    arb_verdict    = arbiter.get("verdict", "neutral")
    arb_confidence = arbiter.get("confidence", 0.5)
    arb_flags      = arbiter.get("flags", [])
    key_risk       = arbiter.get("key_risk", "not specified")
    resolution     = arbiter.get("resolution_summary", "")
    bull_wins      = arbiter.get("bull_wins_on", [])
    bear_wins      = arbiter.get("bear_wins_on", [])
    agreement      = arbiter.get("agreement_level", "mixed")

    # Direction hint
    if macro_score > 0.20:
        direction_hint = f"BUY candidate (score={macro_score:+.3f})"
    elif macro_score < -0.20:
        direction_hint = f"SELL candidate (score={macro_score:+.3f})"
    else:
        direction_hint = f"HOLD candidate (score={macro_score:+.3f}, near-zero conviction)"

    prompt = f"""
RISK CONTROL EVALUATION
=======================
Ticker    : {ticker}
Subsector : {subsector}
Date      : {date_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIGNAL SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Macro-adjusted score  : {macro_score:+.4f}  → {direction_hint}
Arbiter verdict       : {arb_verdict}  (confidence={arb_confidence:.2f})
Agreement level       : {agreement}
Macro regime          : {macro_regime}
Macro summary         : {macro_summary}
Key risk (arbiter)    : {key_risk}
Resolution summary    : {resolution}
Bull wins on          : {bull_wins}
Bear wins on          : {bear_wins}
Arbiter flags         : {arb_flags if arb_flags else 'none'}
Macro override flags  : {override_flags if override_flags else 'none'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRE-COMPUTED RISK SCORES  [0=low risk, 1=high risk]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Volatility risk       : {risk_scores['volatility_risk']:.3f}  (HV-20d={risk_scores['hv_20d']}, ATR%={risk_scores['atr_pct']})
Drawdown risk         : {risk_scores['drawdown_risk']:.3f}  (ROC-20d={risk_scores['roc_20d']}, ROC-60d={risk_scores['roc_60d']})
Tail risk             : {risk_scores['tail_risk']:.3f}  (VIX spike={risk_scores['vix_spike']}, macro flags={risk_scores['macro_flag_count']})
Liquidity risk        : {risk_scores['liquidity_risk']:.3f}  (vol ratio={risk_scores['vol_ratio']})
Market risk           : {risk_scores['market_risk']:.3f}  (VIX regime={risk_scores['vix_regime']})
Signal confidence risk: {risk_scores['signal_confidence_risk']:.3f}  (agreement={risk_scores['agreement_level']}, confidence={risk_scores['confidence']:.2f})
─────────────────────────────────────────────────────
COMPOSITE RISK SCORE  : {risk_scores['composite_risk_score']:.3f}
Preliminary pos. size : {risk_scores['preliminary_position_size']:.3f}  (quantitative pre-estimate)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MARKET CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VIX latest            : {vix.get('vix_latest')}  (20d avg={vix.get('vix_20d_avg')}, regime={vix.get('vix_regime')})
VIX 52-week percentile: {vix.get('vix_52w_pct')}  (0=52w low, 1=52w high)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUNDAMENTAL HEALTH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Revenue YoY growth    : {fundamental.get('revenue_yoy')}
Gross margin          : {fundamental.get('gross_margin')}
Current ratio         : {fundamental.get('current_ratio')}
Debt-to-equity        : {fundamental.get('debt_to_equity')}
Inventory days        : {fundamental.get('inventory_days')}  (>130=oversupply risk)
Composite fund. score : {fundamental.get('composite_score')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR OUTPUT (strict JSON, all fields required)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "final_decision": "buy" | "sell" | "hold",
  "position_size": <float 0.0–1.0>,
  "risk_score": <float 0.0–1.0>,
  "risk_adjustments": [
    "<specific adjustment 1>",
    "<specific adjustment 2>",
    "<specific adjustment 3>"
  ],
  "summary": "<2-3 sentence concise explanation>"
}}
""".strip()
    return prompt


# ── RiskControlAgent Class ─────────────────────────────────────────────────────

class RiskControlAgent:
    """
    Applies risk overlay to MacroAgent output, producing final trading decision.

    Usage:
        agent = RiskControlAgent()
        result = agent.get_signal("NVDA", date_str="2026-03-23")

    Reads from macro_cache, technical_cache, fundamental_cache, arbiter_cache.
    Writes to risk_cache/{TICKER}/risk_{DATE}.json.
    """

    def get_signal(
        self,
        ticker: str,
        date_str: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Optional[dict]:
        """
        Compute risk-adjusted signal for one ticker.

        Returns None if upstream cache is missing (run MacroAgent first).
        """
        today  = date_str or datetime.now().strftime("%Y-%m-%d")
        ticker = ticker.upper()
        cache_path = _cache_dir(ticker) / f"risk_{today}.json"

        if not force_refresh:
            cached = _load_json(cache_path)
            if cached:
                print(f"  [RiskControl] {ticker}: loaded from cache ({today})")
                return cached

        # ── Load upstream caches ──────────────────────────────────────────────
        macro       = _load_json(BASE_DIR / "macro_cache"       / ticker / f"macro_{today}.json")
        technical   = _load_json(BASE_DIR / "technical_cache"   / ticker / f"signal_{today}.json")
        fundamental = _load_json(BASE_DIR / "fundamental_cache" / ticker / f"signal_{today}.json")
        arbiter     = _load_json(BASE_DIR / "arbiter_cache"     / ticker / f"arbiter_{today}.json")

        # Try most recent cached fundamental/technical if today's not available
        if technical is None:
            for days_back in range(1, 8):
                d = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=days_back)).strftime("%Y-%m-%d")
                technical = _load_json(BASE_DIR / "technical_cache" / ticker / f"signal_{d}.json")
                if technical:
                    break

        if fundamental is None:
            for days_back in range(1, 30):
                d = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=days_back)).strftime("%Y-%m-%d")
                fundamental = _load_json(BASE_DIR / "fundamental_cache" / ticker / f"signal_{d}.json")
                if fundamental:
                    break

        if macro is None:
            print(f"  [RiskControl] {ticker}: missing macro cache — run MacroAgent first.")
            return None
        if arbiter is None:
            print(f"  [RiskControl] {ticker}: missing arbiter cache — run ArbiterAgent first.")
            return None

        technical   = technical   or {}
        fundamental = fundamental or {}

        # ── Load VIX ─────────────────────────────────────────────────────────
        vix = _load_vix(today)

        # ── Compute quantitative risk scores ──────────────────────────────────
        print(f"  [RiskControl] {ticker}: computing risk scores...")
        risk_scores = _compute_risk_scores(technical, fundamental, macro, arbiter, vix)

        # ── LLM synthesis ────────────────────────────────────────────────────
        print(f"  [RiskControl] {ticker}: calling LLM (composite_risk={risk_scores['composite_risk_score']:.3f})...")
        prompt  = _build_prompt(ticker, macro, technical, fundamental, arbiter, vix, risk_scores, date_str=today)
        raw     = _call_llm(prompt)
        llm_out = _extract_json(raw)

        result = {
            "ticker":          ticker,
            "date":            today,
            # LLM decision
            "final_decision":  llm_out.get("final_decision", "hold"),
            "position_size":   round(float(llm_out.get("position_size", 0.0)), 4),
            "risk_score":      round(float(llm_out.get("risk_score", risk_scores["composite_risk_score"])), 4),
            "risk_adjustments": llm_out.get("risk_adjustments", []),
            "summary":         llm_out.get("summary", ""),
            # Dimensional risk scores
            "volatility_risk":        risk_scores["volatility_risk"],
            "drawdown_risk":          risk_scores["drawdown_risk"],
            "tail_risk":              risk_scores["tail_risk"],
            "liquidity_risk":         risk_scores["liquidity_risk"],
            "market_risk":            risk_scores["market_risk"],
            "signal_confidence_risk": risk_scores["signal_confidence_risk"],
            "composite_risk_score":   risk_scores["composite_risk_score"],
            # Market context
            "vix_latest":    vix.get("vix_latest"),
            "vix_regime":    vix.get("vix_regime"),
            # Pass-through inputs
            "macro_adjusted_score":  macro.get("macro_adjusted_score"),
            "macro_regime":          macro.get("macro_regime"),
            "subsector":             macro.get("subsector"),
            "arbiter_score":         macro.get("arbiter_score_input"),
            "arbiter_verdict":       macro.get("arbiter_verdict_input"),
        }

        _save_json(result, cache_path)
        print(f"  [RiskControl] {ticker}: done — {result['final_decision'].upper()}, "
              f"size={result['position_size']:.3f}, risk={result['risk_score']:.3f}")
        return result

    def get_all_signals(
        self,
        date_str: Optional[str] = None,
        ticker_list: Optional[list] = None,
        force_refresh: bool = False,
    ) -> dict:
        """Run RiskControlAgent for all tickers."""
        today   = date_str or datetime.now().strftime("%Y-%m-%d")
        tickers = ticker_list or TICKER_LIST
        results = {}

        print(f"\n{'='*60}")
        print(f"RiskControlAgent — {today}")
        print(f"{'='*60}\n")

        for ticker in tickers:
            result = self.get_signal(ticker, date_str=today, force_refresh=force_refresh)
            if result:
                results[ticker] = result

        if results:
            print(f"\n{'─'*76}")
            print(f"{'Ticker':<8} {'Decision':<8} {'Size':>6} {'Risk':>6} "
                  f"{'MacroAdj':>10} {'VIX':>6} {'Regime':<14}")
            print(f"{'─'*76}")
            for t, r in sorted(results.items()):
                print(
                    f"{t:<8} {r['final_decision'].upper():<8} "
                    f"{r['position_size']:>6.3f} {r['risk_score']:>6.3f} "
                    f"{(r['macro_adjusted_score'] or 0):>+10.4f} "
                    f"{(r['vix_latest'] or 0):>6.1f} "
                    f"{r['macro_regime'] or '':14}"
                )
            print(f"{'─'*76}\n")

        return results


# ── CLI Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import re as _re

    _date_pattern = _re.compile(r"^\d{4}-\d{2}-\d{2}$")
    args = sys.argv[1:]

    date_arg  = next((a for a in args if _date_pattern.match(a)), None)
    force_arg = "--force" in args
    ticker_arg = [a.upper() for a in args if a.upper() in TICKER_LIST] or None

    agent = RiskControlAgent()
    agent.get_all_signals(date_str=date_arg, ticker_list=ticker_arg, force_refresh=force_arg)
