"""
Arbiter Agent

Synthesizes outputs from BullAgent and BearAgent into a single structured verdict
for each semiconductor stock.

Architecture (upgraded with multi-round cross-examination debate):
  1. BullAgent and BearAgent each independently produce their thesis.
  2. ArbiterAgent runs N rounds of structured cross-examination:
       Round k, Bear turn  : Bear directly rebuts bull's specific arguments
       Round k, Bull turn  : Bull directly rebuts bear's specific counterarguments
  3. ArbiterAgent evaluates the FULL debate transcript — not just initial theses —
     and judges which arguments survived cross-examination.

This mirrors a real investment committee: written briefs (initial theses) followed
by live cross-examination before the chair delivers a verdict.

Inputs:
  - bull_thesis : BullAgent.get_thesis() output dict
  - bear_thesis : BearAgent.get_thesis() output dict

Debate cache:
  arbiter_cache/{TICKER}/debate_{DATE}.json   ← cross-examination transcript
  arbiter_cache/{TICKER}/arbiter_{DATE}.json  ← final verdict

Output schema (consumed by MacroAgent, RiskControlAgent, and DecisionAgent):
  arbiter_score          float [-1, +1]   net conviction (positive=bullish, negative=bearish)
  verdict                str              "strong_bull" | "mild_bull" | "neutral" | "mild_bear" | "strong_bear"
  confidence             float [0, 1]     arbiter's confidence in the verdict
  bull_wins_on           list[str]        argument categories where bull case survived cross-examination
  bear_wins_on           list[str]        argument categories where bear case survived cross-examination
  contested_points       list[str]        areas with valid arguments on both sides
  argument_survivability dict             {category: "bull_prevailed"|"bear_prevailed"|"contested"|"both_conceded"}
  resolution_summary     str              2-3 sentence explanation of how the debate was resolved
  key_risk               str              single most important risk the Decision Agent must consider
  bull_score_input       float            raw bullish_score passed in
  bear_score_input       float            raw bearish_score passed in
  bull_final_score       float            bull's revised score after cross-examination
  bear_final_score       float            bear's revised score after cross-examination
  agreement_level        str              "aligned" | "mixed" | "divergent"
  flags                  list[str]        e.g. ["data_gap", "high_disagreement", "low_confidence"]
  debate_transcript      list[dict]       full cross-examination log [{round, speaker, content}]
  rounds_conducted       int              number of completed debate rounds

Downstream contract:
  MacroAgent    receives arbiter_score + verdict, applies macro overlay → macro_adjusted_score
  RiskControl   receives arbiter_score + confidence, applies position sizing → risk_signal
  DecisionAgent receives all three signals for final trade decision
"""

import os
import json
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

import re
import anthropic

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_DIR  = Path("d:/Investment_AI_agent")
CACHE_DIR = BASE_DIR / "arbiter_cache"
CACHE_DIR.mkdir(exist_ok=True)

MODEL = "claude-sonnet-4-6"

# Number of cross-examination rounds before the arbiter delivers its verdict.
# Each round = one Bear rebuttal turn + one Bull rebuttal turn.
# 0 = legacy mode (arbiter reads static theses only, no live debate).
# 1 = one round of cross-examination (recommended default).
# 2 = two rounds (higher signal quality, ~2x LLM cost).
DEBATE_ROUNDS_DEFAULT = 1

TICKER_LIST = [
    "TSM", "INTC", "NVDA", "ASML", "AVGO", "TXN", "QCOM", "MU",
    "AMAT", "AMD", "ADI", "LRCX", "KLAC", "MCHP", "SWKS", "MRVL",
    "TER", "ON", "MPWR", "ENTG", "MKSI", "SLAB", "CRUS", "SMTC",
]

client = anthropic.Anthropic()


# ── Cache Utilities ────────────────────────────────────────────────────────────

def _cache_dir(ticker: str) -> Path:
    d = CACHE_DIR / ticker
    d.mkdir(exist_ok=True)
    return d


def _load_json(path: Path) -> Optional[dict | list]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _save_json(data: dict | list, path: Path):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── System Prompts ─────────────────────────────────────────────────────────────

SYSTEM_ARBITER = (
    "You are an impartial senior investment committee chair specializing in semiconductor stocks. "
    "You are given a structured bull thesis, a structured bear thesis, and the full cross-examination "
    "transcript showing how each side responded to the other's arguments. "
    "Your job is to evaluate not just initial positions but which arguments SURVIVED cross-examination. "
    "An argument that was directly rebutted and could not be defended should carry less weight. "
    "An argument the opposing side conceded should carry more weight. "
    "You are NOT biased toward either side — you follow the evidence and the quality of debate. "
    "Always respond with a single valid JSON object exactly matching the specified schema. "
    "Do not include any text outside the JSON object. "
    "Do NOT use double-quote characters inside string values; use single quotes instead if quoting is needed."
)

SYSTEM_BULL_DEBATER = (
    "You are a senior buy-side analyst who has presented a bullish thesis for a semiconductor stock. "
    "The bear analyst has directly challenged specific arguments in your thesis. "
    "Your job is to rebut their specific counterarguments with concrete evidence. "
    "Do NOT simply restate your original thesis — engage directly with what the bear said. "
    "Concede any argument where the bear evidence is genuinely compelling; intellectual honesty "
    "strengthens the arguments you choose to defend. "
    "Always respond with a single valid JSON object exactly matching the specified schema. "
    "Do not include any text outside the JSON object. "
    "Do NOT use double-quote characters inside string values; use single quotes instead if quoting is needed."
)

SYSTEM_BEAR_DEBATER = (
    "You are a senior sell-side analyst who has presented a bearish thesis for a semiconductor stock. "
    "The bull analyst has made their case for investing. Your job is to directly challenge their "
    "specific claims with concrete evidence and expose flawed assumptions. "
    "Do NOT simply restate your original thesis — target the bull's weakest points. "
    "Concede any argument where the bull evidence is genuinely overwhelming; intellectual honesty "
    "strengthens the arguments you choose to defend. "
    "Always respond with a single valid JSON object exactly matching the specified schema. "
    "Do not include any text outside the JSON object. "
    "Do NOT use double-quote characters inside string values; use single quotes instead if quoting is needed."
)


# ── LLM Utilities ──────────────────────────────────────────────────────────────

try:
    from json_repair import repair_json as _json_repair
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False


def _repair_json_strings(s: str) -> str:
    """
    Walk the JSON text character-by-character, tracking string boundaries.

    Inside a string value the following are fixed:
      - Actual newline / carriage-return / tab  → escaped \\n / \\r / \\t
      - Unescaped double-quote that is NOT the closing quote → escaped \\"

    Closing-quote detection (the tricky part):
      A `"` is the *closing* quote when the next non-whitespace char is in  : , } ]
      OR when a newline appears before the next `"` (= next JSON key on a new line).
      Same-line embedded quotes like  "value "quoted" more"  are escaped instead.
    """
    out = []
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if not in_string:
            out.append(c)
            if c == '"':
                in_string = True
        else:
            if c == '\\':
                # Already-escaped sequence — pass both chars through unchanged
                out.append(c)
                i += 1
                if i < len(s):
                    out.append(s[i])
            elif c == '"':
                # Peek at what follows (skipping whitespace) to decide if this
                # is the closing quote or an embedded unescaped quote.
                j = i + 1
                has_newline = False
                while j < len(s) and s[j] in ' \t\r\n':
                    if s[j] == '\n':
                        has_newline = True
                    j += 1
                next_ch = s[j] if j < len(s) else ''
                # Treat as closing quote if:
                #   (a) followed by a structural JSON token  : , } ]
                #   (b) end of input
                #   (c) a newline occurred before the next char — next char is
                #       the opening quote of the following JSON key
                if j >= len(s) or next_ch in ':,}]' or has_newline:
                    out.append(c)
                    in_string = False
                else:
                    # Same-line embedded unescaped quote → escape it
                    out.append('\\')
                    out.append('"')
            elif c == '\n':
                out.append('\\n')
            elif c == '\r':
                out.append('\\r')
            elif c == '\t':
                out.append('\\t')
            else:
                out.append(c)
        i += 1
    return ''.join(out)


def _extract_json(text: str) -> dict:
    # Strip markdown code fences
    if "```" in text:
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines)

    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found in LLM response:\n{text[:300]}")
    raw = text[start:end]

    # Attempt 1: direct parse (fast path — well-formed output)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt 2: json-repair library (handles all common LLM JSON failure modes)
    if _HAS_JSON_REPAIR:
        try:
            result = json.loads(_json_repair(raw))
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    # Attempt 3: trailing commas + // line comments
    cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
    cleaned = re.sub(r"//[^\n]*", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 4: character-level scan — fixes unescaped newlines and
    # same-line unescaped double-quotes inside string values
    repaired = _repair_json_strings(cleaned)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Attempt 5: insert missing commas between adjacent JSON members
    fixed = re.sub(
        r'(["\d\w\]\}])([ \t]*\n[ \t]*)(["\[{])',
        lambda m: m.group(1) + ',' + m.group(2) + m.group(3),
        repaired,
    )
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        # Dump raw and repaired text to a file so the root cause is inspectable
        debug_path = BASE_DIR / "_json_parse_failure.txt"
        try:
            debug_path.write_text(
                f"=== RAW LLM OUTPUT ===\n{raw}\n\n=== AFTER REPAIR ===\n{fixed}",
                encoding="utf-8",
            )
            debug_hint = f"\nFull text written to: {debug_path}"
        except Exception:
            debug_hint = ""

        pos = e.pos if hasattr(e, 'pos') else 0
        snippet = fixed[max(0, pos - 200): pos + 200]
        raise ValueError(
            f"Failed to parse JSON after all repair attempts: {e}\n"
            f"Context around error (char {pos}):\n...{snippet}..."
            f"{debug_hint}"
        )


def _call_llm(
    prompt: str,
    system: str = SYSTEM_ARBITER,
    max_tokens: int = 8000,
    retries: int = 5,
) -> str:
    for attempt in range(retries + 1):
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=max_tokens,
                thinking={"type": "adaptive"},
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
                print(f"    [RETRY {attempt+1}] {e}")
                wait = 2 ** (attempt + 1); time.sleep(wait)
            else:
                raise


# ── Debate Turn Schemas ────────────────────────────────────────────────────────
#
# Each debate turn (bear or bull) returns a structured JSON with this shape:
#
#   rebuttals: list of targeted responses to specific opponent arguments
#     - target_category   : which signal domain is being contested
#     - target_claim      : the specific claim being rebutted (quote closely)
#     - counter_evidence  : concrete data or reasoning used to rebut
#     - outcome           : "refuted" | "weakened" | "inconclusive"
#     - concede           : true if you're conceding this point to the opponent
#
#   concessions         : list of opponent arguments you fully concede
#   reinforced_arguments: your strongest arguments that survived their attack
#   revised_score       : your updated conviction score after seeing their response
#   debate_summary      : 2-3 sentence summary of your position after this round


# ── Debate Prompt Builders ─────────────────────────────────────────────────────

def _format_debate_history(transcript: list[dict]) -> str:
    """Format prior debate rounds into readable text for prompt injection."""
    if not transcript:
        return "(No prior rounds — this is Round 1.)"
    lines = []
    for entry in transcript:
        speaker = entry["speaker"].upper()
        rnd     = entry["round"]
        content = entry["content"]
        lines.append(f"── Round {rnd} · {speaker} ──")
        if isinstance(content, dict):
            # Format key fields for readability
            rebuttals = content.get("rebuttals", [])
            for r in rebuttals:
                concede_marker = " [CONCEDED]" if r.get("concede") else ""
                lines.append(
                    f"  [{r.get('target_category','?')}]{concede_marker} "
                    f"Target: \"{r.get('target_claim','')}\" → "
                    f"{r.get('counter_evidence','')} "
                    f"(outcome: {r.get('outcome','')})"
                )
            if content.get("concessions"):
                lines.append(f"  Concessions: {content['concessions']}")
            if content.get("reinforced_arguments"):
                lines.append(f"  Reinforced: {content['reinforced_arguments']}")
            rev_key = "revised_bullish_score" if speaker == "BULL" else "revised_bearish_score"
            if content.get(rev_key) is not None:
                lines.append(f"  Revised score: {content[rev_key]}")
        else:
            lines.append(f"  {str(content)[:400]}")
        lines.append("")
    return "\n".join(lines)


def _build_bear_debate_prompt(
    ticker: str,
    reference_date: str,
    bull: dict,
    bear: dict,
    debate_history: list[dict],
    round_num: int,
) -> str:
    """Bear analyst rebuts bull's specific arguments."""
    bull_args = json.dumps(bull.get("top_arguments", []), indent=2)
    bear_args = json.dumps(bear.get("top_arguments", []), indent=2)
    history_text = _format_debate_history(debate_history)

    return f"""You are the bear analyst in a structured cross-examination debate for {ticker} as of {reference_date}.

This is Round {round_num} — your turn to respond.

══════════════════════════════════════════════════════
BULL'S ORIGINAL THESIS  (bullish_score = {bull.get('bullish_score', 'N/A')}, conviction = {bull.get('conviction_level', 'N/A')})
══════════════════════════════════════════════════════
{bull.get('debate_summary', 'N/A')}

Bull's Top Arguments:
{bull_args}

Bull's Catalysts: {json.dumps(bull.get('bull_catalysts', []))}
Bull's Acknowledged Risks: {json.dumps(bull.get('acknowledged_risks', []))}

══════════════════════════════════════════════════════
YOUR ORIGINAL BEAR THESIS  (bearish_score = {bear.get('bearish_score', 'N/A')}, conviction = {bear.get('conviction_level', 'N/A')})
══════════════════════════════════════════════════════
{bear.get('debate_summary', 'N/A')}

Your Top Arguments:
{bear_args}

══════════════════════════════════════════════════════
PRIOR DEBATE ROUNDS
══════════════════════════════════════════════════════
{history_text}

══════════════════════════════════════════════════════
TASK
══════════════════════════════════════════════════════
Challenge the bull's SPECIFIC arguments. Target their weakest claims first.
For each rebuttal, cite concrete data or logic. Where the bull data is genuinely
strong, set "concede": true — a strategic concession strengthens your credibility
on the arguments you DO defend.

Return a JSON object with EXACTLY this schema:
{{
  "rebuttals": [
    {{
      "target_category": <"technical" | "fundamental" | "sec_filing" | "sentiment" | "cross_signal">,
      "target_claim": <exact claim from bull's top_arguments you are challenging>,
      "counter_evidence": <specific data, metric, or reasoning that challenges this claim>,
      "outcome": <"refuted" | "weakened" | "inconclusive">,
      "concede": <true if you concede this point to the bull, false otherwise>
    }}
  ],
  "concessions": [<bull arguments you fully concede — quote the category or claim>],
  "reinforced_arguments": [<your strongest bear arguments that survive bull's framing>],
  "revised_bearish_score": <float [-1.0 to +1.0]; update your score after seeing bull's case>,
  "debate_summary": <2-3 sentence summary of your bear position after this round of cross-examination>
}}
"""


def _build_bull_rebuttal_prompt(
    ticker: str,
    reference_date: str,
    bull: dict,
    bear: dict,
    debate_history: list[dict],
    round_num: int,
) -> str:
    """Bull analyst rebuts bear's specific counterarguments."""
    bull_args = json.dumps(bull.get("top_arguments", []), indent=2)
    history_text = _format_debate_history(debate_history)

    return f"""You are the bull analyst in a structured cross-examination debate for {ticker} as of {reference_date}.

This is Round {round_num} — the bear has just responded to your thesis. Now rebut their specific challenges.

══════════════════════════════════════════════════════
YOUR ORIGINAL BULL THESIS  (bullish_score = {bull.get('bullish_score', 'N/A')}, conviction = {bull.get('conviction_level', 'N/A')})
══════════════════════════════════════════════════════
{bull.get('debate_summary', 'N/A')}

Your Top Arguments:
{bull_args}

══════════════════════════════════════════════════════
CROSS-EXAMINATION TRANSCRIPT (including bear's latest response)
══════════════════════════════════════════════════════
{history_text}

══════════════════════════════════════════════════════
TASK
══════════════════════════════════════════════════════
Respond directly to the bear's counterarguments — do NOT simply repeat your original thesis.
Address each specific challenge with concrete evidence. Where the bear raised a genuinely
valid concern, set "concede": true. Conceding weak points makes your defended arguments
more credible to the arbiter.

Return a JSON object with EXACTLY this schema:
{{
  "rebuttals": [
    {{
      "target_category": <"technical" | "fundamental" | "sec_filing" | "sentiment" | "cross_signal">,
      "target_claim": <exact bear counterargument you are responding to>,
      "counter_evidence": <specific data, metric, or reasoning that defends your position>,
      "outcome": <"defended" | "partially_defended" | "conceded">,
      "concede": <true if you concede this point to the bear, false otherwise>
    }}
  ],
  "concessions": [<bear arguments you fully concede — quote the category or claim>],
  "reinforced_arguments": [<your strongest bull arguments that survive bear's cross-examination>],
  "revised_bullish_score": <float [-1.0 to +1.0]; update your score after this exchange>,
  "debate_summary": <2-3 sentence summary of your bull position after this round of cross-examination>
}}
"""


# ── Debate Orchestrator ────────────────────────────────────────────────────────

def run_debate(
    ticker: str,
    reference_date: str,
    bull_thesis: dict,
    bear_thesis: dict,
    n_rounds: int = DEBATE_ROUNDS_DEFAULT,
) -> list[dict]:
    """
    Run n_rounds of structured cross-examination between bull and bear analysts.

    Each round:
      1. Bear responds to bull's arguments (targets specific claims)
      2. Bull rebuts bear's specific counterarguments

    Returns the full transcript as a list of turn dicts:
      [{"round": 1, "speaker": "bear", "content": {...}},
       {"round": 1, "speaker": "bull", "content": {...}},
       ...]

    The transcript is passed to the arbiter so it can evaluate which arguments
    survived cross-examination versus which were conceded or refuted.
    """
    transcript: list[dict] = []

    for round_num in range(1, n_rounds + 1):
        # ── Bear's turn: rebut bull's arguments ─────────────────────────────
        print(f"    [Debate R{round_num}] Bear rebutting bull...", end=" ", flush=True)
        bear_prompt = _build_bear_debate_prompt(
            ticker, reference_date, bull_thesis, bear_thesis, transcript, round_num
        )
        bear_raw      = _call_llm(bear_prompt, system=SYSTEM_BEAR_DEBATER, max_tokens=8000)
        bear_response = _extract_json(bear_raw)
        transcript.append({"round": round_num, "speaker": "bear", "content": bear_response})
        bear_score = bear_response.get("revised_bearish_score", "?")
        concessions_b = len(bear_response.get("concessions", []))
        print(f"revised_score={bear_score}  concessions={concessions_b}")

        # ── Bull's turn: rebut bear's counterarguments ──────────────────────
        print(f"    [Debate R{round_num}] Bull rebutting bear...", end=" ", flush=True)
        bull_prompt = _build_bull_rebuttal_prompt(
            ticker, reference_date, bull_thesis, bear_thesis, transcript, round_num
        )
        bull_raw      = _call_llm(bull_prompt, system=SYSTEM_BULL_DEBATER, max_tokens=8000)
        bull_response = _extract_json(bull_raw)
        transcript.append({"round": round_num, "speaker": "bull", "content": bull_response})
        bull_score = bull_response.get("revised_bullish_score", "?")
        concessions_u = len(bull_response.get("concessions", []))
        print(f"revised_score={bull_score}  concessions={concessions_u}")

    return transcript


def _extract_final_scores(transcript: list[dict]) -> tuple[Optional[float], Optional[float]]:
    """Extract the final revised scores from the last round of the transcript."""
    bull_final = None
    bear_final = None
    for entry in reversed(transcript):
        content = entry.get("content", {})
        if entry["speaker"] == "bull" and bull_final is None:
            bull_final = content.get("revised_bullish_score")
        if entry["speaker"] == "bear" and bear_final is None:
            bear_final = content.get("revised_bearish_score")
        if bull_final is not None and bear_final is not None:
            break
    return bull_final, bear_final


# ── Arbiter Prompt Builder ─────────────────────────────────────────────────────

def _build_prompt(
    ticker: str,
    reference_date: str,
    bull: dict,
    bear: dict,
    debate_transcript: Optional[list[dict]] = None,
) -> str:
    """
    Build the arbiter adjudication prompt.

    When debate_transcript is provided, the arbiter evaluates argument survivability
    across the cross-examination rounds in addition to the initial theses.
    """
    bull_args = json.dumps(bull.get("top_arguments", []), indent=2)
    bear_args = json.dumps(bear.get("top_arguments", []), indent=2)

    has_debate = bool(debate_transcript)
    bull_final, bear_final = (None, None)
    debate_section = ""

    if has_debate:
        bull_final, bear_final = _extract_final_scores(debate_transcript)
        history_text = _format_debate_history(debate_transcript)
        debate_section = f"""
══════════════════════════════════════════════════════
CROSS-EXAMINATION TRANSCRIPT  ({len(debate_transcript)} turns across {len(debate_transcript)//2} rounds)
══════════════════════════════════════════════════════
{history_text}

Post-debate revised scores:
  Bull revised score : {bull_final if bull_final is not None else "N/A"}
  Bear revised score : {bear_final if bear_final is not None else "N/A"}
"""

    bull_final_str = f"{bull_final:.3f}" if bull_final is not None else "N/A (no debate)"
    bear_final_str = f"{bear_final:.3f}" if bear_final is not None else "N/A (no debate)"

    survivability_instruction = ""
    if has_debate:
        survivability_instruction = """
- argument_survivability: for each category ("technical", "fundamental", "sec_filing",
  "sentiment", "cross_signal"), assess which side's arguments in that category survived
  cross-examination best. Use: "bull_prevailed" | "bear_prevailed" | "contested" | "both_conceded"
- Weight arguments that survived cross-examination MORE heavily than arguments that
  were conceded or left unanswered by the defending analyst.
- Where both sides conceded the same point, note it in contested_points."""
    else:
        survivability_instruction = """
- argument_survivability: set to {} since no cross-examination was conducted."""

    prompt = f"""You are adjudicating a bull-bear investment debate for {ticker} as of {reference_date}.

══════════════════════════════════════════════════════
BULL THESIS  (initial bullish_score = {bull.get('bullish_score', 'N/A')},  conviction = {bull.get('conviction_level', 'N/A')})
══════════════════════════════════════════════════════
{bull.get('debate_summary', 'N/A')}

Top Arguments:
{bull_args}

Catalysts: {json.dumps(bull.get('bull_catalysts', []))}
Acknowledged Risks: {json.dumps(bull.get('acknowledged_risks', []))}

══════════════════════════════════════════════════════
BEAR THESIS  (initial bearish_score = {bear.get('bearish_score', 'N/A')},  conviction = {bear.get('conviction_level', 'N/A')})
══════════════════════════════════════════════════════
{bear.get('debate_summary', 'N/A')}

Top Arguments:
{bear_args}

Catalysts: {json.dumps(bear.get('bear_catalysts', []))}
Acknowledged Strengths: {json.dumps(bear.get('acknowledged_strengths', []))}
{debate_section}
══════════════════════════════════════════════════════
TASK
══════════════════════════════════════════════════════
Evaluate both theses and the cross-examination transcript to deliver an impartial verdict.

Key considerations:
- Where the bear's acknowledged_strengths match the bull's top arguments → confirms bull case
- Where the bull's acknowledged_risks match the bear's top arguments → confirms bear case
- Argument strength scores (0-1) indicate how well-supported each argument is
- Higher conviction_level = more certainty in the underlying analysis
{survivability_instruction}

Return a JSON object with EXACTLY these fields:
{{
  "arbiter_score": <float -1.0 to +1.0; positive=net bullish, negative=net bearish, 0=truly neutral>,
  "verdict": <"strong_bull" | "mild_bull" | "neutral" | "mild_bear" | "strong_bear">,
  "confidence": <float 0.0 to 1.0; how confident you are given argument quality and cross-examination results>,
  "bull_wins_on": [<argument categories where bull arguments survived cross-examination most clearly>],
  "bear_wins_on": [<argument categories where bear arguments survived cross-examination most clearly>],
  "contested_points": [<specific claims where both sides landed valid blows — reduce confidence here>],
  "argument_survivability": {{
    <category>: <"bull_prevailed" | "bear_prevailed" | "contested" | "both_conceded">
  }},
  "resolution_summary": <2-3 sentence explanation of how you weighed cross-examination outcomes>,
  "key_risk": <single most important risk the downstream Decision Agent must factor in>,
  "agreement_level": <"aligned" if both scores point same direction | "mixed" | "divergent" if strong contradiction>,
  "flags": [<any of: "data_gap", "high_disagreement", "low_confidence", "forced_bull", "forced_bear", "no_debate">]
}}

Scoring guide:
  arbiter_score ∈ [0.6, 1.0]   → strong_bull
  arbiter_score ∈ [0.2, 0.6)   → mild_bull
  arbiter_score ∈ (-0.2, 0.2)  → neutral
  arbiter_score ∈ (-0.6, -0.2] → mild_bear
  arbiter_score ∈ [-1.0, -0.6] → strong_bear

Bull initial score:   {bull.get('bullish_score', 'N/A')}
Bear initial score:   {bear.get('bearish_score', 'N/A')}
Bull revised score:   {bull_final_str}
Bear revised score:   {bear_final_str}
"""
    return prompt


# ── Score Utilities ────────────────────────────────────────────────────────────

def _preliminary_score(bull: dict, bear: dict) -> float:
    """Simple numeric baseline before LLM adjudication."""
    b_score = bull.get("bullish_score") or 0.0
    r_score = bear.get("bearish_score") or 0.0
    return (b_score + r_score) / 2.0


def _score_divergence(bull: dict, bear: dict) -> float:
    """Absolute difference between bull and bear scores (both on same scale)."""
    b = bull.get("bullish_score") or 0.0
    r = abs(bear.get("bearish_score") or 0.0)
    return abs(b - r)


# ── Core Analysis ──────────────────────────────────────────────────────────────

def analyze_arbiter(
    ticker: str,
    reference_date: str,
    bull: dict,
    bear: dict,
    debate_transcript: Optional[list[dict]] = None,
) -> dict:
    """
    Run the LLM arbiter analysis for a single ticker.

    When debate_transcript is provided (from run_debate()), the arbiter evaluates
    which arguments survived cross-examination. Otherwise, falls back to static
    thesis comparison (legacy mode).

    Returns the structured arbiter verdict dict.
    """
    prompt = _build_prompt(ticker, reference_date, bull, bear, debate_transcript)

    print(f"  [ArbiterAgent] Calling LLM for {ticker} ...", end=" ", flush=True)
    raw    = _call_llm(prompt, system=SYSTEM_ARBITER)
    result = _extract_json(raw)
    print(
        f"score={result.get('arbiter_score', '?'):.3f}  "
        f"verdict={result.get('verdict', '?')}  "
        f"confidence={result.get('confidence', '?'):.2f}"
    )

    # Stamp metadata
    result["ticker"]           = ticker
    result["reference_date"]   = reference_date
    result["analysis_date"]    = datetime.now().strftime("%Y-%m-%d")
    result["bull_score_input"] = bull.get("bullish_score")
    result["bear_score_input"] = bear.get("bearish_score")
    result["preliminary_score"] = _preliminary_score(bull, bear)
    result["score_divergence"]  = _score_divergence(bull, bear)

    # Debate metadata
    bull_final, bear_final = _extract_final_scores(debate_transcript or [])
    result["bull_final_score"]   = bull_final
    result["bear_final_score"]   = bear_final
    result["debate_transcript"]  = debate_transcript or []
    result["rounds_conducted"]   = len(debate_transcript) // 2 if debate_transcript else 0

    # Auto-flag when no debate was run
    if not debate_transcript and "no_debate" not in result.get("flags", []):
        result.setdefault("flags", []).append("no_debate")

    return result


# ── ArbiterAgent Class ─────────────────────────────────────────────────────────

class ArbiterAgent:
    """
    Synthesizes BullAgent and BearAgent theses into a single impartial verdict
    via structured multi-round cross-examination.

    Workflow:
      1. (Optional) run_debate() — Bull and Bear cross-examine each other's arguments.
      2. analyze_arbiter() — LLM evaluates full transcript and delivers verdict.
      3. Cache both the debate transcript and final verdict separately.

    The arbiter_score output is the primary signal consumed by downstream agents:
      - MacroAgent    : applies macro overlay on top of arbiter_score
      - RiskControl   : uses confidence + score for position sizing
      - DecisionAgent : integrates all signals for final trade decision

    Usage:
        arbiter = ArbiterAgent()
        verdict = arbiter.get_verdict(
            ticker="NVDA",
            reference_date="2024-06-30",
            bull_thesis=bull_result,
            bear_thesis=bear_result,
            debate_rounds=1,   # 0 = legacy static mode
        )
    """

    def get_verdict(
        self,
        ticker: str,
        reference_date: str,
        bull_thesis: dict,
        bear_thesis: dict,
        force_refresh: bool = False,
        debate_rounds: int = DEBATE_ROUNDS_DEFAULT,
    ) -> dict:
        """
        Return the arbiter verdict dict for a single ticker.
        Cached per (ticker, reference_date).

        Args:
            ticker         : Stock ticker (e.g. "NVDA")
            reference_date : ISO date string (e.g. "2024-06-30")
            bull_thesis    : Output of BullAgent.get_thesis()
            bear_thesis    : Output of BearAgent.get_thesis()
            force_refresh  : Re-run LLM even if cache exists
            debate_rounds  : Cross-examination rounds (0 = legacy mode)
        """
        verdict_cache = _cache_dir(ticker) / f"arbiter_{reference_date}.json"

        if not force_refresh:
            cached = _load_json(verdict_cache)
            if cached:
                return cached

        # ── Step 1: Run cross-examination debate ────────────────────────────
        debate_transcript: list[dict] = []
        if debate_rounds > 0:
            debate_cache = _cache_dir(ticker) / f"debate_{reference_date}.json"
            if not force_refresh and debate_cache.exists():
                debate_transcript = _load_json(debate_cache)
                print(
                    f"  [ArbiterAgent] Loaded debate transcript from cache "
                    f"({len(debate_transcript)} turns) for {ticker}"
                )
            else:
                print(
                    f"  [ArbiterAgent] Running {debate_rounds}-round cross-examination "
                    f"for {ticker}..."
                )
                debate_transcript = run_debate(
                    ticker=ticker,
                    reference_date=reference_date,
                    bull_thesis=bull_thesis,
                    bear_thesis=bear_thesis,
                    n_rounds=debate_rounds,
                )
                _save_json(debate_transcript, debate_cache)

        # ── Step 2: Arbiter judges the full exchange ────────────────────────
        result = analyze_arbiter(
            ticker=ticker,
            reference_date=reference_date,
            bull=bull_thesis,
            bear=bear_thesis,
            debate_transcript=debate_transcript,
        )
        _save_json(result, verdict_cache)
        return result

    def get_all_verdicts(
        self,
        reference_date: str,
        bull_theses: dict[str, dict],
        bear_theses: dict[str, dict],
        ticker_list: Optional[list[str]] = None,
        force_refresh: bool = False,
        debate_rounds: int = DEBATE_ROUNDS_DEFAULT,
    ) -> dict[str, dict]:
        """
        Run arbiter analysis for all tickers.

        Args:
            reference_date : ISO date string
            bull_theses    : {ticker: thesis} from BullAgent.get_all_theses()
            bear_theses    : {ticker: thesis} from BearAgent.get_all_theses()
            debate_rounds  : Cross-examination rounds per ticker (0 = legacy mode)

        Returns:
            {ticker: arbiter_verdict}
        """
        tickers = ticker_list or TICKER_LIST
        results = {}
        for ticker in tickers:
            bull = bull_theses.get(ticker)
            bear = bear_theses.get(ticker)

            if bull is None or bear is None:
                print(f"[WARNING] {ticker}: missing bull or bear thesis, skipping")
                results[ticker] = {"ticker": ticker, "error": "missing_input_thesis"}
                continue
            if "error" in bull or "error" in bear:
                print(f"[WARNING] {ticker}: upstream thesis has error, skipping")
                results[ticker] = {"ticker": ticker, "error": "upstream_error"}
                continue

            try:
                results[ticker] = self.get_verdict(
                    ticker=ticker,
                    reference_date=reference_date,
                    bull_thesis=bull,
                    bear_thesis=bear,
                    force_refresh=force_refresh,
                    debate_rounds=debate_rounds,
                )
            except Exception as e:
                print(f"[WARNING] {ticker}: {e}")
                results[ticker] = {"ticker": ticker, "error": str(e)}

        return results

    def get_signal(self, ticker: str, reference_date: str) -> Optional[dict]:
        """
        Cache-only retrieval for a single ticker's arbiter signal.
        Returns None if not cached. Used by downstream agents.
        """
        cache_path = _cache_dir(ticker) / f"arbiter_{reference_date}.json"
        return _load_json(cache_path)

    def get_all_signals(
        self,
        reference_date: str,
        ticker_list: Optional[list[str]] = None,
    ) -> dict[str, dict]:
        """
        Cache-only retrieval for all tickers. Returns only cached verdicts.
        Used by downstream agents (MacroAgent, RiskControl, DecisionAgent).
        """
        tickers = ticker_list or TICKER_LIST
        return {
            t: sig
            for t in tickers
            if (sig := self.get_signal(t, reference_date)) is not None
        }

    def get_debate_transcript(
        self, ticker: str, reference_date: str
    ) -> Optional[list[dict]]:
        """
        Cache-only retrieval of the raw cross-examination transcript.
        Useful for debugging or post-analysis of argument quality.
        """
        cache_path = _cache_dir(ticker) / f"debate_{reference_date}.json"
        return _load_json(cache_path)

    def print_summary(self, verdicts: dict[str, dict], top_n: int = 10) -> None:
        """Print ranked table sorted by arbiter_score."""
        rows = []
        for ticker, v in verdicts.items():
            if "error" in v:
                continue
            rows.append({
                "ticker":     ticker,
                "score":      v.get("arbiter_score", 0),
                "verdict":    v.get("verdict", "?"),
                "confidence": v.get("confidence", 0),
                "agreement":  v.get("agreement_level", "?"),
                "rounds":     v.get("rounds_conducted", 0),
                "flags":      ", ".join(v.get("flags", [])) or "—",
            })
        rows.sort(key=lambda x: x["score"], reverse=True)

        print(f"\n{'='*95}")
        print(f"  Arbiter Verdicts — ranked by arbiter_score")
        print(f"{'='*95}")
        print(f"{'Ticker':<8}  {'Score':>6}  {'Verdict':<12}  {'Conf':>5}  {'Rounds':>6}  {'Agreement':<10}  Flags")
        print("-" * 95)
        for r in rows[:top_n]:
            print(
                f"{r['ticker']:<8}  {r['score']:>6.3f}  {r['verdict']:<12}  "
                f"{r['confidence']:>5.2f}  {r['rounds']:>6}  {r['agreement']:<10}  {r['flags']}"
            )

    def export_signals_csv(
        self,
        verdicts: dict[str, dict],
        output_path: Optional[Path] = None,
    ) -> Path:
        """Export arbiter signals to CSV for downstream use."""
        import csv
        path = output_path or BASE_DIR / "arbiter_signals.csv"
        fields = [
            "ticker", "reference_date", "arbiter_score", "verdict",
            "confidence", "agreement_level",
            "bull_score_input", "bear_score_input",
            "bull_final_score", "bear_final_score",
            "preliminary_score", "score_divergence",
            "rounds_conducted", "key_risk", "flags",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for v in verdicts.values():
                if "error" in v:
                    continue
                row = {k: v.get(k) for k in fields}
                row["flags"] = "|".join(v.get("flags", []))
                writer.writerow(row)
        print(f"  [ArbiterAgent] Exported {len(verdicts)} signals → {path}")
        return path


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from bull_agent import BullAgent
    from bear_agent import BearAgent
    from technical_agent   import TechnicalAgent
    from fundamental_agent import FundamentalAgent

    REF_DATE = "2024-06-28"
    ticker   = "NVDA"

    print("Loading upstream agents...")
    tech_agent = TechnicalAgent()
    fund_agent = FundamentalAgent()

    tech_sig = tech_agent.get_signal(ticker, reference_date=REF_DATE)
    fund_sig = fund_agent.get_signal(ticker, reference_date=REF_DATE)

    print(f"\nRunning Bull & Bear agents for {ticker}...")
    bull_agent = BullAgent()
    bear_agent = BearAgent()

    bull_thesis = bull_agent.get_thesis(
        ticker=ticker,
        reference_date=REF_DATE,
        technical_signal=tech_sig,
        fundamental_signal=fund_sig,
    )
    bear_thesis = bear_agent.get_thesis(
        ticker=ticker,
        reference_date=REF_DATE,
        technical_signal=tech_sig,
        fundamental_signal=fund_sig,
    )

    print(f"\nRunning Arbiter (with cross-examination) for {ticker}...")
    arbiter = ArbiterAgent()
    verdict = arbiter.get_verdict(
        ticker=ticker,
        reference_date=REF_DATE,
        bull_thesis=bull_thesis,
        bear_thesis=bear_thesis,
        debate_rounds=1,   # set to 0 for legacy static mode
    )

    print(f"\n{'='*65}")
    print(f"  Arbiter Verdict — {ticker}  ({REF_DATE})")
    print(f"{'='*65}")
    print(f"  arbiter_score    : {verdict['arbiter_score']:.3f}")
    print(f"  verdict          : {verdict['verdict']}")
    print(f"  confidence       : {verdict['confidence']:.2f}")
    print(f"  agreement        : {verdict['agreement_level']}")
    print(f"  rounds_conducted : {verdict['rounds_conducted']}")

    print(f"\n  Score evolution:")
    print(f"    Bull: {verdict['bull_score_input']:.3f} (initial) → "
          f"{verdict['bull_final_score'] or 'N/A'} (post-debate)")
    print(f"    Bear: {verdict['bear_score_input']:.3f} (initial) → "
          f"{verdict['bear_final_score'] or 'N/A'} (post-debate)")

    print(f"\n  Bull wins on   : {verdict.get('bull_wins_on', [])}")
    print(f"  Bear wins on   : {verdict.get('bear_wins_on', [])}")
    print(f"  Contested      : {verdict.get('contested_points', [])}")

    if verdict.get("argument_survivability"):
        print(f"\n  Argument survivability:")
        for cat, outcome in verdict["argument_survivability"].items():
            print(f"    {cat:<20} → {outcome}")

    print(f"\n  Resolution:")
    print(f"  {verdict.get('resolution_summary', '')}")
    print(f"\n  Key Risk:")
    print(f"  {verdict.get('key_risk', '')}")
    if verdict.get("flags"):
        print(f"\n  Flags: {verdict['flags']}")

    print(f"\n  [Downstream contract]")
    print(f"  arbiter_score={verdict['arbiter_score']:.3f} → ready for MacroAgent")
    print(f"  confidence={verdict['confidence']:.2f} → ready for RiskControlAgent")
