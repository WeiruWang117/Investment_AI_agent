"""
Daily Trading Pipeline

一条命令跑完整个 24 只半导体股票的分析流程：

  Stage 1a: TechnicalAgent   → technical_cache/
  Stage 1b: FundamentalAgent → fundamental_cache/
  Stage 1c: SentimentAgent   → sentiment_cache/
  Stage 1d: SECFilingAgent   → sec_analysis/
  Stage 2:  BullAgent + BearAgent  [并行]  → bull_cache/ bear_cache/
  Stage 3:  ArbiterAgent     → arbiter_cache/
  Stage 4:  MacroAgent       → macro_cache/
  Stage 5:  RiskControlAgent → risk_cache/
  Stage 6:  DecisionAgent    → decision_cache/
  Stage 7:  RecordAgent      → portfolio/

每个 Stage 都有缓存：已跑过的 ticker 直接读缓存，不重复调用 LLM。
每个 Stage 内所有 ticker 并行处理（MAX_WORKERS 个线程）。

Usage (Jupyter):
    from pipeline import run_pipeline
    results = run_pipeline("2021-03-01")           # 全部 24 只
    results = run_pipeline("2021-03-01", tickers=["NVDA", "AMD"])  # 指定股票
    results = run_pipeline("2021-03-01", force=True)               # 强制刷新

Usage (命令行):
    python pipeline.py 2021-03-01
    python pipeline.py 2021-03-01 NVDA AMD        # 单股测试
    python pipeline.py 2021-03-01 --force         # 强制重跑
    python pipeline.py 2021-03-01 --from=macro    # 只跑 macro 往后
"""

import json
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

BASE_DIR = Path("d:/Investment_AI_agent")

TICKER_LIST = [
    "TSM", "INTC", "NVDA", "ASML", "AVGO", "TXN", "QCOM", "MU",
    "AMAT", "AMD", "ADI", "LRCX", "KLAC", "MCHP", "SWKS", "MRVL",
    "TER", "ON", "MPWR", "ENTG", "MKSI", "SLAB", "CRUS", "SMTC",
]

STAGE_ORDER = ["technical", "fundamental", "sentiment", "sec", "bull", "bear",
               "arbiter", "macro", "risk", "decision"]

# 每个 Stage 内最多同时跑几个 ticker 的 LLM 调用
# Bull+Bear 会同时运行，实际并发 = MAX_WORKERS * 2，建议不超过 10
MAX_WORKERS = 1


def _load_json(path: Path) -> Optional[dict]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _section(title: str):
    print(f"\n── {title} {'─' * (54 - len(title))}")


def _parallel(fn, tickers: list, label: str, workers: int = MAX_WORKERS) -> dict:
    """
    用 ThreadPoolExecutor 并行调用 fn(ticker)。
    返回 {ticker: result}，失败的 ticker 打印错误后跳过。
    workers=1 退化为串行（适合每个 ticker 内部已有大量并发调用的 stage）。
    """
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fn, t): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                result = fut.result()
                if result is not None:
                    results[t] = result
            except Exception as e:
                print(f"  [{label}] {t}: {e}")
    return results


# ── Stage 1a: TechnicalAgent ───────────────────────────────────────────────────

def run_technical(date_str: str, tickers: list, force: bool = False) -> dict:
    from technical_agent import TechnicalAgent
    agent = TechnicalAgent()
    return _parallel(
        lambda t: agent.get_signal(t, reference_date=date_str, force_refresh=force),
        tickers, "Technical",
    )


# ── Stage 1b: FundamentalAgent ────────────────────────────────────────────────

def run_fundamental(date_str: str, tickers: list, force: bool = False) -> dict:
    from fundamental_agent import FundamentalAgent
    agent = FundamentalAgent()
    return _parallel(
        lambda t: agent.get_signal(t, reference_date=date_str, force_refresh=force),
        tickers, "Fundamental",
    )


# ── Stage 1c: SentimentAgent ──────────────────────────────────────────────────

def run_sentiment(date_str: str, tickers: list, force: bool = False) -> dict:
    from sentiment_agent import SentimentAgent
    agent = SentimentAgent()
    return _parallel(
        lambda t: agent.get_signal(t, reference_date=date_str),
        tickers, "Sentiment",
    )


# ── Stage 1d: SECFilingAgent ──────────────────────────────────────────────────

def run_sec(date_str: str, tickers: list, force: bool = False) -> dict:
    from sec_filing_agent import SECFilingAgent
    agent = SECFilingAgent()
    # workers=1：每个 ticker 内部已有 8 个 chunk 顺序调用，串行处理避免 API 过载
    # SEC 结果永久缓存，慢一点可以接受
    return _parallel(
        lambda t: agent.get_latest_signal(t, reference_date=date_str),
        tickers, "SEC", workers=1,
    )


# ── Stage 2a: BullAgent ───────────────────────────────────────────────────────

def run_bull(date_str: str, tickers: list,
             technical: dict, fundamental: dict,
             sentiment: dict, sec: dict,
             force: bool = False) -> dict:
    from bull_agent import BullAgent
    agent = BullAgent()
    return _parallel(
        lambda t: agent.get_thesis(
            ticker=t,
            reference_date=date_str,
            technical_signal=technical.get(t, {}),
            fundamental_signal=fundamental.get(t, {}),
            sentiment_signal=sentiment.get(t, {}),
            sec_signal=sec.get(t, {}),
            force_refresh=force,
        ),
        tickers, "Bull",
    )


# ── Stage 2b: BearAgent ───────────────────────────────────────────────────────

def run_bear(date_str: str, tickers: list,
             technical: dict, fundamental: dict,
             sentiment: dict, sec: dict,
             force: bool = False) -> dict:
    from bear_agent import BearAgent
    agent = BearAgent()
    return _parallel(
        lambda t: agent.get_thesis(
            ticker=t,
            reference_date=date_str,
            technical_signal=technical.get(t, {}),
            fundamental_signal=fundamental.get(t, {}),
            sentiment_signal=sentiment.get(t, {}),
            sec_signal=sec.get(t, {}),
            force_refresh=force,
        ),
        tickers, "Bear",
    )


# ── Stage 3: ArbiterAgent ─────────────────────────────────────────────────────

def run_arbiter(date_str: str, tickers: list,
                bull: dict, bear: dict,
                force: bool = False) -> dict:
    from arbiter_agent import ArbiterAgent
    agent = ArbiterAgent()

    def _verdict(t):
        if t not in bull or t not in bear:
            print(f"  [Arbiter] {t}: missing bull or bear thesis, skipping.")
            return None
        return agent.get_verdict(
            ticker=t,
            reference_date=date_str,
            bull_thesis=bull[t],
            bear_thesis=bear[t],
            force_refresh=force,
        )

    return _parallel(_verdict, tickers, "Arbiter")


# ── Stage 4: MacroAgent ───────────────────────────────────────────────────────

def run_macro(date_str: str, arbiter: dict, force: bool = False) -> dict:
    from macro_agent import MacroAgent
    agent = MacroAgent()
    return _parallel(
        lambda t: agent.get_signal(t, arbiter[t], date_str=date_str, force_refresh=force),
        list(arbiter.keys()), "Macro",
    )


# ── Stage 5: RiskControlAgent ─────────────────────────────────────────────────

def run_risk(date_str: str, tickers: list, force: bool = False) -> dict:
    from risk_control_agent import RiskControlAgent
    agent = RiskControlAgent()
    return _parallel(
        lambda t: agent.get_signal(t, date_str=date_str, force_refresh=force),
        tickers, "Risk",
    )


# ── Stage 6: DecisionAgent ────────────────────────────────────────────────────

def run_decision(date_str: str, force: bool = False) -> Optional[dict]:
    from decision_agent import DecisionAgent
    agent = DecisionAgent()
    return agent.decide(date_str=date_str, force_refresh=force)


# ── Stage 7: RecordAgent (execute + EOD snapshot) ─────────────────────────────

def run_record(decision_result: dict, prices: dict) -> dict:
    from record_agent import RecordAgent
    recorder = RecordAgent()
    recorder.execute_orders(decision_result)
    snapshot = recorder.end_of_day_snapshot(decision_result["date"], prices)
    return snapshot


# ── Full Pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    date_str: Optional[str] = None,
    tickers: Optional[list] = None,
    force: bool = False,
    start_from: str = "technical",
) -> dict:
    """
    完整日度流水线。

    Args:
        date_str   : 分析日期，格式 "YYYY-MM-DD"，必须显式传入（回测用）
        tickers    : 要分析的股票列表，默认全部 24 只
        force      : True = 强制重新调用 LLM，忽略缓存
        start_from : 从哪个 stage 开始（之前的 stage 直接读缓存不重跑）
                     可选: "technical", "fundamental", "sentiment", "sec",
                           "bull", "bear", "arbiter", "macro", "risk"

    Returns:
        decision dict
    """
    if date_str is None:
        raise ValueError("date_str 必须显式传入，例如 run_pipeline('2021-03-01')")

    targets   = [t.upper() for t in tickers] if tickers else TICKER_LIST
    start_idx = STAGE_ORDER.index(start_from) if start_from in STAGE_ORDER else 0

    def _should_run(stage: str) -> bool:
        return STAGE_ORDER.index(stage) >= start_idx

    print(f"\n{'='*60}")
    print(f"Daily Pipeline — {date_str}  ({len(targets)} tickers, workers={MAX_WORKERS})")
    if start_from != "technical":
        print(f"Starting from: {start_from}  (earlier stages read from cache)")
    print(f"{'='*60}")

    # ── Stage 1a: Technical ──────────────────────────────────────────────────
    _section("Stage 1a: TechnicalAgent")
    technical = run_technical(date_str, targets, force=force and _should_run("technical"))
    print(f"  → {len(technical)} tickers")

    # ── Stage 1b: Fundamental ────────────────────────────────────────────────
    _section("Stage 1b: FundamentalAgent")
    fundamental = run_fundamental(date_str, targets, force=force and _should_run("fundamental"))
    print(f"  → {len(fundamental)} tickers")

    # ── Stage 1c: Sentiment ──────────────────────────────────────────────────
    _section("Stage 1c: SentimentAgent")
    sentiment = run_sentiment(date_str, targets, force=force and _should_run("sentiment"))
    print(f"  → {len(sentiment)} tickers")

    # ── Stage 1d: SEC Filing ─────────────────────────────────────────────────
    _section("Stage 1d: SECFilingAgent")
    sec = run_sec(date_str, targets, force=force and _should_run("sec"))
    print(f"  → {len(sec)} tickers")

    # ── Stage 2: Bull + Bear  [同时并行] ─────────────────────────────────────
    _section("Stage 2: BullAgent + BearAgent  [parallel]")
    bull_force = force and _should_run("bull")
    bear_force = force and _should_run("bear")
    with ThreadPoolExecutor(max_workers=2) as stage_pool:
        bull_fut = stage_pool.submit(
            run_bull, date_str, targets, technical, fundamental, sentiment, sec, bull_force
        )
        bear_fut = stage_pool.submit(
            run_bear, date_str, targets, technical, fundamental, sentiment, sec, bear_force
        )
        bull = bull_fut.result()
        bear = bear_fut.result()
    print(f"  → Bull: {len(bull)} tickers  |  Bear: {len(bear)} tickers")

    # ── Stage 3: Arbiter ─────────────────────────────────────────────────────
    _section("Stage 3: ArbiterAgent")
    arbiter = run_arbiter(date_str, targets, bull, bear,
                          force=force and _should_run("arbiter"))
    print(f"  → {len(arbiter)} tickers")

    if not arbiter:
        print("\n[Pipeline] No arbiter results — cannot continue.")
        return {}

    # ── Stage 4: Macro ───────────────────────────────────────────────────────
    _section("Stage 4: MacroAgent")
    macro = run_macro(date_str, arbiter, force=force and _should_run("macro"))
    print(f"  → {len(macro)} tickers")

    # ── Stage 5: Risk Control ────────────────────────────────────────────────
    _section("Stage 5: RiskControlAgent")
    risk = run_risk(date_str, list(macro.keys()), force=force and _should_run("risk"))
    print(f"  → {len(risk)} tickers")

    # ── Stage 6: Decision ────────────────────────────────────────────────────
    _section("Stage 6: DecisionAgent")
    decision = run_decision(date_str, force=force and _should_run("decision"))
    if not decision:
        print("  [Pipeline] DecisionAgent failed.")
        return {}

    # ── Stage 7: Record ──────────────────────────────────────────────────────
    _section("Stage 7: RecordAgent")
    snapshot = run_record(decision, decision.get("prices_used", {}))

    # ── Final Summary ────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"Pipeline complete — {date_str}")
    print(f"{'═'*60}")
    print(f"  Market stance : {decision['market_stance'].upper()}")
    print(f"  Buy orders    : {decision['n_buy_orders']}  (${decision['total_buy_usd']:,.0f})")
    print(f"  Sell orders   : {decision['n_sell_orders']}  (${decision['total_sell_usd']:,.0f})")
    print(f"  Portfolio val : ${snapshot['total_value']:,.2f}")
    print(f"  Daily P&L     : ${snapshot['daily_pnl']:+,.2f}  ({snapshot['daily_return_pct']:+.2f}%)")
    print(f"  Cum. return   : {snapshot['cumulative_return_pct']:+.2f}%")
    if decision.get("reasoning"):
        print(f"\n  Reasoning: {decision['reasoning']}")
    print(f"{'═'*60}\n")

    return decision


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    args = sys.argv[1:]

    date_arg   = next((a for a in args if _date_pattern.match(a)), None)
    force_arg  = "--force" in args
    from_arg   = next((a.split("=")[1] for a in args if a.startswith("--from=")), "technical")
    ticker_arg = [a.upper() for a in args
                  if a.upper() in TICKER_LIST] or None

    if date_arg is None:
        print("Usage: python pipeline.py YYYY-MM-DD [TICKER ...] [--force] [--from=stage]")
        sys.exit(1)

    run_pipeline(date_str=date_arg, tickers=ticker_arg,
                 force=force_arg, start_from=from_arg)
