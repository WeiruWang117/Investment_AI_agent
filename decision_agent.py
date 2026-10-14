"""
DecisionAgent — Final Portfolio Decision

将所有 ticker 的 RiskControlAgent 输出聚合为一个 portfolio-level 决策，
确定每只股票当天应该买多少、卖多少、持有多少。

与之前 per-ticker agents 的根本区别：
  - 之前每个 agent 只看一只股票
  - DecisionAgent 同时看全部 24 只，做资金分配优化

架构：
  Layer 1 (Python): 从 risk_cache 加载所有信号，获取当日收盘价，
                    计算当前持仓市值，构建信号排名表
  Layer 2 (LLM):   审阅信号全貌和当前持仓，输出每只股票目标仓位比例
  Layer 3 (Python): 将目标比例转换为具体买卖订单（股数 + 美元金额）

Portfolio 约束（写死在 prompt 里，LLM 必须遵守）：
  - 单股最大仓位: 15% of total portfolio
  - 最小现金保留: 5%
  - 同一子行业最大仓位: 35%（ai_datacenter / equipment / analog / memory / foundry）
  - 初始资金: 100,000 USD

输入:
  risk_cache/{TICKER}/risk_{DATE}.json    — 全部 ticker 的风险信号
  portfolio/portfolio_state.json          — 当前持仓和现金

输出:
  decision_cache/decision_{DATE}.json     — LLM 目标配置 + 具体订单
  → 同时返回给 RecordAgent 执行
"""

import json
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

import anthropic

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_DIR        = Path("d:/Investment_AI_agent")
CACHE_DIR       = BASE_DIR / "decision_cache"
PORTFOLIO_DIR   = BASE_DIR / "portfolio"
CACHE_DIR.mkdir(exist_ok=True)
PORTFOLIO_DIR.mkdir(exist_ok=True)

MODEL = "claude-sonnet-4-6"

INITIAL_CAPITAL = 100_000.0   # USD

# Portfolio constraints
MAX_POSITION_PCT   = 0.15   # 单股最大 15%
MIN_CASH_PCT       = 0.05   # 最少保留 5% 现金
MAX_SUBSECTOR_PCT  = 0.35   # 同子行业最多 35%

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

TICKER_LIST = list(SUBSECTOR_MAP.keys())

client = anthropic.Anthropic()


# ── IO Utilities ───────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Optional[dict]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _save_json(data: dict, path: Path):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Price Fetcher ──────────────────────────────────────────────────────────────

# 交易执行价格模式：
#   "close"      — T日收盘价成交（假设 MOC 订单，适合日频模拟）
#   "next_open"  — T+1日开盘价成交（更接近真实，无前视偏差）
EXECUTION_PRICE_MODE = "close"   # 修改这里切换模式


def _get_prices(tickers: list, date_str: str, mode: str = EXECUTION_PRICE_MODE) -> dict:
    """
    获取执行价格。

    mode="close"     → date_str 当天收盘价（T日 close）
    mode="next_open" → date_str 次日开盘价（T+1 open，无前视偏差）

    优先读取本地 CSV，回退到 yfinance。
    返回 {ticker: price}。
    """
    import pandas as pd
    import yfinance as yf

    prices = {}

    # ── 尝试本地 CSV ──────────────────────────────────────────────────────────
    csv_path = BASE_DIR / "semiconductor_tickers_cleaned_2010_2024.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, errors="coerce")
            target = pd.Timestamp(date_str)

            if mode == "next_open":
                # T+1 开盘价：取 date_str 之后第一个交易日的 Open
                future = df[df.index > target]
                if not future.empty:
                    row      = future.iloc[0]
                    col_type = "Open"
                    price_date = future.index[0].strftime("%Y-%m-%d")
                else:
                    future, row, col_type, price_date = None, None, None, None
            else:
                # T 收盘价：取 ≤ date_str 的最近交易日 Close
                available = df[df.index <= target]
                row      = available.iloc[-1] if not available.empty else None
                col_type = "Close"
                price_date = date_str

            if row is not None:
                for t in tickers:
                    try:
                        p = float(row[(col_type, t)])
                        if p > 0:
                            prices[t] = round(p, 4)
                    except (KeyError, ValueError):
                        pass

            if prices:
                label = "T+1 open" if mode == "next_open" else "T close"
                print(f"  [Prices] {label} from CSV ({price_date}): {len(prices)} tickers")
        except Exception as e:
            print(f"  [Prices] CSV read error: {e}")

    # ── 对 CSV 中找不到的 ticker 用 yfinance 补充 ─────────────────────────────
    missing = [t for t in tickers if t not in prices]
    if missing:
        try:
            target = pd.Timestamp(date_str)
            end    = target + pd.Timedelta(days=5)
            start  = target - pd.Timedelta(days=7)
            raw    = yf.download(missing, start=start.strftime("%Y-%m-%d"),
                                 end=end.strftime("%Y-%m-%d"),
                                 progress=False, auto_adjust=True)
            raw.index = pd.to_datetime(raw.index)

            if mode == "next_open":
                col   = "Open"
                frame = raw[col] if col in raw.columns else raw
                frame = frame[frame.index > target]
                if not frame.empty:
                    row = frame.iloc[0]
                    print(f"  [Prices] T+1 open from yfinance ({frame.index[0].date()}): {len(missing)} tickers")
                else:
                    row = None
            else:
                col   = "Close"
                frame = raw[col] if col in raw.columns else raw
                frame = frame[frame.index <= target]
                row   = frame.iloc[-1] if not frame.empty else None
                if row is not None:
                    print(f"  [Prices] T close from yfinance: {len(missing)} tickers")

            if row is not None:
                for t in missing:
                    try:
                        p = float(row[t] if hasattr(row, "__getitem__") and t in row.index else row)
                        if p > 0:
                            prices[t] = round(p, 4)
                    except (KeyError, ValueError, TypeError):
                        pass
        except Exception as e:
            print(f"  [DecisionAgent] yfinance price fetch error: {e}")

    return prices


# ── Portfolio State ────────────────────────────────────────────────────────────

def load_portfolio() -> dict:
    """加载当前持仓，不存在则初始化。"""
    path = PORTFOLIO_DIR / "portfolio_state.json"
    state = _load_json(path)
    if state is None:
        state = {
            "cash":            INITIAL_CAPITAL,
            "positions":       {},   # {ticker: {shares, avg_cost}}
            "initial_capital": INITIAL_CAPITAL,
            "last_updated":    None,
        }
        _save_json(state, path)
    return state


def portfolio_value(state: dict, prices: dict) -> float:
    """计算当前持仓总市值 + 现金。"""
    equity = sum(
        pos["shares"] * prices.get(t, pos.get("avg_cost", 0))
        for t, pos in state["positions"].items()
    )
    return state["cash"] + equity


def positions_summary(state: dict, prices: dict, total_val: float) -> list[dict]:
    """返回每只股票当前持仓的详细信息。"""
    rows = []
    for t, pos in state["positions"].items():
        price    = prices.get(t, pos.get("avg_cost", 0))
        mkt_val  = pos["shares"] * price
        cost_val = pos["shares"] * pos["avg_cost"]
        rows.append({
            "ticker":     t,
            "shares":     pos["shares"],
            "avg_cost":   pos["avg_cost"],
            "price":      price,
            "mkt_value":  round(mkt_val, 2),
            "pct_of_portfolio": round(mkt_val / total_val, 4) if total_val else 0,
            "unrealized_pnl": round(mkt_val - cost_val, 2),
        })
    return sorted(rows, key=lambda x: -x["mkt_value"])


# ── Signal Summary Builder ─────────────────────────────────────────────────────

def _load_all_risk_signals(date_str: str) -> dict:
    """读取当日所有 ticker 的 risk_cache，返回 {ticker: signal}。"""
    signals = {}
    for t in TICKER_LIST:
        s = _load_json(BASE_DIR / "risk_cache" / t / f"risk_{date_str}.json")
        if s:
            signals[t] = s
    return signals


# ── System Prompt ──────────────────────────────────────────────────────────────

SYSTEM_DECISION = """You are the Final Portfolio Decision Agent for a $100,000 semiconductor stock portfolio.

You receive:
1. The current portfolio state (cash + existing positions)
2. Today's risk-adjusted signals for all semiconductor tickers in the pool
3. Current market context (VIX, macro regime)

Your job is to determine the TARGET ALLOCATION for the portfolio — what percentage of
total portfolio value should be in each stock after today's trades.

HARD CONSTRAINTS (you must not violate these):
  - Single stock max: 15% of total portfolio value
  - Minimum cash reserve: 5% of total portfolio value
  - Single subsector max: 35% (ai_datacenter / equipment / analog / memory / foundry)
  - No short selling (position sizes must be ≥ 0)
  - All allocations must sum to exactly 1.0 (including cash)

DECISION PRINCIPLES:
  - Only allocate to stocks with final_decision = "buy" or existing position to hold
  - For "sell" signals: set target allocation to 0 (exit position)
  - For "hold" signals with existing position: keep current allocation (minor rebalance ok)
  - For "hold" signals with no position: do not open new position
  - Allocate more to high conviction (large position_size, low risk_score) signals
  - Prefer stocks with macro_adjusted_score > +0.3 for new positions
  - In defensive macro regimes (contraction / trough): reduce total equity exposure,
    increase cash above minimum

POSITION CHANGE DISCIPLINE:
  - Avoid excessive turnover: do not churn existing positions without strong reason
  - Do not open more than 5 new positions in a single day
  - Do not fully exit more than 3 positions in a single day unless forced by risk

Output ONLY valid JSON. No markdown, no commentary outside the JSON."""


# ── LLM Call ──────────────────────────────────────────────────────────────────

def _call_llm(prompt: str, retries: int = 5) -> str:
    for attempt in range(retries + 1):
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=6000,
                thinking={"type": "adaptive"},
                system=SYSTEM_DECISION,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                message = stream.get_final_message()
                for block in reversed(message.content):
                    if block.type == "text":
                        return block.text
                raise ValueError("No text block in LLM response.")
        except Exception as e:
            if attempt < retries:
                print(f"  [DecisionAgent] RETRY {attempt+1}: {e}")
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
        raise ValueError(f"No JSON in LLM response: {raw[:300]}")
    return json.loads(text[start:end + 1])


# ── Prompt Builder ─────────────────────────────────────────────────────────────

def _build_prompt(
    date_str: str,
    portfolio: dict,
    prices: dict,
    total_val: float,
    positions: list[dict],
    signals: dict,
) -> str:

    # ── Portfolio snapshot ────────────────────────────────────────────────────
    cash_pct = portfolio["cash"] / total_val if total_val else 1.0
    pos_block = ""
    if positions:
        pos_block = "\n".join(
            f"  {p['ticker']:<6} {p['shares']:>6} shares @ ${p['avg_cost']:.2f} "
            f"→ ${p['mkt_value']:,.0f} ({p['pct_of_portfolio']*100:.1f}%)  "
            f"PnL: ${p['unrealized_pnl']:+,.0f}"
            for p in positions
        )
    else:
        pos_block = "  (no positions — all cash)"

    # ── Subsector aggregation ────────────────────────────────────────────────
    subsector_exposure: dict[str, float] = {}
    for p in positions:
        sub = SUBSECTOR_MAP.get(p["ticker"], "other")
        subsector_exposure[sub] = subsector_exposure.get(sub, 0) + p["pct_of_portfolio"]

    # ── Signal table (sorted by conviction) ──────────────────────────────────
    def _row(t: str, s: dict) -> str:
        decision = s.get("final_decision", "hold").upper()
        size     = s.get("position_size", 0)
        risk     = s.get("risk_score", 0.5)
        adj      = s.get("macro_adjusted_score", 0) or 0
        regime   = s.get("macro_regime", "?")
        sub      = s.get("subsector", SUBSECTOR_MAP.get(t, "?"))
        price    = prices.get(t, 0)
        cur_pct  = next((p["pct_of_portfolio"] for p in positions if p["ticker"] == t), 0)
        flag     = "●" if cur_pct > 0 else " "  # ● = currently held
        return (f"  {flag}{t:<6} {sub:<15} {decision:<5} size={size:.2f} "
                f"risk={risk:.2f} adj={adj:+.3f} regime={regime:<12} "
                f"price=${price:.2f}  held={cur_pct*100:.1f}%")

    buy_rows  = sorted([(t, s) for t, s in signals.items() if s.get("final_decision") == "buy"],
                       key=lambda x: -x[1].get("position_size", 0))
    sell_rows = [(t, s) for t, s in signals.items() if s.get("final_decision") == "sell"]
    hold_rows = [(t, s) for t, s in signals.items() if s.get("final_decision") == "hold"]

    signal_block  = "  BUY candidates:\n"
    signal_block += "\n".join(_row(t, s) for t, s in buy_rows) or "    (none)"
    signal_block += "\n\n  SELL signals:\n"
    signal_block += "\n".join(_row(t, s) for t, s in sell_rows) or "    (none)"
    signal_block += "\n\n  HOLD signals (● = currently held):\n"
    signal_block += "\n".join(_row(t, s) for t, s in hold_rows) or "    (none)"

    # ── VIX / macro context ──────────────────────────────────────────────────
    sample  = next(iter(signals.values()), {})
    vix     = sample.get("vix_latest", "N/A")
    vix_reg = sample.get("vix_regime", "N/A")
    regime  = sample.get("macro_regime", "N/A")

    prompt = f"""
FINAL PORTFOLIO DECISION REQUEST
=================================
Date          : {date_str}
Total Value   : ${total_val:,.2f}
Cash          : ${portfolio['cash']:,.2f}  ({cash_pct*100:.1f}%)
Equity        : ${total_val - portfolio['cash']:,.2f}  ({(1-cash_pct)*100:.1f}%)
Initial Cap   : ${portfolio['initial_capital']:,.2f}
Cum. Return   : {(total_val/portfolio['initial_capital']-1)*100:+.2f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT POSITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{pos_block}

Subsector exposure: {dict((k, f"{v*100:.1f}%") for k, v in subsector_exposure.items()) or "none"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MARKET CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VIX: {vix}  ({vix_reg})   Macro regime: {regime}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TODAY'S SIGNALS  (● = currently held)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{signal_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONSTRAINTS REMINDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Max single stock : 15%
  Min cash         : 5%
  Max per subsector: 35%
  Sum of all allocs: must equal 1.00

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIRED OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return EXACTLY this JSON (target_allocations must include ALL tickers + "cash",
and must sum to 1.0):

{{
  "target_allocations": {{
    "NVDA": 0.12,
    "AMD":  0.08,
    ... (only tickers with target > 0),
    "cash": 0.20
  }},
  "market_stance": "aggressive | neutral | defensive",
  "reasoning": "<3-5 sentence portfolio-level rationale>",
  "key_changes": [
    "<most important change 1 and why>",
    "<most important change 2 and why>"
  ]
}}
""".strip()
    return prompt


# ── Trade Order Generator ──────────────────────────────────────────────────────

def _generate_orders(
    target_allocs: dict,
    portfolio: dict,
    prices: dict,
    total_val: float,
) -> list[dict]:
    """
    将目标仓位比例转换为具体的买卖订单。

    逻辑：
      target_value  = target_pct * total_val
      current_value = shares * price
      delta_value   = target - current
      delta_value > 0 → BUY  (delta_value / price) shares
      delta_value < 0 → SELL (delta_value / price) shares
    """
    orders = []
    MIN_ORDER_USD = 50.0   # 小于 $50 的调整忽略，避免产生碎单

    for ticker, target_pct in target_allocs.items():
        if ticker == "cash":
            continue

        price = prices.get(ticker)
        if not price or price <= 0:
            continue

        target_val  = target_pct * total_val
        current_pos = portfolio["positions"].get(ticker, {"shares": 0, "avg_cost": 0})
        current_val = current_pos["shares"] * price
        delta_val   = target_val - current_val

        if abs(delta_val) < MIN_ORDER_USD:
            continue

        if delta_val > 0:
            shares = int(delta_val / price)     # 取整股，向下取整
            if shares <= 0:
                continue
            orders.append({
                "ticker":       ticker,
                "action":       "buy",
                "shares":       shares,
                "price":        price,
                "dollar_amount": round(shares * price, 2),
                "target_pct":   target_pct,
            })
        else:
            shares = min(int(abs(delta_val) / price) + 1,  # 向上取整
                         current_pos["shares"])
            if shares <= 0:
                continue
            orders.append({
                "ticker":       ticker,
                "action":       "sell",
                "shares":       shares,
                "price":        price,
                "dollar_amount": round(shares * price, 2),
                "target_pct":   target_pct,
            })

    # 当前持有但目标为 0 的：全部卖出
    for ticker, pos in portfolio["positions"].items():
        if ticker not in target_allocs or target_allocs.get(ticker, 0) == 0:
            price = prices.get(ticker)
            if not price or pos["shares"] <= 0:
                continue
            orders.append({
                "ticker":       ticker,
                "action":       "sell",
                "shares":       pos["shares"],
                "price":        price,
                "dollar_amount": round(pos["shares"] * price, 2),
                "target_pct":   0.0,
            })

    return orders


# ── DecisionAgent Class ────────────────────────────────────────────────────────

class DecisionAgent:
    """
    Final Portfolio Decision Agent.

    Usage:
        agent = DecisionAgent()
        result = agent.decide(date_str="2026-03-23")
        # result 包含 target_allocations + 具体订单，传给 RecordAgent 执行
    """

    def decide(
        self,
        date_str: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Optional[dict]:
        today = date_str or datetime.now().strftime("%Y-%m-%d")
        cache_path = CACHE_DIR / f"decision_{today}.json"

        if not force_refresh and cache_path.exists():
            print(f"  [DecisionAgent] Loaded from cache ({today})")
            return _load_json(cache_path)

        # ── Load all risk signals ─────────────────────────────────────────────
        signals = _load_all_risk_signals(today)
        if not signals:
            print(f"  [DecisionAgent] No risk signals found for {today}. "
                  f"Run RiskControlAgent first.")
            return None

        print(f"  [DecisionAgent] {len(signals)} tickers with signals")

        # ── Load portfolio + prices ───────────────────────────────────────────
        portfolio  = load_portfolio()
        all_tickers = list(signals.keys()) + list(portfolio["positions"].keys())
        all_tickers = list(set(all_tickers))

        print(f"  [DecisionAgent] Fetching prices for {len(all_tickers)} tickers...")
        prices    = _get_prices(all_tickers, today)
        total_val = portfolio_value(portfolio, prices)
        positions = positions_summary(portfolio, prices, total_val)

        print(f"  [DecisionAgent] Portfolio value: ${total_val:,.2f}  "
              f"Cash: ${portfolio['cash']:,.2f}")

        # ── LLM decision ─────────────────────────────────────────────────────
        print(f"  [DecisionAgent] Calling LLM for portfolio allocation...")
        prompt  = _build_prompt(today, portfolio, prices, total_val, positions, signals)
        raw     = _call_llm(prompt)
        llm_out = _extract_json(raw)

        target_allocs = llm_out.get("target_allocations", {})

        # ── Validate and normalize allocations ────────────────────────────────
        total_alloc = sum(target_allocs.values())
        if abs(total_alloc - 1.0) > 0.02:
            # Normalize if slightly off
            target_allocs = {k: round(v / total_alloc, 4) for k, v in target_allocs.items()}

        # Hard-enforce constraints
        for t, pct in list(target_allocs.items()):
            if t == "cash":
                continue
            if pct > MAX_POSITION_PCT + 0.01:  # allow 1% tolerance
                target_allocs[t] = MAX_POSITION_PCT
        if target_allocs.get("cash", 0) < MIN_CASH_PCT:
            target_allocs["cash"] = MIN_CASH_PCT

        # ── Generate trade orders ─────────────────────────────────────────────
        orders = _generate_orders(target_allocs, portfolio, prices, total_val)

        result = {
            "date":               today,
            "portfolio_value":    round(total_val, 2),
            "cash_before":        round(portfolio["cash"], 2),
            "market_stance":      llm_out.get("market_stance", "neutral"),
            "reasoning":          llm_out.get("reasoning", ""),
            "key_changes":        llm_out.get("key_changes", []),
            "target_allocations": target_allocs,
            "orders":             orders,
            "prices_used":        prices,
            "n_buy_orders":       sum(1 for o in orders if o["action"] == "buy"),
            "n_sell_orders":      sum(1 for o in orders if o["action"] == "sell"),
            "total_buy_usd":      round(sum(o["dollar_amount"] for o in orders if o["action"] == "buy"), 2),
            "total_sell_usd":     round(sum(o["dollar_amount"] for o in orders if o["action"] == "sell"), 2),
        }

        _save_json(result, cache_path)

        print(f"  [DecisionAgent] Done — stance={result['market_stance']}, "
              f"orders: {result['n_buy_orders']} buys / {result['n_sell_orders']} sells")
        print(f"    Buy ${result['total_buy_usd']:,.0f}  |  Sell ${result['total_sell_usd']:,.0f}")
        return result


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, re as _re

    _date_pattern = _re.compile(r"^\d{4}-\d{2}-\d{2}$")
    args      = sys.argv[1:]
    date_arg  = next((a for a in args if _date_pattern.match(a)), None)
    force_arg = "--force" in args

    agent  = DecisionAgent()
    result = agent.decide(date_str=date_arg, force_refresh=force_arg)

    if result:
        print("\n── Orders ──")
        for o in result["orders"]:
            print(f"  {o['action'].upper():<5} {o['ticker']:<6} "
                  f"{o['shares']:>6} shares @ ${o['price']:.2f} = ${o['dollar_amount']:,.2f}")
