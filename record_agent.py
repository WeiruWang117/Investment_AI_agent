"""
RecordAgent — Portfolio Bookkeeper

负责执行交易订单、维护持仓状态、记录每日 P&L，不调用 LLM。

文件结构:
  portfolio/portfolio_state.json    — 当前持仓快照（每次交易后更新）
  portfolio/trade_log.json          — 全部历史交易记录
  portfolio/portfolio_history.json  — 每日组合价值快照（用于绩效分析）

主要方法:
  execute_orders(decision_result)   — 执行 DecisionAgent 的订单，更新持仓
  end_of_day_snapshot(date_str)     — 收盘时记录当日组合价值
  get_performance_summary()         — 返回绩效统计（总收益、最大回撤等）
  print_portfolio()                 — 打印当前持仓表
"""

import json
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_DIR      = Path("d:/Investment_AI_agent")
PORTFOLIO_DIR = BASE_DIR / "portfolio"
PORTFOLIO_DIR.mkdir(exist_ok=True)

STATE_PATH   = PORTFOLIO_DIR / "portfolio_state.json"
TRADELOG_PATH = PORTFOLIO_DIR / "trade_log.json"
HISTORY_PATH  = PORTFOLIO_DIR / "portfolio_history.json"

INITIAL_CAPITAL = 100_000.0


# ── IO Utilities ───────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Optional[dict | list]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _save_json(data, path: Path):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_list(path: Path) -> list:
    data = _load_json(path)
    return data if isinstance(data, list) else []


# ── Portfolio State ────────────────────────────────────────────────────────────

def _load_state() -> dict:
    state = _load_json(STATE_PATH)
    if state is None:
        state = {
            "cash":            INITIAL_CAPITAL,
            "positions":       {},
            "initial_capital": INITIAL_CAPITAL,
            "last_updated":    None,
        }
        _save_json(state, STATE_PATH)
    return state


def _save_state(state: dict):
    _save_json(state, STATE_PATH)


def _portfolio_value(state: dict, prices: dict) -> float:
    equity = sum(
        pos["shares"] * prices.get(t, pos.get("avg_cost", 0))
        for t, pos in state["positions"].items()
    )
    return round(state["cash"] + equity, 2)


# ── Trade Execution ────────────────────────────────────────────────────────────

class RecordAgent:
    """
    Portfolio bookkeeper — no LLM.

    Usage:
        recorder = RecordAgent()

        # After DecisionAgent runs:
        recorder.execute_orders(decision_result)

        # At end of day (to record P&L):
        recorder.end_of_day_snapshot("2026-03-23", prices)

        # Anytime:
        recorder.print_portfolio(prices)
        summary = recorder.get_performance_summary()
    """

    # ── Execute Orders ─────────────────────────────────────────────────────────

    def execute_orders(self, decision_result: dict) -> list[dict]:
        """
        执行 DecisionAgent 输出的订单列表。
        按照先卖后买的顺序执行（确保有足够现金）。
        返回实际成交的订单列表。
        """
        orders  = decision_result.get("orders", [])
        prices  = decision_result.get("prices_used", {})
        date    = decision_result.get("date", datetime.now().strftime("%Y-%m-%d"))

        if not orders:
            print(f"  [RecordAgent] No orders to execute for {date}.")
            return []

        state    = _load_state()
        tradelog = _load_list(TRADELOG_PATH)
        executed = []

        # Sort: sells first, then buys (free up cash before spending)
        sells = [o for o in orders if o["action"] == "sell"]
        buys  = [o for o in orders if o["action"] == "buy"]

        for order in sells + buys:
            t      = order["ticker"]
            action = order["action"]
            shares = order["shares"]
            price  = order.get("price", prices.get(t, 0))
            amount = round(shares * price, 2)

            if price <= 0:
                print(f"  [RecordAgent] {t}: no price available, skipping.")
                continue

            if action == "sell":
                held = state["positions"].get(t, {}).get("shares", 0)
                if held <= 0:
                    print(f"  [RecordAgent] {t}: no position to sell, skipping.")
                    continue
                shares = min(shares, held)  # can't sell more than held
                amount = round(shares * price, 2)

                # Update position
                cost_basis = state["positions"][t]["avg_cost"]
                realized_pnl = round((price - cost_basis) * shares, 2)

                state["cash"] = round(state["cash"] + amount, 2)
                remaining     = state["positions"][t]["shares"] - shares
                if remaining <= 0:
                    del state["positions"][t]
                else:
                    state["positions"][t]["shares"] = remaining

                record = {
                    "date":         date,
                    "ticker":       t,
                    "action":       "sell",
                    "shares":       shares,
                    "price":        price,
                    "dollar_amount": amount,
                    "realized_pnl": realized_pnl,
                    "cash_after":   state["cash"],
                }

            else:  # buy
                if state["cash"] < amount:
                    # Reduce shares to fit available cash (keep $500 buffer)
                    affordable = max(0, int((state["cash"] - 500) / price))
                    if affordable <= 0:
                        print(f"  [RecordAgent] {t}: insufficient cash (${state['cash']:,.0f}), skipping.")
                        continue
                    shares = affordable
                    amount = round(shares * price, 2)

                state["cash"] = round(state["cash"] - amount, 2)

                # Update average cost (weighted average)
                if t in state["positions"]:
                    old_shares = state["positions"][t]["shares"]
                    old_cost   = state["positions"][t]["avg_cost"]
                    new_shares = old_shares + shares
                    new_cost   = round((old_shares * old_cost + shares * price) / new_shares, 4)
                    state["positions"][t] = {"shares": new_shares, "avg_cost": new_cost}
                else:
                    state["positions"][t] = {"shares": shares, "avg_cost": price}

                record = {
                    "date":         date,
                    "ticker":       t,
                    "action":       "buy",
                    "shares":       shares,
                    "price":        price,
                    "dollar_amount": amount,
                    "realized_pnl": 0.0,
                    "cash_after":   state["cash"],
                }

            tradelog.append(record)
            executed.append(record)
            print(f"  [RecordAgent] {action.upper():<5} {t:<6} "
                  f"{shares:>5} shares @ ${price:.2f} = ${amount:,.2f}  "
                  f"cash=${state['cash']:,.0f}")

        state["last_updated"] = date
        _save_state(state)
        _save_json(tradelog, TRADELOG_PATH)

        print(f"  [RecordAgent] {len(executed)} orders executed. "
              f"Cash remaining: ${state['cash']:,.2f}")
        return executed

    # ── End of Day Snapshot ────────────────────────────────────────────────────

    def end_of_day_snapshot(self, date_str: str, prices: dict) -> dict:
        """
        收盘时记录当日组合状态，计算 P&L。
        每天调用一次，建立绩效历史。
        """
        state   = _load_state()
        history = _load_list(HISTORY_PATH)

        total_val = _portfolio_value(state, prices)
        equity_val = total_val - state["cash"]

        # Previous day value
        prev_val = history[-1]["total_value"] if history else INITIAL_CAPITAL
        daily_pnl    = round(total_val - prev_val, 2)
        daily_ret    = round((total_val / prev_val - 1) * 100, 4) if prev_val else 0
        cum_ret      = round((total_val / INITIAL_CAPITAL - 1) * 100, 4)

        # Position details at close
        position_values = {
            t: round(pos["shares"] * prices.get(t, pos["avg_cost"]), 2)
            for t, pos in state["positions"].items()
        }

        snapshot = {
            "date":             date_str,
            "total_value":      total_val,
            "cash":             state["cash"],
            "equity_value":     round(equity_val, 2),
            "n_positions":      len(state["positions"]),
            "position_values":  position_values,
            "daily_pnl":        daily_pnl,
            "daily_return_pct": daily_ret,
            "cumulative_return_pct": cum_ret,
        }

        history.append(snapshot)
        _save_json(history, HISTORY_PATH)

        print(f"  [RecordAgent] EOD snapshot: ${total_val:,.2f}  "
              f"daily={daily_pnl:+,.2f} ({daily_ret:+.2f}%)  "
              f"cum={cum_ret:+.2f}%")
        return snapshot

    # ── Performance Summary ────────────────────────────────────────────────────

    def get_performance_summary(self) -> dict:
        """
        计算完整绩效统计，用于最终资金分析。
        """
        history = _load_list(HISTORY_PATH)
        if not history:
            return {"error": "No portfolio history found."}

        values = [h["total_value"] for h in history]
        dates  = [h["date"]        for h in history]
        rets   = [h["daily_return_pct"] for h in history]

        # Max drawdown
        peak = values[0]
        max_dd = 0.0
        max_dd_start = dates[0]
        max_dd_end   = dates[0]
        cur_peak_date = dates[0]
        for i, v in enumerate(values):
            if v > peak:
                peak = v
                cur_peak_date = dates[i]
            dd = (v - peak) / peak * 100
            if dd < max_dd:
                max_dd       = dd
                max_dd_start = cur_peak_date
                max_dd_end   = dates[i]

        # Streak analysis
        win_days  = sum(1 for r in rets if r > 0)
        lose_days = sum(1 for r in rets if r < 0)
        flat_days = sum(1 for r in rets if r == 0)

        # Sharpe (annualized, assume 252 trading days, risk-free = 4%)
        import statistics
        if len(rets) > 1:
            avg_ret  = statistics.mean(rets)
            std_ret  = statistics.stdev(rets)
            rf_daily = 4.0 / 252   # 4% annual / 252 days
            sharpe   = round((avg_ret - rf_daily) / std_ret * (252 ** 0.5), 3) if std_ret else 0
        else:
            sharpe = 0.0

        tradelog  = _load_list(TRADELOG_PATH)
        realized  = sum(t.get("realized_pnl", 0) for t in tradelog)
        total_trades = len(tradelog)

        return {
            "period_start":         dates[0]  if dates else None,
            "period_end":           dates[-1] if dates else None,
            "trading_days":         len(history),
            "initial_capital":      INITIAL_CAPITAL,
            "final_value":          values[-1] if values else INITIAL_CAPITAL,
            "total_pnl":            round(values[-1] - INITIAL_CAPITAL, 2) if values else 0,
            "cumulative_return_pct": history[-1]["cumulative_return_pct"] if history else 0,
            "max_drawdown_pct":     round(max_dd, 4),
            "max_drawdown_period":  f"{max_dd_start} → {max_dd_end}",
            "avg_daily_return_pct": round(statistics.mean(rets), 4) if rets else 0,
            "daily_return_std":     round(statistics.stdev(rets), 4) if len(rets) > 1 else 0,
            "sharpe_ratio":         sharpe,
            "win_days":             win_days,
            "lose_days":            lose_days,
            "flat_days":            flat_days,
            "win_rate_pct":         round(win_days / len(rets) * 100, 2) if rets else 0,
            "total_trades":         total_trades,
            "total_realized_pnl":   round(realized, 2),
        }

    # ── Print Portfolio ────────────────────────────────────────────────────────

    def print_portfolio(self, prices: Optional[dict] = None):
        """打印当前持仓和绩效概览。"""
        state   = _load_state()
        prices  = prices or {}
        total   = _portfolio_value(state, prices)
        history = _load_list(HISTORY_PATH)
        cum_ret = history[-1]["cumulative_return_pct"] if history else 0

        print(f"\n{'═'*60}")
        print(f"Portfolio as of {state.get('last_updated', 'N/A')}")
        print(f"{'═'*60}")
        print(f"  Total value  : ${total:>12,.2f}")
        print(f"  Cash         : ${state['cash']:>12,.2f}  "
              f"({state['cash']/total*100:.1f}%)")
        print(f"  Cum. return  : {cum_ret:>+11.2f}%")
        print(f"{'─'*60}")

        if state["positions"]:
            print(f"  {'Ticker':<8} {'Shares':>7} {'AvgCost':>9} "
                  f"{'Price':>9} {'Value':>11} {'%Port':>6} {'UnrPnL':>10}")
            print(f"  {'─'*8} {'─'*7} {'─'*9} {'─'*9} {'─'*11} {'─'*6} {'─'*10}")
            for t, pos in sorted(state["positions"].items()):
                price  = prices.get(t, pos["avg_cost"])
                val    = pos["shares"] * price
                pnl    = val - pos["shares"] * pos["avg_cost"]
                pct    = val / total * 100
                print(f"  {t:<8} {pos['shares']:>7} ${pos['avg_cost']:>8.2f} "
                      f"${price:>8.2f} ${val:>10,.2f} {pct:>5.1f}% "
                      f"${pnl:>+9,.2f}")
        else:
            print("  (no positions)")

        print(f"{'═'*60}\n")

    # ── Trade History ──────────────────────────────────────────────────────────

    def print_trade_history(self, last_n: int = 20):
        """打印最近 N 笔交易记录。"""
        tradelog = _load_list(TRADELOG_PATH)
        recent   = tradelog[-last_n:]
        if not recent:
            print("No trade history.")
            return

        print(f"\n{'─'*70}")
        print(f"Last {len(recent)} trades:")
        print(f"{'─'*70}")
        print(f"  {'Date':<12} {'Ticker':<8} {'Action':<6} {'Shares':>7} "
              f"{'Price':>9} {'Amount':>11} {'RealPnL':>10}")
        for t in recent:
            print(f"  {t['date']:<12} {t['ticker']:<8} {t['action'].upper():<6} "
                  f"{t['shares']:>7} ${t['price']:>8.2f} "
                  f"${t['dollar_amount']:>10,.2f} "
                  f"${t.get('realized_pnl', 0):>+9,.2f}")
        print(f"{'─'*70}\n")

    def print_performance(self):
        """打印绩效统计报告。"""
        s = self.get_performance_summary()
        if "error" in s:
            print(s["error"])
            return

        print(f"\n{'═'*50}")
        print(f"Performance Summary  {s['period_start']} → {s['period_end']}")
        print(f"{'═'*50}")
        print(f"  Initial capital    : ${s['initial_capital']:>12,.2f}")
        print(f"  Final value        : ${s['final_value']:>12,.2f}")
        print(f"  Total P&L          : ${s['total_pnl']:>+12,.2f}")
        print(f"  Cumulative return  : {s['cumulative_return_pct']:>+11.2f}%")
        print(f"{'─'*50}")
        print(f"  Sharpe ratio       : {s['sharpe_ratio']:>12.3f}")
        print(f"  Max drawdown       : {s['max_drawdown_pct']:>+11.2f}%  ({s['max_drawdown_period']})")
        print(f"  Avg daily return   : {s['avg_daily_return_pct']:>+11.4f}%")
        print(f"  Daily return std   : {s['daily_return_std']:>12.4f}%")
        print(f"{'─'*50}")
        print(f"  Win / Lose / Flat  : {s['win_days']} / {s['lose_days']} / {s['flat_days']}")
        print(f"  Win rate           : {s['win_rate_pct']:>11.1f}%")
        print(f"  Total trades       : {s['total_trades']:>12}")
        print(f"  Realized P&L       : ${s['total_realized_pnl']:>+12,.2f}")
        print(f"{'═'*50}\n")


# ── CLI / Quick Check ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    recorder = RecordAgent()
    recorder.print_portfolio()
    recorder.print_trade_history()
    recorder.print_performance()
