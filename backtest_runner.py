"""
backtest_runner.py — Full Pipeline Backtest (Quarterly Rebalancing)

Runs the complete multi-agent pipeline on 40 quarterly dates (2015-2024),
then builds a daily return series for comparison with the DMA benchmark.

每个季度首日运行一次完整 pipeline（LLM 调用），季度内每日收益由持仓 × 价格计算（无 LLM）。

Usage:
    python backtest_runner.py                     # 完整 2015-2024，40 个季度
    python backtest_runner.py --start=2020-01-01  # 只跑 2020-2024（20 季度，约 3 小时）
    python backtest_runner.py --resume            # 从上次中断处继续
    python backtest_runner.py --force             # 强制忽略 pipeline 缓存重跑

Outputs (saved to backtest_results/):
    quarterly_decisions.json  — 每季度目标配置权重
    pipeline_returns.csv      — 日度收益（可直接传入 backtesting.py 的 plot_results）
    pipeline_weights.csv      — 日度权重矩阵
"""

import json
import sys
import shutil
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR     = Path("d:/Investment_AI_agent")
RESULTS_DIR  = BASE_DIR / "backtest_results"
RESULTS_DIR.mkdir(exist_ok=True)

DECISIONS_FILE = RESULTS_DIR / "quarterly_decisions.json"
RETURNS_FILE   = RESULTS_DIR / "pipeline_returns.csv"
WEIGHTS_FILE   = RESULTS_DIR / "pipeline_weights.csv"

DEFAULT_START = "2015-01-01"
DEFAULT_END   = "2024-12-31"
RISK_FREE     = 0.04


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    df = pd.read_csv(
        BASE_DIR / "semiconductor_tickers_cleaned_2010_2024.csv",
        skiprows=[3], header=[0, 1], index_col=0,
    )
    df.index = pd.to_datetime(df.index)
    return df["Close"]


def quarterly_dates(close_df: pd.DataFrame, start: str, end: str) -> list:
    """Snap each quarter-start to the nearest actual trading day."""
    q_starts = pd.date_range(start=start, end=end, freq="QS")
    dates = []
    for d in q_starts:
        future = close_df.index[close_df.index >= d]
        if len(future):
            dates.append(future[0])
    return dates


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio state management
# ─────────────────────────────────────────────────────────────────────────────

def reset_portfolio():
    """
    Backup and reset portfolio/portfolio_state.json to 100% cash.
    Required before starting a fresh backtest run.
    """
    portfolio_dir = BASE_DIR / "portfolio"
    portfolio_dir.mkdir(exist_ok=True)

    for fname in ["portfolio_state.json", "trade_log.json", "portfolio_history.json"]:
        p = portfolio_dir / fname
        if p.exists():
            bak = portfolio_dir / f"{fname}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy(p, bak)

    fresh = {
        "cash":            100_000.0,
        "positions":       {},
        "initial_capital": 100_000.0,
        "last_updated":    None,
    }
    (portfolio_dir / "portfolio_state.json").write_text(
        json.dumps(fresh, indent=2), encoding="utf-8"
    )
    print("  Portfolio reset → $100,000 cash  (originals backed up)")


# ─────────────────────────────────────────────────────────────────────────────
# Decision persistence
# ─────────────────────────────────────────────────────────────────────────────

def load_existing_decisions() -> dict:
    """Load previously saved quarterly decisions (for --resume)."""
    if not DECISIONS_FILE.exists():
        return {}
    raw = json.loads(DECISIONS_FILE.read_text(encoding="utf-8"))
    return {pd.Timestamp(k): v for k, v in raw.items()}


def save_decisions(decisions: dict):
    """Persist decisions after every quarter so --resume works on crash."""
    serializable = {str(k.date()): v for k, v in decisions.items()}
    DECISIONS_FILE.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def weights_from_decision(decision: dict, tickers: list) -> dict:
    """Extract per-ticker weights from target_allocations (drop 'cash' key)."""
    allocs = decision.get("target_allocations", {})
    return {t: float(allocs.get(t, 0.0)) for t in tickers}


# ─────────────────────────────────────────────────────────────────────────────
# Daily return construction
# ─────────────────────────────────────────────────────────────────────────────

def build_daily_returns(
    decisions: dict,
    close_df: pd.DataFrame,
    start: str,
    end: str,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Hold weights constant between rebalancing dates.
    daily_return = Σ w(t-1) × price_return(t)   — no LLM, pure math.
    """
    tickers  = close_df.columns.tolist()
    idx      = close_df.loc[start:end].index
    daily_w  = pd.DataFrame(0.0, index=idx, columns=tickers)
    sorted_d = sorted(decisions.keys())

    for i, d_start in enumerate(sorted_d):
        d_end = sorted_d[i + 1] if i + 1 < len(sorted_d) else idx[-1]
        mask  = (idx >= d_start) & (idx <= d_end)
        for ticker, w in decisions[d_start].items():
            if ticker in daily_w.columns:
                daily_w.loc[mask, ticker] = w

    daily_ret = close_df.loc[start:end].pct_change()
    port_ret  = (daily_w.shift(1) * daily_ret).sum(axis=1)
    return port_ret, daily_w


# ─────────────────────────────────────────────────────────────────────────────
# Performance summary
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics(returns: pd.Series, name: str):
    r = returns.dropna()
    if r.empty:
        return
    cum     = (1 + r).cumprod()
    n_years = len(r) / 252
    total   = cum.iloc[-1] - 1
    ann_ret = (1 + total) ** (1 / n_years) - 1
    ann_vol = r.std() * 252**0.5
    sharpe  = (ann_ret - RISK_FREE) / ann_vol if ann_vol else 0
    mdd     = ((cum - cum.cummax()) / cum.cummax()).min()
    monthly = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    win_rt  = (monthly > 0).mean()

    print(f"\n  {name}")
    print(f"  {'─'*40}")
    print(f"  Total return      : {total:+.2%}")
    print(f"  Ann. return       : {ann_ret:+.2%}")
    print(f"  Ann. volatility   : {ann_vol:.2%}")
    print(f"  Sharpe ratio      : {sharpe:.2f}")
    print(f"  Max drawdown      : {mdd:.2%}")
    print(f"  Monthly win rate  : {win_rt:.2%}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline Backtest Runner")
    parser.add_argument("--start",  default=DEFAULT_START, help="Backtest start date")
    parser.add_argument("--end",    default=DEFAULT_END,   help="Backtest end date")
    parser.add_argument("--resume", action="store_true",
                        help="Skip quarters already in quarterly_decisions.json")
    parser.add_argument("--force",  action="store_true",
                        help="Force re-run all LLM agents (ignore agent caches)")
    args = parser.parse_args()

    from pipeline import run_pipeline, TICKER_LIST

    print("=" * 60)
    print("  Full Pipeline Backtest Runner")
    print(f"  {args.start}  →  {args.end}")
    print("=" * 60)

    # ── Load prices ────────────────────────────────────────────────
    print("\n[1] Loading price data...")
    close_df  = load_prices()
    available = [t for t in TICKER_LIST if t in close_df.columns]
    print(f"    {len(available)} tickers loaded")

    # ── Quarterly dates ────────────────────────────────────────────
    q_dates = quarterly_dates(close_df, args.start, args.end)
    print(f"\n[2] {len(q_dates)} quarterly rebalancing dates")
    print(f"    {q_dates[0].strftime('%Y-%m-%d')}  →  {q_dates[-1].strftime('%Y-%m-%d')}")

    # ── Resume or fresh start ──────────────────────────────────────
    if args.resume:
        decisions = load_existing_decisions()
        already   = len(decisions)
        remaining = len(q_dates) - already
        print(f"\n[3] Resume mode: {already} done, {remaining} remaining")
    else:
        decisions = {}
        print("\n[3] Resetting portfolio to clean state...")
        reset_portfolio()

    # ── Pipeline loop ──────────────────────────────────────────────
    total     = len(q_dates)
    completed = len(decisions)
    print(f"\n[4] Running pipeline — {total - completed} quarters to go\n")

    for i, date in enumerate(q_dates):
        date_str = date.strftime("%Y-%m-%d")
        tag      = f"[{i+1:02d}/{total}]"

        if args.resume and date in decisions:
            invested = sum(v for v in decisions[date].values() if v > 0)
            if invested > 0:
                print(f"  {tag} {date_str}  [skip — already done, invested={invested:.0%}]")
                continue
            # invested==0 means a previous failure, retry it

        print(f"  {tag} {date_str}  ← running pipeline...", flush=True)

        try:
            result = run_pipeline(
                date_str=date_str,
                tickers=available,
                force=args.force,
            )

            if result and result.get("target_allocations"):
                decisions[date] = weights_from_decision(result, available)
                save_decisions(decisions)

                allocs   = result["target_allocations"]
                invested = sum(v for k, v in allocs.items() if k != "cash")
                holdings = [k for k, v in allocs.items() if k != "cash" and v > 0]
                print(f"       → {result.get('market_stance','?').upper()} | "
                      f"invested={invested:.0%} | {holdings}")
            else:
                # Carry forward previous quarter's weights on failure
                prev = decisions.get(q_dates[i - 1]) if i > 0 else None
                decisions[date] = prev if prev else {t: 0.0 for t in available}
                save_decisions(decisions)
                print(f"       → pipeline failed, carried forward previous weights")

        except Exception as e:
            print(f"       → ERROR: {e}")
            prev = decisions.get(q_dates[i - 1]) if i > 0 else None
            decisions[date] = prev if prev else {t: 0.0 for t in available}
            save_decisions(decisions)

    # ── Build daily return series ──────────────────────────────────
    print(f"\n[5] Building daily returns from {len(decisions)} quarterly decisions...")
    port_ret, daily_w = build_daily_returns(decisions, close_df, args.start, args.end)

    # ── Save ───────────────────────────────────────────────────────
    port_ret.to_csv(RETURNS_FILE, header=["returns"])
    daily_w.to_csv(WEIGHTS_FILE)
    print(f"    pipeline_returns.csv  ({len(port_ret)} days)")
    print(f"    pipeline_weights.csv")

    # ── Print metrics ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PERFORMANCE SUMMARY  ({args.start} → {args.end})")
    print(f"{'='*60}")
    print_metrics(port_ret, "Full Pipeline (Quarterly Rebalancing)")

    # Compare with DMA if available
    dma_path = BASE_DIR / "benchmark_dma_returns.csv"
    if dma_path.exists():
        dma_df = pd.read_csv(dma_path, index_col=0, parse_dates=True)
        dma = dma_df.iloc[:, 0]  # first column regardless of name
        print_metrics(dma.loc[args.start:args.end], "DMA Benchmark (50/200 MA)")

    print(f"\n{'='*60}")
    print(f"  Next step: load pipeline_returns.csv in backtesting.py")
    print(f"  and call plot_results(dma_returns, pipeline_returns)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
