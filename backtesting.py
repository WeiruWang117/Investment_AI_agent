"""
backtesting.py — AI Agent vs DMA Strategy Backtesting Framework

Compares quarterly-rebalancing AI Agent strategy against the pre-computed
DMA benchmark (50/200 MA) from benchmark_dma_returns.csv.

Period  : 2015-01-01 → 2024-12-31
Tickers : 24 semiconductor stocks (from model.py)
Cost    : ~40 Claude API calls (one per quarter), cached to disk
"""

import re
import os
import json
import anthropic
import numpy  as np
import pandas as pd
import matplotlib.pyplot  as plt
import matplotlib.dates   as mdates
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CLIENT       = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY from env
MODEL        = "claude-haiku-4-5-20251001"    # fast + cheap; Opus not needed for scoring
START_DATE   = "2015-01-01"
END_DATE     = "2024-12-31"
RISK_FREE    = 0.04                           # 4% annual risk-free rate
CACHE_DIR    = "agent_decisions_cache"        # JSON cache for agent decisions

TICKERS = [
    "TSM",  "INTC", "NVDA", "ASML", "AVGO", "TXN",  "QCOM", "MU",
    "AMAT", "AMD",  "ADI",  "LRCX", "KLAC", "MCHP", "SWKS", "MRVL",
    "TER",  "ON",   "MPWR", "ENTG", "MKSI", "SLAB", "CRUS", "SMTC",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_close_prices() -> pd.DataFrame:
    """Load cleaned close prices (multi-index CSV from model.py)."""
    df = pd.read_csv(
        "semiconductor_tickers_cleaned_2010_2024.csv",
        skiprows=[3], header=[0, 1], index_col=0,
    )
    df.index = pd.to_datetime(df.index)
    close = df["Close"].copy()
    available = [t for t in TICKERS if t in close.columns]
    print(f"  Loaded {len(available)} tickers from price CSV")
    return close[available]


def load_dma_benchmark() -> pd.Series:
    """Load pre-computed DMA (50/200 MA) returns from StockData.ipynb output."""
    dma = pd.read_csv("benchmark_dma_returns.csv", index_col=0, parse_dates=True)
    dma.columns = ["returns"]
    return dma["returns"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Context builders (no look-ahead bias — only use data ≤ as_of)
# ─────────────────────────────────────────────────────────────────────────────

def price_summary(ticker: str, close_df: pd.DataFrame, as_of: pd.Timestamp) -> str:
    hist = close_df.loc[:as_of, ticker].dropna().tail(252)
    if len(hist) < 20:
        return "数据不足"

    cur    = hist.iloc[-1]
    ma20   = hist.tail(20).mean()
    ma50   = hist.tail(50).mean()   if len(hist) >= 50  else float("nan")
    ma200  = hist.mean()             if len(hist) >= 200 else float("nan")
    r30    = (hist.iloc[-1] / hist.iloc[max(-31,  -len(hist))] - 1) * 100
    r252   = (hist.iloc[-1] / hist.iloc[max(-253, -len(hist))] - 1) * 100
    vol    = hist.pct_change().std() * 252**0.5 * 100

    parts = [f"P={cur:.1f}", f"MA20={ma20:.1f}"]
    if not np.isnan(ma50):  parts.append(f"MA50={ma50:.1f}")
    if not np.isnan(ma200): parts.append(f"MA200={ma200:.1f}")
    parts += [f"30d={r30:+.1f}%", f"1Y={r252:+.1f}%", f"Vol={vol:.0f}%"]
    return " ".join(parts)


def sentiment_summary(ticker: str, as_of: pd.Timestamp, n: int = 15) -> str:
    path = f"News data/{ticker}_Historical_News_Cleaned.csv"
    if not os.path.exists(path):
        return "无数据"
    try:
        df  = pd.read_csv(path)
        df["time_published"] = pd.to_datetime(df["time_published"], utc=True)
        cut = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of
        rec = (df[df["time_published"] <= cut]
                 .sort_values("time_published", ascending=False)
                 .head(n))
        if rec.empty:
            return "该时期无数据"
        avg = rec["overall_sentiment_score"].mean()
        pos = (rec["overall_sentiment_score"] >  0.2).sum()
        neg = (rec["overall_sentiment_score"] < -0.2).sum()
        return f"均分={avg:+.3f} 正={pos} 负={neg}"
    except Exception:
        return "加载失败"


def sec_summary(ticker: str, as_of: pd.Timestamp) -> str:
    snippets = []
    for ftype in ("10-K", "10-Q"):
        base = Path("SEC filings") / ticker / ftype
        if not base.exists():
            continue
        valid = sorted(
            f for f in base.iterdir()
            if f.is_dir() and f.name <= as_of.strftime("%Y-%m-%d")
        )
        if not valid:
            continue
        latest = valid[-1]
        for fpath in list(latest.rglob("*.htm"))[:1]:
            try:
                text  = fpath.read_text(encoding="utf-8", errors="ignore")
                clean = re.sub(r"<[^>]+>", " ", text)
                clean = re.sub(r"\s+", " ", clean).strip()[:300]
                snippets.append(f"{ftype}: {clean}")
            except Exception:
                pass
    return " | ".join(snippets) if snippets else "无报告"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Agent Decision (quarterly, with disk cache)
# ─────────────────────────────────────────────────────────────────────────────

# Force Claude to return structured portfolio scores via tool use
SCORE_TOOL = {
    "name": "submit_scores",
    "description": "提交每只股票的评分，构建下季度投资组合",
    "input_schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "description": (
                    "每只股票的评分（-1到1）。"
                    "1.0=强买, 0.5=买, 0=中性, -0.5=卖, -1.0=强卖。"
                    "正数=做多候选，负数=回避。"
                ),
                "additionalProperties": {"type": "number"},
            },
            "top_picks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "最看好的3-5只股票代码",
            },
            "macro_view": {
                "type": "string",
                "description": "对当前半导体行业的一句话宏观判断",
            },
        },
        "required": ["scores", "top_picks", "macro_view"],
    },
}


def agent_decide(tickers: list, close_df: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    """
    Make a portfolio decision as of `as_of`.
    Results are cached to JSON — re-running never re-calls the API.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = Path(CACHE_DIR) / f"{as_of.strftime('%Y%m%d')}.json"

    if cache_file.exists():
        print(f"  [cached] {as_of.strftime('%Y-%m')}")
        return json.loads(cache_file.read_text())

    print(f"  [agent ] {as_of.strftime('%Y-%m')} — calling Claude...", flush=True)

    # Build a compact per-ticker summary (no future data)
    lines = []
    for t in tickers:
        p = price_summary(t, close_df, as_of)
        s = sentiment_summary(t, as_of)
        lines.append(f"  {t}: 价格=[{p}] 情感=[{s}]")
    context = "\n".join(lines)

    response = CLIENT.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=(
            "你是专业的半导体行业投资组合经理，带领技术面、情感面分析团队。\n"
            "严格遵守：只使用截至决策日期的历史数据，绝不使用未来信息。\n"
            "策略：多头（long-only），目标最大化夏普比率。"
        ),
        tools=[SCORE_TOOL],
        tool_choice={"type": "tool", "name": "submit_scores"},
        messages=[{
            "role": "user",
            "content": (
                f"决策日期：{as_of.strftime('%Y年%m月%d日')}\n\n"
                f"截至今日的股票数据：\n{context}\n\n"
                "请综合技术面和情感面，对每只股票评分，构建下一季度组合。"
            ),
        }],
    )

    result = {
        "scores":     {t: 0.0 for t in tickers},
        "top_picks":  [],
        "macro_view": "",
        "date":       str(as_of.date()),
    }
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_scores":
            inp = block.input
            result["scores"]     = inp.get("scores",     result["scores"])
            result["top_picks"]  = inp.get("top_picks",  [])
            result["macro_view"] = inp.get("macro_view", "")
            break

    # Fill missing tickers with 0
    for t in tickers:
        result["scores"].setdefault(t, 0.0)

    cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"    picks={result['top_picks']}  view={result['macro_view'][:60]}")
    return result


def scores_to_weights(scores: dict, threshold: float = 0.05) -> dict:
    """Convert raw scores → long-only portfolio weights (score-proportional)."""
    positive = {k: max(v, 0.0) for k, v in scores.items() if v > threshold}
    total = sum(positive.values())
    if total == 0:
        return {k: 0.0 for k in scores}   # all cash this quarter
    return {k: v / total for k, v in positive.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Run Agent Backtest (quarterly rebalancing)
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_backtest(close_df: pd.DataFrame) -> tuple:
    """
    Runs the AI agent strategy with quarterly rebalancing over START_DATE→END_DATE.
    Returns:
        port_returns : pd.Series  — daily portfolio returns
        daily_weights: pd.DataFrame — daily weight matrix
        decisions    : dict       — raw agent decisions per quarter
    """
    tickers = close_df.columns.tolist()

    # Quarterly start dates → snap to nearest trading day
    q_dates = pd.date_range(start=START_DATE, end=END_DATE, freq="QS")
    decision_dates = []
    for d in q_dates:
        future = close_df.index[close_df.index >= d]
        if len(future):
            decision_dates.append(future[0])

    print(f"\n  {len(decision_dates)} quarters to evaluate "
          f"({decision_dates[0].strftime('%Y-%m')} → {decision_dates[-1].strftime('%Y-%m')})")

    # Collect decisions — parallel across quarters (each is independent)
    decisions: dict = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(agent_decide, tickers, close_df, d): d for d in decision_dates}
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                dec          = fut.result()
                decisions[d] = scores_to_weights(dec["scores"])
            except Exception as e:
                print(f"  [agent] {d.strftime('%Y-%m')} failed: {e}")

    # Build daily weight matrix (hold weights until next rebalance)
    daily_w = pd.DataFrame(0.0, index=close_df.index, columns=tickers)
    sorted_d = sorted(decisions)
    for i, start in enumerate(sorted_d):
        end  = sorted_d[i + 1] if i + 1 < len(sorted_d) else close_df.index[-1]
        mask = (close_df.index >= start) & (close_df.index < end)
        for ticker, w in decisions[start].items():
            daily_w.loc[mask, ticker] = w

    # Daily portfolio return = Σ weight(t-1) × return(t)
    daily_ret  = close_df.pct_change()
    port_ret   = (daily_w.shift(1) * daily_ret).sum(axis=1)

    return port_ret, daily_w, decisions


# ─────────────────────────────────────────────────────────────────────────────
# 5. Performance Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(returns: pd.Series, name: str) -> dict:
    """Compute MDD, Sharpe, Sortino, Calmar, annual return, win rate, etc."""
    r = returns.loc[START_DATE:END_DATE].dropna()
    if r.empty:
        return {"Strategy": name}

    cum     = (1 + r).cumprod()
    n_years = len(r) / 252

    total_ret = cum.iloc[-1] - 1
    ann_ret   = (1 + total_ret) ** (1 / n_years) - 1
    ann_vol   = r.std() * 252**0.5
    sharpe    = (ann_ret - RISK_FREE) / ann_vol if ann_vol else 0.0

    # Max Drawdown
    roll_max  = cum.cummax()
    dd        = (cum - roll_max) / roll_max
    mdd       = dd.min()
    mdd_date  = dd.idxmin()

    # Calmar
    calmar    = ann_ret / abs(mdd) if mdd else float("nan")

    # Sortino (downside deviation)
    down_r    = r[r < 0]
    down_vol  = down_r.std() * 252**0.5 if len(down_r) else float("nan")
    sortino   = (ann_ret - RISK_FREE) / down_vol if down_vol else float("nan")

    # Monthly win rate
    monthly   = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    win_rate  = (monthly > 0).mean()

    # Per-year returns
    yearly    = r.resample("YE").apply(lambda x: (1 + x).prod() - 1)

    return {
        "Strategy":           name,
        "Total Return":       f"{total_ret:.2%}",
        "Ann. Return":        f"{ann_ret:.2%}",
        "Ann. Volatility":    f"{ann_vol:.2%}",
        "Sharpe Ratio":       f"{sharpe:.2f}",
        "Sortino Ratio":      f"{sortino:.2f}",
        "Max Drawdown":       f"{mdd:.2%}",
        "MDD Date":           mdd_date.strftime("%Y-%m-%d"),
        "Calmar Ratio":       f"{calmar:.2f}",
        "Win Rate (Monthly)": f"{win_rate:.2%}",
        "Best Year":          f"{yearly.max():.2%}",
        "Worst Year":         f"{yearly.min():.2%}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    dma:   pd.Series,
    agent: pd.Series,
    save:  str = "backtest_comparison.png",
) -> None:
    """4-panel chart: equity curve, drawdown, annual bars, rolling Sharpe."""

    dma   = dma.loc[START_DATE:END_DATE].dropna()
    agent = agent.loc[START_DATE:END_DATE].dropna()
    idx   = dma.index.intersection(agent.index)
    dma, agent = dma[idx], agent[idx]

    dma_cum   = (1 + dma).cumprod()
    agent_cum = (1 + agent).cumprod()
    dma_dd    = (dma_cum   - dma_cum.cummax())   / dma_cum.cummax()
    agent_dd  = (agent_cum - agent_cum.cummax()) / agent_cum.cummax()

    dma_yr   = dma.resample("YE").apply(lambda x: (1+x).prod()-1) * 100
    agent_yr = agent.resample("YE").apply(lambda x: (1+x).prod()-1) * 100

    BLUE, ORANGE = "#2563EB", "#EA580C"
    fmt = mdates.DateFormatter("%Y")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "AI Agent (Quarterly) vs DMA Strategy (50/200 MA)  |  2015–2024",
        fontsize=15, fontweight="bold",
    )

    # ── Panel 1: Equity Curve ──────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(dma_cum,   color=BLUE,   lw=1.8, label="DMA (50/200 MA)")
    ax.plot(agent_cum, color=ORANGE, lw=1.8, label="AI Agent (Quarterly)")
    ax.set_title("Equity Curve")
    ax.set_ylabel("Portfolio Value ($1 initial)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(fmt)

    # ── Panel 2: Drawdown ─────────────────────────────────────────
    ax = axes[0, 1]
    ax.fill_between(dma_dd.index,   dma_dd,   0, color=BLUE,   alpha=0.45, label="DMA")
    ax.fill_between(agent_dd.index, agent_dd, 0, color=ORANGE, alpha=0.45, label="Agent")
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(fmt)

    # ── Panel 3: Annual Returns Bar Chart ─────────────────────────
    ax = axes[1, 0]
    years = [str(d.year) for d in dma_yr.index]
    x, w  = np.arange(len(years)), 0.35
    ax.bar(x - w/2, dma_yr.values,   w, color=BLUE,   alpha=0.8, label="DMA")
    ax.bar(x + w/2, agent_yr.values, w, color=ORANGE, alpha=0.8, label="Agent")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(years, rotation=45)
    ax.set_title("Annual Returns (%)"); ax.set_ylabel("Return (%)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # ── Panel 4: Rolling 1-Year Sharpe ────────────────────────────
    ax = axes[1, 1]

    def rolling_sharpe(r: pd.Series, window: int = 252) -> pd.Series:
        ann_r   = r.rolling(window).mean() * 252
        ann_vol = r.rolling(window).std()  * 252**0.5
        return (ann_r - RISK_FREE) / ann_vol.replace(0, float("nan"))

    ax.plot(rolling_sharpe(dma),   color=BLUE,   lw=1.5, label="DMA")
    ax.plot(rolling_sharpe(agent), color=ORANGE, lw=1.5, label="Agent")
    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.set_title("Rolling 1-Year Sharpe Ratio")
    ax.set_ylabel("Sharpe")
    ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(fmt)

    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved → {save}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Investment AI Agent Backtesting Framework")
    print(f"  {START_DATE}  →  {END_DATE}")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────
    print("\n[1] Loading price data...")
    close_df = load_close_prices()

    print("\n[2] Loading DMA benchmark (50/200 MA)...")
    dma_raw  = load_dma_benchmark()

    # ── Agent strategy ─────────────────────────────────────────────
    print("\n[3] Running AI Agent strategy (quarterly rebalancing)...")
    print("    Cached decisions are loaded instantly; new ones call Claude.")
    agent_returns, agent_weights, decisions = run_agent_backtest(close_df)

    # Slice both series to the backtest period
    dma_returns   = dma_raw.loc[START_DATE:END_DATE]
    agent_returns = agent_returns.loc[START_DATE:END_DATE]

    # ── Metrics ────────────────────────────────────────────────────
    print("\n[4] Computing performance metrics...")
    dma_m   = compute_metrics(dma_returns,   "DMA (50/200 MA)")
    agent_m = compute_metrics(agent_returns, "AI Agent (Quarterly)")

    metrics_df = pd.DataFrame([dma_m, agent_m]).set_index("Strategy")

    print("\n" + "=" * 60)
    print("  PERFORMANCE COMPARISON")
    print("=" * 60)
    print(metrics_df.T.to_string())

    # ── Save results ───────────────────────────────────────────────
    metrics_df.T.to_csv("performance_comparison.csv")
    agent_returns.to_csv("agent_returns.csv", header=["returns"])
    agent_weights.to_csv("agent_weights.csv")

    # Save human-readable decision log
    log_rows = []
    for date, weights in decisions.items():
        for ticker, w in weights.items():
            if w > 0:
                log_rows.append({"quarter": str(date.date()), "ticker": ticker, "weight": round(w, 4)})
    pd.DataFrame(log_rows).to_csv("agent_decision_log.csv", index=False)

    print("\n  Files saved:")
    print("    performance_comparison.csv  ← key metrics table")
    print("    agent_returns.csv           ← daily returns series")
    print("    agent_weights.csv           ← daily weight matrix")
    print("    agent_decision_log.csv      ← quarterly holdings")
    print(f"    {CACHE_DIR}/               ← Claude API call cache")

    # ── Visualise ──────────────────────────────────────────────────
    print("\n[5] Generating comparison charts...")
    plot_results(dma_returns, agent_returns)

    return dma_returns, agent_returns, metrics_df


if __name__ == "__main__":
    main()
