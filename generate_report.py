"""
生成投资AI Agent系统技术报告 Word 文档
"""
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

doc = Document()

# ── 页面设置 ──────────────────────────────────────────────────────────────────
section = doc.sections[0]
section.page_width  = Cm(21)
section.page_height = Cm(29.7)
section.top_margin    = Cm(2.5)
section.bottom_margin = Cm(2.5)
section.left_margin   = Cm(3)
section.right_margin  = Cm(2.5)

# ── 样式辅助函数 ───────────────────────────────────────────────────────────────

def set_font(run, name_en="Calibri", name_cn="宋体", size=11, bold=False, color=None):
    run.font.name = name_en
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name_cn)
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)

def add_heading(text, level=1):
    p = doc.add_heading(level=level)
    p.clear()
    run = p.add_run(text)
    sizes = {1: 16, 2: 14, 3: 12, 4: 11}
    colors = {1: (31, 73, 125), 2: (31, 73, 125), 3: (54, 96, 146), 4: (79, 129, 189)}
    set_font(run, size=sizes.get(level, 12), bold=True, color=colors.get(level, (0,0,0)))
    return p

def add_para(text, indent=0, bold=False, size=11, color=None, align=None):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.left_indent = Cm(indent)
    if align:
        p.alignment = align
    run = p.add_run(text)
    set_font(run, size=size, bold=bold, color=color)
    return p

def add_table(headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    # Header row
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.paragraphs[0].clear()
        run = cell.paragraphs[0].add_run(h)
        set_font(run, size=10, bold=True, color=(255, 255, 255))
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        # Background color
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), "1F497D")
        tcPr.append(shd)
    # Data rows
    for ri, row_data in enumerate(rows):
        row = table.rows[ri + 1]
        fill = "DEEAF1" if ri % 2 == 0 else "FFFFFF"
        for ci, cell_text in enumerate(row_data):
            cell = row.cells[ci]
            cell.paragraphs[0].clear()
            run = cell.paragraphs[0].add_run(str(cell_text))
            set_font(run, size=10)
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:color"), "auto")
            shd.set(qn("w:fill"), fill)
            tcPr.append(shd)
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Cm(w)
    return table

def add_code_block(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.5)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(4)
    # Light gray background via shading
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), "F2F2F2")
    pPr.append(shd)
    run = p.add_run(text)
    run.font.name = "Courier New"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = Pt(9)
    return p

# ══════════════════════════════════════════════════════════════════════════════
# 封面
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
doc.add_paragraph()
doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("半导体股票投资 AI Agent 系统")
set_font(run, size=22, bold=True, color=(31, 73, 125))

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("技术架构与实现报告")
set_font(run, size=18, bold=True, color=(54, 96, 146))

doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Technical Architecture & Implementation Report")
set_font(run, size=13, color=(89, 89, 89))

doc.add_paragraph()
doc.add_paragraph()

add_table(
    ["项目", "内容"],
    [
        ["项目路径", "d:/Investment_AI_agent/"],
        ["研究范围", "24只半导体股票，2010–2024年历史数据"],
        ["核心模型", "Claude claude-opus-4-6（thinking: adaptive + streaming）"],
        ["开发语言", "Python（Anaconda，anthropic SDK v0.86.0）"],
        ["报告日期", "2026-03-26"],
    ],
    col_widths=[4, 11]
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# 一、系统总体架构
# ══════════════════════════════════════════════════════════════════════════════
add_heading("一、系统总体架构", 1)
add_para(
    "本系统采用多智能体串联+并行的七阶段流水线（Pipeline）设计。每个 Agent 职责单一，"
    "通过 JSON 缓存文件传递信号，最终汇聚为组合级交易决策。整个流程从基础信号采集"
    "到最终执行，共分为 7 个 Stage，其中 Stage 1 的四路信号并行运行，Stage 2 中"
    " BullAgent 与 BearAgent 完全并行，DecisionAgent 是唯一拥有全局视角的 Agent。"
)

add_heading("1.1 七阶段流水线", 2)
add_code_block(
"""Stage 1a: TechnicalAgent    ← 价格/成交量技术指标（纯量化，无LLM）
Stage 1b: FundamentalAgent  ← 财务报表指标（纯量化，无LLM）
Stage 1c: SentimentAgent    ← 新闻情绪 + 财报电话会议LLM
Stage 1d: SECFilingAgent    ← 10-K/10-Q MDA & 风险因素LLM
              ↓ 4路信号 → 每只股票
Stage 2:  BullAgent + BearAgent  [完全并行]  ← 多空论点构建
              ↓
Stage 3:  ArbiterAgent      ← 多轮交叉质询辩论 → 裁决
              ↓
Stage 4:  MacroAgent        ← 宏观叠加（利率/通胀/GDP/需求）
              ↓
Stage 5:  RiskControlAgent  ← 六维风险评分 → 仓位规模
              ↓
Stage 6:  DecisionAgent     ← 全局组合优化 → 具体交易订单
              ↓
Stage 7:  RecordAgent       ← 执行订单 → 更新持仓 → EOD快照"""
)

add_heading("1.2 股票池（24只）", 2)
add_table(
    ["子行业", "股票代码"],
    [
        ["AI/数据中心/无晶圆厂", "NVDA, AMD, QCOM, AVGO, MRVL"],
        ["模拟芯片", "ADI, TXN, MCHP, ON, SWKS, SLAB, CRUS, SMTC, MPWR"],
        ["晶圆制造设备", "AMAT, LRCX, KLAC, ENTG, MKSI, TER"],
        ["晶圆代工/IP", "TSM, INTC, ASML, QCOM"],
        ["存储", "MU"],
    ],
    col_widths=[4.5, 11]
)
add_para("注：NXPI 因缺失值过多被剔除，最终保留 24 只。", size=9, color=(89,89,89))

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# 二、数据源
# ══════════════════════════════════════════════════════════════════════════════
add_heading("二、数据源", 1)

add_table(
    ["数据类型", "来源", "覆盖范围"],
    [
        ["股价OHLCV", "yfinance", "24只，2010-01-04 至 2024-12-30"],
        ["新闻情绪评分", "Alpha Vantage News Sentiment API", "20只（5只无数据），预评分 −0.75 至 +0.94"],
        ["财报电话会议文字稿", "Alpha Vantage Earnings Call API", "仅TSM有效（19季度，2015Q1–2020Q4）"],
        ["SEC 10-K / 10-Q", "SEC EDGAR（BeautifulSoup解析）", "22只，2015–2025，MDA + 风险因素段落"],
        ["财务报表", "Alpha Vantage Fundamentals API", "24只，Income / Balance Sheet / Cash Flow"],
        ["宏观经济指标", "Alpha Vantage Economic Indicators", "10项指标（见下表）"],
        ["VIX 恐慌指数", "yfinance", "2020年至今"],
    ],
    col_widths=[4, 5.5, 6]
)

doc.add_paragraph()
add_heading("宏观经济指标（10项）", 3)
add_table(
    ["类别", "指标", "频率"],
    [
        ["核心利率", "10年国债收益率、2年国债收益率", "月度"],
        ["核心利率", "联邦基金利率", "月度"],
        ["通胀", "CPI消费者价格指数", "月度"],
        ["通胀", "通胀率（年度CPI）", "年度"],
        ["增长", "实际GDP", "季度"],
        ["需求", "零售销售额、耐用品订单", "月度"],
        ["就业", "失业率、非农就业人数", "月度"],
    ],
    col_widths=[3.5, 7, 3]
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# 三、各Agent详细设计
# ══════════════════════════════════════════════════════════════════════════════
add_heading("三、各 Agent 详细设计", 1)

# ── 3.1 TechnicalAgent ──
add_heading("3.1 TechnicalAgent（Stage 1a）", 2)
add_para("类型：纯量化，无 LLM。数据源：semiconductor_tickers_cleaned_2010_2024.csv（宽格式 MultiIndex 列）。")

add_heading("技术指标参数", 3)
add_table(
    ["指标", "参数/方法"],
    [
        ["RSI", "14日（Wilder平均法）"],
        ["ATR 真实波幅", "14日，取 max(H−L, |H−C_prev|, |L−C_prev|)"],
        ["MACD", "快线12日EMA，慢线26日EMA，信号线9日EMA，直方图=MACD−Signal"],
        ["布林带", "20日SMA ±2σ，bb_pct=(收盘−下轨)/带宽，bb_width=带宽/中轨"],
        ["历史波动率", "对数收益率×√252，窗口：20日、60日"],
        ["ROC涨跌幅", "windows：20日、60日、120日、252日"],
        ["SMA均线", "SMA50, SMA200；价格/SMA比值；golden/death cross 信号"],
        ["成交量比", "当日成交量 / 20日均量"],
        ["OBV", "On-Balance Volume；20日线性回归斜率（np.polyfit）"],
    ],
    col_widths=[4.5, 11]
)

doc.add_paragraph()
add_heading("截面相对强度（跨股票比较）", 3)
add_para("在 24 只股票中横向比较，计算每只股票相对于全行业/子行业的超额收益：")
add_code_block(
"""rs_sector_60d    = 个股60日收益 − 全行业等权平均60日收益
rs_subsector_60d = 个股60日收益 − 子行业等权平均60日收益
rank_20d/60d/252d = 24只股票中百分位排名（0=最弱，1=最强）"""
)

add_heading("综合动量评分公式（momentum_score，−1 至 +1）", 3)
add_code_block(
"""momentum_score =
    tanh(ROC_20d  / 0.30) × 10%
  + tanh(ROC_60d  / 0.30) × 15%
  + tanh(ROC_120d / 0.30) × 10%
  + tanh(ROC_252d / 0.30) × 15%
  + (RSI − 50) / 50  × 20%
  + tanh(MACD_hist / |MACD| ) × 20%
  + (rank_60d × 2 − 1)       × 20%

注：使用 tanh 软截断将无界收益率映射至 (−1, +1)，避免极端值主导评分。
趋势状态分类：
  uptrend   → 收盘>SMA200×1.02 且 SMA50>SMA200 且 ROC_60d>5%
  downtrend → 收盘<SMA200×0.98 且 SMA50<SMA200 且 ROC_60d<−5%
  sideways  → 其余情况"""
)

# ── 3.2 FundamentalAgent ──
add_heading("3.2 FundamentalAgent（Stage 1b）", 2)
add_para("类型：纯量化，无 LLM。数据源：Financial data/{TICKER}/{INCOME_STATEMENT|BALANCE_SHEET|CASH_FLOW}.json")
add_para("点对点时间安全（Point-in-Time Safety）：季报 +45天，年报 +60天（避免前视偏差）。")

add_heading("计算指标", 3)
add_table(
    ["类别", "指标"],
    [
        ["盈利能力", "毛利率、营业利润率、净利率、ROIC、研发强度（R&D/Revenue）"],
        ["增长", "收入YoY（年度/季度/同季度YoY）、EPS YoY、FCF YoY、OCF YoY"],
        ["现金流质量", "FCF利润率、FCF/净利润（质量代理）、运营现金流利润率、资本开支强度"],
        ["资产负债表", "流动比率、净债务/EBITDA、债务/权益比"],
        ["半导体周期信号", "库存天数（inventory÷(COGS÷90)）、库存YoY"],
        ["利润率趋势", "最近4季度毛利率/营业利润率的线性斜率 → improving/deteriorating/stable"],
    ],
    col_widths=[4.5, 11]
)

add_heading("基本面综合评分公式（fundamental_score，−1 至 +1）", 3)
add_code_block(
"""score =
    tanh((gross_margin    − 0.50) / 0.20) × 12%    # benchmark: 30%~70%
  + tanh((operating_margin− 0.25) / 0.15) × 10%    # benchmark: 10%~40%
  + tanh((ROIC            − 0.175)/ 0.125)×  8%    # benchmark:  5%~30%
  + tanh((revenue_yoy     − 0.10) / 0.20) × 13%    # benchmark: -10%~30%
  + tanh((eps_yoy         − 0.10) / 0.30) × 12%    # benchmark: -20%~40%
  + tanh((fcf_margin      − 0.20) / 0.15) × 13%    # benchmark:  5%~35%
  + tanh((fcf/net_income  − 1.00) / 0.50) × 12%    # benchmark: 0.5~1.5
  + tanh(-(net_debt/ebitda− 1.00) / 2.00) × 10%    # 越低越好（取反）
  + tanh((current_ratio   − 2.00) / 1.00) × 10%    # benchmark: 1.0~3.0

cross-sectional quality_rank：24只股票按 fundamental_score 的百分位排名（0~1）"""
)

# ── 3.3 SentimentAgent ──
add_heading("3.3 SentimentAgent（Stage 1c）", 2)
add_para("由三个子类组成：NewsSentimentAnalyzer（无LLM）+ EarningsCallAnalyzer（LLM）+ SentimentAgent（综合）。")

add_heading("NewsSentimentAnalyzer（无LLM）", 3)
add_table(
    ["字段", "计算方法"],
    [
        ["滚动均分", "7日、30日、90日滑动窗口均值"],
        ["momentum", "7日均分 − 90日均分（短期加速指标）"],
        ["z_score", "(7日均分 − 90日均分) / 90日标准差"],
        ["extreme_positive/negative", "是否超过 ±1.5σ 阈值"],
        ["news_score", "综合输出，范围 −1 至 +1"],
    ],
    col_widths=[5, 10.5]
)

add_heading("EarningsCallAnalyzer（LLM）", 3)
add_para("管理层识别规则：title 含 CEO/CFO/President/Chief/Director/VP/Head/Officer 关键词。")
add_para("每季度2次 LLM 调用：(1) prepared remarks 分析（仅管理层发言）；(2) Q&A 动态分析（分析师+管理层互动）。")
add_para("季度间 delta：比较相邻季度结构化 JSON 输出，非原文对比，减少噪声。")
add_para("注意：由于 Alpha Vantage API 限制，目前仅 TSM 有有效转录数据（19季度，2015Q1–2020Q4）。")

add_heading("综合权重", 3)
add_code_block(
"""有电话会议数据: sentiment_score = 50% news_score + 50% call_score
无电话会议数据: sentiment_score = 100% news_score"""
)

# ── 3.4 SECFilingAgent ──
add_heading("3.4 SECFilingAgent（Stage 1d）", 2)
add_para("基于 SEC EDGAR 下载的原始 HTML 文件（BeautifulSoup 解析），提取 10-K/10-Q 的 MDA 和 Risk Factors 段落。")

add_heading("文本处理策略", 3)
add_table(
    ["文本段落", "处理方法"],
    [
        ["10-K MDA", "截取前 15,000 字符（覆盖战略概述部分）"],
        ["10-K Risk Factors", "原始文本为无换行长字符串 → 按句边界('. ')分块 → 每块≤10,000字 → 取前8块"],
        ["分块合成", "Map-Reduce：每块独立调用LLM → 结构化JSON → 汇总再合成"],
        ["Delta分析", "比较相邻年度的结构化JSON输出（非原文比较）"],
    ],
    col_widths=[4, 11.5]
)

add_heading("输出 Schema", 3)
add_code_block(
"""{
  "mda": {
    "tone_score",           // MDA 整体语气评分
    "business_momentum",    // 业务动能方向
    "forward_guidance",     // 前瞻性指引
    "revenue_tone",         // 收入语气
    "margin_tone",          // 利润率语气
    "management_confidence",// 管理层信心度 (0-1)
    "key_themes",           // 关键主题列表
    "forward_looking_statements", "reasoning"
  },
  "risk": {
    "risk_score",           // 综合风险评分 (0-1)
    "overall_risk_level",   // "high" | "medium" | "low"
    "top_risks",            // 前三大风险列表
    "geopolitical_risk_present", "china_revenue_risk",
    "supply_chain_risk",    "competition_risk",  "reasoning"
  },
  "delta": {
    "mda_direction",        // "improving" | "deteriorating" | "stable"
    "risk_direction",       // 同上
    "new_themes",           // 本期新出现的主题
    "dropped_themes",       // 本期消失的主题
    "key_changes",          "delta_summary",    "vs_prior_period"
  },
  "composite_score":        // −1 至 +1
  "confidence"              // 0 至 1
}""")

add_heading("composite_score 计算公式", 3)
add_code_block(
"""composite_score = tone_score × 0.50
               + (0.5 − risk_score) × 0.30
               + (confidence − 0.5) × 0.20"""
)

doc.add_page_break()

# ── 3.5 BullAgent & BearAgent ──
add_heading("3.5 BullAgent & BearAgent（Stage 2，完全并行）", 2)
add_para(
    "两个 Agent 完全对称、同时运行（ThreadPoolExecutor(max_workers=2)）。"
    "BullAgent 扮演 long-only 分析师，BearAgent 扮演专注下行风险的空头分析师。"
    "两者均使用 claude-opus-4-6 + thinking:adaptive，接收相同的4路信号，独立构建论点。"
)

add_heading("输入：4路上游信号合并", 3)
add_table(
    ["来源", "关键字段"],
    [
        ["TechnicalAgent", "momentum_score, trend_state, rsi14, roc_60d/252d, rank_60d, rs_sector_60d, golden_cross, vol_ratio, bb_pct"],
        ["FundamentalAgent", "fundamental_score, quality_rank, gross/operating_margin, revenue_yoy, fcf_margin, fcf_to_net_income, roic, rd_intensity, net_debt_to_ebitda, inventory_yoy, margin_trend"],
        ["SECFilingAgent", "tone_score, business_momentum, forward_guidance, management_confidence, risk_score, geopolitical_risk_present, top_risks, key_themes"],
        ["SentimentAgent", "news_score, momentum, z_score, extreme_positive, call_tone_score, call_forward_guidance, call_management_confidence"],
    ],
    col_widths=[3.5, 12]
)

add_heading("BullAgent 输出 Schema", 3)
add_code_block(
"""{
  "bullish_score": float,        // −1.0 到 +1.0（负值表示信号整体看空）
  "conviction_level": "high|medium|low|none",
  "top_arguments": [             // 3-5条，按 strength 降序排列
    {
      "category": "technical|fundamental|sec_filing|sentiment|cross_signal",
      "argument": "含具体数字的1-2句论点",
      "evidence_keys": ["支持该论点的指标名称"],
      "strength": float          // 0-1，该条论点的强度
    }
  ],
  "bull_catalysts": [],          // 2-4 个近期潜在催化剂
  "acknowledged_risks": [],      // 必须承认的 2-3 个风险（增强可信度）
  "debate_summary": "2-3句摘要，供辩论阶段使用",
  "data_quality_note": "数据缺口说明"
}"""
)
add_para(
    "关键约束：若大多数信号为负，bullish_score 必须接近0或为负，不允许乐观偏差。"
    "所有论点必须基于实际提供的数据，不得凭空捏造指标。BearAgent 输出结构完全对称。",
    size=10, color=(89,89,89)
)

doc.add_page_break()

# ── 3.6 ArbiterAgent ──
add_heading("3.6 ArbiterAgent（Stage 3）——多轮交叉质询辩论", 2)
add_para(
    "系统最具创新性的组件，模拟投资委员会辩论（Investment Committee）流程。"
    "ArbiterAgent 不是简单平均多空评分，而是主持真实辩论：先让双方独立陈述，"
    "再进行交叉质询，最终基于完整辩论记录（包括哪些论点在质询中存活）作出裁决。"
)

add_heading("辩论流程（默认 DEBATE_ROUNDS_DEFAULT = 1）", 3)
add_code_block(
"""初始阶段：  Bull 提交完整论点（bullish_score, top_arguments, 3-5条）
             Bear 提交完整论点（bearish_score, top_arguments, 3-5条）
                         ↓
第1轮质询：  Bear 针对 Bull 最强的3个具体论点逐一反驳（cross-examination）
                         ↓
第1轮回应：  Bull 针对 Bear 的反驳逐条回应，可以修正自己的评分和论点
                         ↓
裁决：       Arbiter 评估完整辩论记录 → 哪些论点在质询中存活 → 最终 verdict

DEBATE_ROUNDS_DEFAULT 说明：
  0 = 传统模式（仅读静态论点，不进行质询）
  1 = 一轮质询（推荐，兼顾质量与成本）
  2 = 两轮质询（更高信号质量，约 2倍 LLM 成本）"""
)

add_heading("最终 Verdict 输出 Schema", 3)
add_code_block(
"""{
  "arbiter_score":   float,    // −1 至 +1（正=看多）
  "verdict":         "strong_bull|mild_bull|neutral|mild_bear|strong_bear",
  "confidence":      float,    // 0-1，仲裁人对结论的置信度
  "bull_wins_on":    [],       // 多头论点在质询后胜出的类别
  "bear_wins_on":    [],       // 空头论点胜出的类别
  "contested_points":[],       // 两方均有有效论据的争议点
  "argument_survivability": {
    "category": "bull_prevailed|bear_prevailed|contested|both_conceded"
  },
  "resolution_summary": "2-3句，说明辩论如何被裁决",
  "key_risk":        str,      // DecisionAgent 必须关注的最重要风险
  "agreement_level": "aligned|mixed|divergent",
  "flags":           [],       // e.g. ["data_gap","high_disagreement","low_confidence"]
  "debate_transcript": [],     // 完整辩论记录 [{round, speaker, content}]
  "rounds_conducted": int
}"""
)

doc.add_page_break()

# ── 3.7 MacroAgent ──
add_heading("3.7 MacroAgent（Stage 4）", 2)
add_para(
    "宏观信号在所有24只股票间共享（每日只计算一次），但不同子行业对宏观环境的敏感度不同，"
    "需分别评估。量化信号计算为纯 Python（无 LLM），LLM 负责子行业解读和调整量裁决。"
)

add_heading("量化宏观信号计算", 3)
add_table(
    ["信号", "计算逻辑"],
    [
        ["利率环境\n(rate_env, −1~+1)", "FFR>5% → base=−0.8；FFR<2.5% → base=+0.7；"
         "3个月变化方向：下降>0.5ppt → +0.3调整，上升>0.5ppt → −0.2调整"],
        ["通胀趋势\n(infl_trend, −1~+1)", "CPI YoY<2% → +0.8；>4% → −0.8；"
         "3个月年化与年度差值：gap<−1% → +0.2（减速），gap>+1% → −0.2（加速）"],
        ["GDP动能\n(gdp_momentum, −1~+1)", "QoQ年化>3% → +0.8；<0 → −0.5；"
         "季度加速/减速额外 ±0.1"],
        ["需求信号\n(demand_signal, −1~+1)", "(耐用品3个月变化% + 零售3个月变化%) / 2 / 5.0，截断至±1"],
        ["宏观综合评分", "GDP×35% + 通胀×25% + 利率×25% + 需求×15%；"
         "若收益率曲线倒挂（10Y−2Y < −0.3ppt）额外 −0.15"],
    ],
    col_widths=[3.8, 11.7]
)

add_heading("宏观周期分类", 3)
add_code_block(
"""regime_composite_score > +0.3 → "expansion"（扩张）
regime_composite_score  0~+0.3  → "peak"    （顶峰）
regime_composite_score −0.3~0   → "contraction"（收缩）
regime_composite_score < −0.3   → "trough"  （底部）"""
)

add_heading("子行业宏观敏感度（LLM 评估）", 3)
add_table(
    ["子行业", "宏观敏感度特征"],
    [
        ["ai_datacenter", "由超大规模 AI 资本开支驱动，轻度衰退中较抗跌；但对高利率极度敏感（高PE估值）"],
        ["equipment", "最高周期性：资本开支削减时最先被砍（滞后1-2季度）；复苏时也最先反弹"],
        ["analog", "工业/汽车终端市场，产品周期长，波动较数字芯片小，但仍有周期性"],
        ["memory", "最波动：供应过剩时价格崩溃，需求复苏时大幅反弹；对需求信号极其敏感"],
        ["foundry", "使用率驱动，跟随整体芯片需求变化，中等周期性"],
    ],
    col_widths=[3.5, 12]
)
add_para("LLM 输出 subsector_adjustment（−0.3 至 +0.3），最终：macro_adjusted_score = arbiter_score + subsector_adjustment（截断至±1）")

doc.add_page_break()

# ── 3.8 RiskControlAgent ──
add_heading("3.8 RiskControlAgent（Stage 5）", 2)
add_para(
    "接收 MacroAgent 的 macro_adjusted_score，叠加六维量化风险评分（纯 Python），"
    "最终由 LLM 综合裁定交易决策（buy/sell/hold）和仓位规模。"
)

add_heading("六维风险评分", 3)
add_table(
    ["维度", "计算方法", "权重"],
    [
        ["波动率风险", "HV_20d×50% + ATR_pct×30% + BB_width×20%；\nHV>65%→1.0，<25%→0.0", "20%"],
        ["回撤风险", "负ROC_20d×50% + 负ROC_60d×30% + 负ROC_252d×20%;\n20日>20%跌幅→满分", "20%"],
        ["尾部风险", "VIX spike(+0.4) + 危险宏观标志×0.15/个\n+ Arbiter高分歧/低置信×0.15/项", "15%"],
        ["流动性风险", "vol_ratio<0.5→0.7（过薄）；>3.0→0.6（恐慌量）;\n正常区间线性插值", "10%"],
        ["市场风险", "VIX体制评分×55% + 宏观体制评分×45%", "25%"],
        ["信号置信风险", "(1−confidence)×60% + 分歧惩罚×30% + 近零信号×10%", "10%"],
    ],
    col_widths=[3.5, 9.5, 2.5]
)

add_heading("仓位规模规则", 3)
add_table(
    ["综合风险评分", "最大仓位比例"],
    [
        ["< 0.25", "最高 0.9"],
        ["0.25 – 0.45", "0.5 – 0.75"],
        ["0.45 – 0.65", "0.2 – 0.5"],
        ["> 0.65", "≤ 0.2，强烈考虑 hold"],
    ],
    col_widths=[5.5, 10]
)

add_heading("交易决策规则（LLM 最终裁定）", 3)
add_code_block(
"""macro_adjusted_score > +0.20 → BUY 候选
macro_adjusted_score < −0.20 → SELL 候选
|macro_adjusted_score| < 0.20 → HOLD

强制 HOLD 条件（任一触发）：
  - VIX regime = "extreme"
  - GDP 负增长 + 收益率曲线倒挂 同时触发
  - Arbiter high_disagreement + low_confidence 同时出现"""
)

# ── 3.9 DecisionAgent ──
add_heading("3.9 DecisionAgent（Stage 6）——全局组合优化", 2)
add_para(
    "系统中唯一拥有全局视角的 Agent：同时看全部24只股票，进行资金分配优化。"
    "采用三层架构：Python预处理 → LLM裁决目标仓位 → Python生成具体订单。"
)

add_heading("组合硬约束", 3)
add_table(
    ["约束类型", "数值"],
    [
        ["单股最大仓位", "15%"],
        ["最低现金保留", "5%"],
        ["单一子行业最大仓位", "35%（ai_datacenter / equipment / analog / memory / foundry）"],
        ["初始资本", "$100,000 USD"],
        ["禁止做空", "所有 position ≥ 0"],
        ["每日新开仓上限", "5只"],
        ["每日全平仓位上限", "3只（非风险触发时）"],
        ["所有分配比例之和", "必须精确等于 1.00（含现金）"],
    ],
    col_widths=[5, 10.5]
)

add_heading("交易订单生成逻辑", 3)
add_code_block(
"""target_value  = target_pct × total_portfolio_value
current_value = current_shares × current_price
delta_value   = target_value − current_value

delta > 0  → BUY  (delta / price) 股（向下取整）
delta < 0  → SELL (|delta| / price) 股（向上取整）
|delta| < $50 → 忽略（避免碎单）
target_pct = 0 且当前持有 → 全部卖出

执行价格模式（可切换）：
  "close"      — T日收盘价（MOC订单，适合日频模拟）
  "next_open"  — T+1日开盘价（更接近真实，无前视偏差）"""
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# 四、Pipeline流程与并行策略
# ══════════════════════════════════════════════════════════════════════════════
add_heading("四、Pipeline 流程与并行策略", 1)

add_heading("4.1 运行入口", 2)
add_code_block(
"""from pipeline import run_pipeline

# 全部24只股票
run_pipeline(date_str="2021-03-01")

# 指定股票测试
run_pipeline("2021-03-01", tickers=["NVDA", "AMD"])

# 强制重新调用LLM，忽略缓存
run_pipeline("2021-03-01", force=True)

# 从某阶段起跑（之前阶段读缓存）
run_pipeline("2021-03-01", start_from="macro")

# 命令行
python pipeline.py 2021-03-01 NVDA AMD --force --from=arbiter"""
)

add_heading("4.2 并行策略", 2)
add_table(
    ["阶段", "并行方式", "说明"],
    [
        ["Stage 1a–1c Technical/Fundamental/Sentiment", "ThreadPoolExecutor(MAX_WORKERS=3)", "每阶段3个ticker同时处理"],
        ["Stage 1d SEC Filing", "workers=1（串行）", "每个ticker内部已有8个chunk顺序调用，避免API过载"],
        ["Stage 2 Bull+Bear", "stage级并行：2个线程", "Bull和Bear同时运行，实际并发=3×2=6"],
        ["Stage 3–6", "ThreadPoolExecutor(MAX_WORKERS=3)", "Arbiter/Macro/Risk各3并发"],
        ["Stage 4 MacroAgent", "宏观信号共享", "全行业信号只计算一次，所有ticker复用"],
    ],
    col_widths=[5.5, 4.5, 5.5]
)

add_heading("4.3 缓存机制", 2)
add_para("所有 Agent 均按 {ticker}_{date}.json 格式缓存结果，重复运行时直接读缓存，不重复调用 LLM。这使得：")
add_para("① 回测成本可控（历史数据仅付费一次）")
add_para("② 可从任意 Stage 断点续跑")
add_para("③ force=True 时可对特定阶段重新分析")

doc.add_paragraph()
add_heading("缓存目录结构", 3)
add_code_block(
"""d:/Investment_AI_agent/
├── technical_cache/{TICKER}/signal_{DATE}.json
├── fundamental_cache/{TICKER}/signal_{DATE}.json
├── sentiment_cache/{TICKER}/composite_{DATE}.json
├── sec_analysis/{TICKER}/10K_{YEAR}.json  &  10Q_{YEAR}Q{Q}.json
├── bull_cache/{TICKER}/bull_{DATE}.json
├── bear_cache/{TICKER}/bear_{DATE}.json
├── arbiter_cache/{TICKER}/arbiter_{DATE}.json  &  debate_{DATE}.json
├── macro_cache/{TICKER}/macro_{DATE}.json
├── risk_cache/{TICKER}/risk_{DATE}.json
├── decision_cache/decision_{DATE}.json
└── portfolio/portfolio_state.json"""
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# 五、LLM配置
# ══════════════════════════════════════════════════════════════════════════════
add_heading("五、LLM 调用参数配置", 1)

add_table(
    ["参数", "值", "说明"],
    [
        ["模型", "claude-opus-4-6", "Anthropic 最强推理模型"],
        ["思考模式", "thinking: {type: adaptive}", "自适应extended thinking，复杂问题自动增加思考深度"],
        ["输出方式", "client.messages.stream()", "流式接收，取 get_final_message()"],
        ["JSON解析", "_extract_json()", "处理 markdown fence 包裹，取 { 到最后 } 之间内容"],
        ["重试策略", "最多5次，指数退避", "等待时间：2, 4, 8, 16, 32 秒"],
        ["max_tokens (Bull/Bear)", "3000", "论点较详细"],
        ["max_tokens (Arbiter)", "4000（质询）+ 2000（裁决）", "辩论记录较长"],
        ["max_tokens (Macro/Risk)", "2000", "结构化输出"],
        ["max_tokens (Decision)", "3000", "24只股票分配表"],
    ],
    col_widths=[4.5, 5.5, 5.5]
)

add_heading("LLM 角色定义（各Agent System Prompt摘要）", 2)
add_table(
    ["Agent", "角色定位"],
    [
        ["BullAgent", "专注半导体的 long-only 分析师，只引用数据中明确存在的证据，必须承认风险"],
        ["BearAgent", "专注下行风险的空头分析师，重点挖掘估值、周期、地缘政治等风险"],
        ["ArbiterAgent", "投资委员会主席，主持交叉质询，基于论点存活度而非分数平均作裁决"],
        ["MacroAgent", "半导体行业高级宏观策略师，精通各子行业对宏观周期的差异性响应"],
        ["RiskControlAgent", "风控官，不产生新信号，专注下行保护，保守优于进取"],
        ["DecisionAgent", "100,000美元半导体组合经理，全局优化资金分配，遵守硬约束"],
    ],
    col_widths=[3.5, 12]
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# 六、信号流动
# ══════════════════════════════════════════════════════════════════════════════
add_heading("六、端到端信号流动", 1)
add_code_block(
"""TechnicalAgent   → momentum_score(-1~+1), trend_state, rsi14, roc_*, rank_*, rs_*
FundamentalAgent → fundamental_score(-1~+1), margin_trend, revenue_yoy, fcf_margin,
                   inventory_yoy, quality_rank
SECFilingAgent   → composite_score(-1~+1), tone_score, risk_score, delta_direction,
                   key_themes, geopolitical_risk_present
SentimentAgent   → sentiment_score(-1~+1)  [= news_score 或 50% news + 50% call]
        ↓  4路信号 → 每只股票
BullAgent        → bullish_score, top_arguments(3-5条), catalysts, risks
BearAgent        → bearish_score, top_arguments(3-5条), risk_factors
        ↓  辩论
ArbiterAgent     → arbiter_score, verdict, confidence, argument_survivability_map
        ↓  宏观调整
MacroAgent       → macro_adjusted_score = arbiter_score + subsector_adjustment(±0.3)
        ↓  风险控制
RiskControlAgent → final_decision(buy/sell/hold), position_size(0~1), risk_score(0~1),
                   6维风险评分
        ↓  全局优化（24只同时）
DecisionAgent    → target_allocations{ticker: pct, cash: pct},
                   orders[{action, shares, price, dollar_amount}],
                   market_stance(aggressive/neutral/defensive)"""
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# 七、数据局限性
# ══════════════════════════════════════════════════════════════════════════════
add_heading("七、已知数据局限性", 1)
add_table(
    ["问题", "影响与处理"],
    [
        ["财报电话会议仅 TSM 有效数据",
         "EarningsCallAnalyzer 对其他23只均退化为100%新闻情绪；TSM call_score权重50%，其余0%"],
        ["5只股票无新闻数据\n（ASML, MKSI, SLAB, CRUS, SMTC）",
         "SentimentAgent 对这5只信号较弱；news_score 回退至0（中性），需补充其他信号"],
        ["TSM 新闻密度低\n（部分时期每月仅1篇）",
         "短期滚动窗口（7日）信号噪声大，z_score 可靠性降低"],
        ["VIX 数据仅从2020年开始",
         "2010–2019年回测时 RiskControlAgent 的市场风险评分退化（vix_latest=None → 默认0.5）"],
        ["SEC Risk Factors 为无换行长字符串",
         "必须按句边界（'. '）分块，段落分块会失效；已在 SECFilingAgent 中处理"],
        ["point-in-time 对齐",
         "FundamentalAgent 严格执行季报+45天/年报+60天的发布延迟，避免前视偏差"],
    ],
    col_widths=[4.5, 11]
)

# ══════════════════════════════════════════════════════════════════════════════
# 八、文件清单
# ══════════════════════════════════════════════════════════════════════════════
doc.add_page_break()
add_heading("八、核心文件清单", 1)
add_table(
    ["文件", "功能"],
    [
        ["pipeline.py", "主流水线入口，协调7个Stage的串并行执行"],
        ["model.py", "数据采集脚本：股价/新闻/财报电话/SEC/财务报表/宏观指标下载"],
        ["technical_agent.py", "TechnicalAgent：OHLCV → 技术指标 → momentum_score"],
        ["fundamental_agent.py", "FundamentalAgent：三张报表 → 财务指标 → fundamental_score"],
        ["sentiment_agent.py", "SentimentAgent（含NewsAnalyzer + EarningsCallAnalyzer）"],
        ["sec_filing_agent.py", "SECFilingAgent：10-K/10-Q → MDA/Risk LLM分析 → composite_score"],
        ["bull_agent.py", "BullAgent：4路信号 → 多头论点 + bullish_score"],
        ["bear_agent.py", "BearAgent：4路信号 → 空头论点 + bearish_score"],
        ["arbiter_agent.py", "ArbiterAgent：多轮辩论 → 裁决 + arbiter_score"],
        ["macro_agent.py", "MacroAgent：量化宏观信号 + 子行业LLM调整 → macro_adjusted_score"],
        ["risk_control_agent.py", "RiskControlAgent：六维风险评分 → 仓位大小 + 最终决策"],
        ["decision_agent.py", "DecisionAgent：全局组合优化 → 目标仓位 → 具体订单"],
        ["record_agent.py", "RecordAgent：执行订单 + 更新持仓 + EOD快照"],
        ["backtesting.py", "回测框架（已有基础，待完善）"],
        ["parse_10k.py / parse_10q.py", "BeautifulSoup 解析 SEC EDGAR HTML → MDA/Risk文本"],
        ["fetch_financials.py", "财务报表批量下载脚本"],
    ],
    col_widths=[5, 10.5]
)

# 保存文档
out_path = "d:/Investment_AI_agent/Investment_AI_Agent_Report.docx"
doc.save(out_path)
print(f"Report saved: {out_path}")
