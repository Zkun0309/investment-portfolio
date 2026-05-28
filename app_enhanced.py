"""
多项目投资组合选择系统 — 完整增强版
课程：智能决策理论与方法 · 第二次作业 · 选题方向四
整合：输入/计算/推荐/解释/验证 五大模块
扩展：互斥约束 + 项目依赖 + 多目标权衡 + 敏感性分析
"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import copy

# ═══════════════════════════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="投资组合智能选择系统",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# Apple 风格样式 (保持原有精美UI)
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --apple-blue: #0071E3;
        --apple-blue-hover: #0077ED;
        --apple-gray: #86868b;
        --apple-dark: #1D1D1F;
        --apple-light: #F5F5F7;
        --apple-border: #D2D2D7;
        --apple-green: #34C759;
        --apple-red: #FF3B30;
        --apple-orange: #FF9500;
        --apple-purple: #AF52DE;
    }

    .stApp {
        background: linear-gradient(180deg, #FFFFFF 0%, #F5F5F7 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--apple-dark) !important;
        font-weight: 600 !important;
    }

    .apple-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid var(--apple-border);
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08), 0 4px 24px rgba(0,0,0,0.04);
        margin: 12px 0;
    }

    .metric-card {
        background: white;
        border: 1px solid var(--apple-border);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 16px rgba(0,0,0,0.08);
    }

    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: var(--apple-dark);
        margin: 8px 0;
    }

    .metric-label {
        font-size: 14px;
        color: var(--apple-gray);
        font-weight: 500;
    }

    .project-card {
        background: white;
        border: 1px solid var(--apple-border);
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 0 2px 16px rgba(0,0,0,0.08);
        transition: all 0.2s ease;
    }

    .project-card:hover {
        box-shadow: 0 4px 24px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }

    .recommendation-box {
        background: white;
        border: 1px solid var(--apple-border);
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    }

    .recommendation-title {
        font-size: 14px;
        color: var(--apple-blue);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }

    .recommendation-summary {
        font-size: 22px;
        font-weight: 600;
        color: var(--apple-dark);
        line-height: 1.4;
        margin-bottom: 16px;
    }

    .method-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }

    .greedy-badge {
        background: rgba(255,149,0,0.12);
        color: var(--apple-orange);
        border: 1px solid rgba(255,149,0,0.3);
    }

    .dp-badge {
        background: rgba(0,113,227,0.12);
        color: var(--apple-blue);
        border: 1px solid rgba(0,113,227,0.3);
    }

    .exhaustive-badge {
        background: rgba(52,199,89,0.12);
        color: var(--apple-green);
        border: 1px solid rgba(52,199,89,0.3);
    }

    .selected-tag {
        background: var(--apple-green);
        color: white;
        padding: 4px 10px;
        border-radius: 10px;
        font-size: 12px;
        font-weight: 600;
    }

    .rejected-tag {
        background: #F5F5F7;
        color: var(--apple-gray);
        padding: 4px 10px;
        border-radius: 10px;
        font-size: 12px;
        font-weight: 500;
        border: 1px solid var(--apple-border);
    }

    .npv-positive { color: var(--apple-green); }
    .npv-negative { color: var(--apple-red); }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        background: white;
        border: 1px solid var(--apple-border) !important;
        border-radius: 10px;
        padding: 12px 20px;
        font-weight: 500;
        color: var(--apple-gray);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0,113,227,0.06);
        color: var(--apple-blue);
        border-color: var(--apple-blue) !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--apple-blue) !important;
        border-color: var(--apple-blue) !important;
        color: white !important;
    }

    .apple-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--apple-border), transparent);
        margin: 24px 0;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 核心计算函数
# ═══════════════════════════════════════════════════════════════

def calc_npv(rate, cashflows):
    """计算净现值 NPV"""
    return float(npf.npv(rate, cashflows))

def calc_irr(cashflows):
    """计算内部收益率 IRR"""
    try:
        result = npf.irr(cashflows)
        if np.isnan(result):
            return None
        return float(result)
    except Exception:
        return None

def calc_npvr(npv, cost):
    """计算净现值率"""
    if cost == 0:
        return 0.0
    return npv / cost

def calc_payback(cashflows):
    """计算静态回收期（年）"""
    cumulative = 0
    investment = abs(cashflows[0])
    for i, cf in enumerate(cashflows[1:], start=1):
        cumulative += cf
        if cumulative >= investment:
            prev = cumulative - cf
            fraction = (investment - prev) / cf if cf > 0 else 0
            return i - 1 + fraction
    return float("inf")

def analyze_project(project, rate):
    """分析单个项目"""
    cf = project["cashflows"]
    cost = project["cost"]
    npv = calc_npv(rate, cf)
    irr = calc_irr(cf)
    npvr = calc_npvr(npv, cost)
    payback = calc_payback(cf)
    return {
        "id": project["id"],
        "name": project["name"],
        "cost": cost,
        "npv": round(npv, 2),
        "irr": round(irr * 100, 2) if irr is not None else None,
        "npvr": round(npvr, 4),
        "payback": round(payback, 2) if payback != float("inf") else None,
    }

def analyze_all_projects(projects, rate):
    """分析所有项目，按NPVR排序"""
    results = [analyze_project(p, rate) for p in projects]
    results.sort(key=lambda x: x["npvr"], reverse=True)
    return results

# ═══════════════════════════════════════════════════════════════
# 算法1: 贪心法 (NPVR排序)
# ═══════════════════════════════════════════════════════════════

def greedy_selection(projects, budget, rate):
    """按NPVR排序贪心选择"""
    analyzed = analyze_all_projects(projects, rate)
    proj_map = {p["id"]: p for p in projects}
    selected = []
    remaining = budget
    for info in analyzed:
        pid = info["id"]
        cost = proj_map[pid]["cost"]
        if cost <= remaining:
            selected.append(info)
            remaining -= cost
    total_cost = sum(p["cost"] for p in selected)
    total_npv = sum(p["npv"] for p in selected)
    return {
        "method": "贪心法（NPVR排序）",
        "selected": selected,
        "total_cost": round(total_cost, 2),
        "total_npv": round(total_npv, 2),
        "remaining_budget": round(remaining, 2),
        "budget_utilization": round(total_cost / budget * 100, 1) if budget > 0 else 0,
    }

# ═══════════════════════════════════════════════════════════════
# 算法2: 动态规划 (0-1背包, 仅预算约束)
# ═══════════════════════════════════════════════════════════════

def dp_selection(projects, budget, rate):
    """0-1背包DP，仅在预算约束下找最优"""
    analyzed = {p["id"]: analyze_project(p, rate) for p in projects}
    unit = 10000
    W = int(budget // unit)
    n = len(projects)
    costs = [int(p["cost"] // unit) for p in projects]
    values = [analyzed[p["id"]]["npv"] for p in projects]

    dp = [[0.0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        c = costs[i - 1]
        v = values[i - 1]
        for w in range(W + 1):
            if c <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - c] + v)
            else:
                dp[i][w] = dp[i - 1][w]

    selected_ids = []
    w = W
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_ids.append(projects[i - 1]["id"])
            w -= costs[i - 1]

    selected = [analyzed[pid] for pid in selected_ids]
    total_cost = sum(p["cost"] for p in selected)
    total_npv = sum(p["npv"] for p in selected)
    return {
        "method": "动态规划（全局最优·仅预算约束）",
        "selected": selected,
        "total_cost": round(total_cost, 2),
        "total_npv": round(total_npv, 2),
        "remaining_budget": round(budget - total_cost, 2),
        "budget_utilization": round(total_cost / budget * 100, 1) if budget > 0 else 0,
    }

# ═══════════════════════════════════════════════════════════════
# 算法3: 穷举搜索 (预算 + 互斥 + 依赖 约束)
# ═══════════════════════════════════════════════════════════════

def exhaustive_selection(projects, budget, rate,
                         mutual_exclusive_groups=None,
                         dependencies=None,
                         min_projects=0,
                         max_projects=None):
    """
    穷举搜索最优投资组合（处理所有约束条件）
    - mutual_exclusive_groups: [[P1,P2], [P3,P4]] 每组最多选1个
    - dependencies: [(P2, P1)] 选P2必须先选P1
    """
    if max_projects is None:
        max_projects = len(projects)

    analyzed = {p["id"]: analyze_project(p, rate) for p in projects}
    proj_cost = {p["id"]: p["cost"] for p in projects}
    n = len(projects)
    proj_ids = [p["id"] for p in projects]

    best_npv = -float("inf")
    best_combo = []
    best_cost = 0

    for mask in range(1 << n):
        selected = []
        total_cost = 0.0
        valid = True

        for i in range(n):
            if mask & (1 << i):
                selected.append(proj_ids[i])
                total_cost += proj_cost[proj_ids[i]]

        # 预算约束
        if total_cost > budget:
            continue

        # 数量约束
        if len(selected) < min_projects:
            continue
        if len(selected) > max_projects:
            continue

        # 互斥约束
        if mutual_exclusive_groups:
            for group in mutual_exclusive_groups:
                count = sum(1 for pid in selected if pid in group)
                if count > 1:
                    valid = False
                    break
        if not valid:
            continue

        # 依赖约束
        if dependencies:
            for dep_a, dep_b in dependencies:
                if dep_a in selected and dep_b not in selected:
                    valid = False
                    break
        if not valid:
            continue

        # 计算总NPV
        total_npv = sum(analyzed[pid]["npv"] for pid in selected)
        if total_npv > best_npv:
            best_npv = total_npv
            best_combo = selected[:]
            best_cost = total_cost

    if not best_combo:
        return {
            "method": "穷举搜索（含全部约束）",
            "selected": [],
            "total_cost": 0,
            "total_npv": 0,
            "remaining_budget": budget,
            "budget_utilization": 0,
        }

    selected_info = [analyzed[pid] for pid in best_combo]
    return {
        "method": "穷举搜索（含全部约束）",
        "selected": selected_info,
        "total_cost": round(best_cost, 2),
        "total_npv": round(best_npv, 2),
        "remaining_budget": round(budget - best_cost, 2),
        "budget_utilization": round(best_cost / budget * 100, 1) if budget > 0 else 0,
    }

# ═══════════════════════════════════════════════════════════════
# 多目标评分
# ═══════════════════════════════════════════════════════════════

def multi_objective_score(projects, rate, risk_scores, strategic_scores,
                          w_npv=0.5, w_risk=0.2, w_strategic=0.3):
    """综合NPV、风险、战略价值的多目标评分"""
    analyzed = {p["id"]: analyze_project(p, rate) for p in projects}
    scores = {}
    for pid in analyzed:
        npv_norm = max(0, analyzed[pid]["npv"])
        risk_norm = risk_scores.get(pid, 5)  # 1-10, 10最安全
        strategic_norm = strategic_scores.get(pid, 5)  # 1-10
        score = (w_npv * npv_norm / 10000 +
                 w_risk * risk_norm +
                 w_strategic * strategic_norm)
        scores[pid] = round(score, 2)
    return scores

# ═══════════════════════════════════════════════════════════════
# 敏感性分析
# ═══════════════════════════════════════════════════════════════

def sensitivity_rate(projects, budget, rate_range, steps, constraints=None):
    """折现率敏感性分析"""
    if constraints is None:
        constraints = {}
    rates = np.linspace(rate_range[0], rate_range[1], steps)
    results = []
    for r in rates:
        g = greedy_selection(projects, budget, float(r))
        d = dp_selection(projects, budget, float(r))
        e = exhaustive_selection(projects, budget, float(r),
                                 constraints.get("mutual_exclusive"),
                                 constraints.get("dependencies"))
        results.append({
            "rate_pct": round(float(r) * 100, 2),
            "greedy_npv": g["total_npv"],
            "dp_npv": d["total_npv"],
            "exhaustive_npv": e["total_npv"],
            "exhaustive_projects": [p["name"] for p in e["selected"]],
        })
    return results

def sensitivity_budget(projects, rate, budget_range, steps, constraints=None):
    """预算敏感性分析"""
    if constraints is None:
        constraints = {}
    budgets = np.linspace(budget_range[0], budget_range[1], steps)
    results = []
    for b in budgets:
        g = greedy_selection(projects, float(b), rate)
        d = dp_selection(projects, float(b), rate)
        e = exhaustive_selection(projects, float(b), rate,
                                 constraints.get("mutual_exclusive"),
                                 constraints.get("dependencies"))
        results.append({
            "budget_wan": round(b / 10000, 1),
            "greedy_npv": g["total_npv"],
            "dp_npv": d["total_npv"],
            "exhaustive_npv": e["total_npv"],
        })
    return results

# ═══════════════════════════════════════════════════════════════
# 推荐引擎
# ═══════════════════════════════════════════════════════════════

def get_recommendation(greedy, dp, exhaustive, budget):
    """对比三种算法，生成推荐结论"""
    g_npv = greedy["total_npv"]
    d_npv = dp["total_npv"]
    e_npv = exhaustive["total_npv"]

    if e_npv >= d_npv and e_npv > 0:
        rec = exhaustive
        method_tag = "穷举搜索（含全部约束）· 推荐"
    elif d_npv >= g_npv:
        rec = dp
        method_tag = "动态规划 · 推荐"
    else:
        rec = greedy
        method_tag = "贪心法 · 推荐"

    rec_names = [p["name"] for p in rec["selected"]]

    reasoning = []
    if e_npv > d_npv:
        reasoning.append(f"考虑互斥和依赖约束后，最优组合总NPV为 {e_npv:,.0f} 元。")
    elif d_npv > g_npv:
        reasoning.append(f"动态规划在预算 {budget/10000:.0f} 万元约束下找到了理论最优解。")
    else:
        reasoning.append("在本案例参数下，三种算法得出相同的最优组合。")

    reasoning.append(f"推荐组合总投资 {rec['total_cost']/10000:.1f} 万元，预算利用率 {rec['budget_utilization']}%。")

    if d_npv > g_npv and d_npv > 0:
        reasoning.append(f"贪心法可能损失 {d_npv - g_npv:,.0f} 元NPV，建议优先参考精确算法结果。")

    if not rec["selected"]:
        summary = "在当前约束条件下，没有可行的投资组合。"
        warning = "请检查预算设置和项目约束条件。"
    else:
        budget_wan = budget / 10000
        summary = (f"预算 {budget_wan:.0f} 万元下，建议选择【{'、'.join(rec_names)}】，"
                   f"总投资 {rec['total_cost']/10000:.1f} 万元，预期总NPV {rec['total_npv']:,.0f} 元。")
        warning = ""

    return {
        "recommended_method": rec["method"],
        "recommended_projects": rec_names,
        "summary": summary,
        "reasoning": reasoning,
        "warning": warning,
        "npv_gain": e_npv - g_npv,
    }

# ═══════════════════════════════════════════════════════════════
# 格式化工具函数
# ═══════════════════════════════════════════════════════════════

def fmt_wan(value):
    """格式化为万元显示"""
    if value is None:
        return "N/A"
    return f"{value/10000:.1f}万"

def fmt_yuan(value):
    """格式化为元显示"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:,.0f}元"

def fmt_pct(value):
    """格式化百分比"""
    if value is None:
        return "N/A"
    return f"{value:.2f}%"

# ═══════════════════════════════════════════════════════════════
# 默认项目数据
# ═══════════════════════════════════════════════════════════════

DEFAULT_PROJECTS = [
    {"id": "P1", "name": "指数基金定投", "cost": 100000,
     "cashflows": [-100000, 18000, 20000, 22000, 24000, 26000], "years": 5},
    {"id": "P2", "name": "债券型基金", "cost": 80000,
     "cashflows": [-80000, 10000, 10000, 10000, 10000, 90000], "years": 5},
    {"id": "P3", "name": "REITs房地产信托", "cost": 150000,
     "cashflows": [-150000, 12000, 14000, 16000, 18000, 170000], "years": 5},
    {"id": "P4", "name": "黄金定投", "cost": 60000,
     "cashflows": [-60000, 5000, 6000, 8000, 9000, 72000], "years": 5},
    {"id": "P5", "name": "储蓄型保险", "cost": 50000,
     "cashflows": [-50000, 4000, 4000, 4000, 4000, 58000], "years": 5},
    {"id": "P6", "name": "科技股ETF", "cost": 120000,
     "cashflows": [-120000, 8000, 12000, 20000, 28000, 160000], "years": 5},
]

# ═══════════════════════════════════════════════════════════════
# Session State 初始化
# ═══════════════════════════════════════════════════════════════

if "projects" not in st.session_state:
    st.session_state.projects = copy.deepcopy(DEFAULT_PROJECTS)
if "mutual_groups" not in st.session_state:
    st.session_state.mutual_groups = []  # 如 [["P1","P2"], ["P3","P4"]]
if "dependencies" not in st.session_state:
    st.session_state.dependencies = []  # 如 [("P2","P1")] 选P2必须先选P1

# ═══════════════════════════════════════════════════════════════
# 侧边栏 - 输入模块
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 输入模块")

    # 预算和折现率
    budget_wan = st.slider("预算限制（万元）", 10, 100, 30, 5)
    rate_pct = st.slider("折现率", 3, 15, 8, 1, format="%d%%")

    st.markdown("---")

    # 项目管理
    st.markdown("### 项目管理")

    tab_add, tab_constraints = st.tabs(["添加/编辑项目", "约束设置"])

    with tab_add:
        st.markdown("#### 添加新项目")
        with st.form("add_project_form"):
            pname = st.text_input("项目名称", placeholder="例如：新能源基金")
            pcost = st.number_input("初始投资（万元）", 1, 1000, 10, 1)
            pyears = st.number_input("项目期限（年）", 1, 20, 5, 1)
            pcf_str = st.text_input(
                "各年净现金流（万元，逗号分隔）",
                placeholder="例如：-10, 2, 3, 4, 5, 6")
            submitted = st.form_submit_button("添加项目", use_container_width=True)
            if submitted:
                if not pname:
                    st.error("请输入项目名称")
                elif not pcf_str:
                    st.error("请输入各年净现金流")
                else:
                    try:
                        # 支持中英文逗号
                        pcf_cleaned = pcf_str.replace("，", ",")
                        cf_parts = [float(x.strip()) for x in pcf_cleaned.split(",")]
                        if len(cf_parts) >= 2:
                            cost_yuan = int(round(pcost * 10000))
                            cfs_yuan = [int(round(x * 10000)) for x in cf_parts]
                            if abs(cfs_yuan[0] + cost_yuan) > 100:
                                cfs_yuan[0] = -cost_yuan
                            # 生成不重复的ID
                            existing_ids = {p["id"] for p in st.session_state.projects}
                            new_id_num = 1
                            while f"P{new_id_num}" in existing_ids:
                                new_id_num += 1
                            new_id = f"P{new_id_num}"
                            new_proj = {
                                "id": new_id,
                                "name": pname,
                                "cost": cost_yuan,
                                "cashflows": cfs_yuan,
                                "years": int(pyears),
                            }
                            st.session_state.projects.append(new_proj)
                            st.success(f"已添加：{pname}")
                            st.rerun()
                        else:
                            st.error("现金流至少需要2个值（含初始投资）")
                    except ValueError:
                        st.error("现金流格式错误，请使用逗号分隔的数字（如：-10, 2, 3, 4, 5, 6）")

        st.markdown("#### 当前项目列表")
        for i, p in enumerate(st.session_state.projects):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"**{p['name']}** | 投资: {fmt_wan(p['cost'])} | 期限: {p['years']}年")
            with col2:
                if st.button("删除", key=f"del_{p['id']}", use_container_width=True):
                    st.session_state.projects.pop(i)
                    st.rerun()

    with tab_constraints:
        st.markdown("#### 互斥项目组")
        st.caption("同组内最多选1个，每组用逗号分隔项目ID（如：P1,P2）")
        mg_input = st.text_input("互斥组输入", placeholder="P1,P2")
        if st.button("添加互斥组"):
            mg = [x.strip() for x in mg_input.split(",") if x.strip()]
            if len(mg) >= 2:
                st.session_state.mutual_groups.append(mg)
                st.success(f"已添加互斥组: {mg}")
                st.rerun()
        if st.session_state.mutual_groups:
            for i, mg in enumerate(st.session_state.mutual_groups):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.caption(f"互斥组{i+1}: {' 与 '.join(mg)} 最多选1个")
                with col2:
                    if st.button("移除", key=f"rm_mg_{i}", use_container_width=True):
                        st.session_state.mutual_groups.pop(i)
                        st.rerun()

        st.markdown("#### 项目依赖关系")
        st.caption("格式：被依赖项目ID,依赖项目ID（如：P1,P2 表示选P2必须先选P1）")
        dep_input = st.text_input("依赖关系输入", placeholder="P1,P2")
        if st.button("添加依赖关系"):
            parts = [x.strip() for x in dep_input.split(",") if x.strip()]
            if len(parts) == 2:
                st.session_state.dependencies.append((parts[1], parts[0]))
                st.success(f"已添加: 选{parts[1]}必须先选{parts[0]}")
                st.rerun()
        if st.session_state.dependencies:
            for i, (dep_on, dep_need) in enumerate(st.session_state.dependencies):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.caption(f"依赖{i+1}: 选{dep_on}必须先选{dep_need}")
                with col2:
                    if st.button("移除", key=f"rm_dep_{i}", use_container_width=True):
                        st.session_state.dependencies.pop(i)
                        st.rerun()

        if st.button("重置所有约束"):
            st.session_state.mutual_groups = []
            st.session_state.dependencies = []
            st.rerun()

    if st.button("恢复默认项目数据"):
        st.session_state.projects = copy.deepcopy(DEFAULT_PROJECTS)
        st.session_state.mutual_groups = []
        st.session_state.dependencies = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:12px;color:var(--apple-gray);line-height:1.6;">
    <b>算法说明</b><br>
    <span style="color:var(--apple-orange);">● 贪心法</span>：按NPVR排序<br>
    <span style="color:var(--apple-blue);">● 动态规划</span>：0-1背包最优<br>
    <span style="color:var(--apple-green);">● 穷举搜索</span>：含全部约束
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 计算准备
# ═══════════════════════════════════════════════════════════════

budget = budget_wan * 10000
rate = rate_pct / 100.0
projects = st.session_state.projects

if not projects:
    st.warning("请至少添加一个项目。")
    st.stop()

# 运行三种算法
metrics = analyze_all_projects(projects, rate)
greedy = greedy_selection(projects, budget, rate)
dp = dp_selection(projects, budget, rate)
exhaustive = exhaustive_selection(
    projects, budget, rate,
    mutual_exclusive_groups=st.session_state.mutual_groups if st.session_state.mutual_groups else None,
    dependencies=st.session_state.dependencies if st.session_state.dependencies else None,
)
recommendation = get_recommendation(greedy, dp, exhaustive, budget)

# ═══════════════════════════════════════════════════════════════
# 主页面
# ═══════════════════════════════════════════════════════════════

st.markdown("# 多项目投资组合智能选择系统")
st.markdown("*选题方向四：在预算约束下，从多个投资项目中选择最优组合使总NPV最大*")

# ═══════════════════════════════════════════════════════════════
# 五大模块选项卡
# ═══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "项目分析",
    "智能推荐",
    "解释与图表",
    "敏感性分析"
])

# ─────────────────────────────────────────
# Tab 1: 项目分析（计算模块）
# ─────────────────────────────────────────
with tab1:
    st.markdown("### 各项目财务指标一览")

    # 指标概览卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        npv_positive_count = sum(1 for m in metrics if m["npv"] > 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">可行项目数</div>
            <div class="metric-value">{npv_positive_count}/{len(projects)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        best = metrics[0] if metrics else {"name": "-", "npv": 0}
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">最高NPVR项目</div>
            <div class="metric-value" style="font-size:24px;">{best['name']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        total_investment = sum(p["cost"] for p in projects)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">全部项目总投资</div>
            <div class="metric-value" style="font-size:24px;">{fmt_wan(total_investment)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        all_npv = sum(m["npv"] for m in metrics if m["npv"] > 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">全部可行项目NPV和</div>
            <div class="metric-value" style="font-size:24px;">{fmt_yuan(all_npv)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # 项目详情表格
    df = pd.DataFrame(metrics)
    df_display = df.rename(columns={
        "id": "ID", "name": "项目名称", "cost": "投资额(元)",
        "npv": "NPV(元)", "irr": "IRR(%)", "npvr": "NPVR", "payback": "回收期(年)"
    })
    # 添加排名
    df_display.insert(0, "NPVR排名", range(1, len(df_display) + 1))

    # 格式化
    st.dataframe(
        df_display.style.format({
            "投资额(元)": "{:,.0f}",
            "NPV(元)": "{:,.0f}",
            "IRR(%)": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
            "NPVR": "{:.4f}",
            "回收期(年)": lambda x: f"{x:.2f}" if pd.notna(x) else "无法回收",
        }),
        width='stretch', height=400, hide_index=True
    )

    # NPVR排行TOP3
    st.markdown("### NPVR 排行榜 TOP 3")
    medals = ["", "", ""]
    colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
    cols = st.columns(3)
    for i, m in enumerate(metrics[:3]):
        with cols[i]:
            st.markdown(f"""
            <div class="apple-card" style="text-align:center;">
                <div style="font-size:48px;">{medals[i]}</div>
                <div style="font-size:20px;font-weight:600;color:var(--apple-dark);margin:12px 0;">{m['name']}</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                    <div>
                        <div style="font-size:12px;color:var(--apple-gray);">NPVR</div>
                        <div style="font-size:24px;font-weight:700;color:var(--apple-blue);">{m['npvr']:.4f}</div>
                    </div>
                    <div>
                        <div style="font-size:12px;color:var(--apple-gray);">NPV</div>
                        <div style="font-size:24px;font-weight:700;color:var(--apple-green);">{fmt_yuan(m['npv'])}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# Tab 2: 智能推荐（推荐模块）
# ─────────────────────────────────────────
with tab2:
    st.markdown("### 智能推荐结果")

    # 推荐摘要
    st.markdown(f"""
    <div class="recommendation-box">
        <div class="recommendation-title">推荐方案 · {recommendation['recommended_method']}</div>
        <div class="recommendation-summary">{recommendation['summary']}</div>
        <div class="apple-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    # 选中的项目
    rec_result = exhaustive if exhaustive["selected"] else (dp if dp["selected"] else greedy)
    st.markdown("#### 推荐选中的项目")
    if rec_result["selected"]:
        for p in rec_result["selected"]:
            npv_cls = "npv-positive" if p["npv"] > 0 else "npv-negative"
            irr_str = fmt_pct(p["irr"]) if p["irr"] is not None else "N/A"
            pb_str = f"{p['payback']}年" if p["payback"] is not None else "无法回收"
            st.markdown(f"""
            <div class="project-card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                    <span style="font-size:18px;font-weight:600;color:var(--apple-dark);">{p['name']}</span>
                    <span class="selected-tag">推荐</span>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;font-size:14px;">
                    <div><span style="color:var(--apple-gray);">投资:</span> <b>{fmt_yuan(p['cost'])}</b></div>
                    <div><span style="color:var(--apple-gray);">NPV:</span> <b class="{npv_cls}">{fmt_yuan(p['npv'])}</b></div>
                    <div><span style="color:var(--apple-gray);">IRR:</span> <b>{irr_str}</b></div>
                    <div><span style="color:var(--apple-gray);">NPVR:</span> <b>{p['npvr']:.4f}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("当前约束条件下无可行组合。请调整预算或放宽约束。")

    # 预算利用情况
    st.markdown("### 预算使用情况")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">总预算</div>
            <div class="metric-value" style="font-size:24px;">{fmt_wan(budget)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">已投资</div>
            <div class="metric-value" style="font-size:24px;">{fmt_wan(rec_result['total_cost'])}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">剩余预算</div>
            <div class="metric-value" style="font-size:24px;">{fmt_wan(rec_result['remaining_budget'])}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">预算利用率</div>
            <div class="metric-value" style="font-size:24px;">{rec_result['budget_utilization']}%</div>
        </div>
        """, unsafe_allow_html=True)

    # 三种算法对比
    st.markdown("### 三种算法结果对比")

    c1, c2, c3 = st.columns(3)

    with c1:
        g_names = [p["name"] for p in greedy["selected"]] if greedy["selected"] else ["无"]
        st.markdown(f"""
        <div class="apple-card">
            <div style="margin-bottom:12px;">
                <span class="method-badge greedy-badge">贪心法</span>
            </div>
            <div style="font-size:14px;color:var(--apple-gray);margin-bottom:8px;">选中项目：</div>
            <div style="font-weight:600;color:var(--apple-dark);margin-bottom:12px;">{'、'.join(g_names)}</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:13px;">
                <div><span style="color:var(--apple-gray);">总投资</span><br><b>{fmt_wan(greedy['total_cost'])}</b></div>
                <div><span style="color:var(--apple-gray);">总NPV</span><br><b>{fmt_yuan(greedy['total_npv'])}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        d_names = [p["name"] for p in dp["selected"]] if dp["selected"] else ["无"]
        st.markdown(f"""
        <div class="apple-card" style="border:1px solid var(--apple-blue);">
            <div style="margin-bottom:12px;">
                <span class="method-badge dp-badge">动态规划</span>
            </div>
            <div style="font-size:14px;color:var(--apple-gray);margin-bottom:8px;">选中项目：</div>
            <div style="font-weight:600;color:var(--apple-dark);margin-bottom:12px;">{'、'.join(d_names)}</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:13px;">
                <div><span style="color:var(--apple-gray);">总投资</span><br><b>{fmt_wan(dp['total_cost'])}</b></div>
                <div><span style="color:var(--apple-gray);">总NPV</span><br><b>{fmt_yuan(dp['total_npv'])}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        e_names = [p["name"] for p in exhaustive["selected"]] if exhaustive["selected"] else ["无"]
        st.markdown(f"""
        <div class="apple-card" style="border:2px solid var(--apple-green);">
            <div style="margin-bottom:12px;">
                <span class="method-badge exhaustive-badge">穷举搜索</span>
                <span style="color:var(--apple-green);font-size:12px;margin-left:4px;">含全部约束</span>
            </div>
            <div style="font-size:14px;color:var(--apple-gray);margin-bottom:8px;">选中项目：</div>
            <div style="font-weight:600;color:var(--apple-dark);margin-bottom:12px;">{'、'.join(e_names)}</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:13px;">
                <div><span style="color:var(--apple-gray);">总投资</span><br><b>{fmt_wan(exhaustive['total_cost'])}</b></div>
                <div><span style="color:var(--apple-gray);">总NPV</span><br><b>{fmt_yuan(exhaustive['total_npv'])}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # NPV增益提示
    npv_gain = recommendation["npv_gain"]
    if npv_gain > 1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(52,199,89,0.1),rgba(52,199,89,0.05));
                    border-radius:16px;padding:20px;text-align:center;margin-top:16px;">
            <div style="font-size:14px;color:var(--apple-green);font-weight:600;">
                精确算法相比贪心法可额外获得
            </div>
            <div style="font-size:36px;font-weight:700;color:var(--apple-green);margin-top:8px;">
                +{fmt_yuan(npv_gain)}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# Tab 3: 解释与图表（解释模块）
# ─────────────────────────────────────────
with tab3:
    st.markdown("### 决策解释与可视化")

    # 项目NPV对比图
    st.markdown("#### 各项目NPV对比")
    fig1 = go.Figure()
    names = [m["name"] for m in metrics]
    npvs = [m["npv"] for m in metrics]
    colors_bar = ["#34C759" if n > 0 else "#FF3B30" for n in npvs]

    fig1.add_trace(go.Bar(
        x=names, y=npvs,
        marker_color=colors_bar,
        text=[fmt_yuan(n) for n in npvs],
        textposition="outside",
    ))
    fig1.add_hline(y=0, line_dash="dash", line_color="#86868b", opacity=0.5)
    fig1.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1D1D1F"),
        margin=dict(t=10, b=10),
        yaxis_title="NPV（元）",
        showlegend=False,
    )
    fig1.update_yaxes(gridcolor="rgba(0,0,0,0.06)")
    st.plotly_chart(fig1, width='stretch')

    # 推荐组合预算分配饼图
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 推荐组合预算分配")
        if rec_result["selected"]:
            pie_labels = [p["name"] for p in rec_result["selected"]]
            pie_values = [p["cost"] for p in rec_result["selected"]]
            if rec_result["remaining_budget"] > 0:
                pie_labels.append("剩余预算")
                pie_values.append(rec_result["remaining_budget"])

            fig2 = go.Figure(data=[go.Pie(
                labels=pie_labels, values=pie_values,
                hole=0.4,
                marker_colors=["#0071E3", "#34C759", "#FF9500", "#AF52DE",
                               "#FF3B30", "#5AC8FA", "#FFD60A", "#86868b"]
            )])
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig2, width='stretch')

    with col_right:
        st.markdown("#### 推荐理由")
        for i, reason in enumerate(recommendation["reasoning"], 1):
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:12px;padding:8px 0;">
                <div style="width:6px;height:6px;border-radius:50%;background:var(--apple-blue);margin-top:6px;flex-shrink:0;"></div>
                <span style="color:var(--apple-gray);font-size:15px;line-height:1.5;">{reason}</span>
            </div>
            """, unsafe_allow_html=True)
        if recommendation["warning"]:
            st.warning(recommendation["warning"])

    # 当前约束说明
    st.markdown("#### 当前约束条件")
    has_constraints = False
    if st.session_state.mutual_groups:
        has_constraints = True
        for mg in st.session_state.mutual_groups:
            st.info(f"互斥约束：{' 与 '.join(mg)} 中最多选择1个")
    if st.session_state.dependencies:
        has_constraints = True
        for dep_on, dep_need in st.session_state.dependencies:
            st.info(f"依赖约束：选择{dep_on}必须先选择{dep_need}")
    if not has_constraints:
        st.caption("当前仅预算约束，无互斥或依赖约束。可在侧边栏添加约束条件。")

# ─────────────────────────────────────────
# Tab 4: 敏感性分析
# ─────────────────────────────────────────
with tab4:
    st.markdown("### 敏感性分析")

    constraints = {
        "mutual_exclusive": st.session_state.mutual_groups if st.session_state.mutual_groups else None,
        "dependencies": st.session_state.dependencies if st.session_state.dependencies else None,
    }

    analysis_type = st.radio("分析类型", ["折现率敏感性", "预算敏感性"], horizontal=True)

    if analysis_type == "折现率敏感性":
        st.markdown("#### 折现率变化对总NPV的影响")
        sens_data = sensitivity_rate(projects, budget, (0.03, 0.15), 13, constraints)
        sens_df = pd.DataFrame(sens_data)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sens_df["rate_pct"], y=sens_df["greedy_npv"],
            name="贪心法", line=dict(color="#FF9500", width=2), mode="lines+markers"
        ))
        fig.add_trace(go.Scatter(
            x=sens_df["rate_pct"], y=sens_df["dp_npv"],
            name="动态规划", line=dict(color="#0071E3", width=3), mode="lines+markers"
        ))
        fig.add_trace(go.Scatter(
            x=sens_df["rate_pct"], y=sens_df["exhaustive_npv"],
            name="穷举搜索（含约束）", line=dict(color="#34C759", width=3, dash="dot"), mode="lines+markers"
        ))
        fig.update_layout(
            hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1D1D1F"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=30, r=30, b=30, l=30),
        )
        fig.update_xaxes(title_text="折现率 (%)", gridcolor="rgba(0,0,0,0.06)")
        fig.update_yaxes(title_text="总NPV（元）", gridcolor="rgba(0,0,0,0.06)", tickformat=",d")
        st.plotly_chart(fig, width='stretch')

        st.caption("""
        **解读**：折现率升高时，远期现金流的现值降低，总NPV随之下降。
        三种算法的差距在高折现率下可能缩小（因为可选项目减少）。
        穷举搜索线可能等于或高于其他两条线。
        """)

    else:
        st.markdown("#### 预算变化对总NPV的影响")
        max_budget = sum(p["cost"] for p in projects)
        sens_data = sensitivity_budget(projects, rate, (50000, max_budget), 12, constraints)
        sens_df = pd.DataFrame(sens_data)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sens_df["budget_wan"], y=sens_df["greedy_npv"],
            name="贪心法", line=dict(color="#FF9500", width=2), mode="lines+markers"
        ))
        fig.add_trace(go.Scatter(
            x=sens_df["budget_wan"], y=sens_df["dp_npv"],
            name="动态规划", line=dict(color="#0071E3", width=3), mode="lines+markers"
        ))
        fig.add_trace(go.Scatter(
            x=sens_df["budget_wan"], y=sens_df["exhaustive_npv"],
            name="穷举搜索（含约束）", line=dict(color="#34C759", width=3, dash="dot"), mode="lines+markers"
        ))
        fig.update_layout(
            hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1D1D1F"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=30, r=30, b=30, l=30),
        )
        fig.update_xaxes(title_text="预算（万元）", gridcolor="rgba(0,0,0,0.06)")
        fig.update_yaxes(title_text="总NPV（元）", gridcolor="rgba(0,0,0,0.06)", tickformat=",d")
        st.plotly_chart(fig, width='stretch')

        st.caption("""
        **解读**：预算增加时，总NPV呈阶梯式增长（每增加一个项目就跳升一次）。
        预算超过所有项目总投资后，总NPV不再增长。
        """)

# ─────────────────────────────────────────────────────────────
# 注：验证模块已移至独立 Excel 文件（手算验证.xlsx）
# ─────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════
# 页脚
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:var(--apple-gray);font-size:13px;padding:20px 0;">
    <div>多项目投资组合智能选择系统 · 选题方向四</div>
    <div style="margin-top:8px;">智能决策理论与方法 · 第二次作业</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 直接运行入口：python app_enhanced.py 时自动启动服务器并打开浏览器
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__" and not st.runtime.exists():
    import subprocess
    import time
    import webbrowser
    import socket
    import sys

    HOST = "localhost"
    PORT = 8501

    def wait_for_port(host, port, timeout=30):
        start = time.time()
        while time.time() - start < timeout:
            try:
                with socket.create_connection((host, port), timeout=1):
                    return True
            except (socket.timeout, ConnectionRefusedError, OSError):
                time.sleep(0.5)
        return False

    print("=" * 44)
    print("  多项目投资组合智能选择系统")
    print("  正在启动...")
    print("=" * 44)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "app_enhanced.py")

    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", app_path,
         "--server.port", str(PORT),
         "--server.headless", "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=script_dir,
    )

    try:
        print("等待服务器启动...", end="", flush=True)
        if wait_for_port(HOST, PORT, timeout=30):
            print(" 就绪!")
            url = f"http://{HOST}:{PORT}"
            print(f"正在打开浏览器: {url}")
            webbrowser.open(url)
            print("浏览器已打开。按 Ctrl+C 停止服务器。")
            print("=" * 44)
            proc.wait()
        else:
            print(" 超时!")
            print(f"请手动打开: http://{HOST}:{PORT}")
            proc.wait()
    except KeyboardInterrupt:
        print("\n正在停止服务器...")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("已停止。")

