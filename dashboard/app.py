import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import math

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioRisk AI — Health Risk Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global Premium Styles ────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Premium background ── */
  .stApp {
    background: linear-gradient(135deg, #f0f4ff 0%, #f5f3ff 30%, #fdf2f8 60%, #f0f4ff 100%);
    background-attachment: fixed;
  }

  #MainMenu, footer, header { visibility: hidden; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8faff 100%) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.08);
    box-shadow: 4px 0 24px rgba(99, 102, 241, 0.04);
  }
  [data-testid="stSidebar"] * { color: #334155 !important; }

  /* ── Sidebar radio → pill nav ── */
  [data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none; }
  [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    gap: 0 !important;
    flex-direction: column;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 11px 16px !important;
    border-radius: 12px !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    color: #64748b !important;
    cursor: pointer !important;
    margin: 3px 0 !important;
    border: 1px solid transparent !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    background: transparent !important;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: linear-gradient(135deg, rgba(99,102,241,0.06), rgba(168,85,247,0.04)) !important;
    color: #4338ca !important;
    border-color: rgba(99,102,241,0.12) !important;
    transform: translateX(3px);
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label > span:first-child {
    display: none !important;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(168,85,247,0.08) 100%) !important;
    color: #4338ca !important;
    border-color: rgba(99,102,241,0.2) !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.08) !important;
  }

  /* ── General text ── */
  .stMarkdown, .stText, p, h1, h2, h3, h4 { color: #1e293b; }

  /* ── Hero header ── */
  .hero-header {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 40%, #a855f7 70%, #c084fc 100%);
    border-radius: 20px;
    padding: 32px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.25), 0 2px 8px rgba(99, 102, 241, 0.12);
  }
  .hero-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
    animation: float 8s ease-in-out infinite;
  }
  .hero-header::after {
    content: '';
    position: absolute;
    bottom: -40%; left: 10%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    border-radius: 50%;
    animation: float 6s ease-in-out infinite reverse;
  }
  @keyframes float {
    0%, 100% { transform: translateY(0px) scale(1); }
    50% { transform: translateY(-20px) scale(1.05); }
  }
  .hero-title {
    font-size: 28px; font-weight: 800;
    color: white; margin: 0; letter-spacing: -0.5px;
    position: relative; z-index: 1;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
  .hero-sub {
    font-size: 14px; color: rgba(255,255,255,0.85);
    margin: 8px 0 0; position: relative; z-index: 1;
    font-weight: 400;
  }
  .hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.2);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.25);
    padding: 5px 14px; border-radius: 24px;
    font-size: 11.5px; font-weight: 600; color: white;
    margin-left: 12px; vertical-align: middle;
    position: relative; z-index: 1;
  }
  .hero-badge-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 8px rgba(74, 222, 128, 0.6);
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  /* ── KPI cards ── */
  .kpi-card {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.8);
    border-radius: 20px;
    padding: 28px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(99,102,241,0.06);
  }
  .kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    border-radius: 20px 20px 0 0;
    opacity: 0.9;
  }
  .kpi-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 8px 30px rgba(99,102,241,0.15), 0 2px 8px rgba(0,0,0,0.05);
    border-color: rgba(99,102,241,0.15);
  }
  .kpi-icon {
    width: 44px; height: 44px;
    border-radius: 14px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 20px; margin-bottom: 14px;
  }
  .kpi-label {
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: #94a3b8; margin-bottom: 10px;
  }
  .kpi-value {
    font-size: 40px; font-weight: 800;
    letter-spacing: -1.5px; line-height: 1;
    margin-bottom: 6px;
    background-clip: text; -webkit-background-clip: text;
  }
  .kpi-sub {
    font-size: 11px; color: #94a3b8;
    font-weight: 400; line-height: 1.4;
  }

  /* ── Stat row ── */
  .stat-row { display: flex; gap: 14px; margin: 10px 0 24px; }
  .stat-mini {
    flex: 1;
    background: rgba(255,255,255,0.8);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 16px;
    padding: 22px 14px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(99,102,241,0.04);
    transition: all 0.25s ease;
  }
  .stat-mini:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 16px rgba(99,102,241,0.1);
  }
  .stat-num {
    font-size: 28px; font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1;
  }
  .stat-lbl {
    font-size: 10.5px; color: #94a3b8; margin-top: 8px;
    text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;
  }

  /* ── Section divider ── */
  .section-div {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #8b5cf6;
    font-weight: 700;
    padding-bottom: 10px;
    margin: 28px 0 18px;
    border-bottom: 2px solid;
    border-image: linear-gradient(to right, #6366f1, #a855f7, transparent) 1;
  }

  /* ── Risk result ── */
  .risk-level-text { font-size: 24px; font-weight: 800; margin: 10px 0 6px; }
  .risk-desc-text  { font-size: 13px; color: #64748b; margin-bottom: 16px; line-height: 1.6; }

  /* ── Factor row ── */
  .factor-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px; margin: 4px 0;
    border-radius: 12px;
    font-size: 13px; color: #334155;
    background: rgba(255,255,255,0.6);
    border: 1px solid rgba(255,255,255,0.8);
    transition: all 0.2s ease;
  }
  .factor-row:hover {
    background: rgba(255,255,255,0.9);
    transform: translateX(4px);
  }
  .factor-dot {
    width: 8px; height: 8px; border-radius: 50%;
    flex-shrink: 0; margin-right: 10px;
    box-shadow: 0 0 6px currentColor;
  }

  /* ── Input styling ── */
  .stNumberInput input, .stSelectbox select {
    background: rgba(255,255,255,0.9) !important;
    color: #1e293b !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important;
    transition: border-color 0.2s ease !important;
  }
  .stNumberInput input:focus, .stSelectbox select:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.1) !important;
  }
  input::placeholder { color: #94a3b8 !important; }

  /* ── Predict button ── */
  .stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 16px 32px !important;
    width: 100% !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.35), 0 2px 4px rgba(99,102,241,0.2) !important;
    letter-spacing: 0.02em !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.45), 0 4px 8px rgba(99,102,241,0.2) !important;
  }
  .stButton > button:active {
    transform: translateY(0px) !important;
  }

  /* ── Sidebar logo ── */
  .sidebar-logo {
    padding: 12px 0 24px;
    border-bottom: 2px solid;
    border-image: linear-gradient(to right, #6366f1, #a855f7, transparent) 1;
    margin-bottom: 20px;
  }
  .sidebar-logo h2 {
    font-size: 22px; font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; letter-spacing: -0.5px;
  }
  .sidebar-logo p {
    font-size: 11.5px; color: #94a3b8 !important;
    margin: 6px 0 0; font-weight: 500;
  }

  /* ── Sidebar info card ── */
  .sidebar-info {
    background: linear-gradient(135deg, rgba(99,102,241,0.04), rgba(168,85,247,0.04));
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 16px;
    padding: 16px 18px;
    margin-top: 14px;
  }
  .sidebar-info-row {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 12px; padding: 5px 0;
  }
  .sidebar-info-row:not(:last-child) {
    border-bottom: 1px solid rgba(99,102,241,0.06);
    padding-bottom: 8px; margin-bottom: 4px;
  }
  .sinfo-label { color: #94a3b8; font-weight: 500; }
  .sinfo-val   { color: #4338ca; font-weight: 700; }

  /* ── Status badge ── */
  .model-status {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; color: #16a34a; font-weight: 600; margin-top: 16px;
    background: rgba(34,197,94,0.06);
    padding: 6px 14px; border-radius: 20px;
    border: 1px solid rgba(34,197,94,0.12);
  }
  .model-status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #22c55e; box-shadow: 0 0 8px rgba(34,197,94,0.5);
    animation: pulse 2s infinite;
  }

  /* ── Chart container ── */
  .chart-card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.8);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 2px 12px rgba(99,102,241,0.05);
    margin-bottom: 16px;
    transition: all 0.25s ease;
  }
  .chart-card:hover {
    box-shadow: 0 6px 24px rgba(99,102,241,0.1);
  }
  .chart-title {
    font-size: 15px; font-weight: 700; color: #1e293b;
    margin-bottom: 4px;
  }
  .chart-caption {
    font-size: 12px; color: #94a3b8; margin-bottom: 16px;
    line-height: 1.4;
  }

  .cm-wrapper { overflow-x: auto; }

  /* ── Result card ── */
  .result-card {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.8);
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 4px 20px rgba(99,102,241,0.08);
  }

  /* ── Placeholder card ── */
  .placeholder-card {
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(8px);
    border: 2px dashed rgba(99,102,241,0.15);
    border-radius: 20px;
    padding: 60px 30px;
    text-align: center;
  }
  .placeholder-icon {
    font-size: 48px; margin-bottom: 16px;
    opacity: 0.7;
  }

  /* ── Metric highlight ── */
  .metric-highlight {
    display: inline-flex; align-items: center; gap: 8px;
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(168,85,247,0.06));
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 12px;
    padding: 10px 16px;
    font-size: 13px; font-weight: 600;
    color: #4338ca;
    margin: 4px;
  }
</style>
""", unsafe_allow_html=True)

# ── Constants / Model Metrics ─────────────────────────────────────────────────
METRICS = {
    "Accuracy": (91.16, "linear-gradient(135deg, #6366f1, #818cf8)", "#eef2ff", "Correct predictions out of all"),
    "Precision": (91.00, "linear-gradient(135deg, #8b5cf6, #a78bfa)", "#f5f3ff", "Weighted avg — true pos / pred pos"),
    "Recall": (91.00, "linear-gradient(135deg, #10b981, #34d399)", "#ecfdf5", "Weighted avg — true pos / actual pos"),
    "F1 Score": (91.00, "linear-gradient(135deg, #f59e0b, #fbbf24)", "#fffbeb", "Harmonic mean of precision & recall"),
    "AUC-ROC": (96.68, "linear-gradient(135deg, #ec4899, #f472b6)", "#fdf2f8", "Area Under the ROC Curve"),
}

PLOTLY_LIGHT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#64748b", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <h2>CardioRisk AI</h2>
      <p>Cardiovascular Risk Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:10px;text-transform:uppercase;letter-spacing:.14em;color:#8b5cf6;font-weight:700;margin:0 0 8px 2px">Navigation</p>', unsafe_allow_html=True)
    page = st.radio(
        "nav",
        ["KPI Dashboard", "Visual Analysis", "Prediction System"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div class="model-status">
      <div class="model-status-dot"></div> Model Ready
    </div>
    <div class="sidebar-info">
      <div class="sidebar-info-row">
        <span class="sinfo-label">Algorithm</span>
        <span class="sinfo-val">Logistic Reg.</span>
      </div>
      <div class="sidebar-info-row">
        <span class="sinfo-label">Records</span>
        <span class="sinfo-val">9,500</span>
      </div>
      <div class="sidebar-info-row">
        <span class="sinfo-label">Features</span>
        <span class="sinfo-val">10</span>
      </div>
      <div class="sidebar-info-row">
        <span class="sinfo-label">Train / Test</span>
        <span class="sinfo-val">80 / 20</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — KPI DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "KPI Dashboard":
    st.markdown("""
    <div class="hero-header">
      <div class="hero-title">
        KPI Dashboard
        <span class="hero-badge"><span class="hero-badge-dot"></span> Model Trained</span>
      </div>
      <div class="hero-sub">Key performance metrics from the Logistic Regression model trained on 9,500 synthetic health records</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model Performance Cards ───────────────────────────────────────────────
    st.markdown('<div class="section-div">Model Performance</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for i, (name, (val, gradient, bg_color, sub)) in enumerate(METRICS.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: none;">
              <div class="kpi-card" style="border:none;box-shadow:none;padding:0;background:transparent;">
                <div class="kpi-label">{name}</div>
                <div class="kpi-value" style="background:{gradient};-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{val:.1f}%</div>
                <div class="kpi-sub">{sub}</div>
              </div>
              <div style="position:absolute;top:0;left:0;right:0;height:4px;background:{gradient};border-radius:20px 20px 0 0;"></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")  # spacer

    # ── Dataset Stats ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-div">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stat-row">
      <div class="stat-mini"><div class="stat-num">9,500</div><div class="stat-lbl">Total Records</div></div>
      <div class="stat-mini"><div class="stat-num">10</div><div class="stat-lbl">Features Used</div></div>
      <div class="stat-mini"><div class="stat-num">80/20</div><div class="stat-lbl">Train / Test Split</div></div>
      <div class="stat-mini"><div class="stat-num">7,600</div><div class="stat-lbl">Training Samples</div></div>
      <div class="stat-mini"><div class="stat-num">1,900</div><div class="stat-lbl">Test Samples</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Class Distribution Charts ─────────────────────────────────────────────
    st.markdown('<div class="section-div">Class Distribution</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Risk Class Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-caption">Proportion of high-risk vs low-risk patients in the dataset</div>', unsafe_allow_html=True)
        fig_dist = go.Figure(go.Pie(
            labels=["High Risk (1)", "Low Risk (0)"],
            values=[7165, 2335],
            hole=0.7,
            marker=dict(
                colors=["rgba(239,68,68,0.85)", "rgba(16,185,129,0.85)"],
                line=dict(color=["rgba(239,68,68,1)", "rgba(16,185,129,1)"], width=2),
            ),
            textinfo="label+percent",
            textfont=dict(size=12, color="#334155"),
            hovertemplate="%{label}: %{value} patients (%{percent})<extra></extra>",
        ))
        fig_dist.update_layout(**PLOTLY_LIGHT, height=300, showlegend=True,
                               legend=dict(orientation="h", yanchor="bottom", y=-0.12,
                                           font=dict(size=11, color="#64748b")))
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Feature Type Breakdown</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-caption">Numerical vs categorical features used in the model</div>', unsafe_allow_html=True)
        fig_feat = go.Figure(go.Pie(
            labels=["Numerical (5)", "Categorical (5)"],
            values=[5, 5],
            hole=0.7,
            marker=dict(
                colors=["rgba(99,102,241,0.85)", "rgba(168,85,247,0.85)"],
                line=dict(color=["rgba(99,102,241,1)", "rgba(168,85,247,1)"], width=2),
            ),
            textinfo="label+percent",
            textfont=dict(size=12, color="#334155"),
        ))
        fig_feat.update_layout(**PLOTLY_LIGHT, height=300, showlegend=True,
                               legend=dict(orientation="h", yanchor="bottom", y=-0.12,
                                           font=dict(size=11, color="#64748b")))
        st.plotly_chart(fig_feat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Quick metrics bar ─────────────────────────────────────────────────────
    st.markdown('<div class="section-div">Quick Metrics</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 20px;">
      <div class="metric-highlight">Class 0 — Precision: 0.85 · Recall: 0.77 · F1: 0.81</div>
      <div class="metric-highlight">Class 1 — Precision: 0.93 · Recall: 0.96 · F1: 0.94</div>
      <div class="metric-highlight">Support — Class 0: 467 · Class 1: 1,433</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — VISUAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Visual Analysis":
    st.markdown("""
    <div class="hero-header">
      <div class="hero-title">
        Visual Analysis
        <span class="hero-badge">3 Charts</span>
      </div>
      <div class="hero-sub">Model evaluation visualizations — ROC curve, confusion matrix, and logistic regression feature weights</div>
    </div>
    """, unsafe_allow_html=True)

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-div">ROC Curve</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Receiver Operating Characteristic (ROC) Curve</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-caption">Shows trade-off between True Positive Rate and False Positive Rate — AUC = 0.97</div>', unsafe_allow_html=True)

    fpr = [0.0, 0.015, 0.034, 0.051, 0.069, 0.086, 0.116, 0.137, 0.167, 0.191, 0.214, 0.251, 0.298, 0.368, 0.415, 0.507, 1.0]
    tpr = [0.0, 0.761, 0.789, 0.825, 0.861, 0.881, 0.896, 0.918, 0.932, 0.941, 0.954, 0.965, 0.973, 0.980, 0.986, 0.993, 1.0]

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines+markers", name="ROC Curve (AUC = 0.97)",
        line=dict(color="#6366f1", width=3, shape="spline"),
        marker=dict(size=6, color="#6366f1", line=dict(color="white", width=2)),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
        hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random Classifier",
        line=dict(color="rgba(148,163,184,0.5)", width=2, dash="dash"),
        hoverinfo="skip",
    ))
    fig_roc.update_layout(
        **PLOTLY_LIGHT, height=380,
        xaxis=dict(title="False Positive Rate", gridcolor="rgba(99,102,241,0.06)", range=[0, 1]),
        yaxis=dict(title="True Positive Rate", gridcolor="rgba(99,102,241,0.06)", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, font=dict(size=11)),
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Confusion Matrix + Feature Weights ────────────────────────────────────
    st.markdown('<div class="section-div">Confusion Matrix & Feature Importance</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Confusion Matrix</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-caption">Predicted vs actual outcomes on 1,900 test samples</div>', unsafe_allow_html=True)
        # Values: TN=361, FP=106, FN=62, TP=1371
        z = [[361, 106], [62, 1371]]
        text = [["TN: 361", "FP: 106"], ["FN: 62", "TP: 1371"]]
        fig_cm = go.Figure(go.Heatmap(
            z=z,
            x=["No Risk (0)", "At Risk (1)"],
            y=["No Risk (0)", "At Risk (1)"],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=16, color="#1e293b", family="Inter"),
            colorscale=[
                [0.0, "rgba(99,102,241,0.08)"],
                [0.5, "rgba(99,102,241,0.35)"],
                [1.0, "rgba(99,102,241,0.7)"],
            ],
            zmin=0, zmax=1371,
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        ))
        # Highlight error cells
        fig_cm.add_trace(go.Heatmap(
            z=[[0, 106], [62, 0]],
            x=["No Risk (0)", "At Risk (1)"],
            y=["No Risk (0)", "At Risk (1)"],
            colorscale=[
                [0.0, "rgba(0,0,0,0)"],
                [1.0, "rgba(239,68,68,0.6)"],
            ],
            showscale=False,
            hoverinfo="skip",
        ))
        fig_cm.update_layout(
            **PLOTLY_LIGHT, height=320,
            xaxis=dict(title="Predicted Label", side="bottom"),
            yaxis=dict(title="Actual Label", autorange="reversed"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Logistic Regression — Feature Weights</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-caption">Coefficient magnitudes showing each feature\'s influence on risk prediction</div>', unsafe_allow_html=True)
        features = [
            "Max HR", "Sex (Male)", "BMI", "Chest Pain (Non-Anginal)",
            "Exercise Angina", "Chest Pain (Atypical)", "Chest Pain (Typical)",
            "Cholesterol", "Systolic BP", "Age", "Diabetes", "Smoker"
        ]
        coefficients = [-0.002, 0.020, -0.020, -0.035, -0.111, -0.157, -0.202, 1.728, 1.834, 1.950, 4.388, 4.439]
        colors = ["rgba(239,68,68,0.8)" if v > 0 else "rgba(16,185,129,0.8)" for v in coefficients]

        fig_coef = go.Figure(go.Bar(
            x=coefficients,
            y=features,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(
                    color=["rgba(239,68,68,1)" if v > 0 else "rgba(16,185,129,1)" for v in coefficients],
                    width=1.5,
                ),
                cornerradius=6,
            ),
            text=[f"{v:+.2f}" for v in coefficients],
            textposition="outside",
            textfont=dict(size=11, color="#475569", family="Inter"),
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        ))
        fig_coef.update_layout(
            **PLOTLY_LIGHT, height=380,
            xaxis=dict(title="Coefficient Value", gridcolor="rgba(99,102,241,0.06)", zeroline=True,
                       zerolinecolor="#cbd5e1", zerolinewidth=1),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_coef, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div class="hero-header">
      <div class="hero-title">Prediction System</div>
      <div class="hero-sub">Enter patient health data below to estimate cardiovascular risk using the trained model</div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    # ── Input Form ────────────────────────────────────────────────────────────
    with col_form:
        st.markdown('<div class="section-div">Numerical Features</div>', unsafe_allow_html=True)
        n1, n2 = st.columns(2)
        with n1:
            age  = st.number_input("Age (years)",          min_value=18, max_value=100, value=None, placeholder="e.g. 55")
            chol = st.number_input("Cholesterol (mg/dL)",  min_value=100, max_value=400, value=None, placeholder="e.g. 220")
            bmi  = st.number_input("BMI",                  min_value=10.0, max_value=60.0, value=None, placeholder="e.g. 28.5", step=0.1, format="%.1f")
        with n2:
            sbp  = st.number_input("Systolic BP (mmHg)",   min_value=80, max_value=220, value=None, placeholder="e.g. 140")
            hr   = st.number_input("Max Heart Rate (bpm)",  min_value=60, max_value=220, value=None, placeholder="e.g. 150")

        st.markdown('<div class="section-div">Categorical Features</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            sex    = st.selectbox("Sex",                  ["", "Male", "Female"])
            smoker = st.selectbox("Smoker",               ["", "Yes", "No"])
            angina = st.selectbox("Exercise-Induced Angina", ["", "Yes", "No"])
        with c2:
            cp     = st.selectbox("Chest Pain Type",      ["", "Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"])
            diab   = st.selectbox("Diabetes",             ["", "Yes", "No"])

        predict_clicked = st.button("Predict Cardiovascular Risk")

    # ── Result Panel ──────────────────────────────────────────────────────────
    with col_result:
        if not predict_clicked:
            st.markdown("""
            <div class="placeholder-card">
              <div class="placeholder-icon" style="font-size:28px;font-weight:800;color:#6366f1;">CardioRisk</div>
              <div style="font-size:16px;font-weight:600;color:#475569;margin-bottom:8px;">
                Risk Assessment
              </div>
              <div style="font-size:13px;color:#94a3b8;line-height:1.6;max-width:280px;margin:0 auto;">
                Fill in all patient details and click
                <b style="color:#6366f1;">Predict</b>
                to see the cardiovascular risk assessment.
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # ── Validation ───────────────────────────────────────────────────
            errors = []
            if age  is None: errors.append("Age")
            if sbp  is None: errors.append("Systolic BP")
            if chol is None: errors.append("Cholesterol")
            if hr   is None: errors.append("Max Heart Rate")
            if bmi  is None: errors.append("BMI")
            if not sex:    errors.append("Sex")
            if not cp:     errors.append("Chest Pain Type")
            if not smoker: errors.append("Smoker")
            if not diab:   errors.append("Diabetes")
            if not angina: errors.append("Exercise-Induced Angina")

            if errors:
                st.error(f"Please fill in: **{', '.join(errors)}**")
            else:
                # ── Logistic Regression Score ─────────────────────────────
                logit = -2.8
                logit += 1.950 * ((age  - 55)  / 14)
                logit += 1.834 * ((sbp  - 135) / 25)
                logit += 1.728 * ((chol - 225) / 43)
                logit -= 0.020 * ((bmi  - 29)  / 6.4)
                logit -= 0.002 * ((hr   - 145) / 32)

                cp_map = {"Typical Angina": -0.202, "Atypical Angina": -0.157,
                          "Non-Anginal": -0.035, "Asymptomatic": 0.0}
                logit += cp_map.get(cp, 0)
                if angina == "Yes": logit -= 0.111
                if smoker == "Yes": logit += 4.439
                if diab   == "Yes": logit += 4.388
                if sex    == "Male": logit += 0.020

                prob = round((1 / (1 + math.exp(-logit))) * 100)
                score = max(2, min(98, prob))

                # ── Risk level ────────────────────────────────────────────
                if score < 30:
                    color, glow, level, desc = "#10b981", "rgba(16,185,129,0.15)", "Low Risk", \
                        "The patient shows low cardiovascular risk. Maintaining a healthy lifestyle is recommended."
                elif score < 60:
                    color, glow, level, desc = "#f59e0b", "rgba(245,158,11,0.15)", "Moderate Risk", \
                        "Moderate cardiovascular risk detected. Regular check-ups and lifestyle adjustments are advised."
                else:
                    color, glow, level, desc = "#ef4444", "rgba(239,68,68,0.15)", "High Risk", \
                        "High cardiovascular risk detected. Immediate medical consultation is strongly recommended."

                st.markdown('<div class="result-card">', unsafe_allow_html=True)

                # ── Gauge Chart ───────────────────────────────────────────
                fig_gauge = go.Figure(go.Pie(
                    values=[score, 100 - score],
                    hole=0.78,
                    marker=dict(colors=[color, "#f1f5f9"], line=dict(width=0)),
                    hoverinfo="skip",
                    textinfo="none",
                    sort=False,
                ))
                fig_gauge.add_annotation(
                    text=f"<b>{score}%</b>",
                    x=0.5, y=0.55, showarrow=False,
                    font=dict(size=40, color=color, family="Inter"),
                )
                fig_gauge.add_annotation(
                    text="Risk Score",
                    x=0.5, y=0.38, showarrow=False,
                    font=dict(size=13, color="#94a3b8", family="Inter"),
                )
                fig_gauge.update_layout(
                    **{**PLOTLY_LIGHT, "margin": dict(l=20, r=20, t=20, b=0)},
                    height=270,
                    showlegend=False,
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown(f"""
                <div style="text-align:center">
                  <div class="risk-level-text" style="color:{color}">{level}</div>
                  <div class="risk-desc-text">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # ── Key Risk Indicators ───────────────────────────────────
                st.markdown('<div class="section-div">Key Risk Indicators</div>', unsafe_allow_html=True)
                factors = [
                    ("Age",            f"{age} yrs",       age > 65),
                    ("Systolic BP",    f"{sbp} mmHg",      sbp > 140),
                    ("Cholesterol",    f"{chol} mg/dL",    chol > 240),
                    ("BMI",            f"{bmi:.1f}",        bmi > 30),
                    ("Chest Pain",     cp,                  cp in ("Typical Angina", "Asymptomatic")),
                    ("Smoker",         smoker,              smoker == "Yes"),
                    ("Diabetes",       diab,                diab == "Yes"),
                    ("Exer. Angina",   angina,              angina == "Yes"),
                ]
                rows_html = ""
                for label, val, is_risk in factors:
                    dot_color = "#ef4444" if is_risk else "#10b981"
                    val_color = "#ef4444" if is_risk else "#10b981"
                    status_text = "Risk" if is_risk else "Normal"
                    rows_html += f"""
                    <div class="factor-row">
                      <div style="display:flex;align-items:center;gap:10px">
                        <div class="factor-dot" style="background:{dot_color};box-shadow:0 0 6px {dot_color};"></div>
                        <span style="font-weight:500;">{label}</span>
                      </div>
                      <div style="display:flex;align-items:center;gap:12px;">
                        <span style="color:{val_color};font-weight:600;font-size:12px">{val}</span>
                        <span style="color:{val_color};font-size:10px;font-weight:700;background:{dot_color}15;padding:2px 8px;border-radius:6px;">{status_text}</span>
                      </div>
                    </div>"""
                st.markdown(rows_html, unsafe_allow_html=True)
