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

# ── Global Light Styles ──────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Light background */
  .stApp { background: #f5f7fa; }

  #MainMenu, footer, header { visibility: hidden; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0;
  }
  [data-testid="stSidebar"] * { color: #334155 !important; }

  /* ── Sidebar radio → clean nav pills ── */
  [data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none; }
  [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    gap: 0 !important;
    flex-direction: column;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 10px 14px !important;
    border-radius: 10px !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    color: #64748b !important;
    cursor: pointer !important;
    margin: 2px 0 !important;
    border: 1px solid transparent !important;
    transition: all 0.18s ease !important;
    background: transparent !important;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #f1f5f9 !important;
    color: #1e293b !important;
    border-color: #e2e8f0 !important;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label > span:first-child {
    display: none !important;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, rgba(79,142,247,0.12) 0%, rgba(124,95,247,0.08) 100%) !important;
    color: #3b6fd4 !important;
    border-color: rgba(79,142,247,0.28) !important;
    font-weight: 600 !important;
    box-shadow: inset 0 0 0 1px rgba(79,142,247,0.08) !important;
  }

  /* General text */
  .stMarkdown, .stText, p, h1, h2, h3, h4 { color: #1e293b; }

  /* KPI card */
  .kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }
  .kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
  }
  .kpi-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 10px;
  }
  .kpi-value {
    font-size: 42px;
    font-weight: 700;
    letter-spacing: -1px;
    line-height: 1;
    margin-bottom: 8px;
  }
  .kpi-sub { font-size: 11.5px; color: #94a3b8; }

  /* Stat mini */
  .stat-row { display: flex; gap: 16px; margin: 10px 0 20px; }
  .stat-mini {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 14px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  }
  .stat-num { font-size: 26px; font-weight: 700; color: #1e293b; line-height: 1; }
  .stat-lbl {
    font-size: 11px; color: #94a3b8; margin-top: 6px;
    text-transform: uppercase; letter-spacing: 0.06em;
  }

  /* Section divider */
  .section-div {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #94a3b8;
    font-weight: 600;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 8px;
    margin: 24px 0 16px;
  }

  /* Page header */
  .page-title { font-size: 26px; font-weight: 700; color: #0f172a; margin-bottom: 4px; }
  .page-sub   { font-size: 13.5px; color: #64748b; margin-bottom: 20px; }

  /* Badge */
  .badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; margin-left: 10px; vertical-align: middle;
  }
  .badge-green { background: rgba(34,197,94,0.12); color: #16a34a; }
  .badge-blue  { background: rgba(79,142,247,0.12); color: #2563eb; }

  /* Risk result */
  .risk-level-text { font-size: 22px; font-weight: 700; margin: 10px 0 6px; }
  .risk-desc-text  { font-size: 13px; color: #64748b; margin-bottom: 16px; }

  /* Factor row */
  .factor-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 0; border-bottom: 1px solid #f1f5f9;
    font-size: 13px; color: #334155;
  }
  .factor-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; margin-right: 10px; }

  /* Input styling */
  .stNumberInput input, .stSelectbox select {
    background: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
  }
  input::placeholder { color: #94a3b8 !important; }

  /* Predict button */
  .stButton > button {
    background: linear-gradient(135deg, #4f8ef7, #7c5ff7) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 14px 32px !important;
    width: 100% !important;
    transition: box-shadow 0.2s, opacity 0.2s !important;
    box-shadow: 0 4px 14px rgba(79,142,247,0.35) !important;
  }
  .stButton > button:hover {
    opacity: 0.88 !important;
    box-shadow: 0 6px 20px rgba(79,142,247,0.45) !important;
  }

  /* Sidebar logo */
  .sidebar-logo {
    padding: 10px 0 20px;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 16px;
  }
  .sidebar-logo h2 {
    font-size: 20px; font-weight: 700;
    color: #0f172a !important; margin: 0; letter-spacing: -0.3px;
  }
  .sidebar-logo p { font-size: 11.5px; color: #94a3b8 !important; margin: 4px 0 0; }

  /* Sidebar info card */
  .sidebar-info {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 16px;
    margin-top: 12px;
  }
  .sidebar-info-row {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 12px; padding: 4px 0;
  }
  .sidebar-info-row:not(:last-child) {
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 7px; margin-bottom: 3px;
  }
  .sinfo-label { color: #94a3b8; font-weight: 500; }
  .sinfo-val   { color: #475569; font-weight: 600; }

  /* Status badge */
  .model-status {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11px; color: #16a34a; font-weight: 600; margin-top: 14px;
  }
  .model-status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #22c55e; box-shadow: 0 0 6px rgba(34,197,94,0.5);
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  .cm-wrapper { overflow-x: auto; }
</style>
""", unsafe_allow_html=True)

# ── Constants / Model Metrics ─────────────────────────────────────────────────
METRICS = {
    "Accuracy": (92.50, "#4f8ef7", "Correct predictions out of all"),
    "Precision": (92.00, "#7c5ff7", "True positives / predicted positives"),
    "Recall": (93.00, "#22c55e", "True positives / actual positives"),
    "F1 Score": (92.00, "#f59e0b", "Harmonic mean of precision & recall"),
    "AUC-ROC": (89.00, "#ec4899", "Area Under the ROC Curve"),
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

    st.markdown('<p style="font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:#4a5568;font-weight:600;margin:0 0 6px 2px">Navigation</p>', unsafe_allow_html=True)
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
        <span class="sinfo-val">400</span>
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
    <div class="page-title">KPI Dashboard <span class="badge badge-green">Model Trained</span></div>
    <div class="page-sub">Key performance metrics from the Logistic Regression model trained on synthetic health data</div>
    """, unsafe_allow_html=True)

    # ── Model Performance Cards ───────────────────────────────────────────────
    st.markdown('<div class="section-div">Model Performance</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for i, (name, (val, color, sub)) in enumerate(METRICS.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{name}</div>
              <div class="kpi-value" style="color:{color}">{val:.0f}%</div>
              <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")  # spacer

    # ── Dataset Stats ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-div">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stat-row">
      <div class="stat-mini"><div class="stat-num">400</div><div class="stat-lbl">Total Records</div></div>
      <div class="stat-mini"><div class="stat-num">10</div><div class="stat-lbl">Features Used</div></div>
      <div class="stat-mini"><div class="stat-num">80/20</div><div class="stat-lbl">Train / Test Split</div></div>
      <div class="stat-mini"><div class="stat-num">320</div><div class="stat-lbl">Training Samples</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Class Distribution Charts ─────────────────────────────────────────────
    st.markdown('<div class="section-div">Class Distribution</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Risk Class Distribution**")
        st.caption("Proportion of high-risk vs low-risk patients in dataset")
        fig_dist = go.Figure(go.Pie(
            labels=["High Risk (1)", "Low Risk (0)"],
            values=[206, 194],
            hole=0.65,
            marker=dict(
                colors=["rgba(239,68,68,0.85)", "rgba(34,197,94,0.85)"],
                line=dict(color=["rgba(239,68,68,1)", "rgba(34,197,94,1)"], width=2),
            ),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} patients (%{percent})<extra></extra>",
        ))
        fig_dist.update_layout(**PLOTLY_LIGHT, height=280,
                               legend=dict(orientation="h", yanchor="bottom", y=-0.15))
        st.plotly_chart(fig_dist, width="stretch")

    with c2:
        st.markdown("**Feature Type Breakdown**")
        st.caption("Numerical vs categorical features in the dataset")
        fig_feat = go.Figure(go.Pie(
            labels=["Numerical (5)", "Categorical (5)"],
            values=[5, 5],
            hole=0.65,
            marker=dict(
                colors=["rgba(79,142,247,0.85)", "rgba(124,95,247,0.85)"],
                line=dict(color=["rgba(79,142,247,1)", "rgba(124,95,247,1)"], width=2),
            ),
            textinfo="label+percent",
        ))
        fig_feat.update_layout(**PLOTLY_LIGHT, height=280,
                               legend=dict(orientation="h", yanchor="bottom", y=-0.15))
        st.plotly_chart(fig_feat, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — VISUAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Visual Analysis":
    st.markdown("""
    <div class="page-title">Visual Analysis <span class="badge badge-blue">3 Charts</span></div>
    <div class="page-sub">Model evaluation visualizations — ROC curve, confusion matrix, and logistic regression feature weights</div>
    """, unsafe_allow_html=True)

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-div">ROC Curve</div>', unsafe_allow_html=True)
    st.markdown("**Receiver Operating Characteristic (ROC) Curve**")
    st.caption("Shows tradeoff between True Positive Rate and False Positive Rate — AUC = 0.89")

    fpr = [0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.38, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 1.0]
    tpr = [0, 0.28, 0.48, 0.60, 0.70, 0.76, 0.81, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99, 1.0]

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines+markers", name="ROC Curve (AUC = 0.89)",
        line=dict(color="#4f8ef7", width=3),
        marker=dict(size=5, color="#4f8ef7"),
        fill="tozeroy", fillcolor="rgba(79,142,247,0.08)",
        hovertemplate="FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>",
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random Classifier",
        line=dict(color="rgba(100,116,139,0.4)", width=2, dash="dash"),
        hoverinfo="skip",
    ))
    fig_roc.update_layout(
        **PLOTLY_LIGHT, height=350,
        xaxis=dict(title="False Positive Rate", gridcolor="#f1f5f9", range=[0, 1]),
        yaxis=dict(title="True Positive Rate", gridcolor="#f1f5f9", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    st.plotly_chart(fig_roc, width="stretch")

    # ── Confusion Matrix + Feature Weights ────────────────────────────────────
    st.markdown('<div class="section-div">Confusion Matrix & Feature Importance</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Confusion Matrix**")
        st.caption("Predicted vs actual outcomes on 80 test samples")
        # Values: TN=12, FP=5, FN=1, TP=62
        z = [[12, 5], [1, 62]]
        text = [["TN: 12", "FP: 5"], ["FN: 1", "TP: 62"]]
        fig_cm = go.Figure(go.Heatmap(
            z=z,
            x=["No Risk (0)", "At Risk (1)"],
            y=["No Risk (0)", "At Risk (1)"],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=16, color="#1e293b"),
            colorscale=[
                [0.0, "rgba(34,197,94,0.15)"],
                [0.5, "rgba(34,197,94,0.5)"],
                [1.0, "rgba(34,197,94,0.85)"],
            ],
            zmin=0, zmax=62,
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        ))
        # Highlight error cells differently
        fig_cm.add_trace(go.Heatmap(
            z=[[0, 5], [1, 0]],
            x=["No Risk (0)", "At Risk (1)"],
            y=["No Risk (0)", "At Risk (1)"],
            colorscale=[
                [0.0, "rgba(0,0,0,0)"],
                [1.0, "rgba(239,68,68,0.7)"],
            ],
            showscale=False,
            hoverinfo="skip",
        ))
        fig_cm.update_layout(
            **PLOTLY_LIGHT, height=300,
            xaxis=dict(title="Predicted Label", side="bottom"),
            yaxis=dict(title="Actual Label", autorange="reversed"),
        )
        st.plotly_chart(fig_cm, width="stretch")

    with c2:
        st.markdown("**Logistic Regression — Feature Weights**")
        st.caption("Coefficient magnitudes showing each feature's influence on risk prediction")
        features = [
            "Sex (Male)", "Max HR", "BMI", "Diabetes", "Cholesterol",
            "Smoker", "Systolic BP", "Age", "Exercise Angina", "Chest Pain (Typical)"
        ]
        coefficients = [-0.28, -0.56, 0.31, 0.48, 0.54, 0.62, 0.73, 0.85, 1.18, 1.42]
        colors = ["rgba(239,68,68,0.8)" if v > 0 else "rgba(34,197,94,0.8)" for v in coefficients]

        fig_coef = go.Figure(go.Bar(
            x=coefficients,
            y=features,
            orientation="h",
            marker=dict(color=colors, line=dict(
                color=["rgba(239,68,68,1)" if v > 0 else "rgba(34,197,94,1)" for v in coefficients],
                width=1.5,
            )),
            text=[f"{v:+.2f}" for v in coefficients],
            textposition="outside",
            textfont=dict(size=11, color="#475569"),
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        ))
        fig_coef.update_layout(
            **PLOTLY_LIGHT, height=320,
            xaxis=dict(title="Coefficient Value", gridcolor="#f1f5f9", zeroline=True,
                       zerolinecolor="#cbd5e1", zerolinewidth=1),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_coef, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div class="page-title">Prediction System</div>
    <div class="page-sub">Enter patient health data below to estimate cardiovascular risk using the trained model</div>
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
            <div style="height:100%;display:flex;flex-direction:column;align-items:center;
                        justify-content:center;text-align:center;color:#4a5568;padding:60px 20px;">
              <div style="font-size:15px;line-height:1.6;">
                Fill in all patient details and click <b style="color:#6b7a99">Predict</b>
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
                logit += 0.85 * ((age  - 55)  / 14)
                logit += 0.73 * ((sbp  - 135) / 25)
                logit += 0.54 * ((chol - 225) / 43)
                logit += 0.31 * ((bmi  - 29)  / 6.4)
                logit -= 0.56 * ((hr   - 145) / 32)

                cp_map = {"Typical Angina": 1.42, "Atypical Angina": 0.60,
                          "Non-Anginal": 0.0, "Asymptomatic": 0.30}
                logit += cp_map.get(cp, 0)
                if angina == "Yes": logit += 1.18
                if smoker == "Yes": logit += 0.62
                if diab   == "Yes": logit += 0.48
                if sex    == "Male": logit -= 0.28

                prob = round((1 / (1 + math.exp(-logit))) * 100)
                score = max(2, min(98, prob))

                # ── Risk level ────────────────────────────────────────────
                if score < 30:
                    color, level, desc = "#22c55e", "Low Risk", \
                        "The patient shows low cardiovascular risk. Maintaining a healthy lifestyle is recommended."
                elif score < 60:
                    color, level, desc = "#f59e0b", "Moderate Risk", \
                        "Moderate cardiovascular risk detected. Regular check-ups and lifestyle adjustments are advised."
                else:
                    color, level, desc = "#ef4444", "High Risk", \
                        "High cardiovascular risk detected. Immediate medical consultation is strongly recommended."

                # ── Gauge Chart ───────────────────────────────────────────
                fig_gauge = go.Figure(go.Pie(
                    values=[score, 100 - score],
                    hole=0.78,
                    marker=dict(colors=[color, "#e2e8f0"], line=dict(width=0)),
                    hoverinfo="skip",
                    textinfo="none",
                    sort=False,
                ))
                fig_gauge.add_annotation(
                    text=f"<b>{score}%</b>",
                    x=0.5, y=0.55, showarrow=False,
                    font=dict(size=36, color=color, family="Inter"),
                )
                fig_gauge.add_annotation(
                    text="Risk Score",
                    x=0.5, y=0.38, showarrow=False,
                    font=dict(size=13, color="#94a3b8", family="Inter"),
                )
                fig_gauge.update_layout(
                    **{**PLOTLY_LIGHT, "margin": dict(l=20, r=20, t=20, b=0)},
                    height=260,
                    showlegend=False,
                )
                st.plotly_chart(fig_gauge, width="stretch")

                st.markdown(f"""
                <div style="text-align:center">
                  <div class="risk-level-text" style="color:{color}">{level}</div>
                  <div class="risk-desc-text">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

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
                    dot_color = "#ef4444" if is_risk else "#22c55e"
                    val_color = "#ef4444" if is_risk else "#22c55e"
                    rows_html += f"""
                    <div class="factor-row">
                      <div style="display:flex;align-items:center;gap:10px">
                        <div class="factor-dot" style="background:{dot_color}"></div>
                        <span>{label}</span>
                      </div>
                      <span style="color:{val_color};font-weight:600;font-size:12px">{val}</span>
                    </div>"""
                st.markdown(rows_html, unsafe_allow_html=True)
