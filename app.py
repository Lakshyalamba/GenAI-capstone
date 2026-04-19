from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.agent.config import validate_agent_config
from src.agent.workflow import run_agent_workflow
from src.data_processing import load_processed_dataset, summarize_dataset
from src.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_METADATA,
    NUMERIC_FEATURES,
    format_feature_value,
)
from src.inference import load_artifact_bundle, predict_batch, predict_single
from src.utils import get_env_status, humanize_slug


APP_TITLE = "CardioRisk AI: Cardiovascular Risk Prediction & Health Strategist"
APP_SUBTITLE = "Logistic-regression risk scoring with a retrieval-grounded cardiovascular guidance workflow."
PLOT_BACKGROUND = "#ffffff"
SURFACE_COLOR = "#ffffff"
PAGE_BACKGROUND = "#f4f7fb"
TEXT_MUTED = "#5b6577"
ACCENT_BLUE = "#3b82f6"
ACCENT_SLATE = "#475569"
RISK_COLORS = {
    "Low": "#16a34a",
    "Moderate": "#d97706",
    "High": "#dc2626",
}


def default_patient_profile() -> dict[str, Any]:
    return {
        feature: metadata["default"] for feature, metadata in FEATURE_METADATA.items()
    }


@st.cache_data(show_spinner=False)
def get_dashboard_dataset() -> pd.DataFrame:
    frame = load_processed_dataset()
    frame["risk_label"] = frame["risk"].map({0: "Lower risk", 1: "Higher risk"})
    return frame


@st.cache_resource(show_spinner=False)
def get_model_bundle() -> dict[str, Any]:
    return load_artifact_bundle()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
          @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');

          html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #1e293b;
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
          }

          .stApp {
            background-color: #f0f4f8;
          }

          .block-container {
            max-width: 1440px !important;
            margin: 0 auto;
            padding: 2rem 4rem 4rem !important;
          }

          @media (max-width: 1200px) {
            .block-container {
               padding: 1.5rem 2rem 3rem !important;
            }
          }

          @media (max-width: 768px) {
            .block-container {
               padding: 1rem 1rem 2rem !important;
            }
            .panel, div[data-testid="stForm"] {
               padding: 1.25rem 1rem !important;
            }
          }

          [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #e2e8f0 !important;
          }
          
          div.stRadio > label {
            display: none !important;
          }

          /* Hide Streamlit radio circles */
          div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
            display: none !important;
          }

          /* Radio item container acting as nav link */
          div[role="radiogroup"] label[data-baseweb="radio"] {
            background-color: transparent;
            padding: 0.55rem 0.85rem;
            border-radius: 6px;
            margin-bottom: 0.25rem;
            transition: all 0.2s ease;
            cursor: pointer;
            box-shadow: none;
            border: none;
          }

          /* Nav Link Hover */
          div[role="radiogroup"] label[data-baseweb="radio"]:hover {
            background-color: #e2e8f0;
          }

          /* Nav Link Active State */
          div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {
            background-color: #f0fdfa;
            position: relative;
          }

          /* Active Accent Bar */
          div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked)::before {
            content: "";
            position: absolute;
            left: 0;
            top: 15%;
            height: 70%;
            width: 3px;
            background-color: #0d9488;
            border-radius: 4px;
          }

          /* Normal Nav Text */
          div[role="radiogroup"] label[data-baseweb="radio"] div[data-testid="stMarkdownContainer"] p {
            font-size: 0.88rem;
            font-weight: 500;
            color: #334155;
            margin: 0;
            line-height: 1.4;
            transition: color 0.15s ease;
            display: flex;
            align-items: center;
          }

          /* React-style (Material) Icons injected before text */
          div[role="radiogroup"] label[data-baseweb="radio"] div[data-testid="stMarkdownContainer"] p::before {
            font-family: 'Material Symbols Outlined';
            font-weight: normal;
            font-style: normal;
            font-size: 1.15rem;
            line-height: 1;
            letter-spacing: normal;
            text-transform: none;
            display: inline-block;
            white-space: nowrap;
            word-wrap: normal;
            direction: ltr;
            -webkit-font-feature-settings: 'liga';
            -webkit-font-smoothing: antialiased;
            margin-right: 0.55rem;
            color: #64748b;
            transition: color 0.2s ease;
          }

          /* Active & Hover states for the icon */
          div[role="radiogroup"] label[data-baseweb="radio"]:hover div[data-testid="stMarkdownContainer"] p::before {
            color: #0f283d;
          }
          div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) div[data-testid="stMarkdownContainer"] p::before {
            color: #0d9488;
          }

          /* Map specific icons to specific menu items based on their order */
          div[role="radiogroup"] label[data-baseweb="radio"]:nth-child(1) div[data-testid="stMarkdownContainer"] p::before { content: "space_dashboard"; }
          div[role="radiogroup"] label[data-baseweb="radio"]:nth-child(2) div[data-testid="stMarkdownContainer"] p::before { content: "analytics"; }
          div[role="radiogroup"] label[data-baseweb="radio"]:nth-child(3) div[data-testid="stMarkdownContainer"] p::before { content: "account_circle"; }
          div[role="radiogroup"] label[data-baseweb="radio"]:nth-child(4) div[data-testid="stMarkdownContainer"] p::before { content: "folder_open"; }
          div[role="radiogroup"] label[data-baseweb="radio"]:nth-child(5) div[data-testid="stMarkdownContainer"] p::before { content: "medical_services"; }

          /* Hover Nav Text */
          div[role="radiogroup"] label[data-baseweb="radio"]:hover div[data-testid="stMarkdownContainer"] p {
            color: #0f172a;
          }

          /* Active Nav Text */
          div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) div[data-testid="stMarkdownContainer"] p {
            color: #1e3a8a;
            font-weight: 600;
          }
          
          .sidebar-brand {
             margin-bottom: 2.5rem;
             padding-bottom: 1.5rem;
             border-bottom: 1px solid #e2e8f0;
          }
          .sidebar-brand h1 {
             font-family: 'Inter', sans-serif;
             font-size: 1.5rem;
             font-weight: 700;
             color: #0f283d;
             margin: 0;
             line-height: 1.2;
             letter-spacing: -0.02em;
          }
          .sidebar-brand p {
             font-size: 0.85rem;
             color: #64748b;
             margin-top: 0.35rem;
             line-height: 1.4;
          }

          #MainMenu, footer {
            visibility: hidden;
          }
          [data-testid="stToolbar"] {
            visibility: hidden !important;
          }
          header {
            background: transparent !important;
          }
          [data-testid="stSidebarCollapseButton"] {
            display: none !important;
          }

          .app-header {
            background: #ffffff;
            border-bottom: 1px solid #e2e8f0;
            padding: 1rem 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
          }

          .status-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            align-items: center;
          }

          .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.4rem 0.75rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            color: #475569;
          }

          .status-pill.ok {
            background: #d1fae5;
            color: #065f46;
            border-color: #a7f3d0;
          }

          .status-pill.warn {
            background: #fef3c7;
            color: #92400e;
            border-color: #fde68a;
          }

          .status-pill.info {
            background: #ccfbf1;
            color: #115e59;
            border-color: #99f6e4;
          }

          .panel {
            background: #ffffff;
            border: 1px solid rgba(226, 232, 240, 0.8);
            border-radius: 16px;
            padding: 1.75rem 2rem;
            box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.03), 0 2px 4px -2px rgba(15, 23, 42, 0.03), 0 0 0 1px rgba(15, 23, 42, 0.01) inset;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            height: 100%;
            display: flex;
            flex-direction: column;
            margin-bottom: 1.5rem;
            position: relative;
          }

          .panel:hover {
            box-shadow: 0 12px 20px -3px rgba(15, 23, 42, 0.06), 0 4px 6px -4px rgba(15, 23, 42, 0.04), 0 0 0 1px rgba(15, 23, 42, 0.01) inset;
            transform: translateY(-2px);
            border-color: rgba(203, 213, 225, 0.9);
          }

          .metric-tile {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid rgba(226, 232, 240, 0.8);
            border-radius: 14px;
            padding: 1.5rem 1.75rem;
            box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.02), 0 2px 4px -2px rgba(15, 23, 42, 0.02), inset 0 1px 0 rgba(255,255,255,1);
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          }
          
          .metric-tile:hover {
            background: #ffffff;
            box-shadow: 0 12px 20px -3px rgba(15, 23, 42, 0.06), 0 4px 6px -4px rgba(15, 23, 42, 0.04);
            transform: translateY(-3px);
            border-color: rgba(203, 213, 225, 0.9);
          }
          
          .metric-tile::before {
             content: '';
             position: absolute;
             top: 0; left: 0; bottom: 0; width: 5px;
             background: #0d9488;
             border-radius: 14px 0 0 14px;
          }

          .metric-label {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
          }

          .metric-value {
            margin-top: 0.4rem;
            font-family: 'Inter', sans-serif;
            font-size: 2.5rem;
            font-weight: 800;
            color: #0f283d;
            line-height: 1.1;
            letter-spacing: -0.03em;
          }

          .metric-note {
            margin-top: 0.5rem;
            font-size: 0.85rem;
            font-weight: 500;
            color: #475569;
          }

          .section-kicker {
            margin-bottom: 1.25rem;
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f283d;
            display: flex;
            align-items: center;
            letter-spacing: -0.01em;
          }

          .risk-pill {
            display: inline-block;
            padding: 0.35rem 0.85rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 1rem;
            border: 1px solid transparent;
          }

          .risk-low {
            background: #f0fdf4;
            color: #166534;
            border-color: #dcfce7;
          }

          .risk-moderate {
            background: #fffbeb;
            color: #92400e;
            border-color: #fef3c7;
          }

          .risk-high {
            background: #fef2f2;
            color: #b91c1c;
            border-color: #fee2e2;
          }

          .soft-note {
            color: #475569;
            margin-bottom: 1.5rem;
            font-size: 0.95rem;
            line-height: 1.6;
            font-weight: 400;
          }

          div[data-testid="stFileUploader"] section {
            border-radius: 12px;
            border: 2px dashed rgba(203, 213, 225, 0.8);
            background: #f8fafc;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          }

          div[data-testid="stFileUploader"] section:hover {
            border-color: #0d9488;
            background: #f0fdfa;
            box-shadow: inset 0 0 0 1px rgba(13, 148, 136, 0.1);
          }

          div[data-testid="stDataFrame"] {
            border-radius: 12px;
            border: 1px solid rgba(226, 232, 240, 0.8);
            box-shadow: 0 2px 4px rgba(15, 23, 42, 0.02);
            overflow: hidden;
            transition: all 0.3s ease;
          }
          
          div[data-testid="stDataFrame"]:hover {
            box-shadow: 0 6px 12px rgba(15, 23, 42, 0.04);
            border-color: rgba(203, 213, 225, 0.9);
          }

          div[data-testid="stForm"] {
            background: #ffffff;
            border: 1px solid rgba(226, 232, 240, 0.8);
            border-radius: 16px;
            padding: 1.75rem 2rem;
            box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.03), 0 2px 4px -2px rgba(15, 23, 42, 0.03), 0 0 0 1px rgba(15, 23, 42, 0.01) inset;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            margin-bottom: 1.5rem;
          }
          
          div[data-testid="stForm"]:hover {
            box-shadow: 0 12px 20px -3px rgba(15, 23, 42, 0.06), 0 4px 6px -4px rgba(15, 23, 42, 0.04), 0 0 0 1px rgba(15, 23, 42, 0.01) inset;
            transform: translateY(-2px);
            border-color: rgba(203, 213, 225, 0.9);
          }

          .stButton > button {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            padding: 0.65rem 1.5rem;
            font-weight: 600;
            color: #475569;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            font-size: 0.9rem;
            letter-spacing: 0.01em;
          }

          .stButton > button:hover {
            border-color: #94a3b8;
            color: #0f283d;
            box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.05), 0 2px 4px -2px rgba(15, 23, 42, 0.03);
            background: #f8fafc;
          }

          .stButton > button:focus {
            outline: none;
            box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.15);
            border-color: #0d9488;
          }

          button[kind="primaryFormSubmit"],
          .stButton > button[kind="primary"] {
            background: #0d9488 !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            letter-spacing: 0.01em;
            padding: 0.75rem 2rem !important;
            border-radius: 8px !important;
            border: 1px solid #0f766e !important;
            box-shadow: 0 1px 3px rgba(13, 148, 136, 0.15) !important;
            text-transform: none !important;
            font-size: 1rem !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
          }

          button[kind="primaryFormSubmit"]:hover,
          .stButton > button[kind="primary"]:hover {
            background: #0f766e !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 10px -2px rgba(13, 148, 136, 0.25), 0 2px 4px -1px rgba(13, 148, 136, 0.1) !important;
            border-color: #0f766e !important;
            color: #ffffff !important;
          }

          button[kind="primaryFormSubmit"]:focus,
          .stButton > button[kind="primary"]:focus {
            box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.3) !important;
          }

          .stTextInput label p,
          .stNumberInput label p,
          .stSelectbox label p {
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            color: #1e293b !important;
            margin-bottom: 0.3rem !important;
            letter-spacing: 0.01em !important;
          }

          .stTextInput > div > div > input,
          .stNumberInput > div > div > input,
          .stSelectbox > div > div > div:first-child {
            border-radius: 8px;
            border: 1px solid #cbd5e1;
            background: #f8fafc;
            padding: 0.65rem 0.75rem;
            color: #0f283d;
            font-size: 0.95rem;
            font-weight: 500;
            box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.01);
            transition: all 0.2s ease;
          }

          .stTextInput > div > div > input:hover,
          .stNumberInput > div > div > input:hover,
          .stSelectbox > div > div > div:first-child:hover {
            border-color: #94a3b8;
            background: #ffffff;
          }

          .stTextInput > div > div > input:focus,
          .stNumberInput > div > div > input:focus,
          .stSelectbox > div > div > div:first-child:focus-within {
            border-color: #0d9488;
            box-shadow: 0 0 0 1px #0d9488;
            background: #ffffff;
          }

          div[data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid rgba(0,0,0,0.05);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_tile(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-tile">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        <div style="height: 0.5rem;"></div>
        """,
        unsafe_allow_html=True,
    )


def render_panel_open() -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)


def render_panel_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def build_risk_badge(risk_category: str) -> str:
    css = {
        "Low": "risk-low",
        "Moderate": "risk-moderate",
        "High": "risk-high",
    }[risk_category]
    return f'<span class="risk-pill {css}">{risk_category} risk</span>'


def apply_plot_style(figure: go.Figure, height: int = 360) -> go.Figure:
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=20),
        font=dict(family="Inter, sans-serif", color="#334155", size=11),
        title=dict(
            font=dict(size=14, family="Inter, sans-serif", color="#0f283d"),
            pad=dict(b=10),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            gridwidth=1,
            zeroline=False,
            title_font=dict(size=12, color="#475569"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            gridwidth=1,
            zeroline=False,
            title_font=dict(size=12, color="#475569"),
        ),
    )
    return figure


def build_risk_gauge(probability: float, risk_category: str) -> go.Figure:
    gauge_colors = {
        "Low": "#22c55e",
        "Moderate": "#f59e0b",
        "High": "#ef4444",
    }
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={
                "suffix": "%",
                "font": {
                    "size": 36,
                    "family": "Space Grotesk, sans-serif",
                    "color": "#1e1b4b",
                },
            },
            title={
                "text": "<b>Cardiovascular Risk Score</b>",
                "font": {
                    "size": 18,
                    "family": "Manrope, sans-serif",
                    "color": "#64748b",
                },
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#cbd5e1"},
                "bar": {
                    "color": gauge_colors[risk_category],
                    "thickness": 0.75,
                    "line": {"width": 2, "color": "white"},
                },
                "steps": [
                    {"range": [0, 35], "color": "rgba(34,197,94,0.15)"},
                    {"range": [35, 70], "color": "rgba(245,158,11,0.15)"},
                    {"range": [70, 100], "color": "rgba(239,68,68,0.15)"},
                ],
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "bordercolor": "rgba(0,0,0,0)",
            },
        )
    )
    figure.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=60, b=30),
        font=dict(family="Manrope, sans-serif", color="#1e1b4b"),
    )
    return figure


def pretty_transformed_feature(name: str) -> str:
    if "__" not in name:
        return humanize_slug(name)
    _, feature_name = name.split("__", maxsplit=1)
    return humanize_slug(feature_name)


def build_page_header(title: str, subtitle: str, bundle: dict[str, Any] | None, bundle_error: str | None) -> str:
    agent_status = validate_agent_config()
    metrics = bundle["evaluation"]["metrics"] if bundle else None
    app_env = get_env_status()

    metric_message = (
        f"Accuracy {metrics['accuracy']:.1%} • ROC-AUC {metrics['roc_auc']:.1%}"
        if metrics
        else "Model artifacts missing"
    )
    agent_message = (
        f"{agent_status['model_name']}"
        if agent_status["status"] == "llm_enabled"
        else "No API Engine"
    )

    pills = [
        (
            "ok" if not bundle_error else "warn",
            "Artifacts",
            metric_message if not bundle_error else bundle_error,
        ),
        (
            "ok" if agent_status["status"] == "llm_enabled" else "warn",
            "Agent",
            agent_message,
        ),
        (
            "info",
            "Runtime",
            f"{app_env['python_version']} • Local Engine",
        ),
    ]

    pill_markup = "".join(
        f'<span class="status-pill {tone}"><strong>{label}:</strong> {message}</span>'
        for tone, label, message in pills
    )

    return f"""
        <div style="margin-bottom: 2rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1.5rem;">
                <div style="flex: 1; min-width: 300px;">
                    <h2 style="font-family: 'Inter', sans-serif; font-size: 2.1rem; font-weight: 800; color: #0f283d; margin: 0 0 0.4rem 0; letter-spacing: -0.03em; line-height: 1.2;">{title}</h2>
                    <p style="font-size: 1.05rem; color: #475569; margin: 0; max-width: 700px; line-height: 1.6; font-weight: 400;">{subtitle}</p>
                </div>
                <div class="status-strip" style="display:flex; align-items:center; flex-wrap:wrap; gap:0.5rem; justify-content:flex-end;">{pill_markup}</div>
            </div>
        </div>
    """


def build_evaluation_table(
    bundle: dict[str, Any], dataset: pd.DataFrame
) -> pd.DataFrame:
    metrics = bundle["evaluation"]["metrics"]
    summary = summarize_dataset(dataset)
    metadata = bundle.get("metadata", {})

    rows = [
        {
            "Metric": "Model",
            "Value": "Logistic Regression",
            "Context": "Saved deployment artifact",
        },
        {
            "Metric": "Accuracy",
            "Value": format_percent(metrics["accuracy"]),
            "Context": "Hold-out test set",
        },
        {
            "Metric": "Precision",
            "Value": format_percent(metrics["precision"]),
            "Context": "Positive class",
        },
        {
            "Metric": "Recall",
            "Value": format_percent(metrics["recall"]),
            "Context": "Positive class",
        },
        {
            "Metric": "F1 Score",
            "Value": format_percent(metrics["f1_score"]),
            "Context": "Positive class",
        },
        {
            "Metric": "ROC-AUC",
            "Value": format_percent(metrics["roc_auc"]),
            "Context": "Probability quality",
        },
        {
            "Metric": "Clean records",
            "Value": f"{summary['records']:,}",
            "Context": "Processed dataset size",
        },
        {
            "Metric": "Positive class rate",
            "Value": f"{summary['positive_rate']:.1%}",
            "Context": "Higher-risk share",
        },
        {
            "Metric": "Training rows",
            "Value": f"{metadata.get('train_rows', 'N/A')}",
            "Context": "Fit split",
        },
        {
            "Metric": "Test rows",
            "Value": f"{metadata.get('test_rows', 'N/A')}",
            "Context": "Evaluation split",
        },
    ]
    return pd.DataFrame(rows)


def build_dataset_profile_chart(dataset: pd.DataFrame) -> go.Figure:
    summary = pd.DataFrame(
        {
            "Measure": ["Age", "Systolic BP", "Cholesterol", "BMI", "Max Heart Rate"],
            "Average": [
                dataset["age"].mean(),
                dataset["systolic_bp"].mean(),
                dataset["cholesterol"].mean(),
                dataset["bmi"].mean(),
                dataset["max_heart_rate"].mean(),
            ],
        }
    )
    figure = px.bar(
        summary,
        x="Measure",
        y="Average",
        color="Measure",
        color_discrete_sequence=["#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8", "#93c5fd"],
        title="Average Patient Profile",
    )
    figure.update_layout(showlegend=False)
    return apply_plot_style(figure)


def build_roc_chart(evaluation: dict[str, Any]) -> go.Figure:
    metrics = evaluation["metrics"]
    roc_data = evaluation["roc_curve"]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=roc_data["fpr"],
            y=roc_data["tpr"],
            mode="lines",
            name=f"ROC Curve (AUC = {metrics['roc_auc']:.2f})",
            line=dict(color=ACCENT_BLUE, width=3),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline",
            line=dict(color="#94a3b8", dash="dash"),
        )
    )
    figure.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return apply_plot_style(figure)


def build_confusion_chart(evaluation: dict[str, Any]) -> go.Figure:
    figure = px.imshow(
        evaluation["confusion_matrix"],
        text_auto=True,
        color_continuous_scale="Blues",
        x=["Predicted 0", "Predicted 1"],
        y=["Actual 0", "Actual 1"],
        title="Confusion Matrix",
    )
    return apply_plot_style(figure)


def build_coefficient_chart(evaluation: dict[str, Any]) -> go.Figure:
    coefficients = pd.DataFrame(evaluation["coefficients"]["all_coefficients"])
    coefficients["label"] = coefficients["feature"].map(pretty_transformed_feature)
    top_coefficients = coefficients.reindex(
        coefficients["coefficient"].abs().sort_values(ascending=False).index
    ).head(12)
    figure = px.bar(
        top_coefficients.sort_values("coefficient"),
        x="coefficient",
        y="label",
        orientation="h",
        color="coefficient",
        color_continuous_scale="RdBu",
        title="Top Logistic Coefficients",
    )
    figure.update_layout(coloraxis_showscale=False)
    return apply_plot_style(figure, height=460)


def build_risk_distribution_chart(dataset: pd.DataFrame) -> go.Figure:
    risk_distribution = (
        dataset["risk_label"]
        .value_counts()
        .rename_axis("Risk label")
        .reset_index(name="count")
    )
    figure = px.bar(
        risk_distribution,
        x="Risk label",
        y="count",
        color="Risk label",
        color_discrete_map={"Lower risk": "#60a5fa", "Higher risk": "#ef4444"},
        title="Risk Class Distribution",
    )
    figure.update_layout(showlegend=False)
    return apply_plot_style(figure)


def render_model_dashboard(
    dataset: pd.DataFrame, bundle: dict[str, Any] | None, bundle_error: str | None
) -> None:
    st.markdown(
        build_page_header(
            "Key Performance Analytics",
            "The redesigned shell is in place. Dashboard analytics are being layered in next.",
            bundle,
            bundle_error,
        ),
        unsafe_allow_html=True,
    )
    render_panel_open()
    st.info("Dashboard analytics will be added in the next commit.")
    render_panel_close()

def render_eda_tab(
    dataset: pd.DataFrame, bundle: dict[str, Any] | None, bundle_error: str | None
) -> None:
    st.markdown(
        build_page_header(
            "Cohort Data Explorer",
            "The updated EDA workspace will be added in the next commit.",
            bundle,
            bundle_error,
        ),
        unsafe_allow_html=True,
    )
    render_panel_open()
    st.info("EDA charts are staged for the next commit.")
    render_panel_close()

def render_patient_input_fields(
    prefix: str, defaults: dict[str, Any] | None = None
) -> dict[str, Any]:
    patient_defaults = defaults or default_patient_profile()
    values: dict[str, Any] = {}

    st.markdown('<div class="section-kicker" style="border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4rem; margin-top: 0; margin-bottom: 1.5rem;">Clinical Measurements</div>', unsafe_allow_html=True)
    num_cols = st.columns(2)
    for idx, feature in enumerate(NUMERIC_FEATURES):
        with num_cols[idx % 2]:
            metadata = FEATURE_METADATA[feature]
            values[feature] = st.number_input(
                metadata["label"],
                min_value=float(metadata["min"]),
                max_value=float(metadata["max"]),
                value=float(patient_defaults[feature]),
                step=float(metadata["step"]),
                key=f"{prefix}_{feature}",
            )

    st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker" style="border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4rem; margin-top: 1rem; margin-bottom: 1.5rem;">Behavioral & Risk Factors</div>', unsafe_allow_html=True)
    cat_cols = st.columns(2)
    for idx, feature in enumerate(CATEGORICAL_FEATURES):
        with cat_cols[idx % 2]:
            metadata = FEATURE_METADATA[feature]
            current_value = str(patient_defaults[feature])
            values[feature] = st.selectbox(
                metadata["label"],
                options=metadata["options"],
                index=metadata["options"].index(current_value),
                key=f"{prefix}_{feature}",
            )

    return values


def render_prediction_result(result: dict[str, Any], patient: dict[str, Any]) -> None:
    st.markdown(build_risk_badge(result["risk_category"]), unsafe_allow_html=True)
    st.plotly_chart(
        build_risk_gauge(result["probability"], result["risk_category"]),
        use_container_width=True,
    )

    metric_columns = st.columns(2)
    with metric_columns[0]:
        render_metric_tile(
            "Risk Probability",
            f"{result['probability']:.1%}",
            "Predicted positive-class probability",
        )
    with metric_columns[1]:
        render_metric_tile(
            "Predicted Class",
            str(result["predicted_class"]),
            "0 = lower risk, 1 = higher risk",
        )

    st.markdown("#### Key Drivers")
    if result["important_features"]:
        feature_frame = pd.DataFrame(result["important_features"])[
            ["feature", "direction", "contribution"]
        ]
        feature_frame["contribution"] = feature_frame["contribution"].map(
            lambda value: f"{value:+.3f}"
        )
        st.dataframe(feature_frame, use_container_width=True, hide_index=True)
    else:
        st.info("No driver explanation was generated for this prediction.")

    st.markdown("#### Patient Summary")
    summary_frame = pd.DataFrame(
        {
            "Feature": [
                FEATURE_METADATA[feature]["label"] for feature in FEATURE_COLUMNS
            ],
            "Value": [
                format_feature_value(feature, patient[feature])
                for feature in FEATURE_COLUMNS
            ],
        }
    )
    st.dataframe(summary_frame, use_container_width=True, hide_index=True)


def render_single_prediction_tab(
    bundle: dict[str, Any] | None, bundle_error: str | None
) -> None:
    st.markdown(
        build_page_header(
            "Patient Profiling",
            "The new prediction workspace will be enabled in the next commit.",
            bundle,
            bundle_error,
        ),
        unsafe_allow_html=True,
    )
    render_panel_open()
    st.info("Single-patient prediction controls are staged for the next commit.")
    render_panel_close()

def render_batch_scoring_tab(
    bundle: dict[str, Any] | None, bundle_error: str | None, dataset: pd.DataFrame
) -> None:
    st.markdown(
        build_page_header(
            "Batch Inference Payload",
            "Batch scoring will be wired into the redesigned UI in the next commit.",
            bundle,
            bundle_error,
        ),
        unsafe_allow_html=True,
    )
    render_panel_open()
    st.info("Batch CSV scoring is staged for the next commit.")
    render_panel_close()

def render_agentic_health_tab(
    bundle: dict[str, Any] | None, bundle_error: str | None
) -> None:
    st.markdown(
        build_page_header(
            "Clinical Strategist Agent",
            "The grounded strategist experience will be finalized in the next commit.",
            bundle,
            bundle_error,
        ),
        unsafe_allow_html=True,
    )
    render_panel_open()
    st.info("Agent workflow presentation is staged for the final commit.")
    render_panel_close()

def main() -> None:
    st.set_page_config(
        page_title="CardioRisk AI",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    dataset = get_dashboard_dataset()

    bundle: dict[str, Any] | None
    bundle_error: str | None = None
    try:
        bundle = get_model_bundle()
    except FileNotFoundError as exc:
        bundle = None
        bundle_error = str(exc)

    st.sidebar.markdown(
        f'''
        <div class="sidebar-brand">
          <h1>{APP_TITLE.split(":")[0]}</h1>
          <p>{APP_TITLE.split(":")[1].strip()}</p>
        </div>
        ''',
        unsafe_allow_html=True
    )

    NAV_OPTIONS = {
        "System Dashboard": "Model Dashboard",
        "Cohort Explorer": "Exploratory Data Analysis",
        "Patient Profiling": "Single Patient Prediction",
        "Batch Inference": "Batch CSV Scoring",
        "Clinical Strategist": "Agentic Health Strategist",
    }

    selected_nav = st.sidebar.radio(
        "Navigation",
        options=list(NAV_OPTIONS.keys()),
        label_visibility="collapsed",
        key="nav_radio"
    )
    
    page = NAV_OPTIONS[selected_nav]

    if page == "Model Dashboard":
        render_model_dashboard(dataset, bundle, bundle_error)
    elif page == "Exploratory Data Analysis":
        render_eda_tab(dataset, bundle, bundle_error)
    elif page == "Single Patient Prediction":
        render_single_prediction_tab(bundle, bundle_error)
    elif page == "Batch CSV Scoring":
        render_batch_scoring_tab(bundle, bundle_error, dataset)
    elif page == "Agentic Health Strategist":
        render_agentic_health_tab(bundle, bundle_error)


if __name__ == "__main__":
    main()
