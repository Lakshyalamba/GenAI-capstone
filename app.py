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
APP_SUBTITLE = (
    "Logistic-regression risk scoring with a retrieval-grounded cardiovascular guidance workflow."
)
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
    return {feature: metadata["default"] for feature, metadata in FEATURE_METADATA.items()}


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
          @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

          html, body, [class*="css"] {
            font-family: 'Manrope', sans-serif;
          }

          .stApp {
            background:
              radial-gradient(circle at top left, rgba(59,130,246,0.10), transparent 28%),
              linear-gradient(180deg, #f7f9fc 0%, #eef3f9 100%);
          }

          .block-container {
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 2.5rem;
          }

          [data-testid="stSidebar"] {
            display: none;
          }

          #MainMenu, footer, header {
            visibility: hidden;
          }

          .app-header {
            background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(255,255,255,0.88));
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 24px;
            padding: 1.55rem 1.7rem;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
          }

          .app-title {
            font-size: 2rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.04em;
            margin: 0;
          }

          .app-subtitle {
            margin-top: 0.45rem;
            font-size: 1rem;
            color: #475569;
          }

          .status-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1rem;
          }

          .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.48rem 0.85rem;
            border-radius: 999px;
            font-size: 0.83rem;
            font-weight: 700;
            border: 1px solid rgba(148,163,184,0.20);
            background: #f8fafc;
            color: #334155;
          }

          .status-pill.ok {
            background: rgba(22,163,74,0.10);
            color: #166534;
            border-color: rgba(22,163,74,0.15);
          }

          .status-pill.warn {
            background: rgba(217,119,6,0.10);
            color: #92400e;
            border-color: rgba(217,119,6,0.15);
          }

          .status-pill.info {
            background: rgba(59,130,246,0.10);
            color: #1d4ed8;
            border-color: rgba(59,130,246,0.15);
          }

          .panel {
            background: rgba(255,255,255,0.94);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 22px;
            padding: 1.15rem 1.2rem;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.05);
          }

          .metric-tile {
            background: rgba(255,255,255,0.97);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 20px;
            padding: 1rem 1rem 0.95rem;
            box-shadow: 0 10px 22px rgba(15,23,42,0.04);
          }

          .metric-label {
            font-size: 0.76rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
          }

          .metric-value {
            margin-top: 0.35rem;
            font-size: 1.75rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.04em;
          }

          .metric-note {
            margin-top: 0.22rem;
            font-size: 0.88rem;
            color: #64748b;
          }

          .section-kicker {
            margin-top: 0.2rem;
            margin-bottom: 0.8rem;
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #3b82f6;
          }

          .risk-pill {
            display: inline-block;
            padding: 0.42rem 0.85rem;
            border-radius: 999px;
            font-size: 0.84rem;
            font-weight: 800;
            margin-bottom: 0.8rem;
          }

          .risk-low {
            background: rgba(22,163,74,0.10);
            color: #166534;
          }

          .risk-moderate {
            background: rgba(217,119,6,0.10);
            color: #92400e;
          }

          .risk-high {
            background: rgba(220,38,38,0.10);
            color: #991b1b;
          }

          .soft-note {
            color: #64748b;
            margin-bottom: 1rem;
          }

          [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.45rem;
            background: rgba(255,255,255,0.70);
            padding: 0.45rem;
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,0.18);
          }

          [data-testid="stTabs"] [data-baseweb="tab"] {
            border-radius: 14px;
            height: 44px;
            padding-left: 1rem;
            padding-right: 1rem;
            background: transparent;
            color: #475569;
            font-weight: 700;
          }

          [data-testid="stTabs"] [aria-selected="true"] {
            background: #e8f0ff;
            color: #1d4ed8;
          }

          div[data-testid="stFileUploader"] section {
            border-radius: 18px;
            border: 1px dashed rgba(148,163,184,0.45);
            background: rgba(255,255,255,0.94);
          }

          div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
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
        plot_bgcolor=PLOT_BACKGROUND,
        margin=dict(l=18, r=18, t=56, b=18),
        font=dict(family="Manrope, sans-serif", color="#0f172a"),
    )
    return figure


def build_risk_gauge(probability: float, risk_category: str) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 32}},
            title={"text": "Estimated Cardiovascular Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": RISK_COLORS[risk_category]},
                "steps": [
                    {"range": [0, 35], "color": "rgba(22,163,74,0.12)"},
                    {"range": [35, 70], "color": "rgba(217,119,6,0.12)"},
                    {"range": [70, 100], "color": "rgba(220,38,38,0.12)"},
                ],
            },
        )
    )
    return apply_plot_style(figure, height=300)


def pretty_transformed_feature(name: str) -> str:
    if "__" not in name:
        return humanize_slug(name)
    _, feature_name = name.split("__", maxsplit=1)
    return humanize_slug(feature_name)


def render_header(bundle: dict[str, Any] | None, bundle_error: str | None) -> None:
    agent_status = validate_agent_config()
    metrics = bundle["evaluation"]["metrics"] if bundle else None
    app_env = get_env_status()

    metric_message = (
        f"Model ready • Accuracy {metrics['accuracy']:.1%} • ROC-AUC {metrics['roc_auc']:.1%}"
        if metrics
        else "Model artifacts missing"
    )
    agent_message = (
        f"LLM enhancement available via {agent_status['model_name']}"
        if agent_status["status"] == "llm_enabled"
        else "Grounded fallback mode active"
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
            f"Python {app_env['python_version']} • {app_env['app_env']}",
        ),
    ]

    pill_markup = "".join(
        f'<span class="status-pill {tone}"><strong>{label}:</strong> {message}</span>'
        for tone, label, message in pills
    )

    st.markdown(
        f"""
        <div class="app-header">
          <div class="app-title">{APP_TITLE}</div>
          <div class="app-subtitle">{APP_SUBTITLE}</div>
          <div class="status-strip">{pill_markup}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_evaluation_table(bundle: dict[str, Any], dataset: pd.DataFrame) -> pd.DataFrame:
    metrics = bundle["evaluation"]["metrics"]
    summary = summarize_dataset(dataset)
    metadata = bundle.get("metadata", {})

    rows = [
        {"Metric": "Model", "Value": "Logistic Regression", "Context": "Saved deployment artifact"},
        {"Metric": "Accuracy", "Value": format_percent(metrics["accuracy"]), "Context": "Hold-out test set"},
        {"Metric": "Precision", "Value": format_percent(metrics["precision"]), "Context": "Positive class"},
        {"Metric": "Recall", "Value": format_percent(metrics["recall"]), "Context": "Positive class"},
        {"Metric": "F1 Score", "Value": format_percent(metrics["f1_score"]), "Context": "Positive class"},
        {"Metric": "ROC-AUC", "Value": format_percent(metrics["roc_auc"]), "Context": "Probability quality"},
        {"Metric": "Clean records", "Value": f"{summary['records']:,}", "Context": "Processed dataset size"},
        {"Metric": "Positive class rate", "Value": f"{summary['positive_rate']:.1%}", "Context": "Higher-risk share"},
        {"Metric": "Training rows", "Value": f"{metadata.get('train_rows', 'N/A')}", "Context": "Fit split"},
        {"Metric": "Test rows", "Value": f"{metadata.get('test_rows', 'N/A')}", "Context": "Evaluation split"},
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
    figure.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
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


def render_model_dashboard(dataset: pd.DataFrame, bundle: dict[str, Any] | None, bundle_error: str | None) -> None:
    st.markdown('<div class="section-kicker">Model Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="soft-note">A compact summary inspired by the reference Space: evaluation table first, supporting visuals second.</p>',
        unsafe_allow_html=True,
    )

    if bundle_error or bundle is None:
        st.error(bundle_error or "Model artifacts are not available.")
        st.code("python train.py")
        return

    summary = summarize_dataset(dataset)
    metrics = bundle["evaluation"]["metrics"]

    top_metrics = st.columns(4)
    metric_specs = [
        ("Accuracy", format_percent(metrics["accuracy"]), "Hold-out test set"),
        ("ROC-AUC", format_percent(metrics["roc_auc"]), "Probability discrimination"),
        ("Clean records", f"{summary['records']:,}", "Processed synthetic patients"),
        ("Positive rate", f"{summary['positive_rate']:.1%}", "Higher-risk label share"),
    ]
    for column, spec in zip(top_metrics, metric_specs, strict=False):
        with column:
            render_metric_tile(*spec)

    info_columns = st.columns([1.15, 0.85])
    with info_columns[0]:
        render_panel_open()
        st.markdown("#### Evaluation Summary")
        st.dataframe(build_evaluation_table(bundle, dataset), use_container_width=True, hide_index=True)
        render_panel_close()
    with info_columns[1]:
        render_panel_open()
        st.markdown("#### Runtime Notes")
        st.write("- Training and Streamlit runtime are separated.")
        st.write("- The app loads artifacts from `models/` without retraining.")
        st.write("- Agent responses are retrieval-grounded before recommendations are generated.")
        st.write("- The model expects validated cardiovascular profile inputs only.")
        render_panel_close()

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.plotly_chart(build_roc_chart(bundle["evaluation"]), use_container_width=True)
    with chart_columns[1]:
        st.plotly_chart(build_confusion_chart(bundle["evaluation"]), use_container_width=True)

    lower_columns = st.columns(2)
    with lower_columns[0]:
        st.plotly_chart(build_coefficient_chart(bundle["evaluation"]), use_container_width=True)
    with lower_columns[1]:
        st.plotly_chart(build_risk_distribution_chart(dataset), use_container_width=True)
        st.plotly_chart(build_dataset_profile_chart(dataset), use_container_width=True)


def render_eda_tab(dataset: pd.DataFrame) -> None:
    st.markdown('<div class="section-kicker">EDA</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="soft-note">The reference app uses a dedicated exploratory tab. This version keeps the same separation and focuses on the cardiovascular feature profile.</p>',
        unsafe_allow_html=True,
    )

    selector_columns = st.columns([0.55, 0.45])
    with selector_columns[0]:
        numeric_feature = st.selectbox(
            "Inspect numeric feature",
            options=NUMERIC_FEATURES,
            format_func=lambda feature: FEATURE_METADATA[feature]["label"],
        )

    chart_columns = st.columns(2)
    with chart_columns[0]:
        figure = px.histogram(
            dataset,
            x=numeric_feature,
            color="risk_label",
            barmode="overlay",
            nbins=28,
            color_discrete_map={"Lower risk": "#60a5fa", "Higher risk": "#ef4444"},
            title=f"{FEATURE_METADATA[numeric_feature]['label']} Distribution",
        )
        st.plotly_chart(apply_plot_style(figure), use_container_width=True)
    with chart_columns[1]:
        figure = px.box(
            dataset,
            x="risk_label",
            y="cholesterol",
            color="risk_label",
            color_discrete_map={"Lower risk": "#60a5fa", "Higher risk": "#ef4444"},
            title="Cholesterol by Risk Label",
        )
        figure.update_layout(showlegend=False)
        st.plotly_chart(apply_plot_style(figure), use_container_width=True)

    lower_columns = st.columns(2)
    with lower_columns[0]:
        smoker_breakdown = dataset.groupby(["risk_label", "smoker"]).size().reset_index(name="count")
        figure = px.bar(
            smoker_breakdown,
            x="risk_label",
            y="count",
            color="smoker",
            barmode="group",
            color_discrete_map={"Yes": "#ef4444", "No": "#60a5fa"},
            title="Smoking Status by Risk Group",
        )
        st.plotly_chart(apply_plot_style(figure), use_container_width=True)
    with lower_columns[1]:
        scatter_frame = dataset.copy()
        scatter_frame["bmi_marker"] = scatter_frame["bmi"].fillna(scatter_frame["bmi"].median())
        scatter_frame = scatter_frame.dropna(subset=["systolic_bp", "cholesterol", "risk_label"])
        figure = px.scatter(
            scatter_frame,
            x="systolic_bp",
            y="cholesterol",
            color="risk_label",
            size="bmi_marker",
            opacity=0.55,
            color_discrete_map={"Lower risk": "#60a5fa", "Higher risk": "#ef4444"},
            title="Blood Pressure vs Cholesterol",
        )
        st.plotly_chart(apply_plot_style(figure), use_container_width=True)

    correlation = dataset[NUMERIC_FEATURES + ["risk"]].corr(numeric_only=True)
    heatmap = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Blues",
        title="Feature Correlation Heatmap",
    )
    st.plotly_chart(apply_plot_style(heatmap, height=430), use_container_width=True)


def render_patient_input_fields(prefix: str, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    patient_defaults = defaults or default_patient_profile()
    values: dict[str, Any] = {}

    columns = st.columns(2)
    with columns[0]:
        st.markdown("#### Clinical Measurements")
        for feature in NUMERIC_FEATURES:
            metadata = FEATURE_METADATA[feature]
            values[feature] = st.number_input(
                metadata["label"],
                min_value=float(metadata["min"]),
                max_value=float(metadata["max"]),
                value=float(patient_defaults[feature]),
                step=float(metadata["step"]),
                key=f"{prefix}_{feature}",
            )
    with columns[1]:
        st.markdown("#### Risk Factors")
        for feature in CATEGORICAL_FEATURES:
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
    st.plotly_chart(build_risk_gauge(result["probability"], result["risk_category"]), use_container_width=True)

    metric_columns = st.columns(2)
    with metric_columns[0]:
        render_metric_tile("Risk Probability", f"{result['probability']:.1%}", "Predicted positive-class probability")
    with metric_columns[1]:
        render_metric_tile("Predicted Class", str(result["predicted_class"]), "0 = lower risk, 1 = higher risk")

    st.markdown("#### Key Drivers")
    if result["important_features"]:
        feature_frame = pd.DataFrame(result["important_features"])[["feature", "direction", "contribution"]]
        feature_frame["contribution"] = feature_frame["contribution"].map(lambda value: f"{value:+.3f}")
        st.dataframe(feature_frame, use_container_width=True, hide_index=True)
    else:
        st.info("No driver explanation was generated for this prediction.")

    st.markdown("#### Patient Summary")
    summary_frame = pd.DataFrame(
        {
            "Feature": [FEATURE_METADATA[feature]["label"] for feature in FEATURE_COLUMNS],
            "Value": [format_feature_value(feature, patient[feature]) for feature in FEATURE_COLUMNS],
        }
    )
    st.dataframe(summary_frame, use_container_width=True, hide_index=True)


def render_single_prediction_tab(bundle: dict[str, Any] | None, bundle_error: str | None) -> None:
    st.markdown('<div class="section-kicker">Single Patient Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="soft-note">This tab mirrors the reference app flow: complete the patient form, run scoring, and review the result in a clean split layout.</p>',
        unsafe_allow_html=True,
    )

    if bundle_error or bundle is None:
        st.error(bundle_error or "Model artifacts are not available.")
        return

    form_column, output_column = st.columns([1.05, 0.95])
    with form_column:
        defaults = st.session_state.get("single_patient_profile", default_patient_profile())
        with st.form("single_prediction_form"):
            patient = render_patient_input_fields("single", defaults=defaults)
            submitted = st.form_submit_button("Calculate Cardiovascular Risk", use_container_width=True)
        if submitted:
            try:
                result = predict_single(patient, bundle=bundle)
                st.session_state["single_patient_profile"] = result["validated_input"]
                st.session_state["single_prediction_result"] = result
                st.success("Prediction completed.")
            except ValueError as exc:
                st.error(str(exc))

    with output_column:
        result = st.session_state.get("single_prediction_result")
        patient = st.session_state.get("single_patient_profile", default_patient_profile())
        if not result:
            st.info("Submit the form to calculate risk for a patient profile.")
        else:
            render_prediction_result(result, patient)


def render_batch_scoring_tab(bundle: dict[str, Any] | None, bundle_error: str | None, dataset: pd.DataFrame) -> None:
    st.markdown('<div class="section-kicker">Batch CSV Scoring</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="soft-note">The reference Space includes CSV scoring. This tab brings the same workflow to the cardiovascular project using the saved model artifacts.</p>',
        unsafe_allow_html=True,
    )

    if bundle_error or bundle is None:
        st.error(bundle_error or "Model artifacts are not available.")
        return

    st.markdown("**Required columns:** `" + "`, `".join(FEATURE_COLUMNS) + "`")

    template_csv = dataset[FEATURE_COLUMNS].head(12).to_csv(index=False).encode("utf-8")
    actions = st.columns([0.72, 0.28])
    with actions[1]:
        st.download_button(
            "Download Sample CSV",
            data=template_csv,
            file_name="cardio_batch_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    uploaded_file = st.file_uploader("Upload patient CSV", type=["csv"], key="batch_file_uploader")
    score_clicked = st.button("Score Uploaded CSV", use_container_width=True)

    if score_clicked:
        if uploaded_file is None:
            st.warning("Upload a CSV file first.")
        else:
            try:
                input_df = pd.read_csv(uploaded_file)
                scored_df = predict_batch(input_df, bundle=bundle)
                st.session_state["batch_scored_df"] = scored_df
                st.success(
                    f"Scored {len(scored_df)} patient rows. "
                    f"Flagged {(scored_df['risk_category'] == 'High').sum()} as high risk."
                )
            except Exception as exc:
                st.error(f"CSV scoring failed: {exc}")

    scored_df = st.session_state.get("batch_scored_df")
    if scored_df is not None:
        summary_columns = st.columns(3)
        with summary_columns[0]:
            render_metric_tile("Rows scored", f"{len(scored_df):,}", "Uploaded patient profiles")
        with summary_columns[1]:
            render_metric_tile(
                "High risk",
                f"{(scored_df['risk_category'] == 'High').sum():,}",
                "Profiles above the high-risk threshold",
            )
        with summary_columns[2]:
            render_metric_tile(
                "Average probability",
                f"{scored_df['probability'].mean():.1%}",
                "Batch-level mean risk score",
            )

        st.dataframe(scored_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Scored CSV",
            data=scored_df.to_csv(index=False).encode("utf-8"),
            file_name="cardio_batch_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_agentic_health_tab(bundle: dict[str, Any] | None, bundle_error: str | None) -> None:
    st.markdown('<div class="section-kicker">Agentic Health Strategist</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="soft-note">Like the reference app, this section combines model output with an agent-style workflow. Here the agent stays constrained to cardiovascular guidance and uses the local markdown knowledge base before responding.</p>',
        unsafe_allow_html=True,
    )

    agent_status = validate_agent_config()
    if agent_status["status"] == "llm_enabled":
        st.success(f"Optional LLM enhancement is available via `{agent_status['model_name']}`.")
    else:
        st.info("No API key detected. The assistant is using grounded fallback mode.")

    if bundle_error or bundle is None:
        st.error(bundle_error or "Model artifacts are not available.")
        return

    defaults = st.session_state.get("agent_patient_profile", default_patient_profile())
    with st.form("agent_generation_form"):
        focus_query = st.text_input(
            "Health focus",
            value=st.session_state.get("agent_focus_query", ""),
            placeholder="Example: Prioritize the first changes this patient should make to reduce cardiovascular risk.",
        )
        patient = render_patient_input_fields("agent", defaults=defaults)
        generate = st.form_submit_button("Generate Grounded Health Report", use_container_width=True)

    if generate:
        try:
            response = run_agent_workflow(patient_data=patient, question=focus_query, bundle=bundle)
            st.session_state["agent_patient_profile"] = patient
            st.session_state["agent_focus_query"] = focus_query
            st.session_state["agent_response"] = response
            st.success(
                f"Generated a {response['prediction']['risk_category'].lower()}-risk report "
                f"at {response['prediction']['probability']:.1%} using {response['generation_mode']} mode."
            )
        except Exception as exc:
            st.error(f"Agent run failed: {exc}")

    response = st.session_state.get("agent_response")
    if response:
        left, right = st.columns([1.15, 0.85])
        with left:
            render_panel_open()
            st.markdown("#### Structured Health Report")
            st.markdown(build_risk_badge(response["prediction"]["risk_category"]), unsafe_allow_html=True)
            st.write(response["summary"])
            st.write(response["guidance_note"])
            st.markdown("#### Recommended Actions")
            for item in response["recommendations"]:
                st.write(f"- {item}")
            st.caption(response["disclaimer"])
            render_panel_close()
        with right:
            render_panel_open()
            st.markdown("#### Retrieved Guidance")
            for document in response["kb_documents"]:
                st.markdown(
                    f"**{document['title']}**  \n"
                    f"`{document['source']}`  \n"
                    f"{document['snippet']}"
                )
                st.divider()
            render_panel_close()

        st.markdown("#### Workflow Trace")
        trace_df = pd.DataFrame(
            {"Step": list(range(1, len(response["workflow_trace"]) + 1)), "Node": response["workflow_trace"]}
        )
        st.dataframe(trace_df, use_container_width=True, hide_index=True)

        st.markdown("#### Follow-up Q&A")
        follow_up_columns = st.columns([0.82, 0.18])
        with follow_up_columns[0]:
            follow_up_question = st.text_input(
                "Ask a cardiovascular follow-up question",
                key="agent_follow_up_question",
                placeholder="Example: Which lifestyle change matters most for this patient first?",
            )
        with follow_up_columns[1]:
            ask_follow_up = st.button("Ask Follow-up", use_container_width=True)

        if ask_follow_up:
            if not follow_up_question.strip():
                st.warning("Enter a follow-up question first.")
            else:
                try:
                    follow_up_response = run_agent_workflow(
                        patient_data=st.session_state["agent_patient_profile"],
                        question=follow_up_question,
                        bundle=bundle,
                    )
                    st.session_state["agent_follow_up_response"] = follow_up_response
                except Exception as exc:
                    st.error(f"Follow-up failed: {exc}")

        follow_up_response = st.session_state.get("agent_follow_up_response")
        if follow_up_response:
            st.markdown(follow_up_response["follow_up_answer"] or "No follow-up answer generated.")


def main() -> None:
    st.set_page_config(
        page_title="CardioRisk AI",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="collapsed",
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

    render_header(bundle, bundle_error)

    tabs = st.tabs(
        [
            "Model Dashboard",
            "EDA",
            "Single Patient Prediction",
            "Batch CSV Scoring",
            "Agentic Health Strategist",
        ]
    )

    with tabs[0]:
        render_model_dashboard(dataset, bundle, bundle_error)
    with tabs[1]:
        render_eda_tab(dataset)
    with tabs[2]:
        render_single_prediction_tab(bundle, bundle_error)
    with tabs[3]:
        render_batch_scoring_tab(bundle, bundle_error, dataset)
    with tabs[4]:
        render_agentic_health_tab(bundle, bundle_error)


if __name__ == "__main__":
    main()
