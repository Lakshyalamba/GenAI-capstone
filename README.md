---
title: CardioRisk AI
emoji: "🫀"
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.40.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# CardioRisk AI

CardioRisk AI is a deployment-ready cardiovascular health risk prediction system built around a logistic regression model, a saved preprocessing pipeline, a multi-tab Streamlit application, and a retrieval-grounded cardiovascular assistant.

The project uses 10,000 synthetic patient records, cleans them into 9,500 model-ready records, and predicts cardiovascular risk from age, systolic blood pressure, cholesterol, heart-rate capacity, BMI, sex, chest-pain type, smoking status, diabetes status, and exercise-induced angina.

## Problem Statement

The original project demonstrated the prediction task, but it was still organized like a student notebook demo. This refactor separates training from runtime, saves artifacts for deployment, adds testing, introduces a grounded health-guidance workflow, and organizes the codebase so it reads like a real ML product.

## What Is Included

- A standalone `train.py` script for preprocessing, training, evaluation, and artifact generation.
- A root `app.py` Streamlit runtime that only loads saved artifacts and serves the dashboard.
- Modular inference utilities for single-record and batch scoring.
- A grounded cardiovascular agent workflow that retrieves from local markdown knowledge before answering.
- Local markdown knowledge files for diet, exercise, blood pressure, cholesterol, preventive care, warning signs, lifestyle habits, and follow-up monitoring.
- Pytest coverage for inference, retrieval, and agent configuration.

## Architecture

The application is structured around four clean layers:

1. Data and training
   `train.py` loads raw data, cleans it, splits it, fits the preprocessor and logistic regression model, evaluates performance, and saves the artifacts into `models/`.

2. Inference and evaluation
   `src/inference.py` loads saved artifacts and exposes reusable scoring functions such as `predict_single`, `predict_batch`, `get_risk_category`, and `explain_top_risk_factors`.

3. Agent workflow
   `src/agent/workflow.py` runs a step-based cardiovascular assistant workflow:
   `route_request -> prepare_input -> score_patient_risk -> summarize_risk -> retrieve_health_guidance -> generate_recommendations -> validate_output -> answer_follow_up`

4. UI and deployment
   `app.py` renders the production-style Streamlit interface with overview, dataset insights, real-time prediction, grounded agent insight, and model-performance tabs.

## Folder Structure

```text
GenAI-capstone/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   ├── raw/
│   │   └── synthetic_health.csv
│   └── processed/
│       └── cardio_clean.csv
├── models/
│   ├── logistic_regression_model.joblib
│   ├── scaler.joblib
│   ├── feature_columns.json
│   ├── model_metadata.json
│   └── evaluation_summary.json
├── src/
│   ├── data_processing.py
│   ├── evaluation.py
│   ├── features.py
│   ├── inference.py
│   ├── utils.py
│   └── agent/
│       ├── config.py
│       ├── prompts.py
│       ├── retrieval.py
│       └── workflow.py
├── knowledge_base/
│   ├── lifestyle_recommendations.md
│   ├── bp_management.md
│   ├── cholesterol_guidance.md
│   ├── exercise_guidance.md
│   ├── diet_guidance.md
│   ├── preventive_care.md
│   ├── warning_signs.md
│   └── follow_up_monitoring.md
└── tests/
    ├── test_inference.py
    ├── test_agent_config.py
    └── test_retrieval.py
```

## Machine Learning Pipeline

The training workflow includes:

- Raw data loading from `data/raw/synthetic_health.csv`
- Duplicate removal
- Target cleanup and category normalization
- Feature selection through the curated clinical feature list
- Stratified train/test split
- Median imputation and scaling for numeric features
- Most-frequent imputation and one-hot encoding for categorical features
- Logistic regression training
- Evaluation with:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Confusion matrix
  - ROC curve data
  - Coefficient export for explainability

## Streamlit Dashboard

The Streamlit app has five deployment-ready sections:

- `Overview`
  Dataset summary, environment status, artifact readiness, and architecture highlights.
- `Dataset Insights`
  Risk distribution, numeric feature histograms, smoking prevalence, box plots, and a correlation heatmap.
- `Real-Time Prediction`
  Patient input form, real-time risk score, probability gauge, risk tier, contributing feature summary, and validated input display.
- `Agentic Health Insight`
  Step-based cardiovascular assistant with grounded retrieval and optional follow-up answers.
- `Model Performance`
  Core evaluation metrics, confusion matrix, ROC curve, and logistic-regression coefficient visualization.

## Agent Workflow

The assistant is not a generic chatbot. It is constrained to cardiovascular-risk support and grounded recommendations.

Workflow nodes:

- `route_request`
- `prepare_input`
- `score_patient_risk`
- `summarize_risk`
- `retrieve_health_guidance`
- `generate_recommendations`
- `validate_output`
- `answer_follow_up`

The workflow first scores the patient, then retrieves relevant markdown guidance from the local knowledge base, then produces a validated cardiovascular guidance response. If `GEMINI_API_KEY` is available and the `google-genai` package is installed, the app can optionally refine the grounded response with Gemini Flash. If not, it falls back gracefully to a grounded rule-based mode.

## Environment and Secrets

Optional LLM enhancement uses:

- `GEMINI_API_KEY`
- `CARDIO_AGENT_MODEL` (optional, defaults to `gemini-2.5-flash`)
- `APP_ENV` (optional, defaults to `local`)

If the API key is missing, the UI explicitly reports that the assistant is running in grounded fallback mode.

## Setup

Python compatibility: `Python 3.11+`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Generate the cleaned dataset and deployment artifacts:

```bash
python train.py
```

Artifacts are saved to `models/` and the processed dataset is written to `data/processed/cardio_clean.csv`.

## Run the Streamlit App

```bash
streamlit run app.py
```

The legacy path `dashboard/app.py` now forwards to the root app for compatibility with older deployment settings.

## Tests

Run the test suite with:

```bash
pytest
```

Current tests cover:

- Model artifact loading
- Single-record inference
- Batch inference
- Bad input validation
- Missing artifact behavior
- Agent configuration fallback behavior
- Knowledge-base retrieval routing

## Deployment Notes

- The app does not retrain at startup.
- Saved artifacts are loaded from `models/`.
- `.streamlit/config.toml` includes a production-ready theme.
- `.gitignore` excludes secrets and local virtual environments.
- The agent gracefully degrades when LLM credentials are absent.

## Future Improvements

- Add SHAP or calibrated probability interpretation for richer explainability.
- Swap the simple markdown retriever for embeddings-based semantic retrieval.
- Add model monitoring and prediction logging for production observability.
- Support clinician-oriented PDF export for risk reports.
- Introduce CI to run training validation and tests on every push.
