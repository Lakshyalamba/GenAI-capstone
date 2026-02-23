# CardioRisk AI — Cardiovascular Risk Dashboard

**Live Demo:** [huggingface.co/spaces/lakshyalamba/cardiorisk-ai](https://huggingface.co/spaces/lakshyalamba/cardiorisk-ai)

A machine learning project that predicts **cardiovascular health risk** using patient data. Built with Logistic Regression trained on 400 synthetic patient records. Features a Streamlit dashboard for interactive analysis and real-time predictions.

---

## Project Structure

```
genai capstone/
├── Capstone.ipynb        # EDA + model training notebook
├── data/
│   └── synthetic_health.csv  # Dataset (400 records, 11 features)
├── dashboard/
│   ├── app.py            # Streamlit dashboard (3 pages)
│   └── requirements.txt  # Python dependencies
└── README.md
```

---

## Dashboard Pages

| Page | What it shows |
|---|---|
| **KPI Dashboard** | Accuracy (92.5%), Precision (92%), Recall (93%), F1 (92%), AUC-ROC (89%) cards + dataset stats + distribution charts |
| **Visual Analysis** | ROC Curve (AUC=0.89), Confusion Matrix heatmap, Logistic Regression Feature Weights |
| **Prediction System** | Patient input form → real-time risk score gauge + key risk indicators |

---

## Model Performance

| Metric | Value |
|---|---|
| Accuracy | **92.5%** |
| Precision | **92%** |
| Recall | **93%** |
| F1 Score | **92%** |
| AUC-ROC | **0.89** |

---

## Dataset

**File:** `data/synthetic_health.csv` · **400 rows** · **11 columns**

| Feature | Type |
|---|---|
| `age`, `systolic_bp`, `cholesterol`, `max_heart_rate`, `bmi` | Numerical |
| `sex`, `chest_pain`, `smoker`, `diabetes`, `exercise_angina` | Categorical |
| `risk` | Target (0 = Low, 1 = High) |

---

## How to Run

### Option 1 — Streamlit Dashboard (local)

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

### Option 2 — Notebook (Google Colab)

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `Capstone.ipynb` + `data/synthetic_health.csv`
3. Run all cells

### Option 3 — Notebook (local)

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
# Update dataset path in notebook to: pd.read_csv('data/synthetic_health.csv')
jupyter notebook Capstone.ipynb
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `scikit-learn` | Logistic Regression model |
| `pandas` / `numpy` | Data processing |
| `streamlit` | Dashboard UI |
| `plotly` | Interactive charts |
| `matplotlib` / `seaborn` | Notebook visualizations |
