# GenAI Capstone Project

**Live Dashboard:** [https://cardioriskai.netlify.app/](https://cardioriskai.netlify.app/)

A machine learning project that predicts **cardiovascular health risk** using patient data. The model is built using Logistic Regression and trained on a synthetic health dataset. Includes a standalone web dashboard for visual analysis and real-time predictions.

---

## Project Structure

```
genai capstone/
├── Capstone.ipynb        # Main notebook (EDA + ML model)
├── synthetic_health.csv  # Dataset
├── dashboard/
│   ├── index.html        # 3-page web dashboard
│   ├── style.css         # Dark glassmorphism design
│   └── app.js            # Charts + prediction logic
└── README.md
```

---

## Dashboard

A standalone web UI — no server needed, just open `dashboard/index.html` in any browser.

**3 pages:**

| Page | Contents |
|---|---|
| KPI Dashboard | Accuracy, Precision, Recall, F1, AUC-ROC metric cards + dataset stats |
| Visual Analysis | ROC Curve, Confusion Matrix, Logistic Regression Feature Weights |
| Prediction System | Manual input form → real-time cardiovascular risk score |

**Model Results (from notebook):**

| Metric | Value |
|---|---|
| Accuracy | 92.5% |
| Precision | 92% |
| Recall | 93% |
| F1 Score | 92% |
| AUC-ROC | 0.89 |

---

## Dataset

**File:** `synthetic_health.csv`
**Rows:** 400 | **Columns:** 11

| Column | Type | Description |
|---|---|---|
| `age` | Numerical | Patient age |
| `systolic_bp` | Numerical | Systolic blood pressure |
| `cholesterol` | Numerical | Cholesterol level |
| `max_heart_rate` | Numerical | Maximum heart rate |
| `bmi` | Numerical | Body mass index |
| `sex` | Categorical | Male / Female |
| `chest_pain` | Categorical | Type of chest pain |
| `smoker` | Categorical | Yes / No |
| `diabetes` | Categorical | Yes / No |
| `exercise_angina` | Categorical | Yes / No |
| `risk` | Target (0/1) | Cardiovascular risk |

---

## Dependencies

| Library | Purpose |
|---|---|
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `seaborn` | Visualization |
| `scikit-learn` | ML model, preprocessing & evaluation |

---

## How to Run

### Option 1: Google Colab (Recommended)

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `Capstone.ipynb`
3. Upload `synthetic_health.csv` using the Files panel on the left sidebar
4. Run all cells — all libraries are pre-installed in Colab

> **Note:** The dataset path in the notebook is set to `/content/synthetic_health.csv` which works directly in Colab.

### Option 2: Run Locally (VS Code / Jupyter)

1. **Install dependencies:**
   ```bash
   python3 -m pip install pandas numpy matplotlib seaborn scikit-learn --break-system-packages
   ```

2. **Update the dataset path** in `Capstone.ipynb`:
   ```python
   df = pd.read_csv('synthetic_health.csv')
   ```

3. **Select the correct Python kernel** in VS Code (top-right of the notebook)

4. Run all cells

### Option 3: Open the Dashboard

Double-click `dashboard/index.html` or drag it into Chrome/Safari — works offline.

---

## What the Notebook Does

1. **Exploratory Data Analysis (EDA)** — shape, dtypes, missing values, outliers
2. **Data Preprocessing** — fill missing values (median for numerical, mode for categorical), one-hot encoding, feature scaling
3. **Model Training** — Logistic Regression with train/test split (80/20)
4. **Evaluation** — accuracy score, confusion matrix, classification report, ROC-AUC curve
