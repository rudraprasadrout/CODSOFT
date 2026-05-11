<!-- prettier-ignore -->
<p align="center">
  <img alt="Titanic" src="https://raw.githubusercontent.com/rudraprasadrout/CODSOFT/main/assets/titanic-banner.png" width="820" style="max-width:100%;height:auto;"/>
</p>

# 🚢 Titanic Survival Prediction — CODSOFT Task 1

[![Python](https://img.shields.io/badge/python-3.12-blue?logo=python)](https://www.python.org/) [![Flask](https://img.shields.io/badge/flask-2.x-black?logo=flask)](https://flask.palletsprojects.com/) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This project was developed as part of the CodSoft Machine Learning Internship. It demonstrates a complete ML workflow — from EDA and feature engineering to model selection and a deployed Flask prediction API — using the historic Titanic passenger dataset.

---

## 🌐 Live Deployment

- 🚀 Demo Web App: Titanic Survival Predictor (live demo link if deployed)
- 🔌 Prediction API: Render API Endpoint (add your endpoint URL here)

---

## 📖 Overview

On April 15, 1912, the RMS Titanic sank. This project analyzes records of 891 passengers to predict survival (historical survival ≈ 38%) using demographics and ticketing data.

## 💡 Highlights & Surprises

- Engineered `FamilySize` (SibSp + Parch) — revealed solo travelers and very large families (5+) had lower survival rates.
- Integrated the entire preprocessing inside an `sklearn.pipeline.Pipeline` so the exported `.pkl` is deployment-ready with zero leakage.
- Added extra visualizations in `titanic.ipynb` — Age/Survival, Fare skew (log1p), Survival by Cabin class and Embarked, FamilySize heatmap.

---

## 🛠️ Problems Found & Solved

| Issue | Solution |
|---|---|
| Missing Age (177 rows) | Filled with column mean and cast to integer for consistency |
| Missing Embarked (2 rows) | Dropped (minimal impact) |
| Noisy columns | Dropped `PassengerId`, `Name`, `Ticket`, `Cabin` |
| Skewed `Fare` | Applied `log1p` via `ColumnTransformer` |
| Categorical encoding | Label-encoded `Sex` and ordinally mapped `Embarked` |

---

## 📊 Model Comparison & Selection

Evaluated four classifiers with identical `sklearn` Pipelines to avoid leakage:

- XGBoost (100 estimators, lr=0.1, depth=5) — 🏆 Selected (best generalization)
- RandomForest (100 trees) — High accuracy, robust
- LogisticRegression — Good baseline
- DecisionTree (depth=5) — Interpretable, decent

Model artifacts are serialized to `model.pkl` for inference.

---

## ⚙️ Technical Stack

- Backend: Python 3.12, Flask, Flask-CORS
- ML & Data: pandas, numpy, scikit-learn, xgboost
- Visuals: matplotlib, seaborn
- Frontend: HTML5, CSS3, JavaScript

---

## 🏗️ Pipeline Architecture

All preprocessing (imputation, log1p transform on `Fare`, encoding, and `FamilySize` creation) lives inside an `sklearn.pipeline.Pipeline`, enabling the full model to be saved as a single `.pkl` and loaded directly by the Flask app for inference.

### Key features used

- `Pclass` — socioeconomic proxy
- `Sex` — strongest predictor
- `Age` — essential for 'children first' effects
- `FamilySize` — captures social group effects

---

## 🎨 Visualizations (what to look for)

Included in `titanic.ipynb`:

- Age distribution by survival (histograms + KDE)
- Survival rate by `Pclass` and `Sex` (grouped bar charts)
- `Fare` distribution before/after `log1p` (box + violin)
- FamilySize vs Survival heatmap
- Correlation matrix with annotated p-values

Tip: to regenerate visuals run the notebook and export PNGs to `assets/plots/` to embed them here.

---

## 👩‍💻 Credits

Rudra Prasad Rout — Backend Architect & Aspiring Data Scientist

- GitHub: https://github.com/rudraprasadrout
- Email: routrp07@gmail.com

---

## 🚀 Getting Started

Clone, install, and run:

```bash
git clone https://github.com/rudraprasadrout/CODSOFT.git
cd CODSOFT/Task_1_Titanic_Survival
pip install -r requirements.txt
python app.py
```

By default the API listens on `http://0.0.0.0:5000`.

### Example requests

Single prediction (curl):

```bash
curl -s -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Pclass":3,"Sex":"male","Age":22,"Fare":7.25,"Embarked":"S","FamilySize":1}'
```

Batch prediction example also supported at `/predict/batch`.

---

## 🧭 Next steps & Ideas (surprise extras)

- Add interactive Plotly charts to the demo UI for brushing & linking.
- Add SHAP explainability snapshots in `assets/plots/shap_summary.png` so users can inspect feature importance per prediction.
- CI: Add a GitHub Action to run tests and linting on push.

If you want, I can generate the notebook plots and add them to `assets/plots/` and embed them in this README.

---

## 📝 License

MIT — modify as needed.

---

If you'd like visual thumbnails added now, tell me which plots to create and I'll generate and embed them.
