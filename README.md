# 📦 Delivery Risk Analytics & Prediction Dashboard

A multi-page analytics and machine learning web application that analyzes supply chain order data, compares predictive models, and predicts the risk of a shipment arriving late — with a full explanation of *why*, powered by SHAP.

---

## 🎯 Overview

This project turns a supply-chain machine learning pipeline into a complete analytics dashboard, not just a single prediction form. It's built on the **DataCo Smart Supply Chain Dataset** (180,000+ real e-commerce orders) and covers everything from exploratory data analysis to model comparison to individual, explainable predictions.

## ✨ Pages

| Page | What it shows |
|---|---|
| **Dashboard** | Dataset overview stats, on-time vs. late delivery split, and the final model's headline metrics |
| **Data Analysis** | Six exploratory charts — delivery risk distribution, risk by shipping mode, shipping time by region, order profit distribution, monthly sales trends, and top-selling categories |
| **Model Performance** | Accuracy comparison across three models, precision/recall/F1 by class, and the confusion matrix for the final model |
| **Explainability** | Global feature importance for the final model, and an explanation of how per-prediction SHAP explanations work |
| **Prediction** | An interactive form to predict the delay risk for a new order, with a live SHAP breakdown of contributing factors and tailored suggestions |
| **Dataset** | Details on the source data, feature list, and data types used by the model |

---

## 🧠 Machine Learning Approach

- **17 engineered features** covering shipment, order, customer, and date information
- **No data leakage**: `Days for shipping (real)` and `Delivery Status` are excluded from training, since they are only known after a delivery happens
- **One-Hot Encoding** for categorical variables, avoiding false ordinal relationships
- **Three models compared** — Random Forest, XGBoost, and LightGBM — with **XGBoost selected** as the final model for its balance of accuracy and fast, reliable SHAP explainability
- **SHAP (SHapley Additive exPlanations)** used for per-prediction explainability, showing only the categories actually present in a given order

| Model | Accuracy |
|---|---|
| Random Forest | 69.3% |
| **XGBoost (final)** | **70.7%** |
| LightGBM | 70.3% |

A geographic consistency check (Country → Region → Market → State) also runs on the Prediction page, so the model is never asked to reason about impossible input combinations.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data processing & ML | Python, Pandas, Scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Backend | Flask, Jinja2 templates |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Deployment | Docker, Gunicorn |

---

## 📁 Project Structure

```
├── Project_(supply_chain).ipynb    # Full ML pipeline: EDA → cleaning → feature engineering → model training
├── app.py                           # Flask app — page routes + prediction/explainability API
├── templates/                       # Jinja2 templates for all six pages
│   ├── base.html
│   ├── dashboard.html
│   ├── data_analysis.html
│   ├── model_performance.html
│   ├── explainability.html
│   ├── prediction.html
│   └── dataset.html
├── static/
│   ├── css/style.css
│   └── js/chart.umd.js              # Chart.js, bundled locally
├── late_delivery_model.pkl          # Trained XGBoost model
├── model_columns.pkl                # Feature column order used during training
├── dropdown_options.pkl             # Category values for form dropdowns
├── country_region_map.pkl           # Country → Region lookup (input validation)
├── country_market_map.pkl           # Country → Market lookup (input validation)
├── country_states_map.pkl           # Country → valid States lookup (input validation)
├── precomputed_data.json            # Precomputed dataset/model stats powering the dashboard pages
├── requirements.txt
├── Dockerfile
└── Procfile
```

---

## 🚀 Running Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## 🐳 Running with Docker

```bash
docker build -t delivery-risk-dashboard .
docker run -p 7860:7860 delivery-risk-dashboard
```

---

## ⚠️ Limitations

- **70.7% accuracy** reflects a genuinely hard prediction problem — delivery delays depend on real-world factors (weather, traffic, staffing) not captured in this dataset. This is an honest, leakage-free number, not an inflated one.
- Predictions are most reliable for **realistic input combinations** that reflect patterns present in the training data.
- SHAP explanations describe what the model weighted heavily for a specific prediction — they reflect correlation-based historical patterns, not proven causal relationships.
- Country, state, and region names appear in the source dataset's original format, which includes some Spanish-language entries.

---

## 👤 Author

**Wania Naeem**
GitHub: [github.com/Wania-Naeem](https://github.com/Wania-Naeem)
LinkedIn: [Add your LinkedIn URL here]

---

## 📄 Data Source

[DataCo Smart Supply Chain Dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) (Kaggle), used for educational purposes.
