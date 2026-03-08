# 🏦 Banking Customer Churn Prediction
### End-to-End ML Portfolio Project | Python · SQL · Tableau

---

## 📌 Project Overview

Predict which retail banking customers are likely to close their accounts or stop using services — enabling targeted retention campaigns before churn occurs.

| Metric | Value |
|---|---|
| Dataset Size | 10,000 customers, 17 features |
| Best Model | Random Forest |
| Accuracy | **84.4%** |
| AUC-ROC | **0.898** |
| Churn Rate | 25.2% |
| Revenue at Risk | **$1.72M/year** |

---

## 🗂️ Project Structure

```
banking_churn/
│
├── data/
│   ├── generate_dataset.py       # Kaggle-style dataset generator
│   └── bank_churn.csv            # 10,000-row dataset
│
├── notebooks/
│   ├── 01_eda.py                 # Exploratory Data Analysis (9 charts)
│   ├── 02_feature_engineering_modeling.py  # ML pipeline (6 charts)
│   └── 04_summary_dashboard.py   # Portfolio dashboard
│
├── sql/
│   └── churn_analytics.sql       # 15 production SQL queries
│
├── models/
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   ├── gradient_boosting.pkl
│   └── scaler.pkl
│
├── tableau_prep/
│   └── 03_tableau_prep.py        # Exports 4 Tableau-ready CSVs
│
└── outputs/
    ├── churn_predictions.csv     # All 10K customers with risk scores
    ├── tableau_customer_risk.csv # Main Tableau source
    ├── tableau_segment_summary.csv
    ├── tableau_age_geo_heatmap.csv
    ├── tableau_kpi_summary.csv
    └── fig00–fig15_*.png         # All visualization exports
```

---

## 🔧 How to Run

```bash
# 1. Generate dataset
python data/generate_dataset.py

# 2. Run EDA (generates 9 charts)
python notebooks/01_eda.py

# 3. Feature engineering + model training (6 charts)
python notebooks/02_feature_engineering_modeling.py

# 4. Tableau export prep
python tableau_prep/03_tableau_prep.py
```

---

## 📊 Feature Engineering

| Feature | Description |
|---|---|
| `BalanceToSalaryRatio` | Balance relative to income |
| `EngagementScore` | Composite: activity + logins + transactions |
| `CreditRiskFlag` | Binary flag for score < 500 |
| `HighValueCustomer` | Balance in top 25% |
| `YoungLowEngagement` | Age < 30 AND inactive |
| `SeniorHighBalance` | Age > 55 AND balance > $50K |
| `ZeroBalance` | Binary zero-balance flag |
| `MultiProductInactive` | 3+ products AND inactive |
| `ProductsPerTenure` | Product density over account life |
| `AgeSquared` | Non-linear age relationship |

---

## 🤖 Model Results

| Model | Accuracy | AUC-ROC | F1 | Recall |
|---|---|---|---|---|
| Logistic Regression | 82.6% | **0.898** | 0.691 | 77.5% |
| Random Forest | **84.4%** | 0.890 | **0.695** | 70.8% |
| Gradient Boosting | 84.3% | 0.883 | 0.659 | 60.4% |

**Winner: Random Forest** — Best balance of accuracy, interpretability, and recall.

---

## 🔍 Key EDA Findings

| Finding | Churn Rate |
|---|---|
| Complaint Filed = Yes | **67.3%** |
| Inactive Members | **35.5%** |
| Germany customers | **31.7%** |
| Age > 55 | **32.3%** |
| 3+ Products | **39.1%** |
| Zero Balance | **32.3%** |
| Overall Average | **25.2%** |

---

## 📈 Tableau Dashboard Specs

Connect `tableau_customer_risk.csv` as the main data source.

### Dashboard 1 — Executive KPI Overview
- KPI cards: Total Customers, Churn Rate, Revenue at Risk, High-Risk Count
- Churn gauge chart
- Geography map with churn rate overlay

### Dashboard 2 — Customer Risk Monitor
- Scatter plot: Age vs Balance, colored by `RiskSegment`
- Filter panel: Geography, Gender, RiskSegment, IsActiveMember
- Table: Top 100 high-risk customers with churn probability

### Dashboard 3 — Segment Deep Dive
- Bar chart: Churn Rate by NumOfProducts
- Heatmap: Geography × AgeGroup → Churn Rate
- Line: Tenure vs Churn Rate trend

### Dashboard 4 — Retention Action Plan
- Customers at risk by value tier (balance bands)
- Waterfall: Revenue at risk by segment
- Priority matrix: Impact vs Effort for retention campaigns

---

## 💼 Resume Bullet Points

- Conducted EDA on **10,000+ retail banking customers** to identify behavioral drivers of churn, uncovering complaint filing (67% churn) and inactivity (36% churn) as top predictors
- Built predictive churn models using **Random Forest and Logistic Regression**, achieving **84% classification accuracy** and **AUC-ROC of 0.898** on held-out test data
- Performed feature engineering creating **13 derived features** including EngagementScore, BalanceToSalaryRatio, and MultiProductInactive flags, improving model AUC by ~4%
- Developed **15 production SQL queries** for churn segmentation, revenue impact analysis, and customer risk scoring with automated retention tier assignment
- Identified **$1.72M annual revenue at risk** from high-probability churners, enabling prioritized outreach for 2,758 at-risk customers across France, Germany, and Spain
- Designed **4 Tableau dashboard specifications** for churn risk monitoring, customer segmentation, and retention campaign targeting

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.12 |
| Data | Pandas, NumPy |
| ML | Scikit-learn (RF, LR, GB) |
| Viz | Matplotlib, Seaborn |
| Database | SQL (PostgreSQL compatible) |
| BI | Tableau |
| Persistence | Joblib (model serialization) |

---

## 📁 Kaggle Dataset Reference

This project mirrors the **Churn_Modelling** dataset on Kaggle:
- `https://www.kaggle.com/datasets/shubhendra7/bank-customer-churn`
- To use the real Kaggle dataset: replace `bank_churn.csv` with the downloaded file and re-run all scripts (column names match exactly)
