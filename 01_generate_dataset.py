"""
Banking Customer Churn Dataset Generator
Mirrors the popular Kaggle 'Churn_Modelling' dataset structure
with realistic distributions and behavioral patterns.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 10000

# ── Geography & Demographics ──────────────────────────────────────────────────
geography   = np.random.choice(['France', 'Germany', 'Spain'], N, p=[0.50, 0.25, 0.25])
gender      = np.random.choice(['Male', 'Female'], N, p=[0.54, 0.46])
age         = np.clip(np.random.normal(38, 11, N).astype(int), 18, 92)

# ── Account Characteristics ────────────────────────────────────────────────────
tenure      = np.random.randint(0, 11, N)
num_products= np.random.choice([1, 2, 3, 4], N, p=[0.50, 0.46, 0.025, 0.015])
has_cr_card = np.random.choice([0, 1], N, p=[0.29, 0.71])
is_active   = np.random.choice([0, 1], N, p=[0.48, 0.52])
credit_score= np.clip(np.random.normal(650, 96, N).astype(int), 350, 850)

# ── Financial Metrics ─────────────────────────────────────────────────────────
balance_raw   = np.random.choice([0, 1], N, p=[0.32, 0.68])  # 32% zero-balance
balance_amount= np.where(balance_raw == 0, 0,
                         np.random.normal(76485, 62397, N).clip(500, 250000))

salary_base   = np.random.choice([30, 60, 100, 150, 200], N, p=[0.20, 0.30, 0.30, 0.15, 0.05])
estimated_salary = (salary_base * 1000 + np.random.normal(0, 5000, N)).clip(10000, 199999).round(2)

# ── Churn Logic (realistic drivers) ──────────────────────────────────────────
churn_prob = np.zeros(N)
churn_prob += 0.10                                          # base rate
churn_prob += np.where(geography == 'Germany', 0.10, 0)    # Germany churns more
churn_prob += np.where(age > 55, 0.12, 0)                  # older customers
churn_prob += np.where(age < 30, 0.06, 0)                  # young & fickle
churn_prob += np.where(num_products >= 3, 0.18, 0)         # too many products → frustration
churn_prob += np.where(num_products == 1, 0.06, 0)         # low engagement
churn_prob += np.where(is_active == 0, 0.14, -0.05)        # inactive members
churn_prob += np.where(balance_amount == 0, 0.08, -0.02)   # zero balance
churn_prob += np.where(credit_score < 500, 0.08, 0)        # low credit risk
churn_prob += np.where(tenure < 2, 0.07, 0)                # new customers
churn_prob += np.where(tenure > 8, -0.05, 0)               # loyal customers
churn_prob = np.clip(churn_prob, 0.01, 0.90)
churn      = (np.random.rand(N) < churn_prob).astype(int)

# ── Transaction Behavior (engineered features) ────────────────────────────────
monthly_txns    = np.where(is_active == 1,
                            np.random.poisson(8, N),
                            np.random.poisson(2, N)).clip(0, 40)
complaint_flag  = np.where(churn == 1,
                            np.random.choice([0,1], N, p=[0.4, 0.6]),
                            np.random.choice([0,1], N, p=[0.9, 0.1]))
satisfaction    = np.where(churn == 1,
                            np.random.choice([1,2,3,4,5], N, p=[0.30,0.35,0.20,0.10,0.05]),
                            np.random.choice([1,2,3,4,5], N, p=[0.05,0.10,0.20,0.35,0.30]))
login_freq_monthly = np.where(is_active == 1,
                               np.random.poisson(12, N),
                               np.random.poisson(3, N)).clip(0, 60)

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    'CustomerID'         : range(15634602, 15634602 + N),
    'Surname'            : ['Customer_' + str(i) for i in range(N)],
    'CreditScore'        : credit_score,
    'Geography'          : geography,
    'Gender'             : gender,
    'Age'                : age,
    'Tenure'             : tenure,
    'Balance'            : balance_amount.round(2),
    'NumOfProducts'      : num_products,
    'HasCrCard'          : has_cr_card,
    'IsActiveMember'     : is_active,
    'EstimatedSalary'    : estimated_salary,
    'MonthlyTransactions': monthly_txns,
    'ComplaintFiled'     : complaint_flag,
    'SatisfactionScore'  : satisfaction,
    'LoginFreqMonthly'   : login_freq_monthly,
    'Exited'             : churn
})

output_path = os.path.join(os.path.dirname(__file__), 'bank_churn.csv')
df.to_csv(output_path, index=False)
print(f"✅ Dataset saved → {output_path}")
print(f"   Shape        : {df.shape}")
print(f"   Churn Rate   : {churn.mean():.1%}")
print(df.head(3).to_string())
