"""
========================================================
  BANKING CHURN — 03. TABLEAU PREP + DASHBOARD GUIDE
========================================================
Generates Tableau-ready exports and documents all
dashboard specs for building churn monitoring views.
"""

import pandas as pd
import numpy as np
import os

OUT = '/home/claude/banking_churn/outputs'
os.makedirs(OUT, exist_ok=True)

# ── Load predictions ──────────────────────────────────────
df = pd.read_csv(f'{OUT}/churn_predictions.csv')
df['AgeGroup'] = pd.cut(df['Age'], bins=[17,25,35,45,55,65,100],
                         labels=['18-25','26-35','36-45','46-55','56-65','65+'])
print("✅ Loaded predictions:", df.shape)

# ── Tableau Export 1: Customer Risk Dashboard ─────────────
tableau_main = df[[
    'CustomerID', 'Geography', 'Gender', 'Age', 'AgeGroup',
    'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
    'IsActiveMember', 'EstimatedSalary', 'CreditScore',
    'MonthlyTransactions', 'ComplaintFiled', 'SatisfactionScore',
    'LoginFreqMonthly', 'EngagementScore', 'HighValueCustomer',
    'RF_ChurnProb', 'LR_ChurnProb', 'RF_ChurnPred', 'RiskSegment', 'Exited'
]].copy()

tableau_main['ChurnLabel']         = tableau_main['Exited'].map({1:'Churned', 0:'Retained'})
tableau_main['ActiveLabel']        = tableau_main['IsActiveMember'].map({1:'Active', 0:'Inactive'})
tableau_main['HasCardLabel']       = tableau_main['HasCrCard'].map({1:'Has Card', 0:'No Card'})
tableau_main['ComplaintLabel']     = tableau_main['ComplaintFiled'].map({1:'Complaint', 0:'No Complaint'})
tableau_main['RF_ChurnProb_Pct']   = (tableau_main['RF_ChurnProb'] * 100).round(1)
tableau_main['BalanceBand']        = pd.cut(tableau_main['Balance'],
                                             bins=[-1, 0, 25000, 75000, 150000, 300000],
                                             labels=['Zero', '$1-25K', '$25-75K', '$75-150K', '$150K+'])
tableau_main['CreditTier']         = pd.cut(tableau_main['CreditScore'],
                                             bins=[349, 499, 579, 669, 739, 799, 851],
                                             labels=['Very Poor','Poor','Fair','Good','Very Good','Exceptional'])

tableau_main.to_csv(f'{OUT}/tableau_customer_risk.csv', index=False)
print(f"✅ tableau_customer_risk.csv — {len(tableau_main):,} rows")

# ── Tableau Export 2: Segment Summary ────────────────────
segments = []
for geo in df['Geography'].unique():
    for active in [0, 1]:
        sub = df[(df['Geography']==geo) & (df['IsActiveMember']==active)]
        if len(sub) > 0:
            segments.append({
                'Geography'         : geo,
                'ActiveStatus'      : 'Active' if active else 'Inactive',
                'TotalCustomers'    : len(sub),
                'ChurnedCount'      : sub['Exited'].sum(),
                'ChurnRate'         : sub['Exited'].mean().round(4),
                'AvgBalance'        : sub['Balance'].mean().round(2),
                'AvgAge'            : sub['Age'].mean().round(1),
                'AvgCreditScore'    : sub['CreditScore'].mean().round(0),
                'AvgProducts'       : sub['NumOfProducts'].mean().round(2),
                'AvgEngagement'     : sub['EngagementScore'].mean().round(2),
                'HighRiskCount'     : (sub['RF_ChurnProb'] > 0.5).sum(),
                'CriticalRiskCount' : (sub['RF_ChurnProb'] > 0.75).sum(),
                'BalanceAtRisk'     : sub[sub['RF_ChurnProb']>0.5]['Balance'].sum().round(2),
            })

seg_df = pd.DataFrame(segments)
seg_df.to_csv(f'{OUT}/tableau_segment_summary.csv', index=False)
print(f"✅ tableau_segment_summary.csv — {len(seg_df)} segments")

# ── Tableau Export 3: Age × Risk Heatmap Data ─────────────
age_risk = df.groupby(['AgeGroup', 'Geography'], observed=True).agg(
    Customers    = ('Exited', 'count'),
    ChurnRate    = ('Exited', 'mean'),
    AvgChurnProb = ('RF_ChurnProb', 'mean'),
    AvgBalance   = ('Balance', 'mean'),
).reset_index()
age_risk['ChurnRate']    = age_risk['ChurnRate'].round(4)
age_risk['AvgChurnProb'] = age_risk['AvgChurnProb'].round(4)
age_risk['AvgBalance']   = age_risk['AvgBalance'].round(2)
age_risk.to_csv(f'{OUT}/tableau_age_geo_heatmap.csv', index=False)
print(f"✅ tableau_age_geo_heatmap.csv — {len(age_risk)} rows")

# ── Tableau Export 4: KPI Summary ─────────────────────────
kpi_data = {
    'Metric': [
        'Total Customers',
        'Overall Churn Rate',
        'Avg Balance (Churned)',
        'Avg Balance (Retained)',
        'High Risk Customers (>50%)',
        'Critical Risk Customers (>75%)',
        'Balance at Risk (High Risk)',
        'Revenue at Risk (1.5% NIM)',
        'Model Accuracy (RF)',
        'Model AUC-ROC (RF)',
        'Top Churn Driver',
        'Highest Risk Geography',
    ],
    'Value': [
        f"{len(df):,}",
        f"{df['Exited'].mean():.1%}",
        f"${df[df['Exited']==1]['Balance'].mean():,.0f}",
        f"${df[df['Exited']==0]['Balance'].mean():,.0f}",
        f"{(df['RF_ChurnProb']>0.5).sum():,}",
        f"{(df['RF_ChurnProb']>0.75).sum():,}",
        f"${df[df['RF_ChurnProb']>0.5]['Balance'].sum():,.0f}",
        f"${df[df['RF_ChurnProb']>0.5]['Balance'].sum()*0.015:,.0f}",
        "84.4%",
        "0.890",
        "Complaint Filed (67% churn)",
        f"Germany ({df[df['Geography']=='Germany']['Exited'].mean():.1%})",
    ]
}
kpi_df = pd.DataFrame(kpi_data)
kpi_df.to_csv(f'{OUT}/tableau_kpi_summary.csv', index=False)
print(f"✅ tableau_kpi_summary.csv — {len(kpi_df)} KPIs")
print("\n" + "="*60)

# Print KPIs
print("  📊 PROJECT KPI SUMMARY")
print("="*60)
for _, row in kpi_df.iterrows():
    print(f"  {row['Metric']:<38} {row['Value']}")
print("="*60)
