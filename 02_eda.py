"""
========================================================
  BANKING CHURN — 01. EXPLORATORY DATA ANALYSIS
========================================================
Mirrors the EDA you would do on Kaggle's Churn_Modelling
dataset. Generates 10 publication-quality charts.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
PALETTE = {'Churned': '#E74C3C', 'Retained': '#2ECC71'}

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv('/home/claude/banking_churn/data/bank_churn.csv')
df['ChurnLabel'] = df['Exited'].map({1: 'Churned', 0: 'Retained'})
OUT = '/home/claude/banking_churn/outputs'
os.makedirs(OUT, exist_ok=True)

print("=" * 60)
print("  BANKING CHURN — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ─── 1. Basic Info ────────────────────────────────────────────────────────────
print(f"\n📊 Dataset Shape: {df.shape}")
print(f"\n🔍 Data Types & Nulls:\n{df.isnull().sum().to_string()}")
print(f"\n📈 Descriptive Stats:\n{df.describe().round(2).to_string()}")

# ─── 2. Churn Distribution (Figure 1) ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Figure 1 — Churn Class Distribution', fontsize=15, fontweight='bold', y=1.01)

churn_counts = df['ChurnLabel'].value_counts()
colors = ['#2ECC71', '#E74C3C']
axes[0].pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
axes[0].set_title('Churn Proportion')

sns.countplot(data=df, x='ChurnLabel', palette={'Churned':'#E74C3C','Retained':'#2ECC71'}, ax=axes[1])
for p in axes[1].patches:
    axes[1].annotate(f'{int(p.get_height()):,}', (p.get_x()+p.get_width()/2, p.get_height()),
                     ha='center', va='bottom', fontweight='bold')
axes[1].set_title('Churn Count')
axes[1].set_xlabel('')
plt.tight_layout()
plt.savefig(f'{OUT}/fig1_churn_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Fig 1 saved — Churn Distribution")

# ─── 3. Numeric Feature Distributions (Figure 2) ─────────────────────────────
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary',
            'MonthlyTransactions', 'LoginFreqMonthly']

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('Figure 2 — Numeric Feature Distributions by Churn', fontsize=14, fontweight='bold')
axes = axes.flatten()

for i, col in enumerate(num_cols):
    for label, color in [('Retained', '#2ECC71'), ('Churned', '#E74C3C')]:
        axes[i].hist(df[df['ChurnLabel']==label][col], bins=30, alpha=0.6,
                     color=color, label=label, density=True)
    axes[i].set_title(col, fontweight='bold')
    axes[i].legend(fontsize=8)
    axes[i].set_ylabel('Density')

axes[-1].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUT}/fig2_numeric_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 2 saved — Numeric Distributions")

# ─── 4. Categorical Churn Rates (Figure 3) ───────────────────────────────────
cat_cols = ['Geography', 'Gender', 'NumOfProducts', 'HasCrCard',
            'IsActiveMember', 'ComplaintFiled', 'SatisfactionScore']

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Figure 3 — Churn Rate by Categorical Features', fontsize=14, fontweight='bold')
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    rate = df.groupby(col)['Exited'].mean().reset_index()
    rate.columns = [col, 'ChurnRate']
    bars = axes[i].bar(rate[col].astype(str), rate['ChurnRate'],
                       color=['#E74C3C' if r > 0.25 else '#3498DB' for r in rate['ChurnRate']],
                       edgecolor='white', linewidth=0.8)
    for bar, r in zip(bars, rate['ChurnRate']):
        axes[i].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                     f'{r:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[i].axhline(df['Exited'].mean(), color='black', linestyle='--', alpha=0.5, label='Avg')
    axes[i].set_title(f'Churn Rate by {col}', fontweight='bold')
    axes[i].set_ylabel('Churn Rate')
    axes[i].set_ylim(0, min(rate['ChurnRate'].max() + 0.12, 1.0))
    axes[i].legend(fontsize=8)

axes[-1].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUT}/fig3_categorical_churn_rates.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 3 saved — Categorical Churn Rates")

# ─── 5. Correlation Heatmap (Figure 4) ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
num_df = df[['CreditScore','Age','Tenure','Balance','NumOfProducts',
             'HasCrCard','IsActiveMember','EstimatedSalary',
             'MonthlyTransactions','ComplaintFiled',
             'SatisfactionScore','LoginFreqMonthly','Exited']]
corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            annot_kws={'size': 9})
ax.set_title('Figure 4 — Correlation Heatmap\n(Features vs Churn)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 4 saved — Correlation Heatmap")

# ─── 6. Age × Balance Scatter (Figure 5) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
for label, color, alpha in [('Retained','#2ECC71',0.4), ('Churned','#E74C3C',0.6)]:
    sub = df[df['ChurnLabel']==label]
    ax.scatter(sub['Age'], sub['Balance'], c=color, alpha=alpha,
               s=20, label=label, edgecolors='none')
ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('Balance ($)', fontsize=12)
ax.set_title('Figure 5 — Age vs Balance Colored by Churn', fontsize=13, fontweight='bold')
ax.legend(markerscale=2)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x:,.0f}'))
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_age_balance_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 5 saved — Age vs Balance Scatter")

# ─── 7. Churn by Age Group (Figure 6) ────────────────────────────────────────
df['AgeGroup'] = pd.cut(df['Age'], bins=[17,25,35,45,55,65,100],
                         labels=['18-25','26-35','36-45','46-55','56-65','65+'])
age_churn = df.groupby('AgeGroup', observed=True)['Exited'].agg(['mean','count']).reset_index()
age_churn.columns = ['AgeGroup','ChurnRate','Count']

fig, ax1 = plt.subplots(figsize=(11, 6))
ax2 = ax1.twinx()
bars = ax1.bar(age_churn['AgeGroup'].astype(str), age_churn['ChurnRate'],
               color='#E74C3C', alpha=0.75, label='Churn Rate')
ax2.plot(age_churn['AgeGroup'].astype(str), age_churn['Count'],
         color='#2C3E50', marker='o', linewidth=2, label='Customer Count')
ax1.set_ylabel('Churn Rate', color='#E74C3C', fontsize=11)
ax2.set_ylabel('Customer Count', color='#2C3E50', fontsize=11)
ax1.set_title('Figure 6 — Churn Rate & Volume by Age Group', fontsize=13, fontweight='bold')
ax1.tick_params(axis='y', colors='#E74C3C')
ax2.tick_params(axis='y', colors='#2C3E50')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='upper left')
plt.tight_layout()
plt.savefig(f'{OUT}/fig6_age_group_churn.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 6 saved — Age Group Churn")

# ─── 8. Tenure vs Churn (Figure 7) ───────────────────────────────────────────
tenure_churn = df.groupby('Tenure')['Exited'].mean().reset_index()
fig, ax = plt.subplots(figsize=(11, 5))
ax.fill_between(tenure_churn['Tenure'], tenure_churn['Exited'], alpha=0.3, color='#E74C3C')
ax.plot(tenure_churn['Tenure'], tenure_churn['Exited'], marker='o',
        color='#E74C3C', linewidth=2.5, markersize=8)
ax.axhline(df['Exited'].mean(), linestyle='--', color='gray', alpha=0.7, label='Overall Avg')
ax.set_xlabel('Tenure (Years)', fontsize=12)
ax.set_ylabel('Churn Rate', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
ax.set_title('Figure 7 — Churn Rate by Customer Tenure', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/fig7_tenure_churn.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 7 saved — Tenure vs Churn")

# ─── 9. Geography Heatmap (Figure 8) ─────────────────────────────────────────
geo_prod = df.pivot_table(index='Geography', columns='NumOfProducts',
                           values='Exited', aggfunc='mean')
fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(geo_prod, annot=True, fmt='.0%', cmap='YlOrRd',
            linewidths=0.5, ax=ax, cbar_kws={'label':'Churn Rate'})
ax.set_title('Figure 8 — Churn Rate: Geography × Number of Products',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Number of Products')
ax.set_ylabel('Geography')
plt.tight_layout()
plt.savefig(f'{OUT}/fig8_geo_product_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 8 saved — Geography × Product Heatmap")

# ─── 10. Key Insights Summary (Figure 9) ─────────────────────────────────────
insights = {
    'Inactive Members'  : df[df['IsActiveMember']==0]['Exited'].mean(),
    'Active Members'    : df[df['IsActiveMember']==1]['Exited'].mean(),
    'Germany Customers' : df[df['Geography']=='Germany']['Exited'].mean(),
    'France Customers'  : df[df['Geography']=='France']['Exited'].mean(),
    'Age > 55'          : df[df['Age']>55]['Exited'].mean(),
    'Age 26–45'         : df[(df['Age']>=26)&(df['Age']<=45)]['Exited'].mean(),
    '3–4 Products'      : df[df['NumOfProducts']>=3]['Exited'].mean(),
    '1–2 Products'      : df[df['NumOfProducts']<=2]['Exited'].mean(),
    'Zero Balance'      : df[df['Balance']==0]['Exited'].mean(),
    'Positive Balance'  : df[df['Balance']>0]['Exited'].mean(),
    'Complaint Filed'   : df[df['ComplaintFiled']==1]['Exited'].mean(),
    'No Complaint'      : df[df['ComplaintFiled']==0]['Exited'].mean(),
}
labels = list(insights.keys())
values = list(insights.values())
colors_bar = ['#E74C3C' if v > df['Exited'].mean() else '#2ECC71' for v in values]

fig, ax = plt.subplots(figsize=(13, 7))
bars = ax.barh(labels, values, color=colors_bar, edgecolor='white', linewidth=0.8)
ax.axvline(df['Exited'].mean(), color='black', linestyle='--', linewidth=1.5, label=f"Avg {df['Exited'].mean():.0%}")
for bar, v in zip(bars, values):
    ax.text(v + 0.005, bar.get_y()+bar.get_height()/2, f'{v:.1%}',
            va='center', fontweight='bold', fontsize=10)
ax.set_xlabel('Churn Rate', fontsize=12)
ax.set_title('Figure 9 — Key Behavioral Drivers of Churn\n(Red = Above Average Risk)',
             fontsize=13, fontweight='bold')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUT}/fig9_key_churn_drivers.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 9 saved — Key Churn Drivers")

# ─── 11. Print EDA Summary ───────────────────────────────────────────────────
print("\n" + "="*60)
print("  EDA SUMMARY — KEY FINDINGS")
print("="*60)
print(f"  Overall churn rate           : {df['Exited'].mean():.1%}")
print(f"  Germany churn rate           : {df[df['Geography']=='Germany']['Exited'].mean():.1%}")
print(f"  Inactive member churn        : {df[df['IsActiveMember']==0]['Exited'].mean():.1%}")
print(f"  Age > 55 churn rate          : {df[df['Age']>55]['Exited'].mean():.1%}")
print(f"  Customers w/ 3+ products     : {df[df['NumOfProducts']>=3]['Exited'].mean():.1%}")
print(f"  Complaint filed churn rate   : {df[df['ComplaintFiled']==1]['Exited'].mean():.1%}")
print(f"  Zero balance churn rate      : {df[df['Balance']==0]['Exited'].mean():.1%}")
print("="*60)
print("\n✅ EDA Complete — All figures saved to outputs/")
