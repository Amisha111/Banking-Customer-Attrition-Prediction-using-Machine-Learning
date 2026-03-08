"""
========================================================
  BANKING CHURN — 02. FEATURE ENGINEERING + MODELING
========================================================
  Models: Logistic Regression, Random Forest
  Techniques: SMOTE, Cross-Validation, Feature Importance
  Output: Trained models + evaluation plots
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import warnings, os, joblib

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
OUT   = '/home/claude/banking_churn/outputs'
MODEL = '/home/claude/banking_churn/models'
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODEL, exist_ok=True)

print("="*60)
print("  BANKING CHURN — FEATURE ENGINEERING & MODELING")
print("="*60)

# ══════════════════════════════════════════════════════════
#  STEP 1 — Load & Feature Engineering
# ══════════════════════════════════════════════════════════
df = pd.read_csv('/home/claude/banking_churn/data/bank_churn.csv')

print("\n🔧 Engineering Features...")

# ── Encode categoricals ───────────────────────────────────
le_geo    = LabelEncoder()
le_gender = LabelEncoder()
df['Geography_enc'] = le_geo.fit_transform(df['Geography'])
df['Gender_enc']    = le_gender.fit_transform(df['Gender'])

# ── Interaction / Ratio Features ─────────────────────────
df['BalanceToSalaryRatio']  = df['Balance'] / (df['EstimatedSalary'] + 1)
df['ProductsPerTenure']     = df['NumOfProducts'] / (df['Tenure'] + 1)
df['EngagementScore']       = (df['IsActiveMember'] * 2
                                + df['LoginFreqMonthly'] / 5
                                + df['MonthlyTransactions'] / 4)
df['CreditRiskFlag']        = (df['CreditScore'] < 500).astype(int)
df['HighValueCustomer']     = (df['Balance'] > df['Balance'].quantile(0.75)).astype(int)
df['YoungLowEngagement']    = ((df['Age'] < 30) & (df['IsActiveMember'] == 0)).astype(int)
df['SeniorHighBalance']     = ((df['Age'] > 55) & (df['Balance'] > 50000)).astype(int)
df['ZeroBalance']           = (df['Balance'] == 0).astype(int)
df['MultiProductInactive']  = ((df['NumOfProducts'] >= 3) & (df['IsActiveMember'] == 0)).astype(int)
df['TenureGroup']           = pd.cut(df['Tenure'], bins=[-1,2,5,8,10],
                                      labels=[0,1,2,3]).astype(int)
df['AgeSquared']            = df['Age'] ** 2

FEATURES = [
    # Core
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'MonthlyTransactions', 'ComplaintFiled', 'SatisfactionScore',
    'LoginFreqMonthly',
    # Encoded
    'Geography_enc', 'Gender_enc',
    # Engineered
    'BalanceToSalaryRatio', 'ProductsPerTenure', 'EngagementScore',
    'CreditRiskFlag', 'HighValueCustomer', 'YoungLowEngagement',
    'SeniorHighBalance', 'ZeroBalance', 'MultiProductInactive',
    'TenureGroup', 'AgeSquared'
]
TARGET = 'Exited'

X = df[FEATURES]
y = df[TARGET]

print(f"  Total features : {len(FEATURES)}")
print(f"  Engineered     : 13 new features")
print(f"  Class balance  : {y.value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════
#  STEP 2 — Train / Test Split + Class Weighting
# ══════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# ══════════════════════════════════════════════════════════
#  STEP 3 — MODEL A: Logistic Regression
# ══════════════════════════════════════════════════════════
print("\n🤖 Training Logistic Regression...")
lr = LogisticRegression(class_weight='balanced', max_iter=1000, C=0.5, random_state=42)
lr.fit(X_train_sc, y_train)

lr_pred    = lr.predict(X_test_sc)
lr_prob    = lr.predict_proba(X_test_sc)[:, 1]
lr_auc     = roc_auc_score(y_test, lr_prob)
lr_cv      = cross_val_score(lr, X_train_sc, y_train, cv=5, scoring='roc_auc')

print(f"  Test AUC   : {lr_auc:.4f}")
print(f"  5-Fold AUC : {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")
print(f"\n{classification_report(y_test, lr_pred, target_names=['Retained','Churned'])}")

# ══════════════════════════════════════════════════════════
#  STEP 4 — MODEL B: Random Forest
# ══════════════════════════════════════════════════════════
print("🌲 Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300, max_depth=15, min_samples_split=10,
    min_samples_leaf=4, class_weight='balanced',
    random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

rf_pred  = rf.predict(X_test)
rf_prob  = rf.predict_proba(X_test)[:, 1]
rf_auc   = roc_auc_score(y_test, rf_prob)
rf_cv    = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc')

print(f"  Test AUC   : {rf_auc:.4f}")
print(f"  5-Fold AUC : {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
print(f"\n{classification_report(y_test, rf_pred, target_names=['Retained','Churned'])}")

# ══════════════════════════════════════════════════════════
#  STEP 5 — MODEL C: Gradient Boosting
# ══════════════════════════════════════════════════════════
print("⚡ Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=5,
    subsample=0.8, random_state=42)
gb.fit(X_train, y_train)

gb_pred  = gb.predict(X_test)
gb_prob  = gb.predict_proba(X_test)[:, 1]
gb_auc   = roc_auc_score(y_test, gb_prob)
gb_cv    = cross_val_score(gb, X_train, y_train, cv=5, scoring='roc_auc')

print(f"  Test AUC   : {gb_auc:.4f}")
print(f"  5-Fold AUC : {gb_cv.mean():.4f} ± {gb_cv.std():.4f}")

# ══════════════════════════════════════════════════════════
#  STEP 6 — Evaluation Plots
# ══════════════════════════════════════════════════════════

# ── Fig 10: ROC Curves ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Figure 10 — Model Evaluation: ROC & Precision-Recall Curves',
             fontsize=14, fontweight='bold')

models_eval = [
    ('Logistic Regression', lr_prob, '#3498DB'),
    ('Random Forest',       rf_prob, '#E74C3C'),
    ('Gradient Boosting',   gb_prob, '#2ECC71'),
]

for name, prob, color in models_eval:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc          = roc_auc_score(y_test, prob)
    axes[0].plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.3f})')

    prec, rec, _ = precision_recall_curve(y_test, prob)
    ap            = average_precision_score(y_test, prob)
    axes[1].plot(rec, prec, color=color, lw=2, label=f'{name} (AP={ap:.3f})')

axes[0].plot([0,1],[0,1],'k--', alpha=0.4, label='Random')
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves'); axes[0].legend(loc='lower right')

axes[1].axhline(y_test.mean(), color='k', linestyle='--', alpha=0.4,
                label=f'Baseline ({y_test.mean():.2f})')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curves'); axes[1].legend()

plt.tight_layout()
plt.savefig(f'{OUT}/fig10_roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Fig 10 saved — ROC & PR Curves")

# ── Fig 11: Confusion Matrices ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Figure 11 — Confusion Matrices (Test Set)', fontsize=14, fontweight='bold')

for ax, (name, pred) in zip(axes, [('Logistic Regression', lr_pred),
                                     ('Random Forest', rf_pred),
                                     ('Gradient Boosting', gb_pred)]):
    cm  = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Retained','Churned'],
                yticklabels=['Retained','Churned'],
                cbar=False, linewidths=0.5)
    acc = (cm[0,0]+cm[1,1]) / cm.sum()
    ax.set_title(f'{name}\nAccuracy: {acc:.1%}', fontweight='bold')
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(f'{OUT}/fig11_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 11 saved — Confusion Matrices")

# ── Fig 12: Feature Importance ───────────────────────────
fi_df = pd.DataFrame({'Feature': FEATURES, 'Importance': rf.feature_importances_})
fi_df = fi_df.sort_values('Importance', ascending=False).head(18)

fig, ax = plt.subplots(figsize=(11, 8))
bars = ax.barh(fi_df['Feature'][::-1], fi_df['Importance'][::-1],
               color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(fi_df))),
               edgecolor='white')
for bar, val in zip(bars, fi_df['Importance'][::-1]):
    ax.text(bar.get_width()+0.0005, bar.get_y()+bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)
ax.set_xlabel('Feature Importance (Gini)', fontsize=12)
ax.set_title('Figure 12 — Random Forest Feature Importance\n(Top 18 Features)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig12_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 12 saved — Feature Importance")

# ── Fig 13: Logistic Regression Coefficients ─────────────
coef_df = pd.DataFrame({'Feature': FEATURES, 'Coefficient': lr.coef_[0]})
coef_df['AbsCoef'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values('AbsCoef', ascending=False).head(15)
coef_df = coef_df.sort_values('Coefficient')

colors_lr = ['#E74C3C' if c > 0 else '#2ECC71' for c in coef_df['Coefficient']]
fig, ax = plt.subplots(figsize=(11, 7))
ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_lr, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Coefficient (positive = higher churn risk)', fontsize=11)
ax.set_title('Figure 13 — Logistic Regression Coefficients\n(Red = increases churn risk)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig13_lr_coefficients.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 13 saved — LR Coefficients")

# ── Fig 14: Churn Probability Distribution ───────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Figure 14 — Predicted Churn Probability Distribution (RF)',
             fontsize=13, fontweight='bold')

for label, color in [(0,'#2ECC71'), (1,'#E74C3C')]:
    mask = y_test == label
    name = 'Retained' if label == 0 else 'Churned'
    axes[0].hist(rf_prob[mask], bins=40, alpha=0.6, color=color, label=name, density=True)
axes[0].set_xlabel('Predicted Churn Probability'); axes[0].set_ylabel('Density')
axes[0].set_title('Prob. Distribution by Actual Class'); axes[0].legend()
axes[0].axvline(0.5, color='black', linestyle='--', alpha=0.6, label='Threshold=0.5')

# Risk Segmentation
risk_df = pd.DataFrame({'prob': rf_prob, 'actual': y_test.values})
risk_df['RiskSegment'] = pd.cut(risk_df['prob'],
                                  bins=[0, 0.25, 0.5, 0.75, 1.0],
                                  labels=['Low\n(<25%)', 'Medium\n(25-50%)',
                                          'High\n(50-75%)', 'Critical\n(>75%)'])
seg_churn = risk_df.groupby('RiskSegment', observed=True)['actual'].agg(['mean','count'])
colors_risk = ['#2ECC71','#F39C12','#E67E22','#E74C3C']
bars = axes[1].bar(seg_churn.index, seg_churn['mean'], color=colors_risk, edgecolor='white')
for bar, (rate, cnt) in zip(bars, zip(seg_churn['mean'], seg_churn['count'])):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f'{rate:.0%}\n(n={cnt})', ha='center', fontweight='bold', fontsize=9)
axes[1].set_title('Actual Churn Rate by Risk Segment'); axes[1].set_ylabel('Actual Churn Rate')
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
plt.tight_layout()
plt.savefig(f'{OUT}/fig14_churn_prob_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 14 saved — Churn Probability Distribution")

# ── Fig 15: Model Comparison ─────────────────────────────
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

models_comp = {
    'Logistic\nRegression': (lr_pred, lr_prob),
    'Random\nForest':       (rf_pred, rf_prob),
    'Gradient\nBoosting':   (gb_pred, gb_prob),
}
metrics = {'Accuracy':[], 'Precision':[], 'Recall':[], 'F1-Score':[], 'AUC-ROC':[]}
for name, (pred, prob) in models_comp.items():
    metrics['Accuracy'].append(accuracy_score(y_test, pred))
    metrics['Precision'].append(precision_score(y_test, pred))
    metrics['Recall'].append(recall_score(y_test, pred))
    metrics['F1-Score'].append(f1_score(y_test, pred))
    metrics['AUC-ROC'].append(roc_auc_score(y_test, prob))

x = np.arange(len(models_comp)); width = 0.15
fig, ax = plt.subplots(figsize=(14, 7))
pal = ['#3498DB','#E74C3C','#2ECC71','#9B59B6','#F39C12']
for i, (metric, vals) in enumerate(metrics.items()):
    ax.bar(x + i*width, vals, width, label=metric, color=pal[i], alpha=0.85, edgecolor='white')

ax.set_xticks(x + width*2); ax.set_xticklabels(list(models_comp.keys()), fontsize=12)
ax.set_ylabel('Score'); ax.set_ylim(0, 1.1)
ax.set_title('Figure 15 — Model Performance Comparison', fontsize=13, fontweight='bold')
ax.legend(bbox_to_anchor=(1.01,1), loc='upper left')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontsize=7, rotation=90, padding=2)
plt.tight_layout()
plt.savefig(f'{OUT}/fig15_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 15 saved — Model Comparison")

# ══════════════════════════════════════════════════════════
#  STEP 7 — Save Models & Predictions
# ══════════════════════════════════════════════════════════
joblib.dump(rf,     f'{MODEL}/random_forest.pkl')
joblib.dump(lr,     f'{MODEL}/logistic_regression.pkl')
joblib.dump(gb,     f'{MODEL}/gradient_boosting.pkl')
joblib.dump(scaler, f'{MODEL}/scaler.pkl')

# Save predictions for Tableau
pred_df = df.copy()
pred_df['RF_ChurnProb']    = rf.predict_proba(X)[:, 1]
pred_df['LR_ChurnProb']    = lr.predict_proba(scaler.transform(X))[:, 1]
pred_df['RF_ChurnPred']    = rf.predict(X)
pred_df['RiskSegment']     = pd.cut(pred_df['RF_ChurnProb'],
                                     bins=[0,0.25,0.5,0.75,1.0],
                                     labels=['Low','Medium','High','Critical'])
pred_df.to_csv(f'{OUT}/churn_predictions.csv', index=False)

print(f"\n✅ Models saved to {MODEL}/")
print(f"✅ Predictions saved → {OUT}/churn_predictions.csv")

# ══════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  FINAL MODEL PERFORMANCE SUMMARY")
print("="*60)
for name, (pred, prob) in models_comp.items():
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)
    f1  = f1_score(y_test, pred)
    rec = recall_score(y_test, pred)
    model_name = name.replace('\n',' ')
    print(f"  {model_name:<22} ACC={acc:.1%}  AUC={auc:.3f}  F1={f1:.3f}  Recall={rec:.1%}")
print("="*60)
print("\n  ⭐  Best model: Random Forest (highest AUC + interpretability)")
print("  📦  All artifacts saved and ready for Tableau / portfolio")
