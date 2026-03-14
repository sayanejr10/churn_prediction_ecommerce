# ============================================================
# ÉTAPE 5 — MACHINE LEARNING
# Projet : Customer Churn Prediction — E-Commerce
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# ── Chargement ───────────────────────────────────────────────
df = pd.read_csv('outputs/churn_dataset_engineered.csv')

feature_cols = [c for c in df.columns if c not in ['Customer_ID', 'churned']]
X = df[feature_cols]
y = df['churned']

# ── Split Train / Test ────────────────────────────────────────
# stratify=y → on garde le même ratio 15.5% churn dans les deux sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train : {len(X_train)} clients | Test : {len(X_test)} clients")
print(f"Churn rate — Train : {y_train.mean():.1%} | Test : {y_test.mean():.1%}")

# ── Normalisation (nécessaire pour Logistic Regression) ──────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit UNIQUEMENT sur le train
X_test_sc  = scaler.transform(X_test)         # applique la même échelle au test

# ── Définition des 3 modèles ─────────────────────────────────
models = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'X_train': X_train_sc,
        'X_test':  X_test_sc,
    },
    'Random Forest': {
        'model': RandomForestClassifier(n_estimators=100, random_state=42),
        'X_train': X_train,
        'X_test':  X_test,
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'X_train': X_train,
        'X_test':  X_test,
    },
}

# ── Entraînement + évaluation ─────────────────────────────────
results = {}
print("\n" + "=" * 55)
print("ENTRAÎNEMENT DES MODÈLES")
print("=" * 55)

for name, config in models.items():
    m = config['model']
    m.fit(config['X_train'], y_train)

    proba  = m.predict_proba(config['X_test'])[:, 1]
    pred   = m.predict(config['X_test'])
    auc    = roc_auc_score(y_test, proba)
    report = classification_report(y_test, pred, output_dict=True)

    results[name] = {
        'model':     m,
        'proba':     proba,
        'pred':      pred,
        'auc':       auc,
        'precision': report['1']['precision'],
        'recall':    report['1']['recall'],
        'f1':        report['1']['f1-score'],
        'accuracy':  report['accuracy'],
    }

    print(f"\n{name}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Precision : {report['1']['precision']:.4f}")
    print(f"  Recall    : {report['1']['recall']:.4f}")
    print(f"  F1-score  : {report['1']['f1-score']:.4f}")
    print(f"  Accuracy  : {report['accuracy']:.4f}")

# ── GRAPHIQUE 7 : Comparaison des métriques ───────────────────
metrics   = ['auc', 'precision', 'recall', 'f1', 'accuracy']
labels    = ['ROC-AUC', 'Precision', 'Recall', 'F1-Score', 'Accuracy']
model_names = list(results.keys())
colors    = ['#4A90D9', '#E8A838', '#E05A5A']

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 5))
for i, (name, color) in enumerate(zip(model_names, colors)):
    vals = [results[name][m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=name,
                  color=color, edgecolor='white', alpha=0.9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)

ax.set_xticks(x + width)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0.80, 1.02)
ax.set_title('Comparaison des 3 modèles — toutes les métriques',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('outputs/graph7_model_comparison.png', dpi=150)
plt.close()
print("\n✓ graph7_model_comparison.png")

# ── GRAPHIQUE 8 : Courbes ROC ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for (name, color) in zip(model_names, colors):
    fpr, tpr, _ = roc_curve(y_test, results[name]['proba'])
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{name} (AUC = {results[name]['auc']:.4f})")
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aléatoire (AUC = 0.50)')
ax.set_xlabel('Taux de faux positifs (FPR)')
ax.set_ylabel('Taux de vrais positifs (TPR)')
ax.set_title('Courbes ROC — comparaison des modèles', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('outputs/graph8_roc_curves.png', dpi=150)
plt.close()
print("✓ graph8_roc_curves.png")

# ── GRAPHIQUE 9 : Feature importance (Random Forest) ─────────
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=True).tail(12)

fig, ax = plt.subplots(figsize=(8, 6))
colors_imp = ['#E05A5A' if v > 0.08 else '#4A90D9' for v in importances.values]
ax.barh(importances.index, importances.values, color=colors_imp, edgecolor='white')
ax.set_title('Feature Importance — Random Forest (Top 12)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Importance')
ax.spines[['top', 'right']].set_visible(False)
patch_high = mpatches.Patch(color='#E05A5A', label='Importance > 0.08')
patch_low  = mpatches.Patch(color='#4A90D9', label='Importance ≤ 0.08')
ax.legend(handles=[patch_high, patch_low], fontsize=9)
plt.tight_layout()
plt.savefig('outputs/graph9_feature_importance.png', dpi=150)
plt.close()
print("✓ graph9_feature_importance.png")

# ── Meilleur modèle ───────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['auc'])
print()
print("=" * 55)
print(f"MEILLEUR MODÈLE : {best_name}")
print(f"  ROC-AUC   : {results[best_name]['auc']:.4f}")
print(f"  Recall    : {results[best_name]['recall']:.4f}")
print(f"  Precision : {results[best_name]['precision']:.4f}")
print("=" * 55)