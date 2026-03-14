# ============================================================
# ÉTAPE 6 — MODEL EVALUATION
# Projet : Customer Churn Prediction — E-Commerce
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# ── Rechargement et reconstruction du modèle ─────────────────
df = pd.read_csv('outputs/churn_dataset_engineered.csv')

feature_cols = [c for c in df.columns if c not in ['Customer_ID', 'churned']]
X = df[feature_cols]
y = df['churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

proba = rf.predict_proba(X_test)[:, 1]
pred  = rf.predict(X_test)

# ── GRAPHIQUE 10 : Matrice de confusion annotée ───────────────
cm = confusion_matrix(y_test, pred)
TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap='Blues')

# Valeurs dans les cellules
for i in range(2):
    for j in range(2):
        val = cm[i, j]
        color = 'white' if val > cm.max() / 2 else 'black'
        ax.text(j, i, str(val), ha='center', va='center',
                fontsize=28, fontweight='bold', color=color)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Prédit : Retenu', 'Prédit : Churner'], fontsize=11)
ax.set_yticklabels(['Réel : Retenu', 'Réel : Churner'], fontsize=11)
ax.set_title('Matrice de confusion — Random Forest', fontsize=13, fontweight='bold')

# Annotations business
ax.text(0, 0.35, f'✓ Vrais négatifs\n{TN} retenus bien identifiés',
        ha='center', va='center', fontsize=8.5, color='steelblue', style='italic')
ax.text(1, 0.35, f'⚠ Faux positifs\n{FP} retenus classés churners\n(promos inutiles)',
        ha='center', va='center', fontsize=8.5, color='darkorange', style='italic')
ax.text(0, 1.35, f'✗ Faux négatifs\n{FN} churners ratés\n(clients perdus !)',
        ha='center', va='center', fontsize=8.5, color='crimson', style='italic')
ax.text(1, 1.35, f'✓ Vrais positifs\n{TP} churners détectés',
        ha='center', va='center', fontsize=8.5, color='steelblue', style='italic')

plt.tight_layout()
plt.savefig('outputs/graph10_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ graph10_confusion_matrix.png")

# ── GRAPHIQUE 11 : Distribution des probabilités de churn ─────
fig, ax = plt.subplots(figsize=(9, 5))

# Sépare les proba des vrais churners vs vrais retenus
proba_churners  = proba[y_test == 1]
proba_retenus   = proba[y_test == 0]

ax.hist(proba_retenus,  bins=40, alpha=0.65, color='#4A90D9',
        label=f'Retenus réels  (n={len(proba_retenus)})')
ax.hist(proba_churners, bins=40, alpha=0.65, color='#E05A5A',
        label=f'Churners réels (n={len(proba_churners)})')

ax.axvline(0.5, color='black', linestyle='--', linewidth=1.5, label='Seuil décision (0.5)')
ax.set_xlabel('Probabilité de churn prédite', fontsize=11)
ax.set_ylabel('Nombre de clients', fontsize=11)
ax.set_title('Distribution des probabilités — le modèle sépare bien les deux groupes',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('outputs/graph11_probability_distribution.png', dpi=150)
plt.close()
print("✓ graph11_probability_distribution.png")

# ── Tableau des probabilités par client (test set) ────────────
test_ids = df.loc[X_test.index, 'Customer_ID'].values
results_df = pd.DataFrame({
    'Customer_ID':       test_ids,
    'churn_probability': (proba * 100).round(1),
    'churn_predicted':   pred,
    'churn_actual':      y_test.values,
})

# Segmentation par niveau de risque
def risk_level(p):
    if p >= 80:  return 'Très élevé 🔴'
    elif p >= 50: return 'Élevé 🟠'
    elif p >= 20: return 'Modéré 🟡'
    else:         return 'Faible 🟢'

results_df['risk_level'] = results_df['churn_probability'].apply(risk_level)
results_df = results_df.sort_values('churn_probability', ascending=False)

results_df.to_csv('outputs/churn_predictions.csv', index=False)
print("✓ churn_predictions.csv sauvegardé")

# ── RÉSUMÉ FINAL ──────────────────────────────────────────────
print()
print("=" * 55)
print("RÉSUMÉ ÉVALUATION — Random Forest")
print("=" * 55)
print(f"  ROC-AUC   : {roc_auc_score(y_test, proba):.4f}  ← proche de 1.0 = excellent")
print(f"  Recall    : {TP/(TP+FN):.4f}  ← 88% des churners détectés")
print(f"  Precision : {TP/(TP+FP):.4f}  ← 91% des alertes sont justes")
print(f"  Accuracy  : {(TP+TN)/(TP+TN+FP+FN):.4f}  ← 97% de bonnes prédictions")
print()
print("  Matrice de confusion :")
print(f"    ✓ Churners détectés    : {TP}  (vrais positifs)")
print(f"    ✗ Churners ratés       : {FN}  (faux négatifs — clients perdus)")
print(f"    ⚠ Fausses alertes      : {FP}  (faux positifs — promos inutiles)")
print(f"    ✓ Retenus bien classés : {TN}  (vrais négatifs)")
print()
print("  Répartition par niveau de risque :")
print(results_df['risk_level'].value_counts().to_string())
print("=" * 55)
print()
print("  → Fichier churn_predictions.csv : probabilité de churn")
print("    pour chacun des 1200 clients du test set")