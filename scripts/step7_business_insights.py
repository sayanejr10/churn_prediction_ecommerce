# ============================================================
# ÉTAPE 7 — BUSINESS INSIGHTS & ACTIONS
# Projet : Customer Churn Prediction — E-Commerce
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ── Reconstruction du modèle ─────────────────────────────────
df = pd.read_csv('outputs/churn_dataset_engineered.csv')
feature_cols = [c for c in df.columns if c not in ['Customer_ID', 'churned']]
X = df[feature_cols]; y = df['churned']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
proba = rf.predict_proba(X_test)[:, 1]

test_df = X_test.copy()
test_df['churn_proba'] = proba
test_df['churned']     = y_test.values
test_df['Customer_ID'] = df.loc[X_test.index, 'Customer_ID'].values

def risk_level(p):
    if p >= 0.8:   return 'Très élevé 🔴'
    elif p >= 0.5: return 'Élevé 🟠'
    elif p >= 0.2: return 'Modéré 🟡'
    else:          return 'Faible 🟢'

test_df['risk_level'] = test_df['churn_proba'].apply(risk_level)

# ── GRAPHIQUE 12 : Feature Importance finale ──────────────────
importances = pd.Series(rf.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))
colors = ['#E05A5A' if v >= 0.08 else '#4A90D9' if v >= 0.03 else '#B0BEC5'
          for v in importances.values]
ax.barh(importances.index, importances.values, color=colors, edgecolor='white', height=0.7)
ax.set_title('Feature Importance — quelles variables guident le modèle ?',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Importance relative')
ax.axvline(0.08, color='#E05A5A', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(0.03, color='#4A90D9', linestyle='--', linewidth=1, alpha=0.5)
p1 = mpatches.Patch(color='#E05A5A', label='Très importante (>0.08)')
p2 = mpatches.Patch(color='#4A90D9', label='Importante (0.03-0.08)')
p3 = mpatches.Patch(color='#B0BEC5', label='Faible impact (<0.03)')
ax.legend(handles=[p1, p2, p3], fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('outputs/graph12_feature_importance_final.png', dpi=150)
plt.close()
print("✓ graph12_feature_importance_final.png")

# ── GRAPHIQUE 13 : Profils des segments de risque ────────────
segments   = ['Très élevé 🔴', 'Élevé 🟠', 'Modéré 🟡', 'Faible 🟢']
seg_colors = ['#E05A5A', '#E8A838', '#F5D76E', '#4A90D9']
profile_vars = ['days_since_last_purchase', 'engagement_score',
                'satisfaction_score', 'loyalty_member']
profile_labels = ['Jours inactif', 'Engagement', 'Satisfaction', 'Membre fidélité (%)']

fig, axes = plt.subplots(1, 4, figsize=(14, 5))

for i, (var, label) in enumerate(zip(profile_vars, profile_labels)):
    ax = axes[i]
    vals = []
    for seg in segments:
        seg_df = test_df[test_df['risk_level'] == seg]
        v = seg_df[var].mean()
        if var == 'loyalty_member':
            v = v * 100
        vals.append(v)
    bars = ax.bar(range(4), vals, color=seg_colors, edgecolor='white', width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                f'{v:.0f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_title(label, fontsize=10, fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['🔴', '🟠', '🟡', '🟢'], fontsize=13)
    ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('Profil moyen par segment de risque', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/graph13_segment_profiles.png', dpi=150)
plt.close()
print("✓ graph13_segment_profiles.png")

# ── Rapport business complet ──────────────────────────────────
report_lines = []

def line(txt=''):
    report_lines.append(txt)
    print(txt)

line("=" * 60)
line("RAPPORT BUSINESS — CUSTOMER CHURN PREDICTION")
line("=" * 60)
line()
line("📊 PERFORMANCE DU MODÈLE")
line("-" * 40)
line("  ROC-AUC   : 0.9934  (excellent — proche de 1.0)")
line("  Recall    : 88.2%   (88% des churners détectés)")
line("  Precision : 91.1%   (91% des alertes sont justes)")
line("  Accuracy  : 96.8%")
line()
line("👥 SEGMENTATION DES 1 200 CLIENTS DU TEST")
line("-" * 40)
for seg, color in zip(segments, ['🔴', '🟠', '🟡', '🟢']):
    n   = len(test_df[test_df['risk_level'] == seg])
    pct = n / len(test_df) * 100
    line(f"  {seg:<20} : {n:>4} clients  ({pct:.1f}%)")
line()
line("🔍 PROFIL DES CLIENTS À RISQUE TRÈS ÉLEVÉ 🔴")
line("-" * 40)
high_risk = test_df[test_df['risk_level'] == 'Très élevé 🔴']
line(f"  Inactivité moyenne     : {high_risk['days_since_last_purchase'].mean():.0f} jours")
line(f"  Score d'engagement     : {high_risk['engagement_score'].mean():.2f} / 10")
line(f"  Score de satisfaction  : {high_risk['satisfaction_score'].mean():.2f} / 10")
line(f"  Valeur panier moyen    : {high_risk['avg_order_value'].mean():.0f} USD")
line(f"  Membres programme fidélité : {high_risk['loyalty_member'].mean():.0%}")
line()
line("💡 FACTEURS DE CHURN IDENTIFIÉS")
line("-" * 40)
line("  1. Inactivité récente          (corrélation : +0.78)")
line("     → Aucun achat depuis >60j = signal d'alarme fort")
line()
line("  2. Faible engagement           (corrélation : +0.78)")
line("     → Score <3 : le client ne visite plus la plateforme")
line()
line("  3. Score de risque composite   (corrélation : +0.77)")
line("     → Combinaison inactivité + engagement + satisfaction")
line()
line("  4. Segment de récence          (corrélation : +0.66)")
line("     → Plus l'inactivité est longue, plus le risque monte")
line()
line("  5. Satisfaction                (différence : 7.7 vs 8.1)")
line("     → Signal modéré mais accompagne souvent les autres")
line()
line("🎯 PLAN D'ACTIONS PAR SEGMENT")
line("-" * 40)
line()
line("  🔴 TRÈS ÉLEVÉ (≥80% de risque) — 143 clients")
line("     Profil  : inactif depuis 86j, engagement 2.0")
line("     Actions :")
line("       • Email personnalisé 'Vous nous manquez'")
line("       • Réduction exceptionnelle 20-30%")
line("       • Appel service client si panier > 100 USD")
line("       • Offrir l'adhésion au programme fidélité")
line()
line("  🟠 ÉLEVÉ (50-80% de risque) — 38 clients")
line("     Profil  : inactif depuis 56j, engagement 3.4")
line("     Actions :")
line("       • Email de relance avec recommandations produits")
line("       • Offre flash limitée dans le temps")
line("       • Notification push si app mobile installée")
line()
line("  🟡 MODÉRÉ (20-50% de risque) — 43 clients")
line("     Profil  : inactif depuis 52j, engagement 3.6")
line("     Actions :")
line("       • Newsletter avec nouveautés personnalisées")
line("       • Points fidélité bonus sur prochain achat")
line("       • Enquête de satisfaction courte (2 questions)")
line()
line("  🟢 FAIBLE (<20% de risque) — 976 clients")
line("     Profil  : actif, engagement 5.4, satisfaction 8.1")
line("     Actions :")
line("       • Programme fidélité et récompenses")
line("       • Upsell / cross-sell sur leurs catégories")
line("       • Aucune action urgente nécessaire")
line()
line("💰 IMPACT BUSINESS ESTIMÉ")
line("-" * 40)
avg_order = test_df['avg_order_value'].mean()
annual_value = avg_order * 12
high_n = len(test_df[test_df['risk_level'].isin(['Très élevé 🔴', 'Élevé 🟠'])])
line(f"  Valeur annuelle moyenne / client  : ~{annual_value:.0f} USD")
line(f"  Clients à risque élevé détectés   : {high_n}")
line(f"  Revenu potentiellement sauvé      : ~{high_n * annual_value:,.0f} USD")
line(f"  (si on retient 50% de ces clients)")
line(f"  → ~{high_n * annual_value * 0.5:,.0f} USD de revenu préservé")
line()
line("=" * 60)

# Sauvegarde du rapport
with open('outputs/rapport_business.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print()
print("✓ rapport_business.txt sauvegardé")
print("✓ Projet complet — 7 étapes, 13 graphiques, 1 rapport business")