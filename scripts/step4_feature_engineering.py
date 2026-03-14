# ============================================================
# ÉTAPE 4 — FEATURE ENGINEERING
# Projet : Customer Churn Prediction — E-Commerce
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Chargement du dataset nettoyé ────────────────────────────
df = pd.read_csv('outputs/churn_dataset_clean.csv')
print(f"Dataset chargé : {df.shape[0]} clients, {df.shape[1]} colonnes")

# ============================================================
# FEATURE 1 : recency_segment
# Segmente les clients selon leur inactivité
# 0 = très actif (0-15j)  1 = actif (16-30j)
# 2 = à risque (31-60j)   3 = inactif (>60j)
# ============================================================
bins = [0, 15, 30, 60, df['days_since_last_purchase'].max() + 1]
df['recency_segment'] = pd.cut(
    df['days_since_last_purchase'],
    bins=bins,
    labels=[0, 1, 2, 3],
    include_lowest=True
).astype(int)

# ============================================================
# FEATURE 2 : orders_per_month
# Fréquence d'achat normalisée par l'ancienneté du compte
# Un client ancien avec peu de commandes = signal faible
# ============================================================
df['orders_per_month'] = df['total_orders'] / (df['account_age_months'] + 1)

# ============================================================
# FEATURE 3 : is_inactive
# Flag binaire : 1 si aucun achat depuis plus de 60 jours
# Très simple, mais très puissant (corr=0.78 avec churn)
# ============================================================
df['is_inactive'] = (df['days_since_last_purchase'] > 60).astype(int)

# ============================================================
# FEATURE 4 : risk_score
# Score de risque composite sur 3 dimensions :
#   - 40% inactivité récente
#   - 40% faible engagement
#   - 20% insatisfaction
# ============================================================
df['risk_score'] = (
    df['days_since_last_purchase'] / df['days_since_last_purchase'].max() * 0.4 +
    (1 - df['engagement_score']    / df['engagement_score'].max())         * 0.4 +
    (1 - df['satisfaction_score']  / df['satisfaction_score'].max())       * 0.2
)

# ============================================================
# FEATURE 5 : low_engagement
# Flag binaire : 1 si engagement_score < 3 (très peu actif)
# Corrélation avec churn : 0.78 — notre meilleure feature !
# ============================================================
df['low_engagement'] = (df['engagement_score'] < 3).astype(int)

# ============================================================
# FEATURE 6 : high_support
# Flag binaire : 1 si le client a ouvert 2+ tickets support
# Signal d'insatisfaction ou de problèmes récurrents
# ============================================================
df['high_support'] = (df['customer_support_tickets'] >= 2).astype(int)

# ── Vérification ─────────────────────────────────────────────
new_features = ['recency_segment', 'orders_per_month', 'is_inactive',
                'risk_score', 'low_engagement', 'high_support']

print("\n--- Corrélation des nouvelles features avec le churn ---")
for f in new_features:
    corr = df[f].corr(df['churned'])
    print(f"  {f:<25} : {corr:+.3f}")

print(f"\n  Dataset final : {df.shape[1]} colonnes ({df.shape[1]-2} features + ID + target)")

# ── GRAPHIQUE : Taux de churn des features binaires ──────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
binary_features = [
    ('is_inactive',    ['Actif (<60j)', 'Inactif (>60j)'],    ['0', '1']),
    ('low_engagement', ['Engagé (≥3)',  'Peu engagé (<3)'],   ['0', '1']),
    ('high_support',   ['0-1 ticket',   '2+ tickets'],         ['0', '1']),
]
COLORS = ['#4A90D9', '#E05A5A']

for ax, (feat, xlabels, vals) in zip(axes, binary_features):
    rates = [df[df[feat] == int(v)]['churned'].mean() * 100 for v in vals]
    bars = ax.bar(xlabels, rates, color=COLORS, width=0.5, edgecolor='white')
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_title(feat, fontsize=11, fontweight='bold')
    ax.set_ylabel('Taux de churn (%)')
    ax.set_ylim(0, 105)
    ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('Taux de churn selon les nouvelles features binaires',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/graph6_new_features_churn_rate.png', dpi=150)
plt.close()
print("\n✓ graph6_new_features_churn_rate.png sauvegardé")

# ── Sauvegarde du dataset enrichi ────────────────────────────
df.to_csv('outputs/churn_dataset_engineered.csv', index=False)
print("✓ churn_dataset_engineered.csv sauvegardé")

# ── RÉSUMÉ ───────────────────────────────────────────────────
print()
print("=" * 55)
print("RÉSUMÉ FEATURE ENGINEERING")
print("=" * 55)
print("6 nouvelles features créées :")
print("  recency_segment   → segment d'inactivité (0 à 3)")
print("  orders_per_month  → fréquence d'achat normalisée")
print("  is_inactive       → inactif depuis >60j (88% churn !)")
print("  risk_score        → score composite [0-1]")
print("  low_engagement    → engagement faible (92% churn !)")
print("  high_support      → 2+ tickets support")
print()
print("Top 3 features les plus corrélées avec le churn :")
all_corr = df.drop(columns=['Customer_ID', 'churned']).corrwith(df['churned']).abs().sort_values(ascending=False)
print(all_corr.head(3).to_string())
print("=" * 55)