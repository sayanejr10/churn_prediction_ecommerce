# ============================================================
# ÉTAPE 3 — EXPLORATORY DATA ANALYSIS (EDA)
# Projet : Customer Churn Prediction — E-Commerce
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

# ── Chargement du dataset nettoyé ────────────────────────────
df = pd.read_csv('outputs/churn_dataset_clean.csv')

# Couleurs : bleu = retenus, rouge = churners
COLORS = {0: '#4A90D9', 1: '#E05A5A'}
LABELS = {0: 'Retenus', 1: 'Churners'}

# ── GRAPHIQUE 1 : Distribution de la variable cible ──────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = df['churned'].value_counts()
bars = ax.bar(['Retenus (0)', 'Churners (1)'], counts.values,
              color=[COLORS[0], COLORS[1]], width=0.5, edgecolor='white')
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{val}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=11, fontweight='bold')
ax.set_title('Distribution de la variable cible (churned)', fontsize=13, fontweight='bold')
ax.set_ylabel('Nombre de clients')
ax.set_ylim(0, 6000)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('outputs/graph1_target_distribution.png', dpi=150)
plt.close()
print("✓ graph1_target_distribution.png")

# ── GRAPHIQUE 2 : Les 4 variables clés — boxplots ────────────
key_vars = [
    ('days_since_last_purchase', 'Jours depuis dernier achat'),
    ('engagement_score',         'Score d\'engagement'),
    ('satisfaction_score',       'Score de satisfaction'),
    ('customer_support_tickets', 'Tickets support'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, (col, label) in enumerate(key_vars):
    ax = axes[i]
    data_retained = df[df['churned'] == 0][col]
    data_churned  = df[df['churned'] == 1][col]
    bp = ax.boxplot([data_retained, data_churned],
                    patch_artist=True,
                    labels=['Retenus', 'Churners'],
                    medianprops=dict(color='white', linewidth=2))
    bp['boxes'][0].set_facecolor(COLORS[0])
    bp['boxes'][1].set_facecolor(COLORS[1])
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    # Annotation différence de moyenne
    m0 = data_retained.mean()
    m1 = data_churned.mean()
    ax.annotate(f'Moy. retenus : {m0:.1f}\nMoy. churners : {m1:.1f}',
                xy=(0.98, 0.97), xycoords='axes fraction',
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', alpha=0.8))

fig.suptitle('Variables les plus discriminantes — Churners vs Retenus',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/graph2_key_variables_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ graph2_key_variables_boxplot.png")

# ── GRAPHIQUE 3 : Taux de churn par loyalty_member ───────────
fig, ax = plt.subplots(figsize=(6, 4))
churn_by_loyalty = df.groupby('loyalty_member')['churned'].mean() * 100
bars = ax.bar(['Non membre\n(loyalty=0)', 'Membre fidélité\n(loyalty=1)'],
              churn_by_loyalty.values,
              color=[COLORS[1], COLORS[0]], width=0.5, edgecolor='white')
for bar, val in zip(bars, churn_by_loyalty.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
ax.set_title('Taux de churn selon le programme de fidélité', fontsize=13, fontweight='bold')
ax.set_ylabel('Taux de churn (%)')
ax.set_ylim(0, 25)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('outputs/graph3_churn_by_loyalty.png', dpi=150)
plt.close()
print("✓ graph3_churn_by_loyalty.png")

# ── GRAPHIQUE 4 : Matrice de corrélation ─────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
num_cols = [c for c in df.columns if c != 'Customer_ID']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax,
            annot_kws={'size': 8}, linewidths=0.5)
ax.set_title('Matrice de corrélation — toutes les variables', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/graph4_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ graph4_correlation_matrix.png")

# ── GRAPHIQUE 5 : Corrélation de chaque variable avec churned ─
fig, ax = plt.subplots(figsize=(8, 6))
corr_with_churn = df[num_cols].corr()['churned'].drop('churned').sort_values()
colors_bar = [COLORS[1] if v > 0 else COLORS[0] for v in corr_with_churn]
bars = ax.barh(corr_with_churn.index, corr_with_churn.values, color=colors_bar, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Corrélation de chaque variable avec le churn', fontsize=13, fontweight='bold')
ax.set_xlabel('Coefficient de corrélation')
ax.spines[['top', 'right']].set_visible(False)

patch_pos = mpatches.Patch(color=COLORS[1], label='Corrélation positive (↑ churn)')
patch_neg = mpatches.Patch(color=COLORS[0], label='Corrélation négative (↓ churn)')
ax.legend(handles=[patch_pos, patch_neg], loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/graph5_correlation_with_churn.png', dpi=150)
plt.close()
print("✓ graph5_correlation_with_churn.png")

# ── RÉSUMÉ INSIGHTS ──────────────────────────────────────────
print()
print("=" * 55)
print("INSIGHTS CLÉS — EDA")
print("=" * 55)
print("🔴 Signaux FORTS de churn :")
print("   days_since_last_purchase : churners=81j vs retenus=20j")
print("   engagement_score         : churners=2.4 vs retenus=5.3")
print()
print("🟡 Signaux MODÉRÉS de churn :")
print("   satisfaction_score       : churners=7.7 vs retenus=8.1")
print("   customer_support_tickets : churners=1.1 vs retenus=0.8")
print("   browsing_frequency       : churners=2.6 vs retenus=3.2")
print()
print("🟢 Facteur PROTECTEUR :")
print("   loyalty_member           : membres=9% churn vs non-membres=17%")
print()
print("⚪ Variables peu discriminantes :")
print("   avg_order_value, cart_abandonment_rate,")
print("   discount_usage_rate, price_sensitivity_index")
print("=" * 55)
print()
print("✓ 5 graphiques sauvegardés dans outputs/")