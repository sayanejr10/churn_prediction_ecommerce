# ============================================================
# ÉTAPE 2 — DATA PREPARATION
# Projet : Customer Churn Prediction — E-Commerce
# ============================================================

import pandas as pd

# ── 1. Chargement des fichiers ────────────────────────────────
features = pd.read_csv('data/ecommerce_customer_features.csv')
targets  = pd.read_csv('data/ecommerce_customer_targets.csv')

print("Features shape :", features.shape)   # (6000, 15)
print("Targets shape  :", targets.shape)    # (6000, 2)

# ── 2. Fusion sur la clé commune Customer_ID ─────────────────
df = features.merge(targets, on='Customer_ID')

print("\nAprès fusion :", df.shape)          # (6000, 16)

# ── 3. Vérification des valeurs manquantes ────────────────────
missing = df.isnull().sum()
print("\nValeurs manquantes :")
print(missing[missing > 0] if missing.sum() > 0 else "  → Aucune valeur manquante ✓")

# ── 4. Encodage des variables catégorielles ───────────────────

# loyalty_member : Yes → 1 / No → 0
df['loyalty_member'] = df['loyalty_member'].map({'Yes': 1, 'No': 0})

# churned (target) : Yes → 1 / No → 0
df['churned'] = df['churned'].map({'Yes': 1, 'No': 0})

print("\nEncodage loyalty_member :", df['loyalty_member'].value_counts().to_dict())
print("Encodage churned        :", df['churned'].value_counts().to_dict())

# ── 5. Aperçu du dataframe final ─────────────────────────────
print("\nAperçu du dataframe final :")
print(df.head())
print("\nTypes des colonnes :")
print(df.dtypes)

# ── 6. Sauvegarde du dataset nettoyé ─────────────────────────
df.to_csv('churn_dataset_clean.csv', index=False)
print("\n✓ Dataset sauvegardé → churn_dataset_clean.csv")

# ── Résumé ───────────────────────────────────────────────────
print("\n" + "="*50)
print("RÉSUMÉ DATA PREPARATION")
print("="*50)
print(f"  Clients total        : {len(df):,}")
print(f"  Features disponibles : {df.shape[1] - 2} (hors ID et target)")
print(f"  Churners (Yes=1)     : {df['churned'].sum():,}  ({df['churned'].mean()*100:.1f}%)")
print(f"  Retenus  (No=0)      : {(df['churned']==0).sum():,}  ({(df['churned']==0).mean()*100:.1f}%)")
print(f"  Valeurs manquantes   : {df.isnull().sum().sum()}")
print("="*50)