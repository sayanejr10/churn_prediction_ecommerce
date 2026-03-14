# 🛒 Customer Churn Prediction — E-Commerce

> Prédire quels clients vont quitter une plateforme e-commerce, et proposer des actions concrètes pour les retenir.

---

## 📌 Contexte business

Dans le e-commerce, **acquérir un nouveau client coûte 5 à 10 fois plus cher que garder un client existant**.

Le **churn** (attrition client) désigne le phénomène par lequel un client cesse d'utiliser une plateforme. Identifier ces clients *avant* qu'ils partent permet à l'entreprise d'intervenir à temps — via des promotions, un meilleur support, ou des offres de fidélisation.

Ce projet répond à la question :
> **"Quels clients vont partir, pourquoi, et comment les retenir ?"**

---

## 🎯 Objectifs du projet

1. **Analyser** le comportement des clients (achats, engagement, satisfaction)
2. **Identifier** les facteurs qui influencent le churn
3. **Construire** un modèle prédictif (probabilité de churn par client)
4. **Proposer** un plan d'actions business par segment de risque

---

## 📁 Structure du projet

```
churn_project/
├── data/
│   ├── ecommerce_customer_features.csv   # Comportement des clients
│   └── ecommerce_customer_targets.csv    # Labels churn (Yes/No)
├── scripts/
│   ├── step2_data_preparation.py         # Fusion, nettoyage, encodage
│   ├── step3_eda.py                      # Analyse exploratoire + graphiques
│   ├── step4_feature_engineering.py      # Création de nouvelles variables
│   ├── step5_machine_learning.py         # Entraînement des modèles
│   ├── step6_evaluation.py               # Évaluation + matrice de confusion
│   └── step7_business_insights.py        # Insights + plan d'actions
├── outputs/
│   ├── churn_dataset_clean.csv           # Dataset nettoyé
│   ├── churn_dataset_engineered.csv      # Dataset enrichi (20 features)
│   ├── churn_predictions.csv             # Probabilités de churn par client
│   └── rapport_business.txt             # Rapport business complet
└── README.md
```

---

## 📊 Dataset

| Propriété | Valeur |
|---|---|
| Source | Kaggle (synthétique, relations réalistes) |
| Clients | 6 000 |
| Features | 15 (historique achat, engagement, satisfaction) |
| Taux de churn | 15.5% |
| Valeurs manquantes | 0 |

---

## 🔬 Méthodologie — 7 étapes

### Étape 1 — Data Collection
Chargement de deux fichiers séparés (features + targets), structure typique des compétitions Kaggle.

### Étape 2 — Data Preparation
- Fusion des deux fichiers sur `Customer_ID`
- Encodage des variables catégorielles (`loyalty_member`, `churned`) en 0/1
- Vérification : 0 valeur manquante

### Étape 3 — Exploratory Data Analysis (EDA)
Analyse visuelle pour identifier les patterns avant la modélisation.

**Insights clés découverts :**
- Les churners n'ont pas acheté depuis **81 jours** en moyenne (vs 20j pour les retenus)
- L'`engagement_score` des churners est de **2.4** (vs 5.3 pour les retenus)
- Les membres du programme fidélité churne à **9%** vs **17%** pour les non-membres

### Étape 4 — Feature Engineering
Création de 6 nouvelles variables pour améliorer les performances du modèle :

| Feature | Description | Corrélation avec churn |
|---|---|---|
| `is_inactive` | 1 si aucun achat depuis >60j | +0.776 |
| `low_engagement` | 1 si engagement_score < 3 | +0.780 |
| `risk_score` | Score composite [0-1] | +0.772 |
| `recency_segment` | Segment d'inactivité (0 à 3) | +0.659 |
| `orders_per_month` | Fréquence d'achat normalisée | -0.053 |
| `high_support` | 1 si 2+ tickets support | +0.081 |

> Les 3 meilleures features du modèle final sont des features créées — preuve de l'impact du Feature Engineering.

### Étape 5 — Machine Learning
Entraînement de 3 modèles sur 80% des données (4 800 clients), évaluation sur 20% (1 200 clients).

| Modèle | ROC-AUC | Recall | Precision | Accuracy |
|---|---|---|---|---|
| Logistic Regression | **0.9939** | 0.8710 | 0.9153 | 0.9675 |
| **Random Forest** | 0.9934 | **0.8817** | 0.9111 | **0.9683** |
| Gradient Boosting | 0.9912 | 0.8763 | 0.9106 | 0.9675 |

✅ **Modèle retenu : Random Forest** — meilleur Recall (moins de churners ratés)

### Étape 6 — Évaluation
**Matrice de confusion — Random Forest :**

```
                   Prédit Retenu   Prédit Churner
Réel Retenu    [      998               16      ]
Réel Churner   [       22              164      ]
```

- ✅ **164 churners détectés** → actions de rétention possibles
- ❌ **22 churners ratés** → clients perdus sans intervention
- ⚠️ **16 fausses alertes** → promotions envoyées inutilement (coût faible)

### Étape 7 — Business Insights & Actions

**Segmentation des clients par niveau de risque :**

| Segment | Probabilité churn | Clients | Profil moyen | Actions recommandées |
|---|---|---|---|---|
| 🔴 Très élevé | ≥ 80% | 143 | Inactif 86j, engagement 2.0 | Email personnalisé + réduction 20-30% |
| 🟠 Élevé | 50–80% | 38 | Inactif 56j, engagement 3.4 | Offre flash + recommandations produits |
| 🟡 Modéré | 20–50% | 43 | Inactif 52j, engagement 3.6 | Newsletter + points fidélité bonus |
| 🟢 Faible | < 20% | 976 | Actif 19j, engagement 5.4 | Programme fidélité + upsell |

**Impact financier estimé :**
- Valeur annuelle moyenne par client : ~963 USD
- Clients à risque élevé détectés : 181
- Revenu potentiellement préservé (rétention à 50%) : **~87 000 USD**

---

## 📈 Résultats clés

| Métrique | Score | Interprétation |
|---|---|---|
| ROC-AUC | **0.9934** | Excellent (1.0 = parfait, 0.5 = aléatoire) |
| Recall | **88.2%** | 88% des churners détectés avant qu'ils partent |
| Precision | **91.1%** | 91% des alertes envoyées sont justifiées |
| Accuracy | **96.8%** | 97% de prédictions correctes |

---

## 🛠️ Stack technique

| Outil | Usage |
|---|---|
| Python 3.10 | Langage principal |
| pandas | Manipulation des données |
| matplotlib / seaborn | Visualisations |
| scikit-learn | Machine Learning |

---

## ▶️ Lancer le projet

```bash
# 1. Cloner le repo
git clone https://github.com/TON_USERNAME/churn_project.git
cd churn_project

# 2. Installer les dépendances
pip install pandas matplotlib seaborn scikit-learn

# 3. Lancer les étapes dans l'ordre
python3 scripts/step2_data_preparation.py
python3 scripts/step3_eda.py
python3 scripts/step4_feature_engineering.py
python3 scripts/step5_machine_learning.py
python3 scripts/step6_evaluation.py
python3 scripts/step7_business_insights.py
```
