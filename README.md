# 🛒 Customer Churn Prediction — E-Commerce

> Prédire quels clients vont quitter une plateforme e-commerce, et proposer des actions concrètes pour les retenir.

## 🌐 Démo en ligne

| | Lien |
|---|---|
| **Application** | https://churniq.up.railway.app |
| **API docs** | https://churniq.up.railway.app/docs |
| **GitHub** | https://github.com/sayanejr10/churn_prediction_ecommerce |

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
4. **Exposer** le modèle via une API REST
5. **Proposer** une interface web pour utiliser le modèle en temps réel

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
├── notebooks/
│   └── churn_prediction_ecommerce.ipynb  # Notebook complet (EDA → Business)
├── api/
│   └── main.py                           # API FastAPI — endpoint /predict
├── frontend/
│   └── index.html                        # Interface web glassmorphism
├── model/
│   ├── churn_model.pkl                   # Modèle Random Forest entraîné
│   └── feature_cols.pkl                  # Features utilisées
├── outputs/
│   ├── churn_predictions.csv             # Probabilités de churn par client
│   └── rapport_business.txt             # Rapport business complet
├── Dockerfile                            # Image Docker
├── docker-compose.yml                    # Orchestration
├── requirements.txt                      # Dépendances Python
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

| Segment | Probabilité churn | Clients | Actions recommandées |
|---|---|---|---|
| 🔴 Très élevé | ≥ 80% | 143 | Email personnalisé + réduction 20-30% |
| 🟠 Élevé | 50–80% | 38 | Offre flash + recommandations produits |
| 🟡 Modéré | 20–50% | 43 | Newsletter + points fidélité bonus |
| 🟢 Faible | < 20% | 976 | Programme fidélité + upsell |

**Impact financier estimé :**
- Valeur annuelle moyenne par client : ~963 USD
- Clients à risque élevé détectés : 181
- Revenu potentiellement préservé (rétention à 50%) : **~87 000 USD**

---

## 🚀 API REST — FastAPI

Le modèle est exposé via une API REST construite avec **FastAPI**.

### Endpoints

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/` | Interface web (page d'accueil) |
| `POST` | `/predict` | Prédiction du churn |
| `GET` | `/docs` | Documentation interactive |

### Exemple de réponse

```json
{
  "churn_probability": 0.87,
  "churn_percentage": "87.0%",
  "risk_level": "Très élevé 🔴",
  "action": "Email personnalisé + réduction 20-30% + offrir programme fidélité"
}
```

---

## 🖥️ Interface web

Interface **glassmorphism** bleu/violet avec dark/light mode automatique.

- Landing page avec titre accrocheur et stats clés du modèle
- Formulaire épuré — **5 champs uniquement** (les plus impactants)
- Résultat avec niveau de risque + probabilité + action recommandée
- Design responsive — fonctionne sur mobile et PC
- Accessible via : `https://churniq.up.railway.app`

---

## 📝 Guide d'utilisation

Rendez-vous sur [l'application](https://churniq.up.railway.app) et remplissez les 5 champs :

| Champ | Description | Exemple |
|---|---|---|
| **Jours depuis le dernier achat** | Nombre de jours écoulés depuis la dernière commande | `45` |
| **Score d'engagement (0–10)** | Niveau d'activité global du client sur la plateforme | `3.5` |
| **Score de satisfaction (0–10)** | Note de satisfaction générale du client | `7.0` |
| **Tickets support ouverts** | Nombre de demandes d'assistance soumises | `1` |
| **Membre du programme fidélité** | Le client est-il inscrit au programme de fidélité | `Oui` |

Après avoir cliqué sur **"Analyser le profil"**, l'application affiche :
- **Le niveau de risque** : 🔴 Très élevé / 🟠 Élevé / 🟡 Modéré / 🟢 Faible
- **La probabilité de churn** en pourcentage
- **L'action recommandée** pour retenir ce client

---

## 🐳 Déploiement Docker

```bash
# Démarrer
sudo docker-compose up --build

# Arrêter
sudo docker-compose down
```

---

## 📈 Résultats clés

| Métrique | Score |
|---|---|
| ROC-AUC | **0.9934** |
| Recall | **88.2%** |
| Precision | **91.1%** |
| Accuracy | **96.8%** |

---

## 🛠️ Stack technique

| Outil | Usage |
|---|---|
| Python 3.10 | Langage principal |
| pandas / numpy | Manipulation des données |
| matplotlib / seaborn | Visualisations |
| scikit-learn | Machine Learning |
| FastAPI + uvicorn | API REST |
| Docker / docker-compose | Conteneurisation |
| HTML / CSS / JS | Interface web |
| Railway | Hébergement cloud |

---

## ▶️ Lancer le projet

```bash
# 1. Cloner le repo
git clone https://github.com/sayanejr10/churn_prediction_ecommerce.git
cd churn_prediction_ecommerce

# 2. Démarrer avec Docker
sudo docker-compose up --build

# Application → http://localhost:8000
# API docs    → http://localhost:8000/docs
```

### Sans Docker

```bash
pip install -r requirements.txt
python3 -m uvicorn api.main:app --reload
```

---

## 👤 Auteur

**Sayane** — Junior Data Scientist
