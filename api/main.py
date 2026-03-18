# API CHURN PREDICTION — FastAPI 

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Chargement du modèle
model        = joblib.load('model/churn_model.pkl')
feature_cols = joblib.load('model/feature_cols.pkl')

# Initialisation
app = FastAPI(
    title="Churn Prediction API",
    description="Prédit la probabilité de churn d'un client e-commerce",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schéma simplifié — 5 champs uniquement
class CustomerData(BaseModel):
    days_since_last_purchase:  int
    engagement_score:          float
    satisfaction_score:        float
    customer_support_tickets:  int
    loyalty_member:            int

# Endpoint principal
@app.post("/predict")
def predict_churn(customer: CustomerData):

    # Valeurs moyennes pour les champs non saisis
    data = pd.DataFrame([{
        'account_age_months':          31,
        'avg_order_value':             80.5,
        'total_orders':                9,
        'days_since_last_purchase':    customer.days_since_last_purchase,
        'discount_usage_rate':         0.29,
        'return_rate':                 0.07,
        'customer_support_tickets':    customer.customer_support_tickets,
        'loyalty_member':              customer.loyalty_member,
        'browsing_frequency_per_week': 3.1,
        'cart_abandonment_rate':       0.60,
        'product_review_score_avg':    3.9,
        'engagement_score':            customer.engagement_score,
        'satisfaction_score':          customer.satisfaction_score,
        'price_sensitivity_index':     4.5,
    }])

    # Feature Engineering
    bins = [0, 15, 30, 60, 999]
    data['recency_segment'] = pd.cut(
        data['days_since_last_purchase'],
        bins=bins, labels=[0,1,2,3], include_lowest=True).astype(int)
    data['orders_per_month'] = data['total_orders'] / (data['account_age_months'] + 1)
    data['is_inactive']      = (data['days_since_last_purchase'] > 60).astype(int)
    data['risk_score']       = (
        data['days_since_last_purchase'] / 261 * 0.4 +
        (1 - data['engagement_score']    / 8.72) * 0.4 +
        (1 - data['satisfaction_score']  / 10)   * 0.2
    )
    data['low_engagement'] = (data['engagement_score'] < 3).astype(int)
    data['high_support']   = (data['customer_support_tickets'] >= 2).astype(int)

    # Prédiction
    proba = model.predict_proba(data[feature_cols])[0][1]

    # Niveau de risque
    if proba >= 0.8:   risk = "Très élevé 🔴"
    elif proba >= 0.5: risk = "Élevé 🟠"
    elif proba >= 0.2: risk = "Modéré 🟡"
    else:              risk = "Faible 🟢"

    # Action recommandée
    if proba >= 0.8:
        action = "Email personnalisé + réduction 20-30% + offrir programme fidélité"
    elif proba >= 0.5:
        action = "Offre flash limitée + recommandations produits personnalisées"
    elif proba >= 0.2:
        action = "Newsletter + points fidélité bonus sur prochain achat"
    else:
        action = "Aucune action urgente — continuer programme fidélité"

    return {
        "churn_probability": round(float(proba), 4),
        "churn_percentage":  f"{proba*100:.1f}%",
        "risk_level":        risk,
        "action":            action
    }

# Endpoints statiques
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/app")
def serve_frontend():
    return FileResponse("frontend/index.html")

@app.get("/")
def root():
    return FileResponse("frontend/index.html")