# ============================================================
# API CHURN PREDICTION — FastAPI
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# ── Chargement du modèle ─────────────────────────────────────
model        = joblib.load('model/churn_model.pkl')
feature_cols = joblib.load('model/feature_cols.pkl')

# ── Initialisation de l'API ───────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description="Prédit la probabilité de churn d'un client e-commerce",
    version="1.0.0"
)

# ── CORS ─────────────────────────────────────────────────────
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schéma des données d'entrée ───────────────────────────────
class CustomerData(BaseModel):
    account_age_months: int
    avg_order_value: float
    total_orders: int
    days_since_last_purchase: int
    discount_usage_rate: float
    return_rate: float
    customer_support_tickets: int
    loyalty_member: int
    browsing_frequency_per_week: float
    cart_abandonment_rate: float
    product_review_score_avg: float
    engagement_score: float
    satisfaction_score: float
    price_sensitivity_index: float

# ── Endpoint principal ────────────────────────────────────────
@app.post("/predict")
def predict_churn(customer: CustomerData):

    # Conversion en DataFrame
    data = pd.DataFrame([customer.dict()])

    # Feature Engineering (mêmes transformations qu'à l'entraînement)
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

# ── Endpoint de santé ─────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Churn Prediction API is running ✓"}

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/app")
def serve_frontend():
    return FileResponse("frontend/index.html")