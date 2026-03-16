import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

df = pd.read_csv('outputs/churn_dataset_engineered.csv')

feature_cols = [c for c in df.columns if c not in ['Customer_ID', 'churned']]
X = df[feature_cols]
y = df['churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

os.makedirs('model', exist_ok=True)
joblib.dump(rf, 'model/churn_model.pkl')
joblib.dump(feature_cols, 'model/feature_cols.pkl')

print("✓ Modèle sauvegardé → model/churn_model.pkl")
print("✓ Features sauvegardées → model/feature_cols.pkl")
