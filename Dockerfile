# ── Image de base Python ──────────────────────────────────────
FROM python:3.10-slim

# ── Dossier de travail dans le container ─────────────────────
WORKDIR /app

# ── Copie des dépendances et installation ────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copie du code et du modèle ────────────────────────────────
COPY api/ ./api/
COPY model/ ./model/

# ── Port exposé ───────────────────────────────────────────────
EXPOSE 8000

# ── Commande de démarrage ─────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]