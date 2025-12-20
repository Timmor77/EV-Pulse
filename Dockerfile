# Utiliser une image Python officielle
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY src/ ./src/
COPY data/ ./data/

# Exposer le port pour l'API
EXPOSE 8000

# Commande par défaut
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
