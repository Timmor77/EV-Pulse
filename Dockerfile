# Image Python légère et récente
FROM python:3.10-slim

# Installation de curl pour télécharger uv
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Installation de uv (le gestionnaire de paquets ultra-rapide)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Dossier de travail
WORKDIR /app

# 1. On copie d'abord les fichiers de dépendances (Optimisation Cache Docker)
COPY pyproject.toml uv.lock ./

# 2. Installation des dépendances système du projet (sans créer de venv interne)
RUN uv sync --frozen --system

# 3. Copie du code et des modèles
# Note : On ignore 'data' car on le montera ou on le copiera sélectivement si besoin
# Pour ce projet, on a besoin des données processées pour les normales de saison ? 
# Non, les normales sont codées en dur dans l'API maintenant.
# Mais on a besoin du modèle .pkl !
COPY src/ ./src/

# Variables d'environnement pour Python
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Commande par défaut (sera surchargée par docker-compose)
CMD ["python"]