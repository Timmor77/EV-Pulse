# Image Python légère et récente
FROM python:3.10-slim

# Installation de curl pour télécharger uv
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Installation de uv (le gestionnaire de paquets ultra-rapide)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Dossier de travail
WORKDIR /app

COPY pyproject.toml uv.lock ./

# --- CORRECTION ICI ---
# On retire '--system'. uv va créer un dossier .venv dans /app
# On active la compilation du bytecode pour que le démarrage soit plus rapide
ENV UV_COMPILE_BYTECODE=1

# On lance la synchro (création du .venv)
RUN uv sync --frozen --no-install-project

# CRUCIAL : On ajoute le .venv au PATH du système
# Ainsi, quand on tapera "python" ou "uvicorn", il utilisera celui du venv automatiquement
ENV PATH="/app/.venv/bin:$PATH"
# ----------------------

# 3. Copie du code et des modèles
COPY src/ ./src/

# Variables d'environnement pour Python
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Commande par défaut (sera surchargée par docker-compose)
CMD ["python"]