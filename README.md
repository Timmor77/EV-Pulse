# EV-Pulse

Projet de prédiction et d'analyse des données de recharge de véhicules électriques.

## Structure du Projet

```
EV-Pulse/
├── .github/workflows/   # CI/CD
├── data/
│   ├── raw/             # Données brutes (JSON/CSV ACN & Météo)
│   └── processed/       # Données traitées (Parquet)
├── src/
│   ├── data/            # Scripts d'ingestion et processing
│   ├── features/        # Feature engineering (Lags, Dates)
│   ├── models/          # Entraînement et inférence (LightGBM)
│   └── api/             # FastAPI application
├── notebooks/           # Exploration et EDA
├── tests/               # Tests unitaires (Pytest)
├── Dockerfile           # Containerisation
├── requirements.txt     # Dépendances Python
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Traitement des données
```bash
python src/data/process.py
```

### Entraînement du modèle
```bash
python src/models/train.py
```

### Lancement de l'API
```bash
uvicorn src.api.main:app --reload
```

## Développement

### Tests
```bash
pytest tests/
```

### Docker
```bash
docker build -t ev-pulse .
docker run -p 8000:8000 ev-pulse
```

## License

MIT
