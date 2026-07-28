# ⚡ EV-Pulse: EV Charging Load Simulator

![CI](https://github.com/Timmor77/EV-Pulse/actions/workflows/ci.yml/badge.svg) ![Python](https://img.shields.io/badge/Python-3.10-blue) ![LightGBM](https://img.shields.io/badge/Model-LightGBM-green) ![FastAPI](https://img.shields.io/badge/Backend-FastAPI-teal) ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red) ![Docker](https://img.shields.io/badge/Container-Docker-blue)

**EV-Pulse** is a small end-to-end project for exploring day-ahead EV charging load. It keeps Caltech, JPL and Office 001 separate and predicts each daily profile from calendar, weather and a recent historical reference profile.

## 🎯 Business Value & Problem Solved
Charging sites face two practical questions:
1.  **Grid Instability:** Unpredictable peaks in demand can trip circuit breakers.
2.  **Capacity Planning:** Expected peaks help size and operate the electrical connection.

The demo lets a user:
* 🔮 **Forecast** load profiles 24 hours in advance (Day-Ahead).
* ⚠️ **Anticipate** capacity overloads (Peak Shaving alerts).
* 🧪 **Simulate** different temperature and sunlight assumptions.

---

## 🏗️ Technical Architecture

The project deliberately keeps a straightforward architecture:

1.  **Data Pipeline:** Cleaning and processing ACN-Data sessions into one 15-minute load curve per site.
2.  **Core Model:** a simple recent profile plus a **LightGBM residual correction**.
    * The reference is the mean of the last eight training weeks for the same weekday and time slot.
    * LightGBM predicts the remaining difference from calendar and weather context.
    * If cross-validation does not beat the calendar baseline for a site, the API keeps the baseline for that site.
3.  **API (Backend):** **FastAPI** service serving predictions. Includes a **Climatology Fallback** system (uses seasonal averages if no weather data is provided).
4.  **Dashboard (Frontend):** **Streamlit** interface for interactive simulation and visualization.
5.  **Packaging:** **Docker Compose** starts the API and dashboard locally. Package management uses `uv`.

---

## 📊 Model Performance

The evaluation keeps the last 60 complete calendar days untouched (1 January to 29 February 2020). Each site is evaluated separately. Model selection uses three earlier `TimeSeriesSplit` folds; the final holdout is not used to choose between the residual model and the baseline.

| Final holdout | Selected method | MAE | Calendar baseline | Gain |
|---|---|---:|---:|---:|
| Caltech | Recent profile + residual model | **4.78 kW** | 6.81 kW | 29.8% |
| JPL | Recent profile + residual model | **9.61 kW** | 10.35 kW | 7.2% |
| Office 001 | Calendar baseline | **1.67 kW** | 1.67 kW | 0.0% |
| All site rows | Development-selected method | **5.35 kW** | 6.28 kW | 14.7% |

The residual model is retained for Caltech and JPL because it wins on the development folds. Office 001 only contains 1,683 original sessions and its baseline remains slightly better, so the API deliberately keeps that simpler method.

The complete machine-readable result, including every fold, monthly metrics, selected method and source hashes, is stored in [`reports/site_model_evaluation.json`](reports/site_model_evaluation.json). The earlier aggregate context-only benchmark remains available in [`reports/model_evaluation.json`](reports/model_evaluation.json).

To reproduce the site models from the cleaned sessions:

```bash
uv run python -m src.features.make_time_series
uv run python -m src.features.build_features_v3
uv run python -m src.models.train_site_models
```

The final command writes the evaluation report and refits the bundle served by the API.

---

## 🚀 Quick Start

### Option 1: Run with Docker (Recommended)
The easiest way to run the full stack (API + Dashboard).

1.  **Build and Start:**
    ```bash
    docker-compose up --build
    ```
2.  **Access the App:**
    * **Dashboard:** Open [http://localhost:8501](http://localhost:8501)
    * **API Docs:** Open [http://localhost:8000/docs](http://localhost:8000/docs)

3.  **Stop:**
    Press `Ctrl+C` then run `docker-compose down`.

### Option 2: Run Locally (with uv)
If you want to develop without Docker.

1.  **Install dependencies:**
    ```bash
    pip install uv
    uv sync
    ```

2.  **Start the API (Terminal 1):**
    ```bash
    uv run uvicorn src.api.main:app --reload
    ```

3.  **Start the Dashboard (Terminal 2):**
    ```bash
    uv run streamlit run src/dashboard/app.py
    ```

### Running the Tests
```bash
uv sync --extra dev
uv run pytest
```

---

## 📂 Project Structure

```text
EV-Pulse/
├── .github/workflows/   # CI: lint, format and tests
├── data/                # Raw and Processed data (ignored by git)
├── src/
│   ├── api/             # FastAPI Backend
│   │   ├── main.py
│   │   └── schemas.py
│   ├── dashboard/       # Streamlit Frontend
│   │   └── app.py
│   ├── features/        # Feature Engineering Logic (Shared)
│   └── models/          # Model training scripts & saved .pkl
├── reports/             # Reproducible model evaluation result
├── tests/               # Unit & API tests (pytest)
├── Dockerfile           # Multi-stage Docker build
├── docker-compose.yml   # Orchestrator
├── pyproject.toml       # Dependencies (uv)
└── README.md            # You are here
```

## 🛠️ Tools & Technologies
* **Language**: Python 3.10

* **Package Manager**: uv (Astral)

* **ML Framework**: LightGBM, Scikit-Learn, Pandas

* **Web Stack**: FastAPI, Pydantic, Uvicorn

* **Visualization**: Streamlit, Plotly

* **Tooling**: Docker, GitHub Actions (CI; no automated deployment)

## Citations
```text
@inproceedings{lee_acndata_2019,
  author = {Lee, Zachary J. and Li, Tongxin, and Low, Steven H.},
  title = { {ACN}-{Data}: {Analysis} and {Applications} of an {Open} {EV} {Charging} {Dataset} },
  booktitle = {Proceedings of the Tenth International Conference on Future Energy Systems},
  series = {e-Energy '19},
  month = jun,
  year = {2019},
  location = {Phoenix, Arizona}
}
```
