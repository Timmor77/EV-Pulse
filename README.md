# ⚡ EV-Pulse: EV Charging Load Simulator

![CI](https://github.com/Timmor77/EV-Pulse/actions/workflows/ci.yml/badge.svg) ![Python](https://img.shields.io/badge/Python-3.10-blue) ![LightGBM](https://img.shields.io/badge/Model-LightGBM-green) ![FastAPI](https://img.shields.io/badge/Backend-FastAPI-teal) ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red) ![Docker](https://img.shields.io/badge/Container-Docker-blue)

**EV-Pulse** is a small end-to-end project for exploring day-ahead EV charging load. It predicts a daily profile from calendar and weather context, without using recent consumption as an input.

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

1.  **Data Pipeline:** Cleaning and processing of complex JSON time-series (ACN-Data Caltech/JPL).
2.  **Core Model:** **LightGBM Regressor** (Context-Aware).
    * *Strategy:* strictly **no lag features** (past consumption) are used.
    * *Benefit:* The model is robust to sensor failure and can simulate any future date purely based on context (Time + Weather).
3.  **API (Backend):** **FastAPI** service serving predictions. Includes a **Climatology Fallback** system (uses seasonal averages if no weather data is provided).
4.  **Dashboard (Frontend):** **Streamlit** interface for interactive simulation and visualization.
5.  **Packaging:** **Docker Compose** starts the API and dashboard locally. Package management uses `uv`.

---

## 📊 Model Performance

The evaluation keeps the last 60 complete calendar days untouched (1 January to 29 February 2020). Model selection happens before this period: each `TimeSeriesSplit` fold uses the tail of its own training block for early stopping, then refits on the complete fold training block. The comparison baseline is simply the historical mean for the same weekday and 15-minute time slot. The 93 rows from the incomplete 1 March source day are not used for evaluation.

| Final holdout | LightGBM | Calendar baseline |
|---|---:|---:|
| MAE | **11.44 kW** | 13.57 kW |
| RMSE | **17.61 kW** | 24.40 kW |
| R² | **0.907** | 0.822 |

The model improves holdout MAE by **15.7%** over the baseline. Across the five development folds, its mean MAE is 13.32 kW versus 13.66 kW for the baseline; the baseline still wins on one fold, so the gain is modest and not uniform over time.

| Holdout period | Rows | Model MAE | Baseline MAE |
|---|---:|---:|---:|
| January 2020 | 2,976 | 11.74 kW | 14.81 kW |
| February 2020 | 2,784 | 11.11 kW | 12.24 kW |

The complete machine-readable result, including split dates, fold metrics, feature list, dataset SHA-256 and per-period metrics, is stored in [`reports/model_evaluation.json`](reports/model_evaluation.json).

To reproduce it after rebuilding the processed dataset:

```bash
uv run python -m src.models.train_model_v2
```

This command writes the evaluation report and refits the served model on all available rows using the tree count selected before the holdout.

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
