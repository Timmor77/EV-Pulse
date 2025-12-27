# ⚡ EV-Pulse: AI-Powered Smart Charging Simulator

![Python](https://img.shields.io/badge/Python-3.10-blue) ![LightGBM](https://img.shields.io/badge/Model-LightGBM-green) ![FastAPI](https://img.shields.io/badge/Backend-FastAPI-teal) ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red) ![Docker](https://img.shields.io/badge/Deploy-Docker-blue)

**EV-Pulse** is an end-to-end Machine Learning solution designed to optimize Electric Vehicle (EV) charging infrastructures. It acts as a **Digital Twin**, simulating future power consumption based on calendar context and weather conditions, without relying on real-time sensor data history.

## 🎯 Business Value & Problem Solved
Corporate campuses face two major challenges with EV adoption:
1.  **Grid Instability:** Unpredictable peaks in demand can trip circuit breakers.
2.  **Cost Overruns:** Exceeding power capacity (subscribed power) triggers massive financial penalties.

**EV-Pulse empowers Facility Managers to:**
* 🔮 **Forecast** load profiles 24 hours in advance (Day-Ahead).
* ⚠️ **Anticipate** capacity overloads (Peak Shaving alerts).
* 🧪 **Simulate** "What-If" scenarios (e.g., "What if EV adoption doubles next year?").

---

## 🏗️ Technical Architecture

The project follows a modern MLOps microservices architecture:

1.  **Data Pipeline:** Cleaning and processing of complex JSON time-series (ACN-Data Caltech/JPL).
2.  **Core Model:** **LightGBM Regressor** (Context-Aware).
    * *Strategy:* strictly **no lag features** (past consumption) are used.
    * *Benefit:* The model is robust to sensor failure and can simulate any future date purely based on context (Time + Weather).
3.  **API (Backend):** **FastAPI** service serving predictions. Includes a **Climatology Fallback** system (uses seasonal averages if no weather data is provided).
4.  **Dashboard (Frontend):** **Streamlit** interface for interactive simulation and visualization.
5.  **Infrastructure:** Fully containerized with **Docker** & **Docker Compose**. Package management via `uv`.

---

## 📊 Model Performance

* **Algorithm:** LightGBM (Gradient Boosting)
* **Key Features:** `is_business_time`, `solar_radiation` (simulated), `is_holiday` (dynamic US calendar), `hour_sin/cos`.
* **Metrics (Test Set):**
    * **R² Score:** ~83%
    * **MAE (Mean Absolute Error):** ~14 kW (on a 300 kW max load)
* **Capabilities:** Accurately captures weekly cycles, holidays (Christmas, Thanksgiving), and seasonal weather impacts (AC/Heating).

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

---

## 📂 Project Structure

```text
EV-Pulse/
├── .github/workflows/   # CI/CD Pipeline (Linting & Quality Checks)
├── data/                # Raw and Processed data (ignored by git)
├── src/
│   ├── api/             # FastAPI Backend
│   │   ├── main.py
│   │   └── schemas.py
│   ├── dashboard/       # Streamlit Frontend
│   │   └── app.py
│   ├── features/        # Feature Engineering Logic (Shared)
│   └── models/          # Model training scripts & saved .pkl
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

* **DevOps**: Docker, GitHub Actions (CI)

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