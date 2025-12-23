# EV-Pulse

Electric vehicle charging prediction and analysis platform using ACN (Adaptive Charging Network) data.

## Project Structure

```
EV-Pulse/
├── .github/
│   └── workflows/
│       └── ci.yml           # CI/CD pipeline
├── data/
│   ├── raw/                 # Raw data (ACN JSON files)
│   └── processed/           # Processed data (Parquet)
├── src/
│   ├── api/                 # FastAPI application
│   ├── data/                # Data extraction and processing
│   ├── features/            # Feature engineering
│   └── models/              # ML model training and inference
├── notebooks/               # Jupyter notebooks for EDA
├── tests/                   # Unit tests (pytest)
├── Dockerfile               # Container configuration
├── pyproject.toml           # Project dependencies and config
└── README.md
```

## Installation

```bash
# Using uv (recommended)
uv sync

# With optional dependencies
uv sync --extra ml --extra dev

# Using pip
pip install -e .
```

## Usage

### Data Extraction
```bash
python -m src.data.extract_acn
```

### Data Processing
```bash
python -m src.data.process_data
```

### Start API
```bash
uvicorn src.api.main:app --reload
```

## Development

### Run Tests
```bash
pytest
```

### Linting
```bash
ruff check src tests
```

### Docker
```bash
docker build -t ev-pulse .
docker run -p 8000:8000 ev-pulse
```

## Data Source

This project uses data from the ACN-Data platform: https://ev.caltech.edu/

## Citation

```bibtex
@inproceedings{lee_acndata_2019,
  author = {Lee, Zachary J. and Li, Tongxin and Low, Steven H.},
  title = {{ACN}-{Data}: {Analysis} and {Applications} of an {Open} {EV} {Charging} {Dataset}},
  booktitle = {Proceedings of the Tenth International Conference on Future Energy Systems},
  series = {e-Energy '19},
  month = jun,
  year = {2019},
  location = {Phoenix, Arizona}
}
```

## License

MIT