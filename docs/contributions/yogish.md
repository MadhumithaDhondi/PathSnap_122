# Yogish — Frontend, Backend, EDA & Route Reconstruction (Report Sections 2.1, 2.2, 5, and 7)

## What this part covers

- `backend/` — FastAPI asynchronous REST microservice (`main.py`)
  - Endpoints: `GET /health`, `GET /trips`, `POST /predict`
  - Integrated spatial k-d tree candidate lookup, XGBoost inference, and log-space Viterbi decoding
  - Zero reference route access (pure evaluation)
- `frontend/` — Streamlit interactive evaluator UI dashboard (`app.py`)
  - Dynamic trip selection and HTTP prediction requests
  - Real-time KPI cards: Top-1 Accuracy, Route Continuity, Reference Overlap, Inference Latency
  - Embedded multi-layer Folium interactive maps
- `results/eda/` — Exploratory Data Analysis & visual generation:
  - `dataset_inventory.csv` (Per-trip inventory for all 90 trips)
  - `partition_summary.csv` (Train vs student_test aggregates)
  - `dataset_exploration_report.md` (Exploration insights and missing-value report)
  - `gps_observation_distribution.png`, `speed_distribution.png`, `route_length_distribution.png`, `road_network_complexity.png`, `trajectory_length_vs_duration.png`
- `route_reconstruction/` — Sequential HMM + Log-space Viterbi engine (`hmm_viterbi.py`)
  - Blended Gaussian spatial distance & classifier likelihood emissions
  - Directed graph adjacency & 2-hop topological transition matrix
  - Dynamic programming recurrence with backpointer tracking
- `models/` — Saved model binaries (`final_model.joblib`, `xgboost.joblib`, etc.)
- `artifacts/` — Fitted `RobustScaler` pipeline and preprocessing configurations
- `start_all_single_laptop.bat`, `start_wifi_host.ps1` — One-click launchers for single-laptop and distributed Wi-Fi evaluation

## Reproduce & Run

```bash
# Launch both backend and frontend on Wi-Fi network (0.0.0.0):
.\start_wifi_host.ps1

# Or run tests:
python -m unittest tests/test_pipeline.py -v
```

## Be ready to explain in the viva

- Dual-node architecture: Backend running on Laptop 1 (`0.0.0.0:8000`), Streamlit UI running on Laptop 2 (`0.0.0.0:8501`).
- Network decoupling: Streamlit UI makes JSON REST requests over Wi-Fi (`POST /predict`); backend executes feature scaling, spatial candidate projection, and Viterbi dynamic programming.
- Preprocessing and candidate generation: Spatial k-d tree candidate indexing on arc midpoints with orthogonal projection (3.6% candidate miss rate on test).
- Why point-wise classifiers fail continuity (~25%) and how the log-space HMM transition matrix resolves this, driving continuity to 99.00% and Top-1 accuracy to 83.68%.
