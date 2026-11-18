# PathSnap – GPS Map Matching

## Project Overview

PathSnap is a GPS map-matching project that aims to identify the most likely road-network paths corresponding to noisy GPS observations.

The project explores spatial data processing, road-network analysis, candidate generation, feature engineering, and machine-learning-based map matching.

## Phase 1 – Data Understanding and Model Preparation

Phase 1 focuses on preparing and analysing the GPS and road-network datasets.

### Objectives
- Explore GPS trajectories and road-network data.
- Analyse missing values, timestamp gaps, and GPS irregularities.
- Convert geographic coordinates into suitable projected coordinate systems.
- Generate candidate road segments for GPS observations.
- Extract spatial, temporal, and network-based features.
- Compare multiple machine-learning models.

## Dataset Structure

The dataset is organised into:
- `train/` – training trips.
- `student_test/` – student test trips.

Each trip contains GPS observations, road-network nodes, road arcs, and route information.

The original dataset is not included in this repository.

## Technology Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- GeoPandas
- Shapely
- PyProj
- NetworkX
- Matplotlib
- Folium

## Reproducibility

The project uses a fixed random seed of 42 where applicable.

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Project Status

This repository is being developed incrementally. Implementation details, experimental results, and documentation will be added as the project progresses.