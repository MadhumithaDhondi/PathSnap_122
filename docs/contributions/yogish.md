# Yogish — Exploratory Data Analysis & Route Reconstruction (Report Sections 2.1, 2.2, and 4)

## What this part covers

- `results/eda/` — Dataset inventory, partition summaries, distribution plots:
  - `dataset_inventory.csv` (Detailed inventory of all 90 trips)
  - `partition_summary.csv` (Aggregated statistics for train and student_test partitions)
  - `dataset_exploration_report.md` (Exploratory data analysis report)
  - `gps_observation_distribution.png` (Distribution of GPS points per trip)
  - `speed_distribution.png` (Estimated kinematic speed profile)
  - `route_length_distribution.png` (Reference route length distribution)
  - `road_network_complexity.png` (Road graph node and arc counts)
  - `trajectory_length_vs_duration.png` (Spatial trajectory length vs time duration)
- `results/route_reconstruction/` — Sequential HMM + Log-space Viterbi decoding results:
  - `viterbi_comparison_results.csv` (Per-trip continuity and accuracy gains across all 10 student-test trips)
  - `route_continuity_improvement.png` (Route continuity jump from 87.5% to 99.0%)
  - `top1_accuracy_comparison.png` (Observation Top-1 accuracy improvement from 80.1% to 83.7%)
- `scripts/01_eda.py` — Exploratory Data Analysis script

## Reproduce this part

```bash
python scripts/01_eda.py
```

## Be ready to explain in the viva

- Dataset partition: Exactly 80 training trips (104,209 GPS observations, 415,283 graph arcs) and 10 student-testing trips (9,580 GPS observations, 44,611 arcs).
- Zero missing values across all .track, .nodes, .arcs, and .route files.
- Coordinate validity in WGS84 bounding box [46.85, 47.05] N and [7.35, 7.55] E (Bern, Switzerland).
- Kinematic preprocessing: Haversine geodesics, forward azimuth bearing, and temporal gap detection (>10s dropouts).
- Candidate road generation using spatial k-d tree with orthogonal projection and candidate miss rate (3.6% on test, 96.4% recall).
- The topological continuity gap: Why point-wise classifiers achieve only ~25% raw route continuity.
- Log-space HMM & Viterbi decoding formulation: How transition probabilities on the road adjacency graph boost continuity to 99.00% and Top-1 accuracy to 83.68%.
