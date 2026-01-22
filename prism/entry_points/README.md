# PRISM Entry Points

**CLI entry points for the PRISM pipeline**

Entry points are the execution layer of PRISM. Each entry point computes measurements that persist to Parquet files.

---

## Directory Structure

```
prism/entry_points/
├── __init__.py              # Entry point registry
├── README.md                # This file
│
├── fetch.py                 # Data fetching to Parquet
├── cohort.py                # Cohort discovery
├── vector.py                # Layer 1: Vector metrics (51 per signal)
├── geometry.py              # Layer 2: Cohort geometry + modes + wavelet
├── state.py                 # Layer 3: Query-time state derivation
├── load.py                  # Data loading utilities
├── validate_schema.py       # Schema validation
│
├── ml/                      # ML Accelerator entry points
│   ├── __init__.py
│   ├── features.py          # ML feature generation
│   ├── train.py             # Model training (XGBoost, CatBoost, etc.)
│   ├── predict.py           # Run predictions
│   ├── ablation.py          # Feature ablation studies
│   ├── baseline.py          # Baseline XGBoost model
│   └── benchmark.py         # PRISM vs baseline comparison
│
├── derivations/             # Mathematical derivation entry points
│   └── ...
│
├── testing/                 # Testing utilities
│   └── ...
│
└── _archive/                # Archived/deprecated entry points
    ├── cohort_v2.py
    ├── cohort_v3.py
    ├── cohort_v4.py
    ├── geometry_apply.py
    ├── geometry_learn.py
    └── prism_vs_baseline_v2.py
```

**Key Principle:** Modules are building blocks imported by entry points, not run directly.

---

## Pipeline Execution

### Core Pipeline

```bash
# 1. Fetch data
python -m prism.entry_points.fetch cmapss

# 2. Compute vector metrics
python -m prism.entry_points.signal_vector

# 3. Compute geometry
python -m prism.entry_points.geometry

# 4. Compute state
python -m prism.entry_points.state
```

### ML Accelerator

```bash
# Generate ML features
python -m prism.entry_points.ml.features --target RUL

# Train model
python -m prism.entry_points.ml.train --model xgboost

# Run predictions
python -m prism.entry_points.ml.predict

# Feature ablation
python -m prism.entry_points.ml.ablation

# Baseline comparison
python -m prism.entry_points.ml.baseline
python -m prism.entry_points.ml.benchmark
```

---

## Entry Points Summary

### Core Pipeline

| Entry Point | Layer | Purpose |
|-------------|-------|---------|
| `fetch.py` | 0 | Data ingestion from fetchers |
| `cohort.py` | 0.5 | Cohort discovery |
| `vector.py` | 1 | 51 behavioral metrics per signal |
| `geometry.py` | 2 | Cohort geometry + modes + wavelet |
| `state.py` | 3 | Query-time state derivation |

### ML Accelerator

| Entry Point | Purpose |
|-------------|---------|
| `ml/features.py` | Generate ML-ready feature tables |
| `ml/train.py` | Train models (XGBoost, CatBoost, LightGBM) |
| `ml/predict.py` | Run predictions on new data |
| `ml/ablation.py` | Feature ablation studies |
| `ml/baseline.py` | Baseline model without PRISM features |
| `ml/benchmark.py` | Compare PRISM vs baseline performance |

---

## Common Options

| Option | Description |
|--------|-------------|
| `--testing` | Enable testing mode (allows limiting flags) |
| `--force` | Recompute all (clear progress) |
| `--limit N` | [TESTING] Max observations per signal |
| `--signal x,y` | [TESTING] Only process specific signals |
| `--adaptive` | Auto-detect window size from data |

---

## Storage

All storage uses Parquet files in `data/`:

```
data/
├── observations.parquet     # Raw sensor data
├── vector.parquet           # Behavioral metrics
├── geometry.parquet         # Structural snapshots
├── state.parquet            # Temporal dynamics
├── cohorts.parquet          # Entity groupings
├── ml_features.parquet      # ML-ready features
├── ml_results.parquet       # Model predictions
└── ml_model.pkl             # Trained model
```
