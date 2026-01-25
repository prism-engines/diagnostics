# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## CRITICAL: PRISM ↔ ORTHON Architecture

**PRISM is an HTTP service ONLY. NOT a pip install. NO code sharing with ORTHON.**

```
┌─────────────────┐         HTTP          ┌─────────────────┐
│     ORTHON      │ ──────────────────▶   │      PRISM      │
│   (Frontend)    │   POST /compute       │  (Compute API)  │
│   Streamlit     │ ◀──────────────────   │  localhost:8100 │
│                 │   {status, parquets}  │                 │
└─────────────────┘                       └─────────────────┘
        │                                         │
        │ reads                                   │ writes
        ▼                                         ▼
   ~/prism-mac/data/*.parquet              ~/prism-mac/data/*.parquet
```

### Start PRISM API (MUST use venv)
```bash
cd ~/prism_engines-prism
./venv/bin/python -m prism.entry_points.api --port 8100

# Or background:
nohup ./venv/bin/python -m prism.entry_points.api --port 8100 > /tmp/prism-api.log 2>&1 &
```

### Verify PRISM is Running
```bash
curl http://localhost:8100/health
# Should return: {"status":"ok","version":"0.1.0",...}
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Status check |
| `/compute` | POST | Run computation (synchronous) |
| `/files` | GET | List available parquets |
| `/read?path=...` | GET | Read parquet as JSON |
| `/disciplines` | GET | List available disciplines |

### /compute Request Schema
```json
{
  "config": {
    "discipline": "mechanics",
    "window": {"size": 30, "stride": 15},
    "min_samples": 20,
    "engines": {
      "vector": {"enabled": ["hurst_dfa", "sample_entropy", "stationarity", "trend"]},
      "geometry": {"enabled": ["bg_correlation", "bg_distance", "bg_clustering"]}
    }
  },
  "observations_path": "/path/to/observations.parquet"
}
```

### /compute Response Schema
```json
{
  "status": "complete",
  "results_path": "/Users/.../prism-mac/data",
  "parquets": ["vector.parquet", "physics.parquet"],
  "duration_seconds": 6.92,
  "message": null,
  "hint": null,
  "engine": null
}
```

### Observations Format (PRISM expects)
| Column | Type | Description |
|--------|------|-------------|
| entity_id | str | Entity identifier (e.g., "engine_1") |
| signal_id | str | Signal name (e.g., "temperature") |
| index | float | Sequence index (time, cycle, depth) |
| value | float | Measurement value |

### Output Parquets
- `vector.parquet` - Signal-level metrics (hurst, entropy, etc.)
- `geometry.parquet` - Pairwise relationships
- `dynamics.parquet` - Temporal dynamics
- `physics.parquet` - Energy/momentum metrics

### Session Recovery Checklist
If session dies, run these to restore state:
```bash
# 1. Check if PRISM is running
curl http://localhost:8100/health

# 2. If not running, start it (MUST use venv!)
cd ~/prism_engines-prism
nohup ./venv/bin/python -m prism.entry_points.api --port 8100 > /tmp/prism-api.log 2>&1 &

# 3. Verify
curl http://localhost:8100/health

# 4. Check existing data
ls -la ~/prism-mac/data/*.parquet
```

### Common Issues
| Issue | Cause | Fix |
|-------|-------|-----|
| "No module named 'polars'" | Wrong Python | Use `./venv/bin/python` not system python |
| ORTHON shows "fallback" | Cached backend | Restart ORTHON Streamlit |
| 0.5s compute time | Pipeline not running | Check `/tmp/prism-api.log` for errors |

---

## Project Overview

PRISM Diagnostics is a behavioral geometry engine for industrial signal topology analysis. It computes intrinsic properties, relational structure, and temporal dynamics of sensor data from turbofans, bearings, hydraulic systems, and chemical processes.

**Repository:** `prism-engines/diagnostics`

**Architecture: Pure Polars + Parquet**
- All storage via Parquet files (no database)
- All I/O via Polars DataFrames
- Pandas only at engine boundaries (scipy/sklearn compatibility)
- Data stays local (gitignored), only code goes to GitHub

**Core Philosophy: Do It Right, Not Quick**
- Correctness over speed - a wrong answer fast is still wrong
- Complete data, not samples - academic-grade analysis requires full datasets
- Verify before proceeding - check results match expectations
- Run the full pipeline - Vector → Geometry → State → ML

**Design Principles:**
- Record observations faithfully
- Persist all measurements to Parquet
- Explicit time (nothing inferred between steps)
- No implicit execution (importing does nothing)

**Academic Research Standards:**
- **NO SHORTCUTS** - All engines use complete data (no subsampling)
- **NO APPROXIMATIONS** - Peer-reviewed algorithms (antropy, pyrqa)
- **NO SPEED HACKS** - 2-3 hour runs acceptable, 2-3 week runs expected
- **VERIFIED QUALITY** - All engines audited for data integrity
- **Publication-grade** - Suitable for peer-reviewed research

## Directory Structure

```
prism-engines/diagnostics/
├── prism/                      # Core package
│   ├── core/                   # Types and utilities
│   │   ├── domain_clock.py     # DomainClock, DomainInfo, auto_detect_window
│   │   └── signals/            # Signal types (DenseSignal, SparseSignal, LaplaceField)
│   │
│   ├── db/                     # Parquet I/O layer
│   │   └── parquet_store.py    # 5 core files + ML files
│   │
│   ├── engines/                # 33 computation engines
│   │   ├── vector/             # Intrinsic metrics (hurst, entropy, garch, etc.)
│   │   ├── geometry/           # Structural (pca, mst, clustering, coupling, modes)
│   │   ├── state/              # Temporal dynamics (granger, dtw, trajectory, etc.)
│   │   ├── laplace/            # Laplace transform and pairwise
│   │   ├── spectral/           # Wavelet microscope
│   │   ├── pointwise/          # Derivatives, hilbert, statistical
│   │   └── observation/        # Break detector, heaviside, dirac
│   │
│   ├── entry_points/           # CLI entrypoints (python -m prism.entry_points.*)
│   │   ├── fetch.py            # Data fetching
│   │   ├── cohort.py           # Cohort discovery
│   │   ├── signal_vector.py    # Vector computation
│   │   ├── geometry.py         # Geometry computation
│   │   ├── state.py            # State computation
│   │   ├── ml_features.py      # ML feature generation
│   │   └── ml_train.py         # ML model training
│   │
│   └── utils/                  # Utilities (including monitor.py)
│
├── fetchers/                   # Data fetchers
│   ├── cmapss_fetcher.py       # NASA C-MAPSS turbofan
│   ├── femto_fetcher.py        # FEMTO bearing degradation
│   ├── hydraulic_fetcher.py    # UCI hydraulic system
│   ├── cwru_bearing_fetcher.py # CWRU bearing faults
│   ├── tep_fetcher.py          # Tennessee Eastman Process
│   └── yaml/                   # Fetch configurations
│
├── config/                     # YAML configurations
│   ├── engine.yaml             # Engine settings
│   ├── window.yaml             # Window/stride settings
│   ├── stride.yaml             # Legacy stride config
│   └── domain.yaml             # Active domain metadata
│
└── data/                       # LOCAL ONLY (gitignored)
    ├── observations.parquet    # Raw sensor data
    ├── vector.parquet          # Behavioral metrics
    ├── geometry.parquet        # Structural snapshots
    ├── state.parquet           # Temporal dynamics
    ├── cohorts.parquet         # Entity groupings
    ├── ml_features.parquet     # ML-ready features
    ├── ml_results.parquet      # Model predictions
    └── ml_model.pkl            # Trained model
```

## Essential Commands

### Full Pipeline
```bash
# 1. Fetch data (interactive picker or specify source)
python -m prism.entry_points.fetch
python -m prism.entry_points.fetch cmapss

# 2. Compute vector metrics
python -m prism.entry_points.signal_vector

# 3. Compute geometry
python -m prism.entry_points.geometry

# 4. Compute state
python -m prism.entry_points.state

# 5. Generate ML features
python -m prism.entry_points.ml_features --target RUL

# 6. Train ML model
python -m prism.entry_points.ml_train --model xgboost
```

### Testing Mode
All entry points support `--testing` flag for quick iteration:
```bash
python -m prism.entry_points.signal_vector --testing --limit 100
python -m prism.entry_points.geometry --testing
python -m prism.entry_points.state --testing
```

### Common Flags
| Flag | Description |
|------|-------------|
| `--adaptive` | Auto-detect window size from data |
| `--force` | Clear progress and recompute all |
| `--testing` | Enable testing mode (required for --limit, --signal) |
| `--limit N` | [TESTING] Max observations per signal |
| `--signal x,y` | [TESTING] Only process specific signals |

## Pipeline Architecture

```
Layer 0: OBSERVATIONS
         Raw sensor data
         Output: data/observations.parquet

Layer 1: VECTOR
         Raw observations → 51 behavioral metrics per signal
         Output: data/vector.parquet

Layer 2: GEOMETRY
         Vector signals → Laplace fields → structural geometry
         Output: data/geometry.parquet

Layer 3: STATE
         Geometry evolution → temporal dynamics
         Output: data/state.parquet

Layer 4: ML ACCELERATOR
         All layers → denormalized features → trained model
         Output: data/ml_features.parquet, data/ml_model.pkl
```

## ML Accelerator

The ML Accelerator provides end-to-end ML workflow on PRISM features:

### Generate Features
```bash
# Basic feature generation
python -m prism.entry_points.ml_features

# With target variable for supervised learning
python -m prism.entry_points.ml_features --target RUL
python -m prism.entry_points.ml_features --target fault_type
```

### Train Models
```bash
# Train with XGBoost (default)
python -m prism.entry_points.ml_train

# Choose framework
python -m prism.entry_points.ml_train --model catboost
python -m prism.entry_points.ml_train --model lightgbm
python -m prism.entry_points.ml_train --model randomforest

# Hyperparameter tuning
python -m prism.entry_points.ml_train --tune

# Cross-validation
python -m prism.entry_points.ml_train --cv 5
```

### Supported Models
- **xgboost**: XGBoost (default, fast, robust)
- **catboost**: CatBoost (handles categoricals well)
- **lightgbm**: LightGBM (fastest for large data)
- **randomforest**: Scikit-learn Random Forest
- **gradientboosting**: Scikit-learn Gradient Boosting

### Outputs
- `ml_features.parquet`: Denormalized feature table (one row per entity)
- `ml_results.parquet`: Predictions vs actuals for test set
- `ml_importance.parquet`: Feature importance rankings
- `ml_model.pkl`: Serialized trained model

## Engine Categories

**Vector Engines (9)** - Intrinsic properties of single series
- Hurst, Entropy, GARCH, Wavelet, Spectral, Lyapunov, RQA, Realized Vol, Hilbert

**Geometry Engines (9)** - Structural relationships
- PCA, MST, Clustering, LOF, Distance, Convex Hull, Copula, Mutual Information, Barycenter

**State Engines (7)** - Temporal dynamics
- Granger, Cross-Correlation, Cointegration, DTW, DMD, Transfer Entropy, Coupled Inertia

**Temporal Dynamics (5)** - Geometry evolution
- Energy Dynamics, Tension Dynamics, Phase Detector, Cohort Aggregator, Transfer Detector

**Observation Engines (3)** - Discontinuity detection
- Break Detector, Heaviside, Dirac

## Key Patterns

### Reading Data
```python
import polars as pl
from prism.db.parquet_store import get_path, OBSERVATIONS, VECTOR

observations = pl.read_parquet(get_path(OBSERVATIONS))
filtered = observations.filter(pl.col('signal_id') == 'sensor_1')

vector = pl.read_parquet(get_path(VECTOR))
```

### Writing Data
```python
from prism.db.polars_io import upsert_parquet, write_parquet_atomic
from prism.db.parquet_store import get_path, VECTOR

# Upsert (preserves existing rows, updates by key)
upsert_parquet(df, get_path(VECTOR), key_cols=['signal_id', 'timestamp'])

# Atomic write (replaces entire file)
write_parquet_atomic(df, get_path(VECTOR))
```

## Validated Domains

| Domain | Source | Use Case |
|--------|--------|----------|
| **C-MAPSS** | NASA | Turbofan engine degradation (FD001-FD004) |
| **FEMTO** | PHM Society | Bearing degradation (PRONOSTIA) |
| **Hydraulic** | UCI | Hydraulic system condition monitoring |
| **CWRU** | Case Western | Bearing fault classification |
| **TEP** | Tennessee Eastman | Chemical process fault detection |
| **MetroPT** | Metro do Porto | Train compressor failures |

## Technical Stack

- **Language:** Python 3.10+
- **Storage:** Parquet files (columnar, compressed)
- **DataFrame:** Polars (primary), Pandas (engine compatibility)
- **Core:** NumPy, SciPy, scikit-learn
- **ML:** XGBoost, CatBoost, LightGBM (optional)
- **Specialized:** antropy, nolds, pyrqa, arch, PyWavelets, networkx
- **API:** FastAPI, uvicorn, httpx (for ORTHON communication)

---

## Current Session State (Jan 25, 2026)

### What's Working
- PRISM API running on port 8100 with venv Python
- `/compute` endpoint returns real results (tested: 6.92s, 144 rows × 33 cols)
- vector.parquet and physics.parquet being created

### What's In Progress
- CC ORTHON wiring up "Analyze" button to call PRISM `/compute`
- ORTHON needs to send POST to `http://localhost:8100/compute`

### Key Files
- `prism/entry_points/api.py` - HTTP API (FastAPI)
- `prism/entry_points/compute.py` - Pipeline runner
- `prism/entry_points/vector.py` - Vector engine runner
- `~/prism-mac/data/config.yaml` - Current config
- `~/prism-mac/data/observations.parquet` - Input data
- `/tmp/prism-api.log` - API logs

### DO NOT TOUCH
- ORTHON code lives in `~/prism_engines-orthon/` - let CC ORTHON handle it
- Never `pip install prism` - PRISM is HTTP only
