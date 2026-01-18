# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PRISM Diagnostics is a behavioral geometry engine for industrial signal topology analysis. It measures intrinsic properties, relational structure, and temporal dynamics of sensor data from turbofans, bearings, hydraulic systems, and chemical plants. **The math interprets; we don't add narrative.**

**Repository:** `prism-engines/diagnostics`

**Architecture: Pure Polars + Parquet**
- All storage via Parquet files (no database)
- All I/O via Polars DataFrames
- Pandas only at engine boundaries (scipy/sklearn compatibility)
- Data stays local (gitignored), only code goes to GitHub

**Core Philosophy:**
- Record reality faithfully
- Let math speak
- The geometry interprets - we don't add opinion
- Parquet is truth (all measurements persist to Parquet files)
- Explicit time (nothing inferred between steps)
- No implicit execution (importing does nothing)

**Academic Research Standards:**
- **NO SHORTCUTS** - All engines use complete data (no subsampling)
- **NO APPROXIMATIONS** - Peer-reviewed algorithms (antropy, pyrqa)
- **VERIFIED QUALITY** - All engines audited for data integrity
- **Publication-grade** - Suitable for peer-reviewed research

## Directory Structure

```
prism-engines/diagnostics/
├── prism/                      # Core package
│   ├── db/                     # Parquet I/O layer
│   ├── engines/                # 21 computation engines
│   ├── entry_points/           # CLI entrypoints (python -m prism.entry_points.*)
│   ├── modules/                # Reusable computation modules
│   ├── cohorts/                # Cohort definitions
│   ├── state/                  # State tracking
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
│   ├── stride.yaml             # Window/stride settings
│   ├── normalization.yaml      # Normalization per domain
│   └── cohorts/                # Cohort definitions
│
├── scripts/                    # Evaluation/testing scripts
│
├── docs/                       # Documentation
│   ├── notebooks/              # Jupyter notebooks & analysis
│   │   ├── ml_accelerator/     # ML benchmark scripts
│   │   └── cmapss/             # C-MAPSS analysis
│   └── validation/             # Validation studies
│
└── data/                       # LOCAL ONLY (gitignored)
    ├── raw/                    # Raw observations
    ├── vector/                 # Computed metrics
    ├── geometry/               # Structural snapshots
    ├── state/                  # Temporal dynamics
    └── [domain]/               # Domain-specific data (cmapss, femto, etc.)
```

## Essential Commands

### Data Fetching
```bash
# Fetch C-MAPSS turbofan data
python -m prism.entry_points.fetch --cmapss

# Fetch FEMTO bearing data
python -m prism.entry_points.fetch --femto

# Fetch hydraulic system data
python -m prism.entry_points.fetch --hydraulic
```

### Vector Computation
```bash
# Run vector engines on all indicators
python -m prism.entry_points.indicator_vector

# Specific domain
python -m prism.entry_points.indicator_vector --domain cmapss

# Parallel execution
python -m prism.entry_points.indicator_vector --workers 4
```

### Geometry & State
```bash
# Compute geometry
python -m prism.entry_points.geometry --domain cheme

# Compute Laplace field
python -m prism.entry_points.laplace --domain cheme

# Cohort state
python -m prism.entry_points.cohort_state --domain cheme
```

### Monitoring
```bash
# Monitor long-running jobs
python -m prism.utils.monitor
```

## Pipeline Architecture

```
Layer 0: OBSERVATIONS
         Raw sensor data → signal topology
         Output: data/raw/observations.parquet

Layer 1: INDICATOR VECTOR
         Raw observations → 51 behavioral metrics per indicator
         Output: data/vector/indicator.parquet

Layer 2: COHORT GEOMETRY
         Indicator vectors → pairwise relationships + cohort structure
         Output: data/geometry/cohort.parquet

Layer 3: STATE
         Temporal dynamics, transitions, regime tracking
         Output: data/state/cohort.parquet

REGIME CHANGE = geometric deformation at any layer
```

## Engine Types

**Vector Engines (7)** - Intrinsic properties of single series
- Hurst, Entropy, GARCH, Wavelet, Spectral, Lyapunov, RQA

**Geometry Engines** - Structural relationships
- PCA, MST, Clustering, LOF, Distance, Convex Hull

**State Engines (6)** - Temporal dynamics
- Granger, Cross-Correlation, Cointegration, DTW, DMD, Transfer Entropy

## Key Patterns

### Reading Data
```python
import polars as pl
from prism.db.parquet_store import get_parquet_path

observations = pl.read_parquet(get_parquet_path('raw', 'observations'))
filtered = observations.filter(pl.col('indicator_id') == 'sensor_1')
```

### Writing Data
```python
from prism.db.polars_io import upsert_parquet, write_parquet_atomic

# Upsert (preserves existing rows, updates by key)
upsert_parquet(df, target_path, key_cols=['indicator_id', 'obs_date'])

# Atomic write (replaces entire file)
write_parquet_atomic(df, target_path)
```

## Validated Domains

- **C-MAPSS**: NASA turbofan engine degradation (FD001-FD004)
- **FEMTO**: Bearing degradation (PRONOSTIA dataset)
- **Hydraulic**: UCI hydraulic system condition monitoring
- **CWRU**: Case Western bearing fault classification
- **TEP**: Tennessee Eastman chemical process faults
- **MetroPT**: Metro train compressor failures

## Technical Stack

- **Language:** Python 3.10+
- **Storage:** Parquet files (columnar, compressed)
- **DataFrame:** Polars (primary), Pandas (engine compatibility)
- **Core:** NumPy, SciPy, scikit-learn
- **Specialized:** antropy, nolds, pyrqa, arch, PyWavelets, networkx

## What PRISM Does and Does NOT Do

PRISM does NOT:
- Predict timing or outcomes
- Recommend actions
- Add opinion or spin

PRISM DOES:
- Show the shape of structural stress
- Identify when geometry matches historical failure patterns
- Reveal which sensors belong together
- Detect regime boundaries mathematically
