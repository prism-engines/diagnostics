# PRISM Package

**`pip install prism-engine`**

Core Python package for PRISM behavioral signal topology analysis.

---

## Package Structure

```
prism/
├── db/                  # Database utilities (Parquet-based)
│   ├── parquet_store.py # Path management, directory structure
│   ├── polars_io.py     # Atomic writes, upsert operations
│   ├── query.py         # Table introspection utilities
│   └── scratch.py       # TempParquet for parallel workers
│
├── config/              # Configuration
│   └── windows.py       # Domain-aware window configs
│
├── engines/             # 33 measurement engines
│   ├── hurst.py         # Hurst exponent (memory)
│   ├── entropy.py       # Sample/permutation entropy
│   ├── garch.py         # Volatility modeling
│   ├── rqa.py           # Recurrence quantification
│   ├── lyapunov.py      # Chaos signal
│   ├── pca.py           # Principal components
│   ├── granger.py       # Granger causality
│   └── ...              # 14 more engines
│
├── entry_points/        # CLI entry points (15 total)
│   ├── fetch.py         # Data fetching to Parquet
│   ├── characterize.py  # 6-axis signal classification
│   ├── signal_vector.py # Layer 1: Vector metrics
│   ├── laplace.py       # Layer 2: Laplace field
│   ├── geometry.py      # Layer 3: Cohort geometry
│   ├── state.py         # Layer 4: State derivation
│   └── ...
│
├── modules/             # Reusable computation (NOT entry points)
│   ├── characterize.py  # Inline characterization
│   ├── laplace.py       # Laplace field computation
│   ├── modes.py         # Mode discovery from signatures
│   └── wavelet_microscope.py # Frequency-band degradation
│
├── cohorts/             # Cohort definitions
│   ├── climate.py       # Climate cohorts
│   └── cheme.py         # Chemical engineering cohorts
│
└── utils/               # Shared utilities
    ├── stride.py        # Window/stride configuration
    └── monitor.py       # Progress monitoring
```

---

## Usage

### As a Library

```python
# Import engines
from prism.engines import (
    compute_hurst, compute_entropy, compute_rqa,
    PCAEngine, GrangerEngine,
    VECTOR_ENGINES, list_engines
)

# Compute metrics on sensor data
import numpy as np
values = np.random.randn(500)
hurst = compute_hurst(values)
print(hurst['hurst_exponent'])
```

### Database Access

```python
import polars as pl
from prism.db.parquet_store import get_path, OBSERVATIONS, SIGNALS, GEOMETRY, STATE, COHORTS

# Read observations
observations = pl.read_parquet(get_path(OBSERVATIONS))

# Query data
df = observations.filter(pl.col('signal_id') == 'sensor_T30')

# Read signals
signals = pl.read_parquet(get_path(SIGNALS))
```

### CLI Entry Points

```bash
# Fetch industrial data
python -m prism.db.fetch --cmapss
python -m prism.db.fetch --femto

# Run pipeline
python -m prism.entry_points.signal_vector --domain cmapss
python -m prism.entry_points.laplace --domain cmapss
python -m prism.entry_points.geometry --domain cmapss
```

---

## Pipeline Architecture

```
fetch → characterize
  → signal_vector → laplace → geometry → state
```

**Key:** Observations → Vector Metrics → Laplace Field → Geometry → State

---

## Engine Categories

| Category | Count | Purpose | Interface |
|----------|-------|---------|-----------|
| **Vector** | 9 | Single-signal intrinsic properties | `f(array) -> dict` |
| **Geometry** | 9 | Multi-signal relational structure | `Engine.compute(matrix)` |
| **State** | 7 | Temporal dynamics and causality | `Engine.compute(x, y)` |
| **Temporal Dynamics** | 5 | Geometry evolution over time | `Engine.compute(snapshots)` |
| **Observation** | 3 | Discontinuity detection | `f(array) -> dict` |

See [engines/README.md](engines/README.md) for detailed documentation.

---

## Validated Domains

| Domain | Source | Sensors | Use Case |
|--------|--------|---------|----------|
| **C-MAPSS** | NASA | 25 | Turbofan run-to-failure |
| **FEMTO** | PHM Society | 2 | Bearing degradation |
| **Hydraulic** | UCI | 17 | System condition monitoring |
| **CWRU** | CWRU | 4 | Bearing fault diagnosis |
| **TEP** | Tennessee Eastman | 52 | Chemical process faults |

---

## Storage Schema (v2.0 - 5-File Architecture)

```
data/{domain}/
├── observations.parquet  # Raw time series data
├── signals.parquet       # Computed signal metrics
├── geometry.parquet      # Structural relationships
├── state.parquet         # Temporal dynamics
└── cohorts.parquet       # Entity groupings
```

**Schema:**
- `observations`: entity_id, signal_id, timestamp, value
- `signals`: entity_id, signal_id, source_signal, engine, signal_type, timestamp, value, mode_id
- `geometry`: System structure at each timestamp
- `state`: Dynamics at each timestamp
- `cohorts`: Discovered entity groupings

---

## Dependencies

**Required:**
- polars >= 0.20.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0
- scikit-learn >= 1.3.0
- pyyaml >= 6.0
- antropy >= 0.1.6
- arch >= 6.0
- nolds >= 0.5.2
- pyrqa >= 8.0.0
- PyWavelets >= 1.4.0

**Optional:**
- requests (for fetching)

---

## Version

**2.0.0** — Pure Polars + Parquet (no DuckDB)
