# PRISM Engines

**21 behavioral measurement engines for signal topology analysis.**

All engines are pure functions or classes that take numerical data and return structured metrics. No side effects, no database access, no interpretation.

**Architecture Note (v2.0):** Engines are pure mathematical functions with no I/O dependencies. They accept NumPy arrays or pandas DataFrames (for scipy/sklearn compatibility) and return Python dicts. The Polars + Parquet storage layer is handled by runners, not engines.

---

## Installation

```bash
pip install prism-engine
```

---

## Quick Start

```python
import numpy as np
from prism.engines import compute_hurst, compute_entropy, PCAEngine, GrangerEngine

# Generate sample data
values = np.random.randn(500)

# Vector engine (single series)
hurst = compute_hurst(values)
print(f"Hurst exponent: {hurst['hurst_exponent']:.3f}")

# Geometry engine (multiple series)
matrix = np.random.randn(100, 5)  # 100 observations, 5 series
pca = PCAEngine()
result = pca.compute(matrix)
print(f"Variance explained: {result['explained_variance_ratio']}")

# State engine (pairwise dynamics)
x, y = np.random.randn(200), np.random.randn(200)
granger = GrangerEngine(max_lag=5)
result = granger.compute(x, y)
print(f"Granger p-value: {result['p_value']:.4f}")
```

---

## Engine Categories

### Vector Engines (7)

Single-indicator intrinsic properties. Input: 1D array. Output: dict of metrics.

| Engine | Function | Measures |
|--------|----------|----------|
| **Hurst** | `compute_hurst(values)` | Long-term memory (H > 0.5 = trending, H < 0.5 = mean-reverting) |
| **Entropy** | `compute_entropy(values)` | Complexity via sample entropy and permutation entropy |
| **GARCH** | `compute_garch(values)` | Volatility clustering (alpha, beta, persistence) |
| **Wavelet** | `compute_wavelets(values)` | Multi-scale energy distribution |
| **Spectral** | `compute_spectral(values)` | Frequency content via FFT |
| **Lyapunov** | `compute_lyapunov(values)` | Chaos indicator (> 0 = chaotic) |
| **RQA** | `compute_rqa(values)` | Recurrence patterns in phase space |

```python
from prism.engines import (
    compute_hurst, compute_entropy, compute_garch,
    compute_wavelets, compute_spectral, compute_lyapunov, compute_rqa
)

values = np.random.randn(500)

# Each returns a dict of metrics
compute_hurst(values)      # {'hurst_exponent': 0.52, 'std_error': 0.03, ...}
compute_entropy(values)    # {'sample_entropy': 1.8, 'permutation_entropy': 0.95, ...}
compute_garch(values)      # {'alpha': 0.1, 'beta': 0.85, 'persistence': 0.95, ...}
compute_wavelets(values)   # {'energy_by_level': [...], 'dominant_scale': 3, ...}
compute_spectral(values)   # {'dominant_frequency': 0.1, 'spectral_entropy': 0.7, ...}
compute_lyapunov(values)   # {'lyapunov_exponent': 0.02, ...}
compute_rqa(values)        # {'recurrence_rate': 0.15, 'determinism': 0.8, ...}
```

### Geometry Engines (8)

Multi-indicator relational results. Input: 2D matrix (observations x indicators). Output: structural metrics.

| Engine | Class | Measures |
|--------|-------|----------|
| **PCA** | `PCAEngine` | Shared variance, effective dimensionality |
| **Clustering** | `ClusteringEngine` | Natural groupings (K-means, hierarchical) |
| **Distance** | `DistanceEngine` | Pairwise dissimilarity matrix |
| **Mutual Information** | `MutualInformationEngine` | Nonlinear dependence |
| **Copula** | `CopulaEngine` | Tail dependence structure |
| **MST** | `MSTEngine` | Minimum spanning tree topology |
| **LOF** | `LOFEngine` | Local outlier factor scores |
| **Convex Hull** | `ConvexHullEngine` | Geometric extent and centrality |

```python
from prism.engines import (
    PCAEngine, ClusteringEngine, DistanceEngine,
    MutualInformationEngine, CopulaEngine, MSTEngine, LOFEngine, ConvexHullEngine
)

# Matrix: rows = observations, columns = indicators
matrix = np.random.randn(100, 5)

# PCA
pca = PCAEngine()
result = pca.compute(matrix)
# {'explained_variance_ratio': [0.4, 0.25, ...], 'n_components': 5, ...}

# Clustering
cluster = ClusteringEngine(n_clusters=3)
result = cluster.compute(matrix)
# {'labels': [0, 1, 2, 0, ...], 'inertia': 123.4, ...}

# Distance matrix
dist = DistanceEngine(metric='correlation')
result = dist.compute(matrix)
# {'distance_matrix': [[0, 0.3, ...], ...], 'mean_distance': 0.45, ...}

# Mutual information
mi = MutualInformationEngine()
result = mi.compute(matrix)
# {'mi_matrix': [[1.0, 0.2, ...], ...], ...}
```

### State Engines (6)

Multi-indicator temporal dynamics. Input: paired signal topology. Output: dynamic relationship metrics.

| Engine | Class | Measures |
|--------|-------|----------|
| **Granger** | `GrangerEngine` | Predictive causality (does X predict Y?) |
| **Cross-Correlation** | `CrossCorrelationEngine` | Lead/lag synchronization |
| **Cointegration** | `CointegrationEngine` | Long-run equilibrium relationship |
| **DTW** | `DTWEngine` | Shape similarity with time warping |
| **DMD** | `DMDEngine` | Dynamic mode decomposition |
| **Transfer Entropy** | `TransferEntropyEngine` | Directed information flow |

```python
from prism.engines import (
    GrangerEngine, CrossCorrelationEngine, CointegrationEngine,
    DTWEngine, DMDEngine, TransferEntropyEngine
)

x = np.random.randn(200)
y = np.random.randn(200)

# Granger causality
granger = GrangerEngine(max_lag=5)
result = granger.compute(x, y)
# {'p_value': 0.23, 'f_statistic': 1.2, 'optimal_lag': 2, ...}

# Cross-correlation
xcorr = CrossCorrelationEngine(max_lag=20)
result = xcorr.compute(x, y)
# {'peak_correlation': 0.35, 'peak_lag': 3, 'correlations': [...], ...}

# Cointegration
coint = CointegrationEngine()
result = coint.compute(x, y)
# {'is_cointegrated': False, 'p_value': 0.45, ...}

# Dynamic time warping
dtw = DTWEngine()
result = dtw.compute(x, y)
# {'distance': 45.2, 'path': [...], ...}
```

---

## Registry API

Access engines programmatically via the unified registry.

```python
from prism.engines import (
    # Registries
    ENGINES,              # All 21 engines
    VECTOR_ENGINES,       # 7 vector engines
    GEOMETRY_ENGINES,     # 8 geometry engines
    STATE_ENGINES,        # 6 state engines

    # Lookup functions
    get_engine,           # Get any engine by name
    get_vector_engine,    # Get vector engine by name
    get_geometry_engine,  # Get geometry engine by name
    get_state_engine,     # Get state engine by name

    # List functions
    list_engines,         # List all engine names
    list_vector_engines,
    list_geometry_engines,
    list_state_engines,
)

# List all engines
print(list_engines())
# ['clustering', 'cointegration', 'copula', 'cross_correlation', 'distance',
#  'dmd', 'dtw', 'entropy', 'garch', 'granger', 'hurst', 'lof', 'lyapunov',
#  'mst', 'mutual_information', 'pca', 'rqa', 'spectral', 'transfer_entropy',
#  'wavelet']

# Get engine by name
compute_fn = get_engine("hurst")
metrics = compute_fn(values)

# Iterate over all vector engines
for name, fn in VECTOR_ENGINES.items():
    result = fn(values)
    print(f"{name}: {len(result)} metrics")
```

---

## Engine Output Format

All engines return Python dictionaries with typed values.

```python
# Vector engine output
{
    'hurst_exponent': 0.52,       # float
    'std_error': 0.03,            # float
    'r_squared': 0.98,            # float
    'n_points': 500,              # int
}

# Geometry engine output
{
    'explained_variance_ratio': [0.4, 0.25, 0.15, 0.1, 0.1],  # list[float]
    'n_components': 5,            # int
    'total_variance': 1.0,        # float
}

# State engine output
{
    'p_value': 0.03,              # float
    'f_statistic': 4.5,           # float
    'optimal_lag': 2,             # int
    'is_significant': True,       # bool
}
```

---

## Minimum Data Requirements

Each engine has minimum data requirements for reliable results.

| Engine | Minimum Points | Recommended |
|--------|----------------|-------------|
| Hurst | 20 | 100+ |
| Entropy | 20 | 100+ |
| GARCH | 30 | 200+ |
| Wavelet | 32 | 128+ |
| Spectral | 16 | 64+ |
| Lyapunov | 100 | 500+ |
| RQA | 20 | 100+ |

Engines return empty dicts or NaN values when data is insufficient.

---

## Design Principles

1. **Pure Functions**: No side effects, no state, no database access
2. **Consistent Interface**: All vector engines: `f(np.ndarray) -> dict`
3. **Fail Gracefully**: Return empty dict or NaN, never raise on bad data
4. **No Interpretation**: Return raw metrics, don't classify or score
5. **Minimal Dependencies**: Core engines use only NumPy/SciPy

---

## Adding New Engines

1. Create `prism/engines/your_engine.py`
2. Implement compute function or class
3. Add to `__init__.py` registry
4. Add to `__all__` exports

```python
# prism/engines/your_engine.py
import numpy as np

def compute_your_metric(values: np.ndarray) -> dict:
    """Compute your custom metric."""
    if len(values) < 20:
        return {}

    result = your_calculation(values)

    return {
        'your_metric': float(result),
        'n_points': len(values),
    }
```

---

## See Also

- [Main README](../../README.md) - Full project documentation
- [pyproject.toml](../../pyproject.toml) - Package configuration
