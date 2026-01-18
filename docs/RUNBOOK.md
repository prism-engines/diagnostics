# PRISM Runbook

How to run PRISM without AI assistance.

## Quick Reference

```bash
# 1. Compute behavioral metrics (Layer 1)
python -m prism.entry_points.indicator_vector --domain cheme

# 2. Compute geometry/relationships (Layer 2)
python -m prism.entry_points.geometry --domain cheme

# 3. Compute Laplace field + modes (Layer 3)
python -m prism.entry_points.laplace --domain cheme

# 4. Run fault detection assessment
python -m prism.assessments.run --domain cheme
```

---

## What Each Step Does

### Step 1: Indicator Vector (`indicator_vector`)

**What it computes:** 51 behavioral metrics for each indicator (signal topology)

**Key metrics:**
| Category | Metrics | What They Measure |
|----------|---------|-------------------|
| GARCH | alpha, beta, omega | Volatility clustering |
| Spectral | slope, entropy | Frequency composition |
| Entropy | sample, permutation | Randomness/complexity |
| Breaks | break_n, dirac, heaviside | Structural changes |

**Output:** `data/{domain}/vector/indicator.parquet`

**Runtime:** ~5-10 min for 50 indicators, ~2-3 hours for 1000+

---

### Step 2: Geometry (`geometry`)

**What it computes:** Relationships between indicators

**Key outputs:**
- PCA (dimensionality reduction)
- MST (minimum spanning tree - which indicators cluster)
- Distance matrices
- Outlier detection (LOF)

**Output:** `data/{domain}/geometry/cohort.parquet`

**Runtime:** ~10-30 min depending on indicator count

---

### Step 3: Laplace Field (`laplace`)

**What it computes:** Geometric fingerprints from the vector field

**Key outputs:**
- Gradient (rate of change in behavioral space)
- Divergence (expansion/contraction)
- Laplacian (curvature)
- Modes (behavioral clusters via GMM)

**Output:**
- `data/{domain}/vector/indicator_field.parquet`
- `data/{domain}/vector/indicator_modes.parquet`

**Runtime:** ~30 min - 2 hours (memory intensive)

---

### Step 4: Assessment (`assessments.run`)

**What it does:** Combines all layers to detect regime changes

**Three detection layers:**

| Layer | Name | What It Detects | Key Signal |
|-------|------|-----------------|------------|
| WHAT | Classification | Which fault type | GARCH volatility shift |
| WHEN | Break Detection | Timing of change | Structural breaks, impulses |
| MODE | Trajectory | Behavioral state | Affinity drops, Mode 1 precursor |

**Output:** Printed report with detection rates

---

## Interpreting Results

### Detection Rates

```
Overall warning rate: 100%
```

- **>80%**: Strong detection - system reliably warns before events
- **50-80%**: Moderate - useful but not comprehensive
- **<50%**: Weak - need better features or tuning

### Mode 1 Signal (M1!)

When you see `M1!` in the output:

```
IDV05 @ 2000-01-15: breaks=218981, dirac=28778, heaviside=35785 M1! EARLY WARNING
```

This means the **precursor mode** was detected. Mode 1 characteristics:
- Rare (~4% of time)
- Appears BEFORE visible structural breaks
- Indicates system is in uncertain/transitional state

### Break Signals

| Signal | Meaning |
|--------|---------|
| `breaks=X` | Structural breaks detected (changepoints) |
| `dirac=X` | Impulses (sudden spikes) |
| `heaviside=X` | Step changes (level shifts) |

Higher numbers = more discontinuity activity.

### Affinity Delta

```
affinity_delta=-0.150
```

- **Negative**: Mode confidence dropping (destabilization)
- **Near zero**: Stable behavioral state
- **Positive**: Mode confidence increasing (stabilization)

---

## Configuration

All settings are in `config/assessment.yaml`:

```yaml
defaults:
  windows:
    pre_onset: 7          # Days before event to analyze
  thresholds:
    break_zscore: 1.0     # Detection sensitivity

domains:
  cheme:                  # TEP-specific settings
  turbofan:               # CMAPSS-specific settings
```

To add a new domain, copy the `_template` section and fill in:
- `indicator_prefix`: How your indicators are named
- `fault_indicator`: The label/target column
- `exclude_pattern`: What to exclude from analysis

---

## Common Issues

### Out of Memory (OOM)

```bash
# Reduce sample size for mode computation
python -m prism.assessments.tep_modes --domain cheme --sample 25
```

### No Modes Data

```
Modes data: NOT FOUND
```

Run Laplace first:
```bash
python -m prism.entry_points.laplace --domain cheme
```

### Low Detection Rate

1. Check if `when_features` have data:
   ```bash
   python -c "import polars as pl; df = pl.read_parquet('data/cheme/vector/indicator.parquet'); print(df.filter(pl.col('metric_name') == 'break_n')['metric_value'].describe())"
   ```

2. Adjust thresholds in `config/assessment.yaml`

---

## Full Pipeline Example

```bash
# Set domain
export PRISM_DOMAIN=cheme

# Run full pipeline
python -m prism.entry_points.indicator_vector --domain $PRISM_DOMAIN
python -m prism.entry_points.geometry --domain $PRISM_DOMAIN
python -m prism.entry_points.laplace --domain $PRISM_DOMAIN
python -m prism.assessments.run --domain $PRISM_DOMAIN

# Or use the runner script
./scripts/run_pipeline.sh cheme
```

---

## What "Significant" Means

### Compared to Traditional Methods

| Method | Typical Accuracy | PRISM Approach |
|--------|------------------|----------------|
| Simple threshold | 30-50% | Uses 51 behavioral features |
| ML classifier | 60-80% | Adds geometric relationships |
| Deep learning | 70-90% | Adds mode trajectory |

PRISM's value: **Explainability**. You can trace WHY a warning was generated:
- Which features contributed (WHAT)
- When breaks occurred (WHEN)
- What behavioral mode it entered (MODE)

### Statistical Significance

A detection rate of 100% on 30 events is significant (p < 0.001 vs random).

But the real test is:
1. **False positive rate** - Does it cry wolf?
2. **Lead time** - How early does it warn?
3. **Generalization** - Does it work on new data?

---

## Divergence & State Analysis

### Understanding Divergence

Divergence measures stress flow direction:
- **Negative divergence** = Stress SOURCE (originating failure)
- **Positive divergence** = Stress SINK (absorbing stress)
- **More negative** = Worse condition

```python
# Check divergence ranking
import polars as pl
field_df = pl.read_parquet('data/{domain}/vector/indicator_field.parquet')

# Aggregate per entity (bearing, unit, etc.)
stats = field_df.group_by('entity_id').agg([
    pl.col('divergence').mean().alias('div_mean'),
    pl.col('divergence').min().alias('div_min'),
])

# Most negative = predicted worst
print(stats.sort('div_min'))
```

### Computing Second Derivative (Coupling Velocity)

The second derivative (acceleration) often predicts failure better than position:

```python
import numpy as np

# Daily divergence per entity
daily = field_df.group_by(['entity_id', 'window_end']).agg([
    pl.col('divergence').mean()
]).sort(['entity_id', 'window_end'])

# First derivative (velocity)
velocity = np.diff(daily['divergence'].to_numpy())

# Second derivative (acceleration)
acceleration = np.diff(velocity)

# High velocity STD = erratic behavior = may survive longer
# Low velocity STD = settled into failure trajectory
```

### Key Predictors

| Predictor | What it Measures | Failure Signal |
|-----------|------------------|----------------|
| Divergence (min) | Worst stress point | More negative = worse |
| Velocity STD | Behavioral erraticism | Lower = settled into failure |
| Entropy | Mode stability | Higher = chaotic failure |
| Affinity | Mode confidence | Lower = transitioning |

### Combined Analysis

```bash
# Full analysis pipeline
python -m prism.entry_points.indicator_vector --domain femto
python -m prism.entry_points.laplace --domain femto
python -m prism.assessments.run --domain femto
```

---

## Files Reference

```
data/{domain}/
├── raw/
│   └── observations.parquet      # Input signal topology
├── vector/
│   ├── indicator.parquet         # 51 metrics per indicator
│   ├── indicator_field.parquet   # Laplace geometric field
│   └── indicator_modes.parquet   # Behavioral mode assignments
├── geometry/
│   └── cohort.parquet            # Structural relationships
└── state/
    └── bearing_dynamics.parquet  # Velocity/acceleration stats
```

---

## Getting Help

1. Check this runbook
2. Read `TEP_Results.md` for example findings
3. Check `CLAUDE.md` for architecture details
4. File issues at: https://github.com/prism-engines/prism-core/issues
