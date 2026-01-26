# PRISM SQL Pipeline

SQL-first data persistence and analysis. All parquet writing happens here.

## Architecture

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ORCHESTRATORS ARE PURE.                                     ║
║                                                               ║
║   No computation inside orchestrators.                        ║
║   No inline SQL inside orchestrators.                         ║
║   No business logic inside orchestrators.                     ║
║                                                               ║
║   Orchestrators ONLY: load, call, query, pass.                ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

```
prism/sql/
├── orchestrator.py           # Main orchestrator (PURE)
├── run_all.py                # CLI wrapper (PURE)
├── validate_outputs.py       # Post-hoc validation
├── stages/                   # Stage orchestrators (PURE)
│   ├── base.py               # Base class
│   ├── load.py               # 00: Load observations
│   ├── calculus.py           # 01: Derivatives, curvature
│   ├── statistics.py         # 02: Rolling stats, z-scores
│   ├── classification.py     # 03: Signal classification
│   ├── typology.py           # 04: Behavioral typology
│   ├── geometry.py           # 05: Coupling, correlation
│   ├── dynamics.py           # 06: Regimes, transitions
│   ├── causality.py          # 07: Granger, causal roles
│   ├── entropy.py            # 08: Information theory
│   ├── physics.py            # 09: Conservation laws
│   └── manifold.py           # 10: Final assembly, exports
├── sql/                      # SQL files (ALL LOGIC HERE)
│   ├── 00_load.sql
│   ├── 01_calculus.sql
│   ├── 02_statistics.sql
│   ├── 03_signal_class.sql
│   ├── 04_typology.sql
│   ├── 05_geometry.sql
│   ├── 06_dynamics.sql
│   ├── 07_causality.sql
│   ├── 08_entropy.sql
│   ├── 09_physics.sql
│   └── 10_manifold.sql
└── outputs/                  # Generated parquet files
```

## Usage

### CLI

```bash
# Run full pipeline
python -m prism.sql.run_all /path/to/observations.parquet

# Custom output directory
python -m prism.sql.run_all /path/to/observations.parquet ./my_outputs/

# With PRISM primitives
python -m prism.sql.run_all /path/to/observations.parquet ./outputs/ --primitives /path/to/primitives.parquet

# Validate outputs
python validate_outputs.py ./outputs/
```

### Python API

```python
from prism.sql.orchestrator import SQLOrchestrator

# Initialize
orchestrator = SQLOrchestrator()

# Run full pipeline
result = orchestrator.run_pipeline(
    observations_path='data/observations.parquet',
    output_dir='outputs/',
    primitives_path='data/primitives.parquet',  # optional
)

# Or run individual stages
orchestrator.load_observations('data/observations.parquet')
orchestrator.run_stage('load')
orchestrator.run_stage('calculus')
orchestrator.run_stage('statistics')
# ...

# Query views
df = orchestrator.query('v_signal_class')
work_order = orchestrator.get_prism_work_order()
alerts = orchestrator.get_alerts()

# Export
orchestrator.set_output_dir('outputs/')
orchestrator.export('v_signal_summary', 'signal_summary.parquet')
orchestrator.export_all()
```

## Stages

| # | Stage | SQL File | Purpose |
|---|-------|----------|---------|
| 00 | load | 00_load.sql | Load observations, create v_base |
| 01 | calculus | 01_calculus.sql | Derivatives, curvature, arc length |
| 02 | statistics | 02_statistics.sql | Rolling stats, z-scores, autocorrelation |
| 03 | classification | 03_signal_class.sql | Analog/digital/periodic/event |
| 04 | typology | 04_typology.sql | Trending/mean-reverting/chaotic |
| 05 | geometry | 05_geometry.sql | Correlation, coupling, networks |
| 06 | dynamics | 06_dynamics.sql | Regimes, transitions, stability |
| 07 | causality | 07_causality.sql | Granger, transfer entropy, roles |
| 08 | entropy | 08_entropy.sql | Shannon, permutation, mutual info |
| 09 | physics | 09_physics.sql | Conservation laws, balances |
| 10 | manifold | 10_manifold.sql | Assembly, summaries, exports |

## View Dependency Chain

```
v_base (observations)
    │
    ├── v_dy → v_d2y → v_d3y → v_curvature
    │                       └── v_arc_length
    │
    ├── v_stats_global
    │   ├── v_rolling_stats → v_zscore
    │   ├── v_skewness_kurtosis
    │   └── v_autocorrelation
    │
    ├── v_signal_class
    │   └── v_classification_complete
    │
    ├── v_signal_typology
    │   └── v_prism_requests (work order for PRISM)
    │
    ├── v_correlation_matrix → v_optimal_lag → v_lead_lag
    │                                       └── v_coupling_network
    │
    ├── v_regime_assignment → v_regime_stats → v_regime_transitions
    │
    ├── v_granger_proxy → v_causal_roles
    │
    ├── v_shannon_entropy → v_entropy_complete
    │
    └── v_physics_complete
```

## Output Parquets

| File | View | Description |
|------|------|-------------|
| signal_class.parquet | v_export_signal_class | Signal classification |
| signal_typology.parquet | v_export_signal_typology | Behavioral typology |
| behavioral_geometry.parquet | v_export_behavioral_geometry | Coupling relationships |
| dynamical_systems.parquet | v_export_dynamical_systems | Regime dynamics |
| causal_mechanics.parquet | v_export_causal_mechanics | Causal structure |
| manifold.json | v_export_manifold_json | JSON for viewer |

## PRISM Integration

The typology stage generates work orders for PRISM:

```python
# Get work order from SQL
work_order = orchestrator.get_prism_work_order()

# Work order tells PRISM which engines to run per signal
# Logic is in v_prism_requests view in 04_typology.sql
```

## The Rule

All logic lives in SQL files. Orchestrators are pure plumbing.

```python
# ═══════════════════════════════════════════════════════════════
# WRONG: Logic in orchestrator
# ═══════════════════════════════════════════════════════════════

def classify_signals(self):
    sql = """SELECT signal_id, CASE WHEN x < 0.05 THEN 'digital' END..."""
    return self.conn.execute(sql)  # INLINE SQL = VIOLATION

# ═══════════════════════════════════════════════════════════════
# RIGHT: Logic in SQL file, orchestrator just queries view
# ═══════════════════════════════════════════════════════════════

def classify_signals(self):
    return self.query('v_signal_class')  # View defined in SQL file
```

If you're writing logic in an orchestrator, you're in the wrong file.

## Dependencies

- DuckDB >= 0.9.0
- Python >= 3.10
- PRISM engines (optional, for enhanced analysis)
