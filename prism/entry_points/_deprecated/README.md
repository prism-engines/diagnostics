# Deprecated Entry Points

These entry points have been replaced by the unified pipeline runner.

## Migration

**Old way:**
```bash
python -m prism.entry_points.signal_typology
python -m prism.entry_points.behavioral_geometry
python -m prism.entry_points.dynamical_systems
```

**New way:**
```bash
python -m prism.entry_points.run              # Full pipeline
python -m prism.entry_points.run --only typology
python -m prism.entry_points.run --from geometry
```

## Deprecated Files

| File | Replaced By |
|------|-------------|
| `signal_typology.py` | `run.py` calls `layers/signal_typology.py` |
| `behavioral_geometry.py` | `run.py` calls `layers/behavioral_geometry.py` |
| `dynamical_systems.py` | `run.py` calls `layers/dynamical_systems.py` |
| `phase_state.py` | Merged into `layers/dynamical_systems.py` |
| `orchestrator.py` | Replaced by `run.py` |
| `vector.py` | Replaced by `signal_typology.py` |

## Architecture

```
fetch.py → observations.parquet (standalone)

run.py → calls layers/ in sequence:
    └── layers/signal_typology.py      (Layer 1: WHAT)
    └── layers/behavioral_geometry.py  (Layer 2: HOW)
    └── layers/dynamical_systems.py    (Layer 3: WHEN)
    └── layers/causal_mechanics.py     (Layer 4: WHY)
```

## Do Not Delete

These files are kept for reference and potential rollback.
They may be removed in a future version.
