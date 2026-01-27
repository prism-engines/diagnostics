"""
PRISM SQL Pipeline

SQL-first data persistence and analysis.

CANONICAL RULE: Orchestrators are PURE.
All logic lives in SQL files, not in Python.

Fast Primitives: DuckDB vectorized computations for instant metrics.
"""

from .orchestrator import SQLOrchestrator
from .fast_primitives import (
    compute_all_fast,
    compute_correlation_matrix,
    compute_typology,
    compute_typology_complete,
    compute_derivatives,
)

__all__ = [
    'SQLOrchestrator',
    'compute_all_fast',
    'compute_correlation_matrix',
    'compute_typology',
    'compute_typology_complete',
    'compute_derivatives',
]
