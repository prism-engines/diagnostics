"""
Orthon Cohort Discovery Module

Three-layer hierarchical cohort discovery:
1. SIGNAL TYPES: Cluster signals by behavioral fingerprint (from vector)
2. STRUCTURAL GROUPS: Cluster entities by correlation patterns (from geometry)
3. TEMPORAL COHORTS: Cluster entities by trajectory dynamics (from state)

Key insight: Temporal cohorts must be discovered WITHIN structural groups,
otherwise structural differences confound trajectory differences.

Usage:
    from orthon.cohort import discover_cohorts, CohortResult

    result = discover_cohorts(
        vector_path="data/vector.parquet",
        geometry_path="data/geometry.parquet",
        state_path="data/state.parquet",
    )

    print(result.summary())
    result.save("data/cohorts.parquet")
"""

from orthon.cohort.discovery import (
    discover_cohorts,
    discover_signal_types,
    discover_structural_groups,
    discover_temporal_cohorts,
    CohortResult,
)

from orthon.cohort.confidence import (
    compute_clustering_confidence,
    ConfidenceMetrics,
)

__all__ = [
    "discover_cohorts",
    "discover_signal_types",
    "discover_structural_groups",
    "discover_temporal_cohorts",
    "CohortResult",
    "compute_clustering_confidence",
    "ConfidenceMetrics",
]
