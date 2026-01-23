"""
prism/modules/ - Pure computation modules

No I/O, no side effects. Just computation.

Usage:
    from prism.modules.vector import compute_vector_features
    from prism.modules.geometry import compute_geometry_features
    from prism.modules.state import compute_state_features
    from prism.modules.discovery import discover_cohorts
"""

from prism.modules.vector import compute_vector_features
from prism.modules.geometry import compute_geometry_features
from prism.modules.state import compute_state_features
from prism.modules.discovery import discover_cohorts

__all__ = [
    "compute_vector_features",
    "compute_geometry_features",
    "compute_state_features",
    "discover_cohorts",
]
