"""
Calculus Stage Orchestrator

PURE: Loads 01_calculus.sql, creates derivative views.
NO computation. NO inline SQL.
"""

from .base import StageOrchestrator


class CalculusStage(StageOrchestrator):
    """Derivatives, curvature, arc length, velocity, divergence."""

    SQL_FILE = '01_calculus.sql'

    VIEWS = [
        'v_dy',               # First derivative
        'v_d2y',              # Second derivative
        'v_d3y',              # Third derivative (jerk)
        'v_curvature',        # Curvature κ
        'v_arc_length',       # Cumulative arc length
        'v_velocity',         # Velocity magnitude
        'v_laplacian',        # Laplacian (d2y approximation)
        'v_divergence',       # Divergence (cross-signal)
        'v_smoothness_index', # Smoothness metric
        'v_calculus_complete', # Combined view
    ]

    DEPENDS_ON = ['v_base']
