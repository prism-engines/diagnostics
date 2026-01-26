"""
Classification Stage Orchestrator

PURE: Loads 03_signal_class.sql, creates classification views.
NO computation. NO inline SQL.
"""

from .base import StageOrchestrator


class ClassificationStage(StageOrchestrator):
    """Signal classification: analog, digital, periodic, event."""

    SQL_FILE = '03_signal_class.sql'

    VIEWS = [
        'v_unit_class',           # Classification from units
        'v_continuity_class',     # Continuous vs discrete
        'v_smoothness_class',     # Smooth vs rough
        'v_periodicity_detection', # Periodic detection
        'v_sparsity_detection',   # Sparse/event detection
        'v_signal_class',         # Final classification
        'v_classification_complete', # Combined view
    ]

    DEPENDS_ON = ['v_base', 'v_stats_global', 'v_curvature']
