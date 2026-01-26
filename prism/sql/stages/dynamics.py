"""
Dynamics Stage Orchestrator

PURE: Loads 06_dynamics.sql, creates dynamical systems views.
NO computation. NO inline SQL.
"""

from .base import StageOrchestrator


class DynamicsStage(StageOrchestrator):
    """Regime detection, transitions, stability, basins, attractors."""

    SQL_FILE = '06_dynamics.sql'

    VIEWS = [
        'v_rolling_regime_stats',  # Rolling stats for regime detection
        'v_regime_changes',        # Change point detection
        'v_regime_boundaries',     # Regime boundary identification
        'v_regime_assignment',     # Per-point regime assignment
        'v_regime_stats',          # Per-regime statistics
        'v_regime_transitions',    # Transition characterization
        'v_transition_matrix',     # Transition probabilities
        'v_stability',             # Local stability analysis
        'v_basins',                # Basin of attraction detection
        'v_attractors',            # Attractor identification
        'v_recurrence_proxy',      # Recurrence quantification
        'v_bifurcation_candidates', # Bifurcation detection
        'v_phase_velocity',        # Phase space velocity
        'v_dynamics_complete',     # Combined view
        'v_system_regime',         # System-level regime changes
    ]

    DEPENDS_ON = ['v_base', 'v_curvature', 'v_d2y', 'v_local_extrema', 'v_stats_global']
