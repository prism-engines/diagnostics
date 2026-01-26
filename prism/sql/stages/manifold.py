"""
Manifold Stage Orchestrator

PURE: Loads 10_manifold.sql, creates final assembly and export views.
NO computation. NO inline SQL.
"""

from .base import StageOrchestrator


class ManifoldStage(StageOrchestrator):
    """Phase space assembly, trajectory, summaries, exports."""

    SQL_FILE = '10_manifold.sql'

    VIEWS = [
        # Assembly views
        'v_phase_space',           # Phase space coordinates
        'v_trajectory',            # Trajectory through phase space

        # Summary views
        'v_signal_summary',        # One row per signal
        'v_regime_summary',        # Regime statistics
        'v_coupling_summary',      # Pairwise relationships
        'v_system_summary',        # System-level aggregates

        # Alerts
        'v_alerts',                # Anomalies and warnings

        # Export views (for parquet output)
        'v_export_signal_class',
        'v_export_signal_typology',
        'v_export_behavioral_geometry',
        'v_export_dynamical_systems',
        'v_export_causal_mechanics',

        # JSON export for viewer
        'v_export_manifold_json',
    ]

    DEPENDS_ON = [
        'v_base',
        'v_curvature',
        'v_signal_class',
        'v_signal_typology',
        'v_regime_assignment',
        'v_regime_stats',
        'v_causal_roles',
        'v_entropy_complete',
        'v_geometry_complete',
        'v_dynamics_complete',
        'v_causality_complete',
        'v_physics_complete',
    ]

    def get_manifold_json(self) -> dict:
        """
        Get manifold data for viewer.

        PURE: Just queries the view. Assembly logic is in SQL.
        """
        result = self.conn.execute("SELECT manifold_data FROM v_export_manifold_json").fetchone()
        if result:
            import json
            return json.loads(result[0])
        return {}
