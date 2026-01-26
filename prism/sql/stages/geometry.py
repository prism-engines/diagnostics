"""
Geometry Stage Orchestrator

PURE: Loads 05_geometry.sql, creates behavioral geometry views.
NO computation. NO inline SQL.
"""

from .base import StageOrchestrator


class GeometryStage(StageOrchestrator):
    """Coupling, correlation, networks, lead-lag relationships."""

    SQL_FILE = '05_geometry.sql'

    VIEWS = [
        'v_correlation_matrix',       # Pairwise Pearson correlation
        'v_lagged_correlation',       # Correlation at multiple lags
        'v_optimal_lag',              # Best lag per pair
        'v_lead_lag',                 # Lead-lag direction
        'v_coupling_network',         # Network edges
        'v_node_degree',              # Node centrality
        'v_directional_degree',       # In/out degree
        'v_correlation_clusters',     # Signal clusters
        'v_derivative_correlation',   # Velocity coupling
        'v_covariance_matrix',        # Covariance
        'v_partial_correlation_proxy', # Partial correlation
        'v_mutual_info_proxy',        # Mutual information
        'v_geometry_complete',        # Combined view
    ]

    DEPENDS_ON = ['v_base', 'v_d2y', 'v_stats_global']
