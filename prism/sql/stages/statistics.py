"""
Statistics Stage Orchestrator

PURE: Loads 02_statistics.sql, creates statistical views.
NO computation. NO inline SQL.
"""

from .base import StageOrchestrator


class StatisticsStage(StageOrchestrator):
    """Rolling statistics, z-scores, autocorrelation, extrema."""

    SQL_FILE = '02_statistics.sql'

    VIEWS = [
        'v_stats_global',       # Global per-signal statistics
        'v_rolling_stats',      # Rolling window statistics
        'v_zscore',             # Z-score normalization
        'v_local_extrema',      # Local peaks and valleys
        'v_skewness_kurtosis',  # Higher moments
        'v_autocorrelation',    # ACF at multiple lags
        'v_runs_test',          # Runs test for randomness
        'v_statistics_complete', # Combined view
    ]

    DEPENDS_ON = ['v_base']
