"""
Typology Stage Orchestrator

PURE: Loads 04_typology.sql, creates behavioral typology views.
NO computation. NO inline SQL.
"""

from .base import StageOrchestrator


class TypologyStage(StageOrchestrator):
    """Behavioral typology: trending, mean-reverting, chaotic, random."""

    SQL_FILE = '04_typology.sql'

    VIEWS = [
        'v_trend_detection',       # Trend analysis
        'v_mean_reversion',        # Mean reversion detection
        'v_stationarity_test',     # Stationarity proxy
        'v_chaos_proxy',           # Chaos indicators
        'v_volatility_clustering', # GARCH-like clustering
        'v_signal_typology',       # Final typology
        'v_prism_requests',        # PRISM work order generation
    ]

    DEPENDS_ON = ['v_base', 'v_signal_class', 'v_stats_global', 'v_autocorrelation']

    def get_prism_work_order(self) -> list:
        """
        Query the PRISM work order view.

        PURE: Just queries the view. Logic is in SQL.
        """
        return self.query('v_prism_requests').to_dict(orient='records')
