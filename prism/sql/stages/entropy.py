"""
Entropy Stage Orchestrator

PURE: Loads 08_entropy.sql, creates information theory views.
NO computation. NO inline SQL.
"""

from .base import StageOrchestrator


class EntropyStage(StageOrchestrator):
    """Shannon entropy, permutation entropy, mutual information."""

    SQL_FILE = '08_entropy.sql'

    VIEWS = [
        'v_shannon_entropy',          # Shannon entropy (binned)
        'v_permutation_entropy',      # Permutation entropy
        'v_spectral_entropy_proxy',   # Spectral entropy approximation
        'v_mutual_information_pairwise', # Pairwise mutual information
        'v_conditional_entropy_proxy', # Conditional entropy
        'v_entropy_complete',         # Combined view
    ]

    DEPENDS_ON = ['v_base', 'v_dy', 'v_stats_global']
