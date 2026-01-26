"""
Causality Stage Orchestrator

PURE: Loads 07_causality.sql, creates causal mechanics views.
NO computation. NO inline SQL.
"""

from .base import StageOrchestrator


class CausalityStage(StageOrchestrator):
    """Granger causality, transfer entropy, causal roles."""

    SQL_FILE = '07_causality.sql'

    VIEWS = [
        'v_granger_proxy',           # Granger causality proxy
        'v_bidirectional_causality', # A→B vs B→A
        'v_transfer_entropy_proxy',  # Transfer entropy approximation
        'v_causal_roles',            # SOURCE/SINK/CONDUIT/ISOLATED
        'v_causal_chains',           # A→B→C chains
        'v_causal_timing',           # Response lag estimation
        'v_intervention_effects',    # Intervention impact
        'v_root_cause_candidates',   # Root cause identification
        'v_causal_strength',         # Combined causal strength
        'v_causal_graph',            # Graph edges for visualization
        'v_causality_complete',      # Per-signal summary
        'v_system_causal_structure', # System-level summary
    ]

    DEPENDS_ON = ['v_base', 'v_optimal_lag', 'v_regime_changes', 'v_dy']
