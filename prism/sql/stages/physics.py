"""
Physics Stage Orchestrator

PURE: Loads 09_physics.sql, creates conservation law views.
NO computation. NO inline SQL.
"""

from .base import StageOrchestrator


class PhysicsStage(StageOrchestrator):
    """Conservation laws, mass balance, energy, thermodynamics."""

    SQL_FILE = '09_physics.sql'

    VIEWS = [
        'v_conservation_check',        # General conservation
        'v_mass_balance',              # Mass in = mass out
        'v_energy_balance',            # P × Q power estimation
        'v_thermodynamic_consistency', # PVT relationships
        'v_second_law_check',          # Entropy increase
        'v_continuity_equation',       # ∂ρ/∂t + ∇·(ρv) = 0
        'v_diffusion_check',           # Fick's law
        'v_bernoulli_check',           # P + ½ρv² + ρgh = const
        'v_heat_transfer_check',       # Temperature gradients
        'v_momentum_balance',          # F = ma
        'v_physics_complete',          # Summary of violations
    ]

    DEPENDS_ON = ['v_base', 'v_d2y', 'v_shannon_entropy', 'v_signal_class']
