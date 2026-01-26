"""
Stefan Problem - Phase Change Heat Transfer

Moving boundary problems, melting/solidification.
"""

import numpy as np
from typing import Dict, Any, Optional


def compute(T_hot: float, T_cold: float, T_melt: float,
            k: float, rho: float, cp: float, L_fusion: float,
            t: float) -> Dict[str, Any]:
    """
    Stefan problem for 1D melting/solidification.
    
    Args:
        T_hot: Hot boundary temperature [K]
        T_cold: Initial/cold temperature [K]
        T_melt: Melting temperature [K]
        k: Thermal conductivity [W/mK]
        rho: Density [kg/m³]
        cp: Specific heat [J/kgK]
        L_fusion: Latent heat of fusion [J/kg]
        t: Time [s]
    
    Returns:
        s: Interface position [m]
        lambda_: Stefan number root
        T_profile: Temperature function
    """
    alpha = k / (rho * cp)  # Thermal diffusivity
    Ste = cp * (T_hot - T_melt) / L_fusion  # Stefan number
    
    # Neumann solution: s(t) = 2 * λ * √(αt)
    # λ satisfies: λ * exp(λ²) * erf(λ) = Ste / √π
    
    # Solve for λ iteratively
    from scipy.special import erf
    from scipy.optimize import brentq
    
    def residual(lam):
        return lam * np.exp(lam**2) * erf(lam) - Ste / np.sqrt(np.pi)
    
    try:
        lambda_ = brentq(residual, 0.01, 5.0)
    except ValueError:
        lambda_ = np.sqrt(Ste / 2)  # Approximate for small Ste
    
    s = 2 * lambda_ * np.sqrt(alpha * t)
    
    return {
        's': float(s),
        'lambda': float(lambda_),
        'stefan_number': float(Ste),
        'thermal_diffusivity': float(alpha),
    }
