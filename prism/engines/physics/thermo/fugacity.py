"""
Fugacity Calculations

Fugacity, fugacity coefficient, chemical potential.
"""

import numpy as np
from typing import Dict, Any, Optional


R = 8.314  # J/mol/K


def compute(T: float, P: float, Z: float, y: float = 1.0,
            A: float = 0, B: float = 0) -> Dict[str, Any]:
    """
    Compute fugacity from compressibility.
    
    Args:
        T: Temperature [K]
        P: Pressure [Pa]
        Z: Compressibility factor
        y: Mole fraction
        A, B: EOS parameters
    
    Returns:
        f: Fugacity [Pa]
        phi: Fugacity coefficient
        mu: Chemical potential deviation from ideal
    """
    # From Peng-Robinson
    sqrt2 = np.sqrt(2)
    
    if B > 0 and A > 0:
        ln_phi = Z - 1 - np.log(Z - B) - A/(2*sqrt2*B) * np.log((Z + (1+sqrt2)*B)/(Z + (1-sqrt2)*B))
    else:
        ln_phi = Z - 1 - np.log(Z)  # Ideal gas limit
    
    phi = np.exp(ln_phi)
    f = phi * y * P
    
    # Chemical potential: μ - μ_ig = RT ln(φ)
    mu_deviation = R * T * ln_phi
    
    return {
        'fugacity': float(f),
        'fugacity_coeff': float(phi),
        'mu_deviation': float(mu_deviation),
    }
