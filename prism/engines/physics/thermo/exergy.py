"""
Exergy Analysis

Available work, irreversibility, second law efficiency.
"""

import numpy as np
from typing import Dict, Any, Optional


R = 8.314  # J/mol/K


def compute(T: float, T0: float, P: float, P0: float,
            h: float, h0: float, s: float, s0: float,
            n: float = 1.0) -> Dict[str, Any]:
    """
    Compute specific exergy (availability).
    
    Args:
        T, P: State temperature [K] and pressure [Pa]
        T0, P0: Dead state (environment) conditions
        h, h0: Specific enthalpy [J/mol]
        s, s0: Specific entropy [J/mol/K]
        n: Molar flow rate [mol/s]
    
    Returns:
        ex_physical: Physical exergy [J/mol]
        ex_thermal: Thermal component
        ex_mechanical: Mechanical component
        Ex_flow: Exergy flow rate [W]
    """
    # Physical exergy = (h - h0) - T0*(s - s0)
    ex_physical = (h - h0) - T0 * (s - s0)
    
    # Split into thermal and mechanical
    # Thermal: from temperature difference
    # Approximate for ideal gas
    cp = 5/2 * R  # Monatomic ideal gas
    ex_thermal = cp * (T - T0) - T0 * cp * np.log(T/T0)
    
    # Mechanical: from pressure difference
    ex_mechanical = R * T0 * np.log(P/P0) if P > 0 and P0 > 0 else 0
    
    Ex_flow = n * ex_physical
    
    return {
        'ex_physical': float(ex_physical),
        'ex_thermal': float(ex_thermal),
        'ex_mechanical': float(ex_mechanical),
        'Ex_flow': float(Ex_flow),
    }
