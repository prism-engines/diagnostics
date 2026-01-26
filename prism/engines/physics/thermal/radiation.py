"""
Thermal Radiation

Stefan-Boltzmann, view factors, radiative exchange.
"""

import numpy as np
from typing import Dict, Any, Optional


STEFAN_BOLTZMANN = 5.67e-8  # W/m²K⁴


def compute(T1: float, T2: float = 300.0, epsilon1: float = 0.9, 
            epsilon2: float = 0.9, F12: float = 1.0,
            A1: float = 1.0) -> Dict[str, Any]:
    """
    Compute radiative heat transfer.
    
    Args:
        T1, T2: Surface temperatures [K]
        epsilon1, epsilon2: Emissivities
        F12: View factor from 1 to 2
        A1: Area of surface 1 [m²]
    
    Returns:
        Q: Heat transfer rate [W]
        q: Heat flux [W/m²]
        Eb1, Eb2: Blackbody emissive powers
    """
    Eb1 = STEFAN_BOLTZMANN * T1**4
    Eb2 = STEFAN_BOLTZMANN * T2**4
    
    # Radiative resistance network for two surfaces
    # Q = (Eb1 - Eb2) / ((1-ε1)/(ε1*A1) + 1/(A1*F12) + (1-ε2)/(ε2*A1))
    R_total = (1-epsilon1)/(epsilon1*A1) + 1/(A1*F12) + (1-epsilon2)/(epsilon2*A1)
    
    Q = (Eb1 - Eb2) / R_total
    q = Q / A1
    
    return {
        'Q': float(Q),
        'q': float(q),
        'Eb1': float(Eb1),
        'Eb2': float(Eb2),
        'R_total': float(R_total),
    }
