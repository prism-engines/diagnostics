"""
Convective Heat Transfer

Newton's law of cooling, Nusselt correlations, fin efficiency.
"""

import numpy as np
from typing import Dict, Any, Optional


def compute(T_surface: float, T_fluid: float, h: float = None,
            L: float = 1.0, k_fluid: float = 0.6, nu: float = 1e-6,
            Pr: float = 7.0, U: float = 1.0, 
            geometry: str = 'flat_plate') -> Dict[str, Any]:
    """
    Compute convective heat transfer coefficient and Nusselt number.
    
    Args:
        T_surface, T_fluid: Surface and bulk fluid temperatures [K]
        h: Heat transfer coefficient (if known) [W/m²K]
        L: Characteristic length [m]
        k_fluid: Fluid thermal conductivity [W/mK]
        nu: Kinematic viscosity [m²/s]
        Pr: Prandtl number
        U: Fluid velocity [m/s]
        geometry: 'flat_plate', 'cylinder', 'sphere'
    
    Returns:
        h: Heat transfer coefficient
        Nu: Nusselt number
        q: Heat flux [W/m²]
        Re: Reynolds number
    """
    Re = U * L / nu
    
    if h is None:
        # Correlations
        if geometry == 'flat_plate':
            if Re < 5e5:  # Laminar
                Nu = 0.664 * Re**0.5 * Pr**(1/3)
            else:  # Turbulent
                Nu = 0.037 * Re**0.8 * Pr**(1/3)
        elif geometry == 'cylinder':
            # Churchill-Bernstein
            Nu = 0.3 + (0.62 * Re**0.5 * Pr**(1/3)) / (1 + (0.4/Pr)**(2/3))**0.25
            if Re > 2e5:
                Nu *= (1 + (Re/2.82e5)**0.625)**0.8
        elif geometry == 'sphere':
            # Whitaker
            Nu = 2 + (0.4*Re**0.5 + 0.06*Re**(2/3)) * Pr**0.4
        else:
            Nu = 0.023 * Re**0.8 * Pr**0.4  # Dittus-Boelter (pipe)
        
        h = Nu * k_fluid / L
    else:
        Nu = h * L / k_fluid
    
    q = h * (T_surface - T_fluid)
    
    return {
        'h': float(h),
        'Nu': float(Nu),
        'q': float(q),
        'Re': float(Re),
        'Pr': float(Pr),
    }
