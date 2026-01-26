"""
Two-Phase Flow Analysis

Void fraction, drift-flux, flow regime identification.
"""

import numpy as np
from typing import Dict, Any, Optional


def compute(alpha: np.ndarray = None, j_g: float = None, j_l: float = None,
            rho_g: float = 1.2, rho_l: float = 1000.0,
            sigma: float = 0.072, D: float = 0.05, g: float = 9.81) -> Dict[str, Any]:
    """
    Two-phase flow analysis.
    
    Args:
        alpha: Void fraction time series
        j_g, j_l: Superficial gas/liquid velocities [m/s]
        rho_g, rho_l: Gas/liquid densities [kg/m³]
        sigma: Surface tension [N/m]
        D: Pipe diameter [m]
        g: Gravity [m/s²]
    
    Returns:
        flow_regime: Predicted flow pattern
        drift_velocity: Drift-flux velocity
        void_fraction: Mean void fraction
        slip_ratio: S = v_g / v_l
    """
    result = {}
    
    if j_g is not None and j_l is not None:
        j_total = j_g + j_l
        beta = j_g / j_total if j_total > 0 else 0
        
        # Drift-flux model: j_g = C_0 * α * j + V_gj * α
        # Wallis correlation for drift velocity
        V_gj = 1.53 * (g * sigma * (rho_l - rho_g) / rho_l**2)**0.25
        
        # Distribution parameter (Zuber-Findlay)
        C_0 = 1.2  # typical value for bubbly flow
        
        # Solve for α: j_g = C_0 * α * j + V_gj * α
        # α = j_g / (C_0 * j + V_gj)
        alpha_mean = j_g / (C_0 * j_total + V_gj) if (C_0 * j_total + V_gj) > 0 else 0
        
        # Flow regime identification (Taitel-Dukler style)
        Fr = j_total / np.sqrt(g * D)
        We = rho_l * j_total**2 * D / sigma
        
        if beta < 0.25 and Fr < 1:
            regime = 'bubbly'
        elif beta < 0.8 and Fr < 3:
            regime = 'slug'
        elif beta > 0.8 and Fr > 2:
            regime = 'annular'
        else:
            regime = 'churn'
        
        result.update({
            'flow_regime': regime,
            'drift_velocity': float(V_gj),
            'void_fraction': float(alpha_mean),
            'beta': float(beta),
            'froude': float(Fr),
            'weber': float(We),
        })
    
    if alpha is not None:
        alpha = np.asarray(alpha)
        result.update({
            'void_fraction_mean': float(np.mean(alpha)),
            'void_fraction_std': float(np.std(alpha)),
            'void_fraction_pdf': np.histogram(alpha, bins=50, density=True)[0].tolist(),
        })
    
    return result
