"""
Reynolds Stress Tensor Analysis

τ_ij = -ρ <u'_i u'_j>

Turbulent momentum transport, anisotropy, realizability.
"""

import numpy as np
from typing import Dict, Any, Optional


def compute(u: np.ndarray, v: np.ndarray, w: np.ndarray = None,
            rho: float = 1.0) -> Dict[str, Any]:
    """
    Compute Reynolds stress tensor from velocity fluctuations.
    
    Args:
        u, v, w: Velocity component time series
        rho: Density [kg/m³]
    
    Returns:
        tau: Reynolds stress tensor (3x3 or 2x2)
        tke: Turbulent kinetic energy
        anisotropy: Anisotropy tensor
        invariants: II, III (anisotropy invariants)
    """
    # Remove mean (get fluctuations)
    u_prime = u - np.mean(u)
    v_prime = v - np.mean(v)
    
    if w is not None:
        w_prime = w - np.mean(w)
        # 3D Reynolds stress tensor
        tau = -rho * np.array([
            [np.mean(u_prime**2), np.mean(u_prime*v_prime), np.mean(u_prime*w_prime)],
            [np.mean(v_prime*u_prime), np.mean(v_prime**2), np.mean(v_prime*w_prime)],
            [np.mean(w_prime*u_prime), np.mean(w_prime*v_prime), np.mean(w_prime**2)]
        ])
        tke = 0.5 * (np.mean(u_prime**2) + np.mean(v_prime**2) + np.mean(w_prime**2))
    else:
        # 2D Reynolds stress tensor
        tau = -rho * np.array([
            [np.mean(u_prime**2), np.mean(u_prime*v_prime)],
            [np.mean(v_prime*u_prime), np.mean(v_prime**2)]
        ])
        tke = 0.5 * (np.mean(u_prime**2) + np.mean(v_prime**2))
    
    # Anisotropy tensor: b_ij = tau_ij/(2*rho*k) - δ_ij/3
    if tke > 0:
        b = -tau / (2 * rho * tke) - np.eye(tau.shape[0]) / 3
        # Invariants
        II = -0.5 * np.trace(b @ b)
        III = np.linalg.det(b) if b.shape[0] == 3 else 0
    else:
        b = np.zeros_like(tau)
        II, III = 0, 0
    
    return {
        'tau': tau.tolist(),
        'tke': float(tke),
        'anisotropy': b.tolist(),
        'II': float(II),
        'III': float(III),
    }
