"""
Vorticity Analysis

ω = ∇ × v

Vortex identification, enstrophy, helicity.
"""

import numpy as np
from typing import Dict, Any, Optional


def compute(u: np.ndarray, v: np.ndarray, w: np.ndarray = None,
            dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> Dict[str, Any]:
    """
    Compute vorticity from velocity field.
    
    Args:
        u, v, w: Velocity components (2D or 3D arrays)
        dx, dy, dz: Grid spacing
    
    Returns:
        omega: Vorticity components
        enstrophy: 0.5 * <ω·ω>
        helicity: <v·ω>
        q_criterion: Q = 0.5*(||Ω||² - ||S||²)
    """
    if w is None:
        # 2D: ω_z = ∂v/∂x - ∂u/∂y
        dvdx = np.gradient(v, dx, axis=1) if v.ndim > 1 else np.gradient(v, dx)
        dudy = np.gradient(u, dy, axis=0) if u.ndim > 1 else 0
        omega_z = dvdx - dudy
        
        enstrophy = 0.5 * np.mean(omega_z**2)
        
        return {
            'omega_z': omega_z,
            'enstrophy': float(enstrophy),
            'max_vorticity': float(np.max(np.abs(omega_z))),
        }
    else:
        # 3D vorticity
        dudy = np.gradient(u, dy, axis=1)
        dudz = np.gradient(u, dz, axis=2)
        dvdx = np.gradient(v, dx, axis=0)
        dvdz = np.gradient(v, dz, axis=2)
        dwdx = np.gradient(w, dx, axis=0)
        dwdy = np.gradient(w, dy, axis=1)
        
        omega_x = dwdy - dvdz
        omega_y = dudz - dwdx
        omega_z = dvdx - dudy
        
        omega_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        enstrophy = 0.5 * np.mean(omega_mag**2)
        helicity = np.mean(u*omega_x + v*omega_y + w*omega_z)
        
        return {
            'omega_x': omega_x,
            'omega_y': omega_y,
            'omega_z': omega_z,
            'enstrophy': float(enstrophy),
            'helicity': float(helicity),
            'max_vorticity': float(np.max(omega_mag)),
        }
