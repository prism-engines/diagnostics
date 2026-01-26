"""
Heat Equation Solver

∂T/∂t = α ∇²T + Q/(ρ*cp)

1D, 2D, 3D diffusion with sources.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import Dict, Any, Optional


def compute(T_initial: np.ndarray, alpha: float, dt: float, dx: float,
            n_steps: int = 100, bc_left: float = None, bc_right: float = None,
            source: np.ndarray = None) -> Dict[str, Any]:
    """
    Solve 1D heat equation using implicit scheme (Crank-Nicolson).
    
    Args:
        T_initial: Initial temperature distribution
        alpha: Thermal diffusivity [m²/s]
        dt: Time step [s]
        dx: Grid spacing [m]
        n_steps: Number of time steps
        bc_left, bc_right: Dirichlet boundary conditions
        source: Heat source term [K/s]
    
    Returns:
        T_final: Final temperature distribution
        T_history: Temperature at all time steps
        max_temp, min_temp: Extremes
    """
    n = len(T_initial)
    T = T_initial.copy()
    T_history = [T.copy()]
    
    r = alpha * dt / (2 * dx**2)
    
    # Tridiagonal matrix for Crank-Nicolson
    main_diag = (1 + 2*r) * np.ones(n)
    off_diag = -r * np.ones(n-1)
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')
    
    main_diag_rhs = (1 - 2*r) * np.ones(n)
    B = diags([r*np.ones(n-1), main_diag_rhs, r*np.ones(n-1)], [-1, 0, 1], format='csr')
    
    for step in range(n_steps):
        rhs = B @ T
        
        if source is not None:
            rhs += dt * source
        
        # Apply BCs
        if bc_left is not None:
            rhs[0] = bc_left
        if bc_right is not None:
            rhs[-1] = bc_right
        
        T = spsolve(A, rhs)
        
        if bc_left is not None:
            T[0] = bc_left
        if bc_right is not None:
            T[-1] = bc_right
            
        T_history.append(T.copy())
    
    return {
        'T_final': T,
        'T_history': np.array(T_history),
        'max_temp': float(np.max(T)),
        'min_temp': float(np.min(T)),
        'mean_temp': float(np.mean(T)),
    }
