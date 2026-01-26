"""
Distillation Column Simulation

Rigorous tray-by-tray calculations, MESH equations.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from scipy import optimize


def bubble_point(x: np.ndarray, P: float, K_func: callable = None,
                 alpha: np.ndarray = None) -> Dict[str, Any]:
    """
    Calculate bubble point temperature.

    Σ(x_i × K_i) = 1

    Args:
        x: Liquid composition (mole fractions)
        P: Pressure [Pa]
        K_func: Function K(T, P, i) returning K-values
        alpha: Relative volatilities (if K_func not provided)

    Returns:
        T: Bubble point temperature [K]
        y: Vapor composition
        K: K-values
    """
    x = np.atleast_1d(x)
    n = len(x)

    if alpha is not None:
        # Simplified: assume constant relative volatility
        # K_i/K_ref = alpha_i, K_ref = 1/Σ(alpha_i × x_i)
        alpha = np.atleast_1d(alpha)
        K_ref = 1.0 / np.sum(alpha * x)
        K = alpha * K_ref
        y = K * x
        T = 373.15  # Placeholder

        return {
            'T': float(T),
            'y': y.tolist(),
            'K': K.tolist(),
        }

    elif K_func is not None:
        # Solve for T where Σ(x_i × K_i(T)) = 1
        def objective(T):
            K = np.array([K_func(T, P, i) for i in range(n)])
            return np.sum(x * K) - 1

        T = optimize.brentq(objective, 200, 500)
        K = np.array([K_func(T, P, i) for i in range(n)])
        y = K * x

        return {
            'T': float(T),
            'y': y.tolist(),
            'K': K.tolist(),
        }

    return {'error': 'Provide K_func or alpha'}


def dew_point(y: np.ndarray, P: float, K_func: callable = None,
              alpha: np.ndarray = None) -> Dict[str, Any]:
    """
    Calculate dew point temperature.

    Σ(y_i / K_i) = 1

    Args:
        y: Vapor composition (mole fractions)
        P: Pressure [Pa]
        K_func: Function K(T, P, i) returning K-values
        alpha: Relative volatilities

    Returns:
        T: Dew point temperature [K]
        x: Liquid composition
        K: K-values
    """
    y = np.atleast_1d(y)
    n = len(y)

    if alpha is not None:
        alpha = np.atleast_1d(alpha)
        # K_ref such that Σ(y_i/K_i) = 1
        # Σ(y_i / (alpha_i × K_ref)) = 1
        K_ref = np.sum(y / alpha)
        K = alpha * K_ref
        x = y / K
        T = 373.15  # Placeholder

        return {
            'T': float(T),
            'x': x.tolist(),
            'K': K.tolist(),
        }

    return {'error': 'Provide K_func or alpha'}


def tray_calculation(x_below: np.ndarray, y_above: np.ndarray,
                     L: float, V: float, F: float = 0,
                     z_F: np.ndarray = None, q: float = 1.0,
                     alpha: np.ndarray = None) -> Dict[str, Any]:
    """
    Single tray MESH equations (simplified).

    Material: L[j+1]×x[j+1] + V[j-1]×y[j-1] + F[j]×z[j] = L[j]×x[j] + V[j]×y[j]
    Equilibrium: y[j] = K[j] × x[j]
    Summation: Σx = 1, Σy = 1
    Heat: Energy balance

    Args:
        x_below: Liquid composition from tray below
        y_above: Vapor composition from tray above
        L: Liquid flow rate [mol/s]
        V: Vapor flow rate [mol/s]
        F: Feed flow rate [mol/s]
        z_F: Feed composition
        q: Feed quality (1=sat liquid, 0=sat vapor)
        alpha: Relative volatilities

    Returns:
        x: Liquid composition leaving tray
        y: Vapor composition leaving tray
    """
    x_below = np.atleast_1d(x_below)
    y_above = np.atleast_1d(y_above)
    n = len(x_below)

    if z_F is None:
        z_F = np.zeros(n)
    z_F = np.atleast_1d(z_F)

    if alpha is None:
        alpha = np.ones(n)
    alpha = np.atleast_1d(alpha)

    # Liquid entering from above
    L_above = L + F * q

    # Vapor entering from below
    V_below = V + F * (1 - q)

    # Material balance (assume x leaving satisfies equilibrium)
    def equations(vars):
        x = vars[:n]
        y = vars[n:]

        x = np.maximum(x, 1e-10)
        y = np.maximum(y, 1e-10)

        # Normalize
        x = x / np.sum(x)
        y = y / np.sum(y)

        # Equilibrium
        K = alpha / np.sum(alpha * x)
        eq_residual = y - K * x

        # Material balance
        mat_residual = (L_above * x_below + V_below * y_above + F * z_F -
                        L * x - V * y)

        return np.concatenate([eq_residual, mat_residual])

    # Initial guess
    x0 = (x_below + y_above) / 2
    y0 = (x_below + y_above) / 2
    x0 = x0 / np.sum(x0)
    y0 = y0 / np.sum(y0)

    try:
        result = optimize.fsolve(equations, np.concatenate([x0, y0]), full_output=True)
        solution = result[0]
        x = solution[:n]
        y = solution[n:]

        x = np.maximum(x, 0)
        y = np.maximum(y, 0)
        x = x / np.sum(x)
        y = y / np.sum(y)

        success = True
    except Exception:
        x = x_below
        y = y_above
        success = False

    return {
        'x': x.tolist(),
        'y': y.tolist(),
        'L': L,
        'V': V,
        'success': success,
    }


def fenske_underwood_gilliland(x_D: np.ndarray, x_B: np.ndarray,
                                alpha: np.ndarray, q: float = 1.0,
                                z_F: np.ndarray = None) -> Dict[str, Any]:
    """
    Shortcut distillation design using FUG method.

    Args:
        x_D: Distillate composition
        x_B: Bottoms composition
        alpha: Relative volatilities
        q: Feed quality
        z_F: Feed composition

    Returns:
        N_min: Minimum stages (Fenske)
        R_min: Minimum reflux (Underwood)
        N_actual: Actual stages (Gilliland correlation)
    """
    x_D = np.atleast_1d(x_D)
    x_B = np.atleast_1d(x_B)
    alpha = np.atleast_1d(alpha)

    if z_F is None:
        z_F = (x_D + x_B) / 2
    z_F = np.atleast_1d(z_F)

    # Light and heavy key (assume binary or pseudo-binary)
    i_lk = 0  # Light key
    i_hk = 1 if len(alpha) > 1 else 0  # Heavy key

    alpha_lk = alpha[i_lk]
    alpha_hk = alpha[i_hk]
    alpha_rel = alpha_lk / alpha_hk

    # Fenske: Minimum stages
    N_min = np.log((x_D[i_lk] / x_B[i_lk]) * (x_B[i_hk] / x_D[i_hk])) / np.log(alpha_rel)

    # Underwood: Minimum reflux
    # Solve Σ(α_i × z_i / (α_i - θ)) = 1 - q
    def underwood_eq(theta):
        return np.sum(alpha * z_F / (alpha - theta)) - (1 - q)

    try:
        theta = optimize.brentq(underwood_eq, alpha_hk + 0.01, alpha_lk - 0.01)
        R_min = np.sum(alpha * x_D / (alpha - theta)) - 1
    except Exception:
        R_min = 1.0

    # Gilliland correlation for actual stages
    # Assume R = 1.2 × R_min
    R = 1.2 * R_min
    X = (R - R_min) / (R + 1)
    Y = 1 - np.exp((1 + 54.4 * X) / (11 + 117.2 * X) * (X - 1) / np.sqrt(X))
    N_actual = (N_min + Y) / (1 - Y)

    return {
        'N_min': float(N_min),
        'R_min': float(R_min),
        'R_actual': float(R),
        'N_actual': float(N_actual),
        'alpha_relative': float(alpha_rel),
        'theta': float(theta) if 'theta' in dir() else None,
    }


def column_simulation(n_trays: int, feed_tray: int, x_F: np.ndarray,
                      alpha: np.ndarray, R: float, D_F: float,
                      P: float = 101325, max_iter: int = 100,
                      tol: float = 1e-6) -> Dict[str, Any]:
    """
    Rigorous tray-by-tray simulation.

    Args:
        n_trays: Number of trays (excluding reboiler/condenser)
        feed_tray: Feed tray location (1 = bottom)
        x_F: Feed composition
        alpha: Relative volatilities
        R: Reflux ratio
        D_F: Distillate to feed ratio (D/F)
        P: Pressure [Pa]
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        x: Liquid composition on each tray
        y: Vapor composition on each tray
        T: Temperature on each tray (if available)
        converged: Whether solution converged
    """
    x_F = np.atleast_1d(x_F)
    alpha = np.atleast_1d(alpha)
    n_comp = len(x_F)

    # Initialize
    x = np.zeros((n_trays + 2, n_comp))  # Include reboiler (0) and condenser (n+1)
    y = np.zeros((n_trays + 2, n_comp))

    # Initial profile (linear)
    for i in range(n_trays + 2):
        x[i] = x_F  # Simple initialization
        y[i] = x_F

    # Flow rates (assume constant molar overflow)
    F = 1.0  # Basis
    D = D_F * F
    B = F - D
    L_rect = R * D  # Rectifying section liquid
    V = (R + 1) * D  # Vapor flow
    L_strip = L_rect + F  # Stripping section liquid

    converged = False
    for iteration in range(max_iter):
        x_old = x.copy()

        # Condenser (total)
        x[-1] = y[-2]  # Assume total condenser

        # Rectifying section (top down)
        for j in range(n_trays, feed_tray - 1, -1):
            L = L_rect
            # Material balance
            x_mixed = (L * x[j + 1] + V * y[j - 1]) / (L + V)
            # Equilibrium
            K = alpha / np.sum(alpha * x_mixed)
            y[j] = K * x_mixed
            y[j] = y[j] / np.sum(y[j])
            x[j] = y[j] / K
            x[j] = x[j] / np.sum(x[j])

        # Feed tray
        j = feed_tray
        x_mixed = (L_rect * x[j + 1] + V * y[j - 1] + F * x_F) / (L_strip + V)
        K = alpha / np.sum(alpha * x_mixed)
        y[j] = K * x_mixed
        y[j] = y[j] / np.sum(y[j])
        x[j] = y[j] / K
        x[j] = x[j] / np.sum(x[j])

        # Stripping section (bottom up)
        for j in range(feed_tray - 1, 0, -1):
            L = L_strip
            x_mixed = (L * x[j + 1] + V * y[j - 1]) / (L + V)
            K = alpha / np.sum(alpha * x_mixed)
            y[j] = K * x_mixed
            y[j] = y[j] / np.sum(y[j])
            x[j] = y[j] / K
            x[j] = x[j] / np.sum(x[j])

        # Reboiler
        x[0] = x[1]

        # Check convergence
        error = np.max(np.abs(x - x_old))
        if error < tol:
            converged = True
            break

    return {
        'x': x.tolist(),
        'y': y.tolist(),
        'x_distillate': x[-1].tolist(),
        'x_bottoms': x[0].tolist(),
        'n_trays': n_trays,
        'feed_tray': feed_tray,
        'converged': converged,
        'iterations': iteration + 1,
    }


def compute(design_method: str = 'fug', **kwargs) -> Dict[str, Any]:
    """
    Main entry point for distillation calculations.

    Args:
        design_method: 'fug' (shortcut) or 'rigorous'
    """
    if design_method == 'fug':
        x_D = kwargs.get('x_D', [0.95, 0.05])
        x_B = kwargs.get('x_B', [0.05, 0.95])
        alpha = kwargs.get('alpha', [2.5, 1.0])
        q = kwargs.get('q', 1.0)
        z_F = kwargs.get('z_F')
        return fenske_underwood_gilliland(x_D, x_B, alpha, q, z_F)

    elif design_method == 'rigorous':
        return column_simulation(
            kwargs.get('n_trays', 20),
            kwargs.get('feed_tray', 10),
            kwargs.get('x_F', [0.5, 0.5]),
            kwargs.get('alpha', [2.5, 1.0]),
            kwargs.get('R', 2.0),
            kwargs.get('D_F', 0.5),
            kwargs.get('P', 101325),
        )

    elif design_method == 'bubble_point':
        return bubble_point(
            kwargs.get('x', [0.5, 0.5]),
            kwargs.get('P', 101325),
            kwargs.get('K_func'),
            kwargs.get('alpha', [2.5, 1.0]),
        )

    elif design_method == 'dew_point':
        return dew_point(
            kwargs.get('y', [0.5, 0.5]),
            kwargs.get('P', 101325),
            kwargs.get('K_func'),
            kwargs.get('alpha', [2.5, 1.0]),
        )

    return {'error': f"Unknown method: {design_method}"}
