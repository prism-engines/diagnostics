"""
Stability Analysis

Lyapunov stability, Routh-Hurwitz, Nyquist criterion.
"""

import numpy as np
from typing import Dict, Any, List, Optional


def routh_hurwitz(coefficients: List[float]) -> Dict[str, Any]:
    """
    Routh-Hurwitz stability criterion.

    For polynomial: a_n*s^n + a_{n-1}*s^{n-1} + ... + a_1*s + a_0 = 0

    Args:
        coefficients: Polynomial coefficients [a_n, a_{n-1}, ..., a_1, a_0]

    Returns:
        stable: Boolean stability result
        routh_array: The Routh array
        sign_changes: Number of sign changes (= RHP poles)
    """
    coeffs = np.array(coefficients, dtype=float)
    n = len(coeffs) - 1

    if n < 1:
        return {'stable': True, 'routh_array': [[coeffs[0]]], 'sign_changes': 0}

    # Build Routh array
    rows = n + 1
    cols = (n + 2) // 2
    routh = np.zeros((rows, cols))

    # First two rows
    routh[0, :] = coeffs[::2][:cols]
    routh[1, :len(coeffs[1::2])] = coeffs[1::2]

    # Compute remaining rows
    for i in range(2, rows):
        for j in range(cols - 1):
            a = routh[i - 1, 0]
            if abs(a) < 1e-10:
                # Handle zero in first column
                a = 1e-10

            b = routh[i - 2, j + 1] if j + 1 < cols else 0
            c = routh[i - 1, j + 1] if j + 1 < cols else 0
            d = routh[i - 2, 0]

            routh[i, j] = (a * b - d * c) / a

    # Count sign changes in first column
    first_col = routh[:, 0]
    sign_changes = np.sum(np.diff(np.sign(first_col)) != 0)

    stable = sign_changes == 0 and np.all(first_col > 0)

    return {
        'stable': bool(stable),
        'routh_array': routh.tolist(),
        'first_column': first_col.tolist(),
        'sign_changes': int(sign_changes),
        'rhp_poles': int(sign_changes),
    }


def nyquist_stability(num: List[float], den: List[float],
                      omega: np.ndarray = None) -> Dict[str, Any]:
    """
    Nyquist stability criterion.

    N = P - Z, where N = encirclements of -1, P = open-loop RHP poles,
    Z = closed-loop RHP poles.

    Args:
        num: Numerator coefficients
        den: Denominator coefficients
        omega: Frequency range [rad/s]

    Returns:
        stable: Closed-loop stability
        encirclements: Number of -1 encirclements
        open_loop_unstable_poles: Number of RHP poles in open-loop
    """
    from scipy import signal as sig

    num = np.atleast_1d(num)
    den = np.atleast_1d(den)

    # Open-loop poles
    poles = np.roots(den)
    P = np.sum(np.real(poles) > 0)

    # Generate frequency response
    if omega is None:
        omega = np.logspace(-3, 3, 10000)

    system = sig.TransferFunction(num, den)
    _, H = sig.freqresp(system, omega)

    # Count encirclements of -1
    # Simple method: count crossings of negative real axis
    real_part = np.real(H)
    imag_part = np.imag(H)

    # Find crossings where real < -1 and imag changes sign
    N = 0
    for i in range(len(omega) - 1):
        if real_part[i] < -1 or real_part[i + 1] < -1:
            if imag_part[i] * imag_part[i + 1] < 0:
                # Sign change in imaginary part
                if imag_part[i] > 0:
                    N += 1  # Counterclockwise
                else:
                    N -= 1  # Clockwise

    Z = P - N
    stable = Z == 0

    # Gain margin (distance from -1)
    min_distance = np.min(np.abs(H + 1))

    return {
        'stable': bool(stable),
        'encirclements': int(N),
        'open_loop_unstable_poles': int(P),
        'closed_loop_unstable_poles': int(Z),
        'min_distance_to_minus_one': float(min_distance),
        'nyquist_real': np.real(H).tolist(),
        'nyquist_imag': np.imag(H).tolist(),
    }


def lyapunov_stability_linear(A: np.ndarray) -> Dict[str, Any]:
    """
    Lyapunov stability analysis for linear system dx/dt = Ax.

    Solves A'P + PA = -Q for P (Q = I).
    System is stable if P is positive definite.

    Args:
        A: System matrix (n x n)

    Returns:
        stable: Boolean stability
        P: Lyapunov matrix
        eigenvalues_P: Eigenvalues of P (all positive if stable)
    """
    from scipy import linalg

    A = np.atleast_2d(A)
    n = A.shape[0]

    # Eigenvalues of A
    eig_A = np.linalg.eigvals(A)
    stable_eigenvalues = np.all(np.real(eig_A) < 0)

    # Solve Lyapunov equation A'P + PA = -Q
    Q = np.eye(n)
    try:
        P = linalg.solve_continuous_lyapunov(A.T, -Q)

        # Check positive definiteness
        eig_P = np.linalg.eigvals(P)
        positive_definite = np.all(np.real(eig_P) > 0)

        stable = positive_definite
    except Exception:
        P = np.zeros((n, n))
        eig_P = np.zeros(n)
        positive_definite = False
        stable = stable_eigenvalues

    return {
        'stable': bool(stable),
        'P': P.tolist(),
        'eigenvalues_P': np.real(eig_P).tolist(),
        'positive_definite': bool(positive_definite),
        'eigenvalues_A': np.real(eig_A).tolist(),
        'eigenvalues_A_imag': np.imag(eig_A).tolist(),
        'max_real_eigenvalue': float(np.max(np.real(eig_A))),
    }


def pole_placement(A: np.ndarray, B: np.ndarray,
                   desired_poles: List[complex]) -> Dict[str, Any]:
    """
    State feedback pole placement.

    Find K such that eig(A - BK) = desired_poles.

    Args:
        A: System matrix (n x n)
        B: Input matrix (n x m)
        desired_poles: Desired closed-loop poles

    Returns:
        K: Feedback gain matrix
        achieved_poles: Actual closed-loop poles
    """
    from scipy import signal as sig

    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    desired_poles = np.atleast_1d(desired_poles)

    try:
        K = sig.place_poles(A, B, desired_poles).gain_matrix

        # Verify poles
        A_cl = A - B @ K
        achieved_poles = np.linalg.eigvals(A_cl)

        # Controllability check
        n = A.shape[0]
        C = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
        controllable = np.linalg.matrix_rank(C) == n

        success = True
    except Exception as e:
        K = np.zeros((B.shape[1], A.shape[0]))
        achieved_poles = np.linalg.eigvals(A)
        controllable = False
        success = False

    return {
        'K': K.tolist(),
        'desired_poles_real': np.real(desired_poles).tolist(),
        'desired_poles_imag': np.imag(desired_poles).tolist(),
        'achieved_poles_real': np.real(achieved_poles).tolist(),
        'achieved_poles_imag': np.imag(achieved_poles).tolist(),
        'controllable': bool(controllable),
        'success': success,
    }


def margins(num: List[float], den: List[float]) -> Dict[str, Any]:
    """
    Calculate stability margins.

    Args:
        num: Numerator coefficients
        den: Denominator coefficients

    Returns:
        gain_margin: Gain margin [dB]
        phase_margin: Phase margin [deg]
        gain_crossover: Gain crossover frequency [rad/s]
        phase_crossover: Phase crossover frequency [rad/s]
    """
    from scipy import signal as sig

    num = np.atleast_1d(num)
    den = np.atleast_1d(den)

    system = sig.TransferFunction(num, den)
    omega = np.logspace(-3, 3, 10000)
    _, H = sig.freqresp(system, omega)

    magnitude_db = 20 * np.log10(np.abs(H) + 1e-20)
    phase_deg = np.degrees(np.unwrap(np.angle(H)))

    # Gain crossover (|H| = 1, magnitude = 0 dB)
    gain_cross_idx = np.where(np.diff(np.sign(magnitude_db)))[0]
    if len(gain_cross_idx) > 0:
        omega_gc = omega[gain_cross_idx[0]]
        phase_at_gc = phase_deg[gain_cross_idx[0]]
        phase_margin = 180 + phase_at_gc
    else:
        omega_gc = np.nan
        phase_margin = np.nan

    # Phase crossover (phase = -180)
    phase_cross_idx = np.where(np.diff(np.sign(phase_deg + 180)))[0]
    if len(phase_cross_idx) > 0:
        omega_pc = omega[phase_cross_idx[0]]
        mag_at_pc = magnitude_db[phase_cross_idx[0]]
        gain_margin = -mag_at_pc
    else:
        omega_pc = np.nan
        gain_margin = np.inf

    return {
        'gain_margin_db': float(gain_margin),
        'phase_margin_deg': float(phase_margin),
        'gain_crossover_freq': float(omega_gc),
        'phase_crossover_freq': float(omega_pc),
        'stable': bool(gain_margin > 0 and phase_margin > 0),
    }


def compute(coefficients: List[float] = None, num: List[float] = None,
            den: List[float] = None, A: np.ndarray = None,
            B: np.ndarray = None, desired_poles: List[complex] = None,
            **kwargs) -> Dict[str, Any]:
    """
    Main entry point for stability analysis.
    """
    results = {}

    # Routh-Hurwitz for characteristic polynomial
    if coefficients is not None:
        results['routh_hurwitz'] = routh_hurwitz(coefficients)

    # Transfer function analysis
    if num is not None and den is not None:
        results['nyquist'] = nyquist_stability(num, den)
        results['margins'] = margins(num, den)

    # State space analysis
    if A is not None:
        results['lyapunov'] = lyapunov_stability_linear(A)

        if B is not None and desired_poles is not None:
            results['pole_placement'] = pole_placement(A, B, desired_poles)

    return results if results else {'error': 'Provide coefficients, num/den, or A matrix'}
