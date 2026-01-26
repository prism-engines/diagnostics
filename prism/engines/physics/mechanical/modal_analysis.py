"""
Modal Analysis

Eigenvalue analysis for structural vibration modes.
Natural frequencies, mode shapes, modal participation factors.
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy import linalg


def compute(M: np.ndarray, K: np.ndarray, C: np.ndarray = None,
            n_modes: int = None) -> Dict[str, Any]:
    """
    Compute natural frequencies and mode shapes.

    Solves: (K - ω²M)φ = 0

    Args:
        M: Mass matrix [kg] (n x n)
        K: Stiffness matrix [N/m] (n x n)
        C: Damping matrix [N·s/m] (n x n), optional
        n_modes: Number of modes to return (default: all)

    Returns:
        natural_frequencies: ωn [rad/s]
        frequencies_hz: fn [Hz]
        mode_shapes: Normalized eigenvectors
        modal_mass: Generalized mass for each mode
        modal_stiffness: Generalized stiffness
    """
    M = np.atleast_2d(M)
    K = np.atleast_2d(K)
    n = M.shape[0]

    if n_modes is None:
        n_modes = n

    # Solve generalized eigenvalue problem
    eigenvalues, eigenvectors = linalg.eigh(K, M)

    # Natural frequencies (eigenvalues are ω²)
    omega_n = np.sqrt(np.maximum(eigenvalues, 0))
    freq_hz = omega_n / (2 * np.pi)

    # Sort by frequency
    idx = np.argsort(omega_n)[:n_modes]
    omega_n = omega_n[idx]
    freq_hz = freq_hz[idx]
    mode_shapes = eigenvectors[:, idx]

    # Mass-normalize mode shapes
    for i in range(n_modes):
        phi = mode_shapes[:, i]
        modal_mass_i = phi @ M @ phi
        if modal_mass_i > 0:
            mode_shapes[:, i] = phi / np.sqrt(modal_mass_i)

    # Compute modal properties
    modal_mass = np.array([mode_shapes[:, i] @ M @ mode_shapes[:, i]
                          for i in range(n_modes)])
    modal_stiffness = np.array([mode_shapes[:, i] @ K @ mode_shapes[:, i]
                               for i in range(n_modes)])

    # Damping ratios if C provided
    if C is not None:
        C = np.atleast_2d(C)
        modal_damping = np.array([mode_shapes[:, i] @ C @ mode_shapes[:, i]
                                 for i in range(n_modes)])
        zeta = modal_damping / (2 * omega_n * modal_mass + 1e-10)
        damped_freq = omega_n * np.sqrt(1 - zeta**2)
    else:
        zeta = np.zeros(n_modes)
        damped_freq = omega_n

    return {
        'natural_frequencies': omega_n.tolist(),
        'frequencies_hz': freq_hz.tolist(),
        'mode_shapes': mode_shapes.tolist(),
        'modal_mass': modal_mass.tolist(),
        'modal_stiffness': modal_stiffness.tolist(),
        'damping_ratios': zeta.tolist(),
        'damped_frequencies': damped_freq.tolist(),
        'n_dof': n,
        'n_modes': n_modes,
    }


def modal_participation(mode_shapes: np.ndarray, M: np.ndarray,
                        direction: np.ndarray = None) -> Dict[str, Any]:
    """
    Compute modal participation factors.

    How much each mode participates in response to excitation.

    Args:
        mode_shapes: Mode shape matrix (n x n_modes)
        M: Mass matrix
        direction: Excitation direction vector (default: all 1s)

    Returns:
        participation_factors: Γ for each mode
        effective_mass: Modal effective mass
        cumulative_mass: Cumulative effective mass ratio
    """
    mode_shapes = np.atleast_2d(mode_shapes)
    M = np.atleast_2d(M)
    n = M.shape[0]
    n_modes = mode_shapes.shape[1]

    if direction is None:
        direction = np.ones(n)

    total_mass = np.sum(np.diag(M))

    # Participation factors
    gamma = np.zeros(n_modes)
    eff_mass = np.zeros(n_modes)

    for i in range(n_modes):
        phi = mode_shapes[:, i]
        m_modal = phi @ M @ phi
        if m_modal > 0:
            gamma[i] = (phi @ M @ direction) / m_modal
            eff_mass[i] = gamma[i]**2 * m_modal

    # Cumulative mass ratio
    cumulative = np.cumsum(eff_mass) / total_mass

    return {
        'participation_factors': gamma.tolist(),
        'effective_mass': eff_mass.tolist(),
        'effective_mass_ratio': (eff_mass / total_mass).tolist(),
        'cumulative_mass_ratio': cumulative.tolist(),
        'total_mass': float(total_mass),
    }


def frequency_response(omega: np.ndarray, M: np.ndarray, K: np.ndarray,
                       C: np.ndarray, F: np.ndarray,
                       dof_out: int = 0) -> Dict[str, Any]:
    """
    Compute frequency response function (FRF).

    H(ω) = (K - ω²M + jωC)⁻¹

    Args:
        omega: Frequency range [rad/s]
        M, K, C: Mass, stiffness, damping matrices
        F: Force amplitude vector
        dof_out: Output DOF to track

    Returns:
        magnitude: |H(ω)|
        phase: arg(H(ω)) [rad]
        real: Re(H)
        imag: Im(H)
    """
    omega = np.atleast_1d(omega)
    n_freq = len(omega)

    H_mag = np.zeros(n_freq)
    H_phase = np.zeros(n_freq)
    H_real = np.zeros(n_freq)
    H_imag = np.zeros(n_freq)

    for i, w in enumerate(omega):
        # Dynamic stiffness matrix
        D = K - w**2 * M + 1j * w * C

        try:
            H = np.linalg.solve(D, F)
            H_out = H[dof_out]

            H_mag[i] = np.abs(H_out)
            H_phase[i] = np.angle(H_out)
            H_real[i] = np.real(H_out)
            H_imag[i] = np.imag(H_out)
        except np.linalg.LinAlgError:
            H_mag[i] = np.nan
            H_phase[i] = np.nan
            H_real[i] = np.nan
            H_imag[i] = np.nan

    return {
        'omega': omega.tolist(),
        'magnitude': H_mag.tolist(),
        'phase': H_phase.tolist(),
        'real': H_real.tolist(),
        'imag': H_imag.tolist(),
        'magnitude_dB': (20 * np.log10(H_mag + 1e-20)).tolist(),
    }
