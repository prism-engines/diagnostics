"""
Rotor Dynamics

Critical speeds, unbalance response, stability analysis.
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy import linalg


def critical_speeds(m: np.ndarray, k: np.ndarray,
                    rpm_range: tuple = (0, 10000)) -> Dict[str, Any]:
    """
    Calculate critical speeds from rotor model.

    Critical speed = natural frequency at that speed.

    Args:
        m: Mass matrix or array of station masses [kg]
        k: Stiffness matrix or array of bearing stiffnesses [N/m]
        rpm_range: Range to search for critical speeds

    Returns:
        critical_speeds: List of critical speeds [RPM]
        mode_shapes: Corresponding mode shapes
    """
    m = np.atleast_1d(m)
    k = np.atleast_1d(k)

    # Simple Jeffcott rotor model if scalar
    if m.size == 1 and k.size == 1:
        omega_n = np.sqrt(k[0] / m[0])
        critical_rpm = omega_n * 60 / (2 * np.pi)
        return {
            'critical_speeds': [float(critical_rpm)],
            'natural_frequencies': [float(omega_n)],
            'mode_shapes': [[1.0]],
            'model_type': 'jeffcott',
        }

    # Multi-DOF model
    M = np.diag(m) if m.ndim == 1 else m
    K = np.diag(k) if k.ndim == 1 else k

    eigenvalues, eigenvectors = linalg.eigh(K, M)
    omega_n = np.sqrt(np.maximum(eigenvalues, 0))
    critical_rpm = omega_n * 60 / (2 * np.pi)

    # Filter to range
    mask = (critical_rpm >= rpm_range[0]) & (critical_rpm <= rpm_range[1])

    return {
        'critical_speeds': critical_rpm[mask].tolist(),
        'natural_frequencies': omega_n[mask].tolist(),
        'mode_shapes': eigenvectors[:, mask].T.tolist(),
        'model_type': 'multi_dof',
        'n_dof': len(m),
    }


def unbalance_response(m: float, k: float, c: float, e: float,
                       omega: np.ndarray) -> Dict[str, Any]:
    """
    Steady-state unbalance response (Jeffcott rotor).

    |X| = m·e·ω² / √[(k - m·ω²)² + (c·ω)²]

    Args:
        m: Rotor mass [kg]
        k: Bearing stiffness [N/m]
        c: Damping coefficient [N·s/m]
        e: Mass eccentricity [m]
        omega: Angular velocity array [rad/s]

    Returns:
        amplitude: Response amplitude [m]
        phase: Phase angle [rad]
        critical_speed: Undamped critical speed [rad/s]
    """
    omega = np.atleast_1d(omega)
    omega_n = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(k * m))

    # Frequency ratio
    r = omega / omega_n

    # Normalized amplitude
    X_normalized = r ** 2 / np.sqrt((1 - r ** 2) ** 2 + (2 * zeta * r) ** 2)

    # Actual amplitude
    amplitude = m * e / m * X_normalized  # = e * X_normalized

    # Phase
    phase = np.arctan2(2 * zeta * r, 1 - r ** 2)

    # Resonance amplitude
    X_resonance = e / (2 * zeta) if zeta > 0 else float('inf')

    return {
        'omega': omega.tolist(),
        'amplitude': amplitude.tolist(),
        'phase': phase.tolist(),
        'critical_speed': float(omega_n),
        'critical_speed_rpm': float(omega_n * 60 / (2 * np.pi)),
        'damping_ratio': float(zeta),
        'resonance_amplitude': float(X_resonance),
        'amplification_factor': float(1 / (2 * zeta)) if zeta > 0 else float('inf'),
    }


def orbit_analysis(x: np.ndarray, y: np.ndarray, fs: float,
                   shaft_rpm: float) -> Dict[str, Any]:
    """
    Analyze shaft orbit from two perpendicular probes.

    Args:
        x: Horizontal displacement [m]
        y: Vertical displacement [m]
        fs: Sampling frequency [Hz]
        shaft_rpm: Shaft speed [RPM]

    Returns:
        major_axis: Major axis of orbit ellipse [m]
        minor_axis: Minor axis [m]
        ellipticity: minor/major ratio
        orientation: Major axis angle [rad]
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Synchronous filtering at shaft frequency
    shaft_freq = shaft_rpm / 60
    n = len(x)
    t = np.arange(n) / fs

    # Extract 1X component (fundamental)
    cos_1x = np.cos(2 * np.pi * shaft_freq * t)
    sin_1x = np.sin(2 * np.pi * shaft_freq * t)

    x_cos = 2 * np.mean(x * cos_1x)
    x_sin = 2 * np.mean(x * sin_1x)
    y_cos = 2 * np.mean(y * cos_1x)
    y_sin = 2 * np.mean(y * sin_1x)

    # 1X amplitude and phase
    x_amp = np.sqrt(x_cos ** 2 + x_sin ** 2)
    y_amp = np.sqrt(y_cos ** 2 + y_sin ** 2)
    x_phase = np.arctan2(x_sin, x_cos)
    y_phase = np.arctan2(y_sin, y_cos)

    # Orbit ellipse parameters
    phase_diff = y_phase - x_phase

    # Semi-axes
    a = np.sqrt(0.5 * (x_amp ** 2 + y_amp ** 2 +
                       np.sqrt((x_amp ** 2 - y_amp ** 2) ** 2 +
                              (2 * x_amp * y_amp * np.cos(phase_diff)) ** 2)))

    b = np.sqrt(0.5 * (x_amp ** 2 + y_amp ** 2 -
                       np.sqrt((x_amp ** 2 - y_amp ** 2) ** 2 +
                              (2 * x_amp * y_amp * np.cos(phase_diff)) ** 2)))

    # Orientation
    theta = 0.5 * np.arctan2(2 * x_amp * y_amp * np.cos(phase_diff),
                             x_amp ** 2 - y_amp ** 2)

    # Direction (forward or backward whirl)
    direction = 'forward' if np.sin(phase_diff) > 0 else 'backward'

    return {
        'major_axis': float(a),
        'minor_axis': float(b),
        'ellipticity': float(b / a) if a > 0 else 0,
        'orientation': float(theta),
        'orientation_deg': float(np.degrees(theta)),
        'x_amplitude': float(x_amp),
        'y_amplitude': float(y_amp),
        'phase_difference': float(phase_diff),
        'whirl_direction': direction,
    }


def stability_analysis(m: float, k: float, c: float,
                       k_cross: float = 0, c_cross: float = 0) -> Dict[str, Any]:
    """
    Rotor stability analysis with cross-coupled stiffness.

    Instability onset when real part of eigenvalue > 0.

    Args:
        m: Rotor mass [kg]
        k: Direct stiffness [N/m]
        c: Direct damping [N·s/m]
        k_cross: Cross-coupled stiffness [N/m]
        c_cross: Cross-coupled damping [N·s/m]

    Returns:
        threshold_speed: Onset of instability [rad/s]
        stable: Whether currently stable
        log_decrement: Measure of stability margin
    """
    omega_n = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(k * m))

    # Threshold for instability (simplified model)
    # Instability when k_cross > c * omega
    if k_cross <= 0:
        threshold = float('inf')
        stable = True
    else:
        threshold = c * omega_n / k_cross if k_cross > 0 else float('inf')
        stable = True  # Would need operating speed to determine

    # Log decrement (stability measure)
    log_dec = 2 * np.pi * zeta / np.sqrt(1 - zeta ** 2) if zeta < 1 else float('inf')

    return {
        'natural_frequency': float(omega_n),
        'damping_ratio': float(zeta),
        'log_decrement': float(log_dec),
        'threshold_speed': float(threshold),
        'threshold_speed_rpm': float(threshold * 60 / (2 * np.pi)) if threshold < float('inf') else float('inf'),
        'stable': stable,
        'stability_margin': float(log_dec / (2 * np.pi)) if log_dec < float('inf') else float('inf'),
    }


def compute(m: float = None, k: float = None, c: float = 0,
            e: float = None, omega: np.ndarray = None,
            x: np.ndarray = None, y: np.ndarray = None,
            fs: float = None, shaft_rpm: float = None,
            **kwargs) -> Dict[str, Any]:
    """
    Main entry point for rotor dynamics analysis.
    """
    results = {}

    # Critical speeds
    if m is not None and k is not None:
        m_arr = np.atleast_1d(m)
        k_arr = np.atleast_1d(k)
        results['critical_speeds'] = critical_speeds(m_arr, k_arr)

        # Unbalance response
        if e is not None and omega is not None:
            results['unbalance_response'] = unbalance_response(
                float(m_arr[0]), float(k_arr[0]), c, e, omega)

        # Stability
        results['stability'] = stability_analysis(
            float(m_arr[0]), float(k_arr[0]), c,
            kwargs.get('k_cross', 0), kwargs.get('c_cross', 0))

    # Orbit analysis
    if x is not None and y is not None and fs is not None and shaft_rpm is not None:
        results['orbit'] = orbit_analysis(x, y, fs, shaft_rpm)

    return results if results else {'error': 'Provide rotor parameters (m, k) or orbit data (x, y, fs, shaft_rpm)'}
