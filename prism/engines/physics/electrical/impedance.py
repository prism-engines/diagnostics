"""
Impedance Analysis

Electrochemical impedance spectroscopy (EIS), equivalent circuits.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from scipy import optimize


def impedance_randles(omega: np.ndarray, R_s: float, R_ct: float,
                      C_dl: float, sigma: float = 0) -> Dict[str, Any]:
    """
    Randles circuit impedance.

    Z = R_s + 1/(1/Z_ct + jωC_dl)
    where Z_ct = R_ct + σ/√ω - jσ/√ω (with Warburg)

    Args:
        omega: Angular frequency array [rad/s]
        R_s: Solution resistance [Ω]
        R_ct: Charge transfer resistance [Ω]
        C_dl: Double layer capacitance [F]
        sigma: Warburg coefficient [Ω/√s] (0 for no diffusion)

    Returns:
        Z_real: Real impedance [Ω]
        Z_imag: Imaginary impedance [Ω]
        Z_mag: Impedance magnitude [Ω]
        Z_phase: Phase angle [rad]
    """
    omega = np.atleast_1d(omega)

    # Warburg impedance
    if sigma > 0:
        sqrt_omega = np.sqrt(omega)
        Z_w = sigma / sqrt_omega - 1j * sigma / sqrt_omega
    else:
        Z_w = 0

    # Charge transfer with Warburg
    Z_ct = R_ct + Z_w

    # Double layer capacitor
    Z_C = 1 / (1j * omega * C_dl)

    # Parallel combination of R_ct and C_dl
    Z_parallel = 1 / (1 / Z_ct + 1 / Z_C)

    # Total impedance
    Z = R_s + Z_parallel

    return {
        'omega': omega.tolist(),
        'frequency': (omega / (2 * np.pi)).tolist(),
        'Z_real': np.real(Z).tolist(),
        'Z_imag': np.imag(Z).tolist(),
        'Z_mag': np.abs(Z).tolist(),
        'Z_phase': np.angle(Z).tolist(),
        'Z_phase_deg': np.degrees(np.angle(Z)).tolist(),
        'R_s': R_s,
        'R_ct': R_ct,
        'C_dl': C_dl,
        'sigma': sigma,
    }


def impedance_cpe(omega: np.ndarray, R_s: float, R_ct: float,
                  Q: float, n: float) -> Dict[str, Any]:
    """
    Randles circuit with Constant Phase Element (CPE).

    Z_CPE = 1 / (Q × (jω)^n)

    Args:
        omega: Angular frequency array [rad/s]
        R_s: Solution resistance [Ω]
        R_ct: Charge transfer resistance [Ω]
        Q: CPE parameter [S·s^n]
        n: CPE exponent (0-1, 1=capacitor, 0.5=Warburg)

    Returns:
        Z_real, Z_imag, Z_mag, Z_phase
    """
    omega = np.atleast_1d(omega)

    # CPE impedance
    Z_cpe = 1 / (Q * (1j * omega) ** n)

    # Parallel R_ct and CPE
    Z_parallel = 1 / (1 / R_ct + 1 / Z_cpe)

    # Total
    Z = R_s + Z_parallel

    return {
        'omega': omega.tolist(),
        'frequency': (omega / (2 * np.pi)).tolist(),
        'Z_real': np.real(Z).tolist(),
        'Z_imag': np.imag(Z).tolist(),
        'Z_mag': np.abs(Z).tolist(),
        'Z_phase': np.angle(Z).tolist(),
        'Z_phase_deg': np.degrees(np.angle(Z)).tolist(),
        'R_s': R_s,
        'R_ct': R_ct,
        'Q': Q,
        'n': n,
        'effective_C': float(Q ** (1/n) / R_ct ** ((1-n)/n)) if n > 0 else 0,
    }


def fit_randles(freq: np.ndarray, Z_real: np.ndarray, Z_imag: np.ndarray,
                initial_guess: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Fit Randles circuit to experimental EIS data.

    Args:
        freq: Frequency array [Hz]
        Z_real: Real impedance data [Ω]
        Z_imag: Imaginary impedance data [Ω]
        initial_guess: Initial parameter guesses

    Returns:
        Fitted parameters: R_s, R_ct, C_dl, sigma
        fit_quality: R-squared
    """
    freq = np.asarray(freq)
    Z_real = np.asarray(Z_real)
    Z_imag = np.asarray(Z_imag)
    omega = 2 * np.pi * freq

    Z_exp = Z_real + 1j * Z_imag

    # Initial guesses
    if initial_guess is None:
        R_s_init = Z_real[-1]  # High frequency
        R_ct_init = Z_real[0] - R_s_init  # Low frequency
        C_dl_init = 1e-5
        sigma_init = 1.0
    else:
        R_s_init = initial_guess.get('R_s', Z_real[-1])
        R_ct_init = initial_guess.get('R_ct', 100)
        C_dl_init = initial_guess.get('C_dl', 1e-5)
        sigma_init = initial_guess.get('sigma', 1.0)

    def model(params):
        R_s, R_ct, C_dl, sigma = params

        if sigma > 0:
            sqrt_omega = np.sqrt(omega)
            Z_w = sigma / sqrt_omega - 1j * sigma / sqrt_omega
        else:
            Z_w = 0

        Z_ct = R_ct + Z_w
        Z_C = 1 / (1j * omega * C_dl)
        Z_parallel = 1 / (1 / Z_ct + 1 / Z_C)
        return R_s + Z_parallel

    def residual(params):
        Z_calc = model(params)
        diff = Z_calc - Z_exp
        return np.concatenate([np.real(diff), np.imag(diff)])

    # Bounds
    bounds = ([0, 0, 1e-12, 0], [np.inf, np.inf, 1, 1000])

    try:
        result = optimize.least_squares(
            residual,
            [R_s_init, R_ct_init, C_dl_init, sigma_init],
            bounds=bounds
        )

        R_s, R_ct, C_dl, sigma = result.x

        # Calculate R-squared
        Z_fit = model(result.x)
        ss_res = np.sum(np.abs(Z_exp - Z_fit) ** 2)
        ss_tot = np.sum(np.abs(Z_exp - np.mean(Z_exp)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        success = True
    except Exception as e:
        R_s, R_ct, C_dl, sigma = R_s_init, R_ct_init, C_dl_init, sigma_init
        r_squared = 0
        success = False

    return {
        'R_s': float(R_s),
        'R_ct': float(R_ct),
        'C_dl': float(C_dl),
        'sigma': float(sigma),
        'r_squared': float(r_squared),
        'success': success,
        'time_constant': float(R_ct * C_dl),
    }


def nyquist_features(Z_real: np.ndarray, Z_imag: np.ndarray) -> Dict[str, Any]:
    """
    Extract features from Nyquist plot.

    Args:
        Z_real: Real impedance [Ω]
        Z_imag: Imaginary impedance [Ω] (typically negative)

    Returns:
        R_s: Solution resistance (high freq intercept)
        R_p: Polarization resistance (low freq intercept - R_s)
        f_max: Frequency at maximum -Z_imag
    """
    Z_real = np.asarray(Z_real)
    Z_imag = np.asarray(Z_imag)

    # High frequency limit (R_s)
    R_s = np.min(Z_real)

    # Low frequency limit (R_s + R_p)
    R_total = np.max(Z_real)
    R_p = R_total - R_s

    # Maximum -Z_imag
    idx_max = np.argmin(Z_imag)  # Most negative
    Z_real_at_max = Z_real[idx_max]
    Z_imag_max = -Z_imag[idx_max]

    # Semicircle diameter (if single arc)
    diameter = 2 * Z_imag_max

    # Semicircle center
    center_real = R_s + diameter / 2
    center_imag = 0

    return {
        'R_s': float(R_s),
        'R_p': float(R_p),
        'R_total': float(R_total),
        'Z_imag_max': float(Z_imag_max),
        'Z_real_at_max': float(Z_real_at_max),
        'semicircle_diameter': float(diameter),
        'semicircle_center': [float(center_real), float(center_imag)],
        'depressed_angle': float(np.arctan2(Z_imag_max, center_real - R_s)),
    }


def bode_features(freq: np.ndarray, Z_mag: np.ndarray,
                  Z_phase: np.ndarray) -> Dict[str, Any]:
    """
    Extract features from Bode plot.

    Args:
        freq: Frequency [Hz]
        Z_mag: Impedance magnitude [Ω]
        Z_phase: Phase angle [rad or deg]

    Returns:
        corner_freq: Corner frequency [Hz]
        low_freq_slope: Slope at low frequency
        high_freq_slope: Slope at high frequency
    """
    freq = np.asarray(freq)
    Z_mag = np.asarray(Z_mag)
    Z_phase = np.asarray(Z_phase)

    log_freq = np.log10(freq)
    log_mag = np.log10(Z_mag)

    # Find corner frequency (maximum phase change rate)
    if len(Z_phase) > 2:
        phase_deriv = np.gradient(Z_phase, log_freq)
        idx_corner = np.argmax(np.abs(phase_deriv))
        corner_freq = freq[idx_corner]
    else:
        corner_freq = freq[len(freq) // 2]

    # Slopes
    n_low = min(5, len(freq) // 4)
    n_high = min(5, len(freq) // 4)

    if n_low > 1:
        low_slope = np.polyfit(log_freq[:n_low], log_mag[:n_low], 1)[0]
    else:
        low_slope = 0

    if n_high > 1:
        high_slope = np.polyfit(log_freq[-n_high:], log_mag[-n_high:], 1)[0]
    else:
        high_slope = 0

    # DC and high frequency limits
    Z_dc = Z_mag[0]
    Z_hf = Z_mag[-1]

    return {
        'corner_freq': float(corner_freq),
        'low_freq_slope': float(low_slope),
        'high_freq_slope': float(high_slope),
        'Z_dc': float(Z_dc),
        'Z_hf': float(Z_hf),
        'phase_min': float(np.min(Z_phase)),
        'phase_max': float(np.max(Z_phase)),
    }


def compute(omega: np.ndarray = None, freq: np.ndarray = None,
            Z_real: np.ndarray = None, Z_imag: np.ndarray = None,
            R_s: float = None, R_ct: float = None,
            C_dl: float = None, sigma: float = 0,
            Q: float = None, n: float = None,
            **kwargs) -> Dict[str, Any]:
    """
    Main entry point for impedance analysis.
    """
    results = {}

    # Convert freq to omega if needed
    if omega is None and freq is not None:
        omega = 2 * np.pi * np.asarray(freq)

    # Generate model impedance
    if omega is not None and R_s is not None and R_ct is not None:
        if Q is not None and n is not None:
            results['model'] = impedance_cpe(omega, R_s, R_ct, Q, n)
        elif C_dl is not None:
            results['model'] = impedance_randles(omega, R_s, R_ct, C_dl, sigma)

    # Analyze experimental data
    if Z_real is not None and Z_imag is not None:
        Z_real = np.asarray(Z_real)
        Z_imag = np.asarray(Z_imag)

        results['nyquist_features'] = nyquist_features(Z_real, Z_imag)

        if freq is not None:
            freq = np.asarray(freq)
            Z_mag = np.sqrt(Z_real ** 2 + Z_imag ** 2)
            Z_phase = np.arctan2(Z_imag, Z_real)
            results['bode_features'] = bode_features(freq, Z_mag, Z_phase)

            # Fit Randles circuit
            results['fit'] = fit_randles(freq, Z_real, Z_imag)

    return results if results else {'error': 'Provide omega/freq and circuit parameters or Z_real/Z_imag data'}
