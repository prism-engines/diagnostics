"""
Transfer Function Analysis

Frequency response, step response, system identification.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from scipy import signal as sig


def frequency_response(num: List[float], den: List[float],
                       omega: np.ndarray = None) -> Dict[str, Any]:
    """
    Compute frequency response H(jω).

    Args:
        num: Numerator coefficients [b0, b1, ..., bm]
        den: Denominator coefficients [a0, a1, ..., an]
        omega: Frequency range [rad/s] (default: auto)

    Returns:
        magnitude: |H(jω)| [dB]
        phase: ∠H(jω) [deg]
        omega: Frequency points
    """
    num = np.atleast_1d(num)
    den = np.atleast_1d(den)

    if omega is None:
        # Auto-generate frequency range
        system = sig.TransferFunction(num, den)
        omega, H = sig.freqresp(system)
    else:
        omega = np.atleast_1d(omega)
        _, H = sig.freqresp(sig.TransferFunction(num, den), omega)

    magnitude_db = 20 * np.log10(np.abs(H) + 1e-20)
    phase_deg = np.degrees(np.unwrap(np.angle(H)))

    # Bandwidth (-3dB point)
    dc_gain_db = magnitude_db[0]
    bw_idx = np.where(magnitude_db < dc_gain_db - 3)[0]
    bandwidth = omega[bw_idx[0]] if len(bw_idx) > 0 else omega[-1]

    # Gain and phase margins
    gain_margin = np.nan
    phase_margin = np.nan

    # Phase crossover (where phase = -180)
    phase_cross_idx = np.where(np.diff(np.sign(phase_deg + 180)))[0]
    if len(phase_cross_idx) > 0:
        gain_margin = -magnitude_db[phase_cross_idx[0]]

    # Gain crossover (where magnitude = 0 dB)
    gain_cross_idx = np.where(np.diff(np.sign(magnitude_db)))[0]
    if len(gain_cross_idx) > 0:
        phase_margin = 180 + phase_deg[gain_cross_idx[0]]

    return {
        'omega': omega.tolist(),
        'frequency_hz': (omega / (2 * np.pi)).tolist(),
        'magnitude_db': magnitude_db.tolist(),
        'magnitude': np.abs(H).tolist(),
        'phase_deg': phase_deg.tolist(),
        'phase_rad': np.angle(H).tolist(),
        'bandwidth': float(bandwidth),
        'bandwidth_hz': float(bandwidth / (2 * np.pi)),
        'dc_gain': float(10 ** (dc_gain_db / 20)),
        'dc_gain_db': float(dc_gain_db),
        'gain_margin_db': float(gain_margin),
        'phase_margin_deg': float(phase_margin),
    }


def step_response(num: List[float], den: List[float],
                  t: np.ndarray = None, n_points: int = 500) -> Dict[str, Any]:
    """
    Compute step response.

    Args:
        num: Numerator coefficients
        den: Denominator coefficients
        t: Time array [s] (default: auto)
        n_points: Number of points if t not specified

    Returns:
        t: Time points
        y: Response values
        rise_time: 10% to 90% rise time
        settling_time: 2% settling time
        overshoot: Percent overshoot
    """
    num = np.atleast_1d(num)
    den = np.atleast_1d(den)

    system = sig.TransferFunction(num, den)

    if t is None:
        t, y = sig.step(system, N=n_points)
    else:
        t = np.atleast_1d(t)
        _, y = sig.step(system, T=t)

    # Steady state value
    y_final = y[-1]
    y_normalized = y / y_final if abs(y_final) > 1e-10 else y

    # Rise time (10% to 90%)
    t_10 = t[np.where(y_normalized >= 0.1)[0][0]] if np.any(y_normalized >= 0.1) else t[0]
    t_90 = t[np.where(y_normalized >= 0.9)[0][0]] if np.any(y_normalized >= 0.9) else t[-1]
    rise_time = t_90 - t_10

    # Overshoot
    y_peak = np.max(y_normalized)
    overshoot = (y_peak - 1) * 100 if y_peak > 1 else 0

    # Peak time
    peak_idx = np.argmax(y)
    peak_time = t[peak_idx]

    # Settling time (2% band)
    settling_idx = np.where(np.abs(y_normalized - 1) > 0.02)[0]
    settling_time = t[settling_idx[-1]] if len(settling_idx) > 0 else t[-1]

    return {
        't': t.tolist(),
        'y': y.tolist(),
        'y_final': float(y_final),
        'rise_time': float(rise_time),
        'settling_time': float(settling_time),
        'overshoot_percent': float(overshoot),
        'peak_time': float(peak_time),
        'peak_value': float(y[peak_idx]),
    }


def impulse_response(num: List[float], den: List[float],
                     t: np.ndarray = None, n_points: int = 500) -> Dict[str, Any]:
    """
    Compute impulse response.

    Args:
        num: Numerator coefficients
        den: Denominator coefficients
        t: Time array [s] (default: auto)
        n_points: Number of points if t not specified

    Returns:
        t: Time points
        y: Response values
    """
    num = np.atleast_1d(num)
    den = np.atleast_1d(den)

    system = sig.TransferFunction(num, den)

    if t is None:
        t, y = sig.impulse(system, N=n_points)
    else:
        t = np.atleast_1d(t)
        _, y = sig.impulse(system, T=t)

    return {
        't': t.tolist(),
        'y': y.tolist(),
        'peak_value': float(np.max(np.abs(y))),
        'peak_time': float(t[np.argmax(np.abs(y))]),
    }


def poles_zeros(num: List[float], den: List[float]) -> Dict[str, Any]:
    """
    Compute poles and zeros.

    Args:
        num: Numerator coefficients
        den: Denominator coefficients

    Returns:
        poles: System poles
        zeros: System zeros
        stability: Whether all poles in LHP
    """
    num = np.atleast_1d(num)
    den = np.atleast_1d(den)

    zeros = np.roots(num)
    poles = np.roots(den)

    # Stability check
    stable = np.all(np.real(poles) < 0)

    # Natural frequencies and damping ratios
    pole_info = []
    for p in poles:
        omega_n = np.abs(p)
        zeta = -np.real(p) / omega_n if omega_n > 0 else 1.0
        pole_info.append({
            'real': float(np.real(p)),
            'imag': float(np.imag(p)),
            'magnitude': float(omega_n),
            'damping_ratio': float(zeta),
            'time_constant': float(-1 / np.real(p)) if np.real(p) < 0 else float('inf'),
        })

    return {
        'poles_real': np.real(poles).tolist(),
        'poles_imag': np.imag(poles).tolist(),
        'zeros_real': np.real(zeros).tolist(),
        'zeros_imag': np.imag(zeros).tolist(),
        'n_poles': len(poles),
        'n_zeros': len(zeros),
        'stable': bool(stable),
        'pole_info': pole_info,
        'dominant_pole': pole_info[np.argmax([p['real'] for p in pole_info])] if pole_info else None,
    }


def system_identification_arx(y: np.ndarray, u: np.ndarray,
                              na: int = 2, nb: int = 2,
                              nk: int = 1) -> Dict[str, Any]:
    """
    ARX model identification.

    y[k] = a1*y[k-1] + ... + an*y[k-na] + b0*u[k-nk] + ... + bm*u[k-nk-nb+1]

    Args:
        y: Output signal
        u: Input signal
        na: Number of output lags
        nb: Number of input lags
        nk: Input delay

    Returns:
        a: Output coefficients [a1, ..., ana]
        b: Input coefficients [b0, ..., bnb-1]
        fit_percent: Model fit percentage
    """
    y = np.asarray(y)
    u = np.asarray(u)

    N = len(y)
    n = max(na, nb + nk - 1)

    # Build regression matrix
    n_data = N - n
    phi = np.zeros((n_data, na + nb))

    for i in range(n_data):
        k = i + n
        # Output lags
        for j in range(na):
            phi[i, j] = -y[k - j - 1]
        # Input lags
        for j in range(nb):
            phi[i, na + j] = u[k - nk - j]

    Y = y[n:]

    # Least squares
    try:
        theta, residuals, rank, s = np.linalg.lstsq(phi, Y, rcond=None)

        a = theta[:na]
        b = theta[na:]

        # Model fit
        y_pred = phi @ theta
        ss_res = np.sum((Y - y_pred) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        fit_percent = 100 * (1 - np.sqrt(ss_res / ss_tot)) if ss_tot > 0 else 0

        # Convert to transfer function
        den = np.concatenate([[1], a])
        num = np.concatenate([np.zeros(nk), b])

        success = True
    except Exception:
        a = np.zeros(na)
        b = np.zeros(nb)
        den = [1]
        num = [0]
        fit_percent = 0
        success = False

    return {
        'a': a.tolist(),
        'b': b.tolist(),
        'num': num.tolist(),
        'den': den.tolist(),
        'na': na,
        'nb': nb,
        'nk': nk,
        'fit_percent': float(fit_percent),
        'success': success,
    }


def compute(num: List[float] = None, den: List[float] = None,
            y: np.ndarray = None, u: np.ndarray = None,
            omega: np.ndarray = None, t: np.ndarray = None,
            **kwargs) -> Dict[str, Any]:
    """
    Main entry point for transfer function analysis.
    """
    results = {}

    # Transfer function analysis
    if num is not None and den is not None:
        results['frequency_response'] = frequency_response(num, den, omega)
        results['step_response'] = step_response(num, den, t)
        results['impulse_response'] = impulse_response(num, den, t)
        results['poles_zeros'] = poles_zeros(num, den)

    # System identification
    if y is not None and u is not None:
        na = kwargs.get('na', 2)
        nb = kwargs.get('nb', 2)
        nk = kwargs.get('nk', 1)
        results['arx_identification'] = system_identification_arx(y, u, na, nb, nk)

    return results if results else {'error': 'Provide num/den or y/u for identification'}
