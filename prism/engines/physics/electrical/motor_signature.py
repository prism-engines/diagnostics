"""
Motor Current Signature Analysis (MCSA)

Induction motor fault detection from stator current.
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy import signal as sig


def motor_frequencies(line_freq: float, n_poles: int,
                      slip: float = 0.03) -> Dict[str, Any]:
    """
    Calculate characteristic motor frequencies.

    Args:
        line_freq: Line frequency [Hz] (50 or 60)
        n_poles: Number of poles
        slip: Slip ratio (typically 0.01-0.05)

    Returns:
        f_sync: Synchronous speed [Hz]
        f_rotor: Rotor speed [Hz]
        f_slot: Slot passing frequency [Hz]
        f_eccentricity: Eccentricity frequencies [Hz]
        f_broken_bar: Broken rotor bar sidebands [Hz]
    """
    # Synchronous frequency
    f_sync = line_freq * 2 / n_poles

    # Rotor frequency
    f_rotor = f_sync * (1 - slip)

    # RPM
    rpm_sync = f_sync * 60
    rpm_rotor = f_rotor * 60

    # Slip frequency
    f_slip = slip * line_freq

    # Broken rotor bar frequencies
    # f = f_line × (1 ± 2s)
    f_brb_lower = line_freq * (1 - 2 * slip)
    f_brb_upper = line_freq * (1 + 2 * slip)

    # Eccentricity frequencies
    # f_ecc = f_line ± k × f_rotor
    f_ecc = [line_freq - f_rotor, line_freq + f_rotor]

    # Bearing frequencies (approximate, depends on bearing)
    # Typical: f_bearing ≈ 0.4 × n_balls × f_rotor
    f_bearing_approx = 0.4 * 8 * f_rotor  # Assuming 8 balls

    return {
        'line_freq': float(line_freq),
        'sync_freq': float(f_sync),
        'rotor_freq': float(f_rotor),
        'slip_freq': float(f_slip),
        'slip': float(slip),
        'rpm_sync': float(rpm_sync),
        'rpm_rotor': float(rpm_rotor),
        'broken_bar_lower': float(f_brb_lower),
        'broken_bar_upper': float(f_brb_upper),
        'eccentricity_freqs': [float(f) for f in f_ecc],
        'bearing_freq_approx': float(f_bearing_approx),
        'n_poles': n_poles,
    }


def broken_rotor_bar(current: np.ndarray, fs: float, line_freq: float,
                     slip: float = 0.03) -> Dict[str, Any]:
    """
    Detect broken rotor bars from current spectrum.

    Broken bars create sidebands at f_line ± 2×s×f_line.

    Args:
        current: Stator current waveform [A]
        fs: Sampling frequency [Hz]
        line_freq: Line frequency [Hz]
        slip: Motor slip

    Returns:
        brb_indicator: Broken bar indicator [dB]
        lower_sideband: Amplitude at f - 2sf
        upper_sideband: Amplitude at f + 2sf
        severity: Fault severity estimate
    """
    current = np.asarray(current)
    n = len(current)

    # High-resolution FFT with zero padding
    n_fft = max(n, int(fs * 10))  # At least 10s resolution
    fft_vals = np.fft.rfft(current, n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)
    magnitude_db = 20 * np.log10(np.abs(fft_vals) + 1e-20)

    df = freqs[1] - freqs[0]

    def get_amplitude_db(f_target):
        idx = int(round(f_target / df))
        if 0 <= idx < len(magnitude_db):
            return float(magnitude_db[idx])
        return -100.0

    # Fundamental amplitude
    fund_db = get_amplitude_db(line_freq)

    # Sideband frequencies
    f_lower = line_freq * (1 - 2 * slip)
    f_upper = line_freq * (1 + 2 * slip)

    lower_db = get_amplitude_db(f_lower)
    upper_db = get_amplitude_db(f_upper)

    # BRB indicator (dB below fundamental)
    brb_lower = fund_db - lower_db
    brb_upper = fund_db - upper_db

    # Average indicator
    brb_indicator = (brb_lower + brb_upper) / 2

    # Severity assessment
    if brb_indicator > 54:
        severity = 'healthy'
    elif brb_indicator > 48:
        severity = 'slight_fault'
    elif brb_indicator > 42:
        severity = 'moderate_fault'
    else:
        severity = 'severe_fault'

    return {
        'brb_indicator_db': float(brb_indicator),
        'lower_sideband_db': float(lower_db),
        'upper_sideband_db': float(upper_db),
        'fundamental_db': float(fund_db),
        'brb_lower_offset_db': float(brb_lower),
        'brb_upper_offset_db': float(brb_upper),
        'lower_freq': float(f_lower),
        'upper_freq': float(f_upper),
        'severity': severity,
        'slip_used': slip,
    }


def eccentricity(current: np.ndarray, fs: float, line_freq: float,
                 f_rotor: float) -> Dict[str, Any]:
    """
    Detect rotor eccentricity from current spectrum.

    Eccentricity creates sidebands at f_line ± f_rotor.

    Args:
        current: Stator current waveform [A]
        fs: Sampling frequency [Hz]
        line_freq: Line frequency [Hz]
        f_rotor: Rotor frequency [Hz]

    Returns:
        static_ecc: Static eccentricity indicator [dB]
        dynamic_ecc: Dynamic eccentricity indicator [dB]
        mixed_ecc: Mixed eccentricity indicator [dB]
    """
    current = np.asarray(current)
    n = len(current)

    n_fft = max(n, int(fs * 10))
    fft_vals = np.fft.rfft(current, n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)
    magnitude_db = 20 * np.log10(np.abs(fft_vals) + 1e-20)

    df = freqs[1] - freqs[0]

    def get_amplitude_db(f_target):
        idx = int(round(f_target / df))
        if 0 <= idx < len(magnitude_db):
            return float(magnitude_db[idx])
        return -100.0

    fund_db = get_amplitude_db(line_freq)

    # Eccentricity frequencies
    f_ecc_lower = line_freq - f_rotor
    f_ecc_upper = line_freq + f_rotor

    ecc_lower_db = get_amplitude_db(f_ecc_lower)
    ecc_upper_db = get_amplitude_db(f_ecc_upper)

    # Indicators (dB below fundamental)
    static_ecc = fund_db - ecc_lower_db
    dynamic_ecc = fund_db - ecc_upper_db
    mixed_ecc = (static_ecc + dynamic_ecc) / 2

    return {
        'static_eccentricity_db': float(static_ecc),
        'dynamic_eccentricity_db': float(dynamic_ecc),
        'mixed_eccentricity_db': float(mixed_ecc),
        'lower_freq': float(f_ecc_lower),
        'upper_freq': float(f_ecc_upper),
        'lower_amplitude_db': float(ecc_lower_db),
        'upper_amplitude_db': float(ecc_upper_db),
        'condition': 'healthy' if mixed_ecc > 40 else 'possible_eccentricity',
    }


def bearing_fault_mcsa(current: np.ndarray, fs: float, line_freq: float,
                       f_bearing: float) -> Dict[str, Any]:
    """
    Detect bearing faults from motor current.

    Bearing faults modulate at bearing defect frequency.

    Args:
        current: Stator current waveform [A]
        fs: Sampling frequency [Hz]
        line_freq: Line frequency [Hz]
        f_bearing: Bearing characteristic frequency [Hz]

    Returns:
        bearing_indicator: Bearing fault indicator [dB]
        sideband_amplitudes: Amplitudes at f_line ± k×f_bearing
    """
    current = np.asarray(current)
    n = len(current)

    n_fft = max(n, int(fs * 10))
    fft_vals = np.fft.rfft(current, n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)
    magnitude_db = 20 * np.log10(np.abs(fft_vals) + 1e-20)

    df = freqs[1] - freqs[0]

    def get_amplitude_db(f_target):
        idx = int(round(f_target / df))
        if 0 <= idx < len(magnitude_db):
            return float(magnitude_db[idx])
        return -100.0

    fund_db = get_amplitude_db(line_freq)

    # Bearing sidebands at f_line ± k×f_bearing
    sidebands = []
    for k in range(1, 4):
        f_lower = line_freq - k * f_bearing
        f_upper = line_freq + k * f_bearing
        sidebands.append({
            'k': k,
            'lower_freq': float(f_lower),
            'upper_freq': float(f_upper),
            'lower_db': float(get_amplitude_db(f_lower)),
            'upper_db': float(get_amplitude_db(f_upper)),
            'lower_offset_db': float(fund_db - get_amplitude_db(f_lower)),
            'upper_offset_db': float(fund_db - get_amplitude_db(f_upper)),
        })

    # Average indicator
    avg_offset = np.mean([s['lower_offset_db'] + s['upper_offset_db']
                          for s in sidebands]) / 2

    return {
        'bearing_indicator_db': float(avg_offset),
        'sidebands': sidebands,
        'fundamental_db': float(fund_db),
        'f_bearing': f_bearing,
        'condition': 'healthy' if avg_offset > 50 else 'possible_bearing_fault',
    }


def compute(current: np.ndarray = None, fs: float = None,
            line_freq: float = 50.0, n_poles: int = 4,
            slip: float = 0.03, f_bearing: float = None,
            **kwargs) -> Dict[str, Any]:
    """
    Main entry point for motor signature analysis.
    """
    results = {}

    # Motor frequencies
    motor_freqs = motor_frequencies(line_freq, n_poles, slip)
    results['motor_frequencies'] = motor_freqs

    if current is not None and fs is not None:
        current = np.asarray(current)

        # Broken rotor bar
        results['broken_rotor_bar'] = broken_rotor_bar(current, fs, line_freq, slip)

        # Eccentricity
        results['eccentricity'] = eccentricity(
            current, fs, line_freq, motor_freqs['rotor_freq'])

        # Bearing (if frequency provided or use approximate)
        f_bear = f_bearing if f_bearing else motor_freqs['bearing_freq_approx']
        results['bearing'] = bearing_fault_mcsa(current, fs, line_freq, f_bear)

    return results
