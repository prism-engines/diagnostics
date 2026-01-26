"""
Gear Mesh Analysis

Gear mesh frequencies, sidebands, tooth fault detection.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from scipy import signal as sig


def mesh_frequencies(n_teeth_pinion: int, n_teeth_gear: int,
                     shaft_rpm: float) -> Dict[str, Any]:
    """
    Calculate gear mesh frequency and harmonics.

    GMF = N_teeth × shaft_frequency

    Args:
        n_teeth_pinion: Number of teeth on pinion
        n_teeth_gear: Number of teeth on gear
        shaft_rpm: Pinion shaft speed [RPM]

    Returns:
        GMF: Gear mesh frequency [Hz]
        pinion_shaft_freq: Pinion rotation frequency [Hz]
        gear_shaft_freq: Gear rotation frequency [Hz]
        gear_ratio: Speed reduction ratio
    """
    f_pinion = shaft_rpm / 60

    # Gear mesh frequency
    GMF = n_teeth_pinion * f_pinion

    # Gear shaft frequency
    gear_ratio = n_teeth_pinion / n_teeth_gear
    f_gear = f_pinion * gear_ratio

    return {
        'GMF': float(GMF),
        'GMF_harmonics': [float(GMF * i) for i in range(1, 6)],
        'pinion_shaft_freq': float(f_pinion),
        'gear_shaft_freq': float(f_gear),
        'gear_ratio': float(gear_ratio),
        'n_teeth_pinion': n_teeth_pinion,
        'n_teeth_gear': n_teeth_gear,
        'hunting_tooth_freq': float(GMF / (n_teeth_pinion * n_teeth_gear)),
    }


def sideband_analysis(signal: np.ndarray, fs: float, GMF: float,
                      shaft_freq: float, n_sidebands: int = 5) -> Dict[str, Any]:
    """
    Analyze sidebands around gear mesh frequency.

    Sidebands indicate modulation from shaft rotation.

    Args:
        signal: Vibration signal
        fs: Sampling frequency [Hz]
        GMF: Gear mesh frequency [Hz]
        shaft_freq: Shaft rotation frequency [Hz]
        n_sidebands: Number of sidebands to check

    Returns:
        sideband_amplitudes: Amplitudes at GMF ± n×shaft_freq
        sideband_ratio: Sideband energy / GMF energy
    """
    signal = np.asarray(signal)
    n = len(signal)

    # FFT
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    magnitude = np.abs(fft_vals) * 2 / n

    # Frequency resolution
    df = freqs[1] - freqs[0]

    def get_amplitude(f_target):
        idx = int(round(f_target / df))
        if 0 <= idx < len(magnitude):
            return float(magnitude[idx])
        return 0.0

    # GMF amplitude
    gmf_amp = get_amplitude(GMF)

    # Sideband amplitudes
    lower_sidebands = []
    upper_sidebands = []

    for i in range(1, n_sidebands + 1):
        lower_sidebands.append(get_amplitude(GMF - i * shaft_freq))
        upper_sidebands.append(get_amplitude(GMF + i * shaft_freq))

    # Sideband energy ratio
    sideband_energy = sum(lower_sidebands) + sum(upper_sidebands)
    sideband_ratio = sideband_energy / gmf_amp if gmf_amp > 0 else 0

    return {
        'GMF_amplitude': gmf_amp,
        'lower_sidebands': lower_sidebands,
        'upper_sidebands': upper_sidebands,
        'sideband_frequencies_lower': [GMF - i * shaft_freq for i in range(1, n_sidebands + 1)],
        'sideband_frequencies_upper': [GMF + i * shaft_freq for i in range(1, n_sidebands + 1)],
        'sideband_ratio': float(sideband_ratio),
        'fault_indicator': 'possible_fault' if sideband_ratio > 0.5 else 'healthy',
    }


def time_synchronous_average(signal: np.ndarray, fs: float,
                             shaft_rpm: float, n_averages: int = None) -> Dict[str, Any]:
    """
    Time synchronous averaging to extract periodic components.

    Averages over multiple shaft revolutions.

    Args:
        signal: Vibration signal
        fs: Sampling frequency [Hz]
        shaft_rpm: Shaft speed [RPM]
        n_averages: Number of revolutions to average

    Returns:
        tsa: Time synchronous average (one revolution)
        residual: Signal minus TSA (for random/impact analysis)
    """
    signal = np.asarray(signal)
    period_samples = int(fs * 60 / shaft_rpm)

    if n_averages is None:
        n_averages = len(signal) // period_samples

    # Reshape into complete revolutions
    n_complete = n_averages * period_samples
    if n_complete > len(signal):
        n_averages = len(signal) // period_samples
        n_complete = n_averages * period_samples

    reshaped = signal[:n_complete].reshape(n_averages, period_samples)

    # Average
    tsa = np.mean(reshaped, axis=0)

    # Residual (original minus repeated TSA)
    tsa_repeated = np.tile(tsa, n_averages)
    residual = signal[:n_complete] - tsa_repeated

    # Statistics
    tsa_rms = np.sqrt(np.mean(tsa ** 2))
    residual_rms = np.sqrt(np.mean(residual ** 2))

    return {
        'tsa': tsa.tolist(),
        'residual_rms': float(residual_rms),
        'tsa_rms': float(tsa_rms),
        'snr': float(20 * np.log10(tsa_rms / residual_rms)) if residual_rms > 0 else float('inf'),
        'n_averages': n_averages,
        'samples_per_revolution': period_samples,
    }


def gear_condition_indicators(signal: np.ndarray, fs: float, GMF: float,
                              n_harmonics: int = 5) -> Dict[str, Any]:
    """
    Gear-specific condition indicators.

    Args:
        signal: Vibration signal
        fs: Sampling frequency [Hz]
        GMF: Gear mesh frequency [Hz]
        n_harmonics: Number of GMF harmonics to consider

    Returns:
        FM0: Frequency modulation indicator
        FM4: Kurtosis of difference signal
        NA4: Normalized fourth moment
    """
    signal = np.asarray(signal)
    n = len(signal)

    # FFT
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    magnitude = np.abs(fft_vals) * 2 / n
    df = freqs[1] - freqs[0]

    def get_amplitude(f_target):
        idx = int(round(f_target / df))
        if 0 <= idx < len(magnitude):
            return float(magnitude[idx])
        return 0.0

    # Sum of GMF harmonics
    gmf_sum = sum(get_amplitude(GMF * i) for i in range(1, n_harmonics + 1))

    # Total RMS
    total_rms = np.sqrt(np.mean(signal ** 2))

    # FM0: difference signal method
    # Create bandpass around GMF
    nyq = fs / 2
    try:
        low = max(GMF - GMF * 0.1, 1) / nyq
        high = min(GMF + GMF * 0.1, nyq - 1) / nyq
        b, a = sig.butter(4, [low, high], btype='band')
        diff_signal = sig.filtfilt(b, a, signal)

        std_diff = np.std(diff_signal)
        FM0 = std_diff / total_rms if total_rms > 0 else 0

        # FM4: kurtosis of difference signal
        if std_diff > 0:
            FM4 = np.mean((diff_signal - np.mean(diff_signal)) ** 4) / std_diff ** 4
        else:
            FM4 = 3
    except Exception:
        FM0 = 0
        FM4 = 3

    # NA4: average of normalized fourth moment
    std_sig = np.std(signal)
    if std_sig > 0:
        NA4 = np.mean((signal - np.mean(signal)) ** 4) / std_sig ** 4
    else:
        NA4 = 3

    return {
        'FM0': float(FM0),
        'FM4': float(FM4),
        'NA4': float(NA4),
        'GMF_energy_ratio': float(gmf_sum / total_rms) if total_rms > 0 else 0,
        'total_rms': float(total_rms),
        'condition': 'healthy' if NA4 < 4 and FM4 < 4 else 'possible_fault',
    }


def compute(signal: np.ndarray = None, fs: float = 1.0,
            n_teeth_pinion: int = None, n_teeth_gear: int = None,
            shaft_rpm: float = None, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for gear mesh analysis.
    """
    results = {}

    # Calculate mesh frequencies if gear geometry provided
    if all(v is not None for v in [n_teeth_pinion, n_teeth_gear, shaft_rpm]):
        results['mesh_frequencies'] = mesh_frequencies(
            n_teeth_pinion, n_teeth_gear, shaft_rpm)
        GMF = results['mesh_frequencies']['GMF']
        shaft_freq = results['mesh_frequencies']['pinion_shaft_freq']

        # Signal analysis
        if signal is not None:
            signal = np.asarray(signal)
            results['sidebands'] = sideband_analysis(signal, fs, GMF, shaft_freq)
            results['tsa'] = time_synchronous_average(signal, fs, shaft_rpm)
            results['condition_indicators'] = gear_condition_indicators(signal, fs, GMF)

    elif signal is not None:
        return {'error': 'Provide gear geometry (n_teeth_pinion, n_teeth_gear, shaft_rpm)'}

    return results if results else {'error': 'Provide gear parameters'}
