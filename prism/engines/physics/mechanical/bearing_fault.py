"""
Bearing Fault Analysis

Characteristic defect frequencies, envelope analysis, bearing condition indicators.
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy import signal as sig


def defect_frequencies(n_balls: int, d_ball: float, d_pitch: float,
                       contact_angle: float, shaft_rpm: float) -> Dict[str, Any]:
    """
    Calculate bearing characteristic defect frequencies.

    Args:
        n_balls: Number of rolling elements
        d_ball: Ball/roller diameter [m]
        d_pitch: Pitch diameter [m]
        contact_angle: Contact angle [radians]
        shaft_rpm: Shaft speed [RPM]

    Returns:
        BPFO: Ball pass frequency outer race [Hz]
        BPFI: Ball pass frequency inner race [Hz]
        BSF: Ball spin frequency [Hz]
        FTF: Fundamental train frequency [Hz]
    """
    f_shaft = shaft_rpm / 60  # Convert to Hz
    cos_angle = np.cos(contact_angle)

    # Fundamental train frequency (cage)
    FTF = (f_shaft / 2) * (1 - d_ball / d_pitch * cos_angle)

    # Ball pass frequency outer race
    BPFO = (n_balls / 2) * f_shaft * (1 - d_ball / d_pitch * cos_angle)

    # Ball pass frequency inner race
    BPFI = (n_balls / 2) * f_shaft * (1 + d_ball / d_pitch * cos_angle)

    # Ball spin frequency
    BSF = (d_pitch / (2 * d_ball)) * f_shaft * (1 - (d_ball / d_pitch * cos_angle) ** 2)

    return {
        'BPFO': float(BPFO),
        'BPFI': float(BPFI),
        'BSF': float(BSF),
        'FTF': float(FTF),
        'shaft_frequency': float(f_shaft),
        'n_balls': n_balls,
        'd_ball': d_ball,
        'd_pitch': d_pitch,
        'contact_angle_deg': float(np.degrees(contact_angle)),
    }


def envelope_spectrum(signal: np.ndarray, fs: float,
                      bandpass: tuple = None) -> Dict[str, Any]:
    """
    Envelope (demodulation) analysis for bearing faults.

    Hilbert transform to extract amplitude modulation.

    Args:
        signal: Vibration signal
        fs: Sampling frequency [Hz]
        bandpass: (low, high) frequency band for filtering [Hz]

    Returns:
        envelope_spectrum: FFT of envelope
        frequencies: Frequency axis
        peak_frequencies: Dominant peaks
    """
    signal = np.asarray(signal)
    n = len(signal)

    # Bandpass filter if specified
    if bandpass is not None:
        low, high = bandpass
        nyq = fs / 2
        b, a = sig.butter(4, [low / nyq, high / nyq], btype='band')
        signal_filtered = sig.filtfilt(b, a, signal)
    else:
        signal_filtered = signal

    # Hilbert transform for envelope
    analytic = sig.hilbert(signal_filtered)
    envelope = np.abs(analytic)

    # Remove DC
    envelope = envelope - np.mean(envelope)

    # FFT of envelope
    fft_env = np.fft.rfft(envelope)
    freqs = np.fft.rfftfreq(n, 1 / fs)

    magnitude = np.abs(fft_env) * 2 / n

    # Find peaks
    peak_idx, _ = sig.find_peaks(magnitude, height=np.max(magnitude) * 0.1)
    peak_freqs = freqs[peak_idx]
    peak_mags = magnitude[peak_idx]

    # Sort by magnitude
    sort_idx = np.argsort(peak_mags)[::-1]
    peak_freqs = peak_freqs[sort_idx[:10]].tolist()
    peak_mags = peak_mags[sort_idx[:10]].tolist()

    return {
        'frequencies': freqs.tolist(),
        'envelope_spectrum': magnitude.tolist(),
        'peak_frequencies': peak_freqs,
        'peak_magnitudes': peak_mags,
        'fs': fs,
    }


def bearing_condition_indicators(signal: np.ndarray, fs: float) -> Dict[str, Any]:
    """
    Calculate bearing condition monitoring indicators.

    Args:
        signal: Vibration signal (acceleration)
        fs: Sampling frequency [Hz]

    Returns:
        rms: Root mean square
        peak: Peak value
        crest_factor: Peak/RMS
        kurtosis: 4th moment (healthy ~3, faulty >3)
        skewness: 3rd moment
    """
    signal = np.asarray(signal)

    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    mean_val = np.mean(signal)
    std_val = np.std(signal)

    # Crest factor
    crest = peak / rms if rms > 0 else 0

    # Higher order statistics
    if std_val > 0:
        centered = signal - mean_val
        skewness = np.mean(centered ** 3) / std_val ** 3
        kurtosis = np.mean(centered ** 4) / std_val ** 4
    else:
        skewness = 0
        kurtosis = 3

    # Clearance factor
    clearance = peak / (np.mean(np.sqrt(np.abs(signal))) ** 2) if np.mean(np.sqrt(np.abs(signal))) > 0 else 0

    # Impulse factor
    impulse = peak / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) > 0 else 0

    # Shape factor
    shape = rms / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) > 0 else 0

    return {
        'rms': float(rms),
        'peak': float(peak),
        'crest_factor': float(crest),
        'kurtosis': float(kurtosis),
        'skewness': float(skewness),
        'clearance_factor': float(clearance),
        'impulse_factor': float(impulse),
        'shape_factor': float(shape),
        'healthy_kurtosis_range': [2.5, 3.5],
        'condition': 'healthy' if 2.5 < kurtosis < 3.5 else 'possible_fault',
    }


def spectral_kurtosis(signal: np.ndarray, fs: float,
                      n_bands: int = 16) -> Dict[str, Any]:
    """
    Spectral kurtosis for optimal demodulation band selection.

    Args:
        signal: Vibration signal
        fs: Sampling frequency [Hz]
        n_bands: Number of frequency bands

    Returns:
        kurtosis_per_band: Kurtosis in each frequency band
        optimal_band: Frequency band with highest kurtosis
    """
    signal = np.asarray(signal)
    nyq = fs / 2

    band_edges = np.linspace(0, nyq, n_bands + 1)
    band_centers = (band_edges[:-1] + band_edges[1:]) / 2
    band_kurtosis = []

    for i in range(n_bands):
        low = band_edges[i]
        high = band_edges[i + 1]

        if low < 1:
            low = 1

        try:
            b, a = sig.butter(4, [low / nyq, min(high / nyq, 0.99)], btype='band')
            filtered = sig.filtfilt(b, a, signal)
            std_f = np.std(filtered)
            if std_f > 0:
                k = np.mean((filtered - np.mean(filtered)) ** 4) / std_f ** 4
            else:
                k = 3
        except Exception:
            k = 3

        band_kurtosis.append(k)

    band_kurtosis = np.array(band_kurtosis)
    optimal_idx = np.argmax(band_kurtosis)

    return {
        'band_centers': band_centers.tolist(),
        'band_edges': band_edges.tolist(),
        'kurtosis_per_band': band_kurtosis.tolist(),
        'optimal_band_center': float(band_centers[optimal_idx]),
        'optimal_band_range': [float(band_edges[optimal_idx]),
                               float(band_edges[optimal_idx + 1])],
        'max_kurtosis': float(band_kurtosis[optimal_idx]),
    }


def compute(signal: np.ndarray = None, fs: float = 1.0,
            n_balls: int = None, d_ball: float = None,
            d_pitch: float = None, contact_angle: float = 0.0,
            shaft_rpm: float = None, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for bearing fault analysis.
    """
    results = {}

    # Calculate defect frequencies if bearing geometry provided
    if all(v is not None for v in [n_balls, d_ball, d_pitch, shaft_rpm]):
        results['defect_frequencies'] = defect_frequencies(
            n_balls, d_ball, d_pitch, contact_angle, shaft_rpm)

    # Signal analysis if signal provided
    if signal is not None:
        signal = np.asarray(signal)
        results['condition_indicators'] = bearing_condition_indicators(signal, fs)
        results['spectral_kurtosis'] = spectral_kurtosis(signal, fs)

        # Envelope spectrum
        sk = results['spectral_kurtosis']
        bandpass = tuple(sk['optimal_band_range'])
        results['envelope_spectrum'] = envelope_spectrum(signal, fs, bandpass)

    return results if results else {'error': 'Provide signal or bearing geometry'}
