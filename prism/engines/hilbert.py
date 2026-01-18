"""
PRISM Hilbert Transform Engine
==============================

Computes instantaneous amplitude, phase, and frequency using the Hilbert transform.

The Hilbert transform extracts the analytic signal, giving us:
- Instantaneous amplitude (envelope): magnitude of oscillations
- Instantaneous phase: position in oscillation cycle
- Instantaneous frequency: rate of phase change

Useful for detecting:
- Amplitude modulation (envelope changes)
- Phase dynamics and synchronization
- Frequency variations over time

Mathematical Basis:
    analytic_signal = signal + i * hilbert(signal)
    amplitude = |analytic_signal|
    phase = angle(analytic_signal)
    inst_freq = d(phase)/dt / (2*pi)
"""

import numpy as np
from scipy.signal import hilbert
from typing import Dict, Optional


def compute_hilbert(values: np.ndarray, min_obs: int = 20) -> Dict[str, Optional[float]]:
    """
    Compute Hilbert transform metrics for a signal topology.

    Args:
        values: 1D array of signal topology values
        min_obs: Minimum observations required

    Returns:
        Dict with metrics:
            - hilbert_amp_mean: Mean instantaneous amplitude
            - hilbert_amp_std: Std of instantaneous amplitude
            - hilbert_amp_cv: Coefficient of variation of amplitude
            - hilbert_phase_mean: Mean instantaneous phase (circular)
            - hilbert_phase_std: Std of instantaneous phase
            - hilbert_inst_freq_mean: Mean instantaneous frequency
            - hilbert_inst_freq_std: Std of instantaneous frequency
    """
    result = {
        'hilbert_amp_mean': None,
        'hilbert_amp_std': None,
        'hilbert_amp_cv': None,
        'hilbert_phase_mean': None,
        'hilbert_phase_std': None,
        'hilbert_inst_freq_mean': None,
        'hilbert_inst_freq_std': None,
    }

    # Validate input
    if len(values) < min_obs:
        return result

    values = np.asarray(values, dtype=np.float64)

    # Remove NaN/Inf
    mask = np.isfinite(values)
    if np.sum(mask) < min_obs:
        return result

    clean_values = values[mask]

    # Demean the signal (Hilbert works better on zero-mean signals)
    signal = clean_values - np.mean(clean_values)

    # Handle constant signal
    if np.std(signal) < 1e-10:
        return result

    try:
        # Compute analytic signal via Hilbert transform
        analytic_signal = hilbert(signal)

        # Instantaneous amplitude (envelope)
        amplitude = np.abs(analytic_signal)

        # Instantaneous phase
        phase = np.angle(analytic_signal)

        # Instantaneous frequency (derivative of phase)
        # Unwrap phase to avoid discontinuities at +/- pi
        unwrapped_phase = np.unwrap(phase)
        inst_freq = np.diff(unwrapped_phase) / (2.0 * np.pi)

        # Compute statistics
        # Amplitude metrics
        amp_mean = float(np.mean(amplitude))
        amp_std = float(np.std(amplitude))
        amp_cv = amp_std / amp_mean if amp_mean > 1e-10 else 0.0

        result['hilbert_amp_mean'] = amp_mean
        result['hilbert_amp_std'] = amp_std
        result['hilbert_amp_cv'] = amp_cv

        # Phase metrics (circular statistics)
        # Mean phase using circular mean
        phase_mean = float(np.arctan2(np.mean(np.sin(phase)), np.mean(np.cos(phase))))
        # Circular standard deviation
        r = np.sqrt(np.mean(np.cos(phase))**2 + np.mean(np.sin(phase))**2)
        phase_std = float(np.sqrt(-2.0 * np.log(r))) if r > 1e-10 else np.pi

        result['hilbert_phase_mean'] = phase_mean
        result['hilbert_phase_std'] = min(phase_std, np.pi)  # Cap at pi

        # Instantaneous frequency metrics
        if len(inst_freq) > 0:
            # Filter out extreme values (numerical artifacts)
            freq_clean = inst_freq[np.abs(inst_freq) < 0.5]  # Max freq = 0.5 (Nyquist)
            if len(freq_clean) > 0:
                result['hilbert_inst_freq_mean'] = float(np.mean(freq_clean))
                result['hilbert_inst_freq_std'] = float(np.std(freq_clean))

    except Exception:
        # Return partial results on error
        pass

    return result


# Alias for consistency with other engines
def get_hilbert_metrics(values: np.ndarray, min_obs: int = 20) -> Dict[str, Optional[float]]:
    """Alias for compute_hilbert."""
    return compute_hilbert(values, min_obs)
