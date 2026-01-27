"""
Complete Signal Typology - Complex Classifications

Classifications that require FFT, Hurst, Lyapunov, Entropy results:
- Predictability (deterministic / stochastic / chaotic)
- Repetition pattern (periodic / quasi_periodic / aperiodic)
- Standard form (step / ramp / sinusoid / exponential / noise_* / complex)
- Symmetry (even / odd / none)
- Frequency content (lowpass / highpass / bandpass / broadband / narrowband)

CANONICAL INTERFACE:
    Input:  observations DataFrame [entity_id, signal_id, I, y]
            primitives DataFrame (optional) [entity_id, signal_id, hurst, lyapunov, ...]
    Output: DataFrame with complex typology classifications
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ENGINE_TYPE = 'signal'


def compute(observations: pd.DataFrame, primitives: pd.DataFrame = None) -> pd.DataFrame:
    """
    Complete signal typology using all available information.

    Computes complex classifications that require algorithmic analysis:
    - predictability: deterministic / stochastic / chaotic / mixed
    - repetition_pattern: periodic / quasi_periodic / aperiodic / cyclostationary
    - standard_form: impulse / step / ramp / exponential_* / sinusoidal / noise_* / complex
    - symmetry: even / odd / none
    - frequency_content: lowpass / highpass / bandpass / broadband / narrowband

    Args:
        observations: Raw signal data [entity_id, signal_id, I, y]
        primitives: Optional precomputed metrics [hurst, lyapunov, entropy, fft results]

    Returns:
        DataFrame with complex typology per signal
    """
    results = []

    for (entity_id, signal_id), group in observations.groupby(['entity_id', 'signal_id']):
        group = group.sort_values('I')
        y = group['y'].values
        I = group['I'].values
        n = len(y)

        if n < 10:
            continue

        row = {
            'entity_id': entity_id,
            'signal_id': signal_id,
        }

        # Get primitives if available
        prim = None
        if primitives is not None:
            prim_match = primitives[
                (primitives['entity_id'] == entity_id) &
                (primitives['signal_id'] == signal_id)
            ]
            if len(prim_match) > 0:
                prim = prim_match.iloc[0].to_dict()

        # =====================================================================
        # PREDICTABILITY (needs hurst, lyapunov, entropy)
        # =====================================================================
        if prim and 'lyapunov' in prim and 'hurst' in prim:
            lyap = prim.get('lyapunov')
            hurst = prim.get('hurst')
            entropy = prim.get('sample_entropy', prim.get('entropy', 0.5))

            if lyap is not None and not np.isnan(lyap):
                if lyap > 0.1:
                    row['predictability'] = 'chaotic'
                elif entropy is not None and entropy < 0.3 and hurst is not None and hurst > 0.9:
                    row['predictability'] = 'deterministic'
                elif entropy is not None and entropy > 0.7:
                    row['predictability'] = 'stochastic'
                else:
                    row['predictability'] = 'mixed'
            else:
                row['predictability'] = 'unknown'
        else:
            # Estimate from signal properties
            row['predictability'] = _estimate_predictability(y)

        # =====================================================================
        # REPETITION PATTERN (needs FFT)
        # =====================================================================
        row['repetition_pattern'], row['period'] = _detect_repetition_pattern(y, I, prim)

        # =====================================================================
        # STANDARD FORM
        # =====================================================================
        row['standard_form'], row['standard_form_confidence'] = _detect_standard_form(y, I)

        # =====================================================================
        # SYMMETRY
        # =====================================================================
        row['symmetry'] = _detect_symmetry(y)

        # =====================================================================
        # FREQUENCY CONTENT
        # =====================================================================
        row['frequency_content'] = _classify_frequency_content(y, prim)

        results.append(row)

    return pd.DataFrame(results)


def _estimate_predictability(y: np.ndarray) -> str:
    """Estimate predictability without precomputed metrics."""
    n = len(y)
    if n < 20:
        return 'unknown'

    # Simple autocorrelation check
    y_centered = y - np.mean(y)
    autocorr = np.correlate(y_centered, y_centered, mode='full')
    autocorr = autocorr[n-1:] / (autocorr[n-1] + 1e-10)

    # High persistence = more deterministic
    if len(autocorr) > 10:
        lag10_corr = autocorr[10] if len(autocorr) > 10 else 0
        if lag10_corr > 0.8:
            return 'likely_deterministic'
        elif lag10_corr < 0.1:
            return 'likely_stochastic'

    return 'mixed'


def _detect_repetition_pattern(y: np.ndarray, I: np.ndarray, prim: dict = None) -> tuple:
    """Detect periodicity / quasi-periodicity / aperiodicity."""
    n = len(y)
    if n < 20:
        return 'unknown', None

    # Check dominant frequency from primitives
    if prim:
        dominant_freq = prim.get('dominant_freq')
        dominant_power_ratio = prim.get('dominant_power_ratio', 0)

        if dominant_power_ratio is not None and dominant_power_ratio > 0.5:
            period = 1.0 / dominant_freq if dominant_freq and dominant_freq > 0 else None
            return 'periodic', period
        elif dominant_power_ratio is not None and dominant_power_ratio > 0.2:
            period = 1.0 / dominant_freq if dominant_freq and dominant_freq > 0 else None
            return 'quasi_periodic', period

    # Compute from FFT
    try:
        y_centered = y - np.mean(y)
        fft_result = np.fft.fft(y_centered)
        power = np.abs(fft_result[:n//2]) ** 2
        total_power = power.sum()

        if total_power > 0:
            dominant_idx = np.argmax(power[1:]) + 1  # Skip DC
            dominant_power_ratio = power[dominant_idx] / total_power

            if dominant_power_ratio > 0.5:
                freqs = np.fft.fftfreq(n)
                dominant_freq = abs(freqs[dominant_idx])
                period = 1.0 / dominant_freq if dominant_freq > 0 else None
                return 'periodic', period
            elif dominant_power_ratio > 0.2:
                freqs = np.fft.fftfreq(n)
                dominant_freq = abs(freqs[dominant_idx])
                period = 1.0 / dominant_freq if dominant_freq > 0 else None
                return 'quasi_periodic', period
    except Exception:
        pass

    # Check autocorrelation for cyclostationary
    y_centered = y - np.mean(y)
    autocorr = np.correlate(y_centered, y_centered, mode='full')
    autocorr = autocorr[n-1:] / (autocorr[n-1] + 1e-10)

    # Find peaks in autocorrelation
    try:
        peaks, _ = find_peaks(autocorr[1:], height=0.3)
        if len(peaks) > 2:
            # Check if peaks are regularly spaced
            peak_diffs = np.diff(peaks)
            if len(peak_diffs) > 0 and np.std(peak_diffs) / (np.mean(peak_diffs) + 1e-10) < 0.2:
                return 'cyclostationary', float(np.mean(peak_diffs))
    except Exception:
        pass

    return 'aperiodic', None


def _detect_standard_form(y: np.ndarray, I: np.ndarray) -> tuple:
    """Match signal to standard forms with confidence."""
    n = len(y)
    if n < 10:
        return 'unknown', 0.0

    dy = np.diff(y)
    d2y = np.diff(dy) if len(dy) > 1 else np.array([0])

    y_std = np.std(y)
    dy_std = np.std(dy)

    if y_std < 1e-10:
        return 'constant', 1.0

    # Impulse: single large value, rest near zero
    above_threshold = np.abs(y - np.median(y)) > 3 * y_std
    if above_threshold.sum() <= 3 and above_threshold.sum() > 0:
        return 'impulse_like', 0.8

    # Step: constant derivative after single change
    if dy_std < 0.05 * y_std:
        transitions = np.where(np.abs(dy) > 3 * dy_std)[0]
        if len(transitions) <= 3:
            return 'step_like', 0.7

    # Ramp: constant non-zero derivative
    if dy_std < 0.1 * np.abs(np.mean(dy)) and np.abs(np.mean(dy)) > 0.01 * y_std:
        return 'ramp_like', 0.7

    # Parabolic: constant second derivative
    if len(d2y) > 5:
        d2y_std = np.std(d2y)
        d2y_mean = np.mean(d2y)
        if d2y_std < 0.1 * np.abs(d2y_mean) and np.abs(d2y_mean) > 1e-10:
            return 'parabolic', 0.6

    # Exponential: dy proportional to y
    if len(y) > 10 and len(dy) > 10:
        try:
            corr = np.corrcoef(y[:-1], dy)[0, 1]
            if not np.isnan(corr) and corr > 0.9:
                return 'exponential_growth' if np.mean(dy) > 0 else 'exponential_decay', 0.8
        except Exception:
            pass

    # Sinusoidal: strong single frequency + zero crossings
    try:
        y_centered = y - np.mean(y)
        fft_result = np.fft.fft(y_centered)
        power = np.abs(fft_result[:n//2]) ** 2
        total_power = power.sum()

        if total_power > 0:
            dominant_idx = np.argmax(power[1:]) + 1
            dominant_power_ratio = power[dominant_idx] / total_power

            if dominant_power_ratio > 0.5:
                # Check for damping
                envelope = np.abs(y_centered)
                if len(envelope) > 20:
                    first_quarter = envelope[:n//4].mean()
                    last_quarter = envelope[-n//4:].mean()
                    if last_quarter < 0.5 * first_quarter:
                        return 'damped_sinusoid', 0.7
                return 'sinusoidal', 0.8
    except Exception:
        pass

    # Noise classification by spectral slope
    try:
        y_centered = y - np.mean(y)
        fft_result = np.fft.fft(y_centered)
        freqs = np.fft.fftfreq(n)
        power = np.abs(fft_result) ** 2

        positive_mask = freqs > 0
        if positive_mask.sum() > 10:
            log_freq = np.log(freqs[positive_mask] + 1e-10)
            log_power = np.log(power[positive_mask] + 1e-10)

            # Remove infinities
            valid = np.isfinite(log_freq) & np.isfinite(log_power)
            if valid.sum() > 5:
                slope, _ = np.polyfit(log_freq[valid], log_power[valid], 1)

                if slope > -0.5:
                    return 'noise_white', 0.6
                elif slope > -1.5:
                    return 'noise_pink', 0.6
                elif slope > -2.5:
                    return 'noise_brown', 0.6
    except Exception:
        pass

    return 'complex', 0.5


def _detect_symmetry(y: np.ndarray) -> str:
    """Check for even/odd symmetry."""
    n = len(y)
    if n < 10:
        return 'unknown'

    mid = n // 2

    # Center the signal
    y_centered = y - np.mean(y)

    # For even symmetry check
    y_forward = y_centered[:mid]
    y_backward = y_centered[-mid:][::-1]

    if len(y_forward) != len(y_backward) or len(y_forward) < 3:
        return 'none'

    try:
        # Check even: f(-t) = f(t)
        even_corr = np.corrcoef(y_forward, y_backward)[0, 1]

        # Check odd: f(-t) = -f(t)
        odd_corr = np.corrcoef(y_forward, -y_backward)[0, 1]

        if not np.isnan(even_corr) and even_corr > 0.9:
            return 'even'
        elif not np.isnan(odd_corr) and odd_corr > 0.9:
            return 'odd'
    except Exception:
        pass

    return 'none'


def _classify_frequency_content(y: np.ndarray, prim: dict = None) -> str:
    """Classify based on spectral characteristics."""
    n = len(y)
    if n < 20:
        return 'unknown'

    # Use precomputed if available
    if prim:
        centroid = prim.get('spectral_centroid')
        bandwidth = prim.get('spectral_bandwidth')
        if centroid is not None:
            nyquist = n / 2
            relative_centroid = centroid / nyquist if nyquist > 0 else 0

            if relative_centroid < 0.1:
                return 'lowpass'
            elif relative_centroid > 0.7:
                return 'highpass'

            if bandwidth is not None:
                relative_bandwidth = bandwidth / nyquist if nyquist > 0 else 0
                if relative_bandwidth < 0.1:
                    return 'narrowband'
                elif relative_bandwidth > 0.5:
                    return 'broadband'
            return 'bandpass'

    # Compute from signal
    try:
        y_centered = y - np.mean(y)
        fft_result = np.fft.fft(y_centered)
        power = np.abs(fft_result[:n//2]) ** 2
        freqs = np.arange(n//2)

        total_power = power.sum()
        if total_power > 0:
            # Spectral centroid
            centroid = np.sum(freqs * power) / total_power
            nyquist = n / 2

            relative_centroid = centroid / nyquist

            if relative_centroid < 0.1:
                return 'lowpass'
            elif relative_centroid > 0.7:
                return 'highpass'

            # Check bandwidth
            variance = np.sum(((freqs - centroid) ** 2) * power) / total_power
            bandwidth = np.sqrt(variance)
            relative_bandwidth = bandwidth / nyquist

            if relative_bandwidth < 0.1:
                return 'narrowband'
            elif relative_bandwidth > 0.5:
                return 'broadband'

            return 'bandpass'
    except Exception:
        pass

    return 'unknown'
