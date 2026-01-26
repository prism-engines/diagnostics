"""
Power Quality Analysis

THD, power factor, harmonics, voltage sags/swells.
"""

import numpy as np
from typing import Dict, Any, Optional


def harmonic_analysis(signal: np.ndarray, fs: float,
                      fundamental_freq: float = 50.0,
                      n_harmonics: int = 40) -> Dict[str, Any]:
    """
    Analyze harmonic content of voltage/current waveform.

    Args:
        signal: Voltage or current waveform
        fs: Sampling frequency [Hz]
        fundamental_freq: Fundamental frequency [Hz] (50 or 60)
        n_harmonics: Number of harmonics to analyze

    Returns:
        harmonics: Amplitude of each harmonic
        thd: Total harmonic distortion [%]
        individual_hd: Individual harmonic distortion [%]
    """
    signal = np.asarray(signal)
    n = len(signal)

    # FFT
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    magnitude = np.abs(fft_vals) * 2 / n

    # Extract harmonics
    df = freqs[1] - freqs[0]
    harmonics = []
    harmonic_freqs = []

    for h in range(1, n_harmonics + 1):
        f_h = h * fundamental_freq
        idx = int(round(f_h / df))
        if 0 <= idx < len(magnitude):
            harmonics.append(float(magnitude[idx]))
        else:
            harmonics.append(0.0)
        harmonic_freqs.append(f_h)

    # Fundamental amplitude (1st harmonic)
    V1 = harmonics[0] if harmonics[0] > 0 else 1e-10

    # THD calculation
    harmonic_sum_sq = sum(h ** 2 for h in harmonics[1:])
    THD = 100 * np.sqrt(harmonic_sum_sq) / V1

    # Individual harmonic distortion
    individual_hd = [100 * h / V1 for h in harmonics]

    return {
        'harmonics': harmonics,
        'harmonic_frequencies': harmonic_freqs,
        'fundamental': float(V1),
        'thd': float(THD),
        'individual_hd': individual_hd,
        'dominant_harmonic': int(np.argmax(harmonics[1:]) + 2),
        'n_harmonics': n_harmonics,
    }


def power_factor(v: np.ndarray, i: np.ndarray, fs: float,
                 fundamental_freq: float = 50.0) -> Dict[str, Any]:
    """
    Calculate power factor and power components.

    Args:
        v: Voltage waveform [V]
        i: Current waveform [A]
        fs: Sampling frequency [Hz]
        fundamental_freq: Fundamental frequency [Hz]

    Returns:
        pf: Power factor (total)
        pf_displacement: Displacement power factor (cos φ)
        pf_distortion: Distortion power factor
        P: Active power [W]
        Q: Reactive power [VAr]
        S: Apparent power [VA]
        D: Distortion power [VA]
    """
    v = np.asarray(v)
    i = np.asarray(i)

    # RMS values
    V_rms = np.sqrt(np.mean(v ** 2))
    I_rms = np.sqrt(np.mean(i ** 2))

    # Apparent power
    S = V_rms * I_rms

    # Active power (average of instantaneous power)
    P = np.mean(v * i)

    # Power factor
    pf = P / S if S > 0 else 1.0

    # Fundamental components
    n = len(v)
    t = np.arange(n) / fs
    omega = 2 * np.pi * fundamental_freq

    # Extract fundamental using correlation
    v_cos = 2 * np.mean(v * np.cos(omega * t))
    v_sin = 2 * np.mean(v * np.sin(omega * t))
    i_cos = 2 * np.mean(i * np.cos(omega * t))
    i_sin = 2 * np.mean(i * np.sin(omega * t))

    V1 = np.sqrt(v_cos ** 2 + v_sin ** 2) / np.sqrt(2)
    I1 = np.sqrt(i_cos ** 2 + i_sin ** 2) / np.sqrt(2)

    phi_v = np.arctan2(v_sin, v_cos)
    phi_i = np.arctan2(i_sin, i_cos)
    phi = phi_v - phi_i

    # Displacement power factor
    pf_displacement = np.cos(phi)

    # Distortion power factor
    pf_distortion = pf / pf_displacement if abs(pf_displacement) > 0.001 else 1.0

    # Reactive power (fundamental)
    Q = V1 * I1 * np.sin(phi)

    # Distortion power
    D = np.sqrt(max(0, S ** 2 - P ** 2 - Q ** 2))

    return {
        'pf': float(pf),
        'pf_displacement': float(pf_displacement),
        'pf_distortion': float(pf_distortion),
        'phase_angle': float(phi),
        'phase_angle_deg': float(np.degrees(phi)),
        'P': float(P),
        'Q': float(Q),
        'S': float(S),
        'D': float(D),
        'V_rms': float(V_rms),
        'I_rms': float(I_rms),
        'pf_type': 'lagging' if phi > 0 else 'leading',
    }


def voltage_events(signal: np.ndarray, fs: float,
                   nominal_voltage: float, fundamental_freq: float = 50.0,
                   sag_threshold: float = 0.9,
                   swell_threshold: float = 1.1) -> Dict[str, Any]:
    """
    Detect voltage sags, swells, and interruptions.

    Args:
        signal: Voltage waveform [V]
        fs: Sampling frequency [Hz]
        nominal_voltage: Nominal RMS voltage [V]
        fundamental_freq: Fundamental frequency [Hz]
        sag_threshold: Threshold for sag (fraction of nominal)
        swell_threshold: Threshold for swell

    Returns:
        sags: List of sag events (start_time, duration, depth)
        swells: List of swell events
        interruptions: List of interruption events (<0.1 pu)
    """
    signal = np.asarray(signal)

    # Calculate RMS in sliding windows (1 cycle)
    samples_per_cycle = int(fs / fundamental_freq)
    n_cycles = len(signal) // samples_per_cycle

    rms_per_cycle = []
    for i in range(n_cycles):
        start = i * samples_per_cycle
        end = start + samples_per_cycle
        rms = np.sqrt(np.mean(signal[start:end] ** 2))
        rms_per_cycle.append(rms)

    rms_per_cycle = np.array(rms_per_cycle)
    pu = rms_per_cycle / nominal_voltage

    # Time for each cycle
    cycle_times = np.arange(n_cycles) / fundamental_freq

    # Detect events
    sags = []
    swells = []
    interruptions = []

    in_sag = False
    in_swell = False
    event_start = 0

    for i, v in enumerate(pu):
        if v < 0.1:  # Interruption
            if not in_sag:
                in_sag = True
                event_start = i
        elif v < sag_threshold:  # Sag
            if not in_sag:
                in_sag = True
                event_start = i
        elif v > swell_threshold:  # Swell
            if not in_swell:
                in_swell = True
                event_start = i
        else:
            if in_sag:
                depth = 1 - np.min(pu[event_start:i])
                duration = (i - event_start) / fundamental_freq
                if np.min(pu[event_start:i]) < 0.1:
                    interruptions.append({
                        'start_time': float(cycle_times[event_start]),
                        'duration': float(duration),
                        'min_voltage_pu': float(np.min(pu[event_start:i])),
                    })
                else:
                    sags.append({
                        'start_time': float(cycle_times[event_start]),
                        'duration': float(duration),
                        'depth_pu': float(depth),
                        'min_voltage_pu': float(np.min(pu[event_start:i])),
                    })
                in_sag = False

            if in_swell:
                magnitude = np.max(pu[event_start:i]) - 1
                duration = (i - event_start) / fundamental_freq
                swells.append({
                    'start_time': float(cycle_times[event_start]),
                    'duration': float(duration),
                    'magnitude_pu': float(magnitude),
                    'max_voltage_pu': float(np.max(pu[event_start:i])),
                })
                in_swell = False

    return {
        'sags': sags,
        'swells': swells,
        'interruptions': interruptions,
        'n_sags': len(sags),
        'n_swells': len(swells),
        'n_interruptions': len(interruptions),
        'rms_per_cycle': rms_per_cycle.tolist(),
        'pu_per_cycle': pu.tolist(),
    }


def flicker(signal: np.ndarray, fs: float,
            nominal_voltage: float) -> Dict[str, Any]:
    """
    Calculate flicker indicators (simplified).

    Pst: Short-term flicker severity (10-minute)

    Args:
        signal: Voltage waveform [V]
        fs: Sampling frequency [Hz]
        nominal_voltage: Nominal voltage [V]

    Returns:
        pst: Short-term flicker severity
        max_fluctuation: Maximum voltage fluctuation [%]
    """
    signal = np.asarray(signal)

    # Demodulate to extract envelope
    from scipy import signal as sig

    # Bandpass for flicker range (0.5 to 35 Hz)
    nyq = fs / 2
    if nyq > 35:
        b, a = sig.butter(4, [0.5 / nyq, 35 / nyq], btype='band')
        envelope = np.abs(sig.filtfilt(b, a, signal))
    else:
        envelope = np.abs(signal)

    # Normalize
    envelope = envelope / nominal_voltage

    # Simplified Pst (would need full IEC 61000-4-15 for accurate)
    fluctuation = np.std(envelope) * 100  # %
    max_fluct = (np.max(envelope) - np.min(envelope)) / np.mean(envelope) * 100

    # Very simplified Pst estimate
    Pst = fluctuation * 0.1  # Rough approximation

    return {
        'pst': float(Pst),
        'max_fluctuation_percent': float(max_fluct),
        'std_fluctuation_percent': float(fluctuation),
        'note': 'Simplified flicker calculation, full IEC 61000-4-15 not implemented',
    }


def compute(signal: np.ndarray = None, fs: float = None,
            fundamental_freq: float = 50.0, nominal_voltage: float = None,
            voltage: np.ndarray = None, current: np.ndarray = None,
            **kwargs) -> Dict[str, Any]:
    """
    Main entry point for power quality analysis.
    """
    results = {}

    if signal is not None and fs is not None:
        results['harmonics'] = harmonic_analysis(signal, fs, fundamental_freq)

        if nominal_voltage is not None:
            results['voltage_events'] = voltage_events(
                signal, fs, nominal_voltage, fundamental_freq)
            results['flicker'] = flicker(signal, fs, nominal_voltage)

    if voltage is not None and current is not None and fs is not None:
        results['power_factor'] = power_factor(voltage, current, fs, fundamental_freq)
        results['voltage_harmonics'] = harmonic_analysis(voltage, fs, fundamental_freq)
        results['current_harmonics'] = harmonic_analysis(current, fs, fundamental_freq)

    return results if results else {'error': 'Provide signal and fs'}
