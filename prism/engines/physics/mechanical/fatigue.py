"""
Fatigue Analysis

Stress-life, strain-life, crack growth models.
Rainflow counting, damage accumulation.
"""

import numpy as np
from typing import Dict, Any, List, Optional


def sn_curve(S: float, S_f: float = 1000e6, b: float = -0.1,
             N_f: float = 1e6) -> Dict[str, Any]:
    """
    Basquin's S-N curve for high-cycle fatigue.

    S = S_f * (2N)^b

    Args:
        S: Stress amplitude [Pa]
        S_f: Fatigue strength coefficient [Pa]
        b: Fatigue strength exponent (typically -0.05 to -0.15)
        N_f: Reference cycles (for endurance limit)

    Returns:
        N: Cycles to failure
        damage_per_cycle: 1/N
    """
    if S <= 0:
        return {'N': float('inf'), 'damage_per_cycle': 0.0}

    # Solve for N: S = S_f * (2N)^b
    # (S/S_f)^(1/b) = 2N
    # N = 0.5 * (S/S_f)^(1/b)
    N = 0.5 * (S / S_f) ** (1 / b)

    return {
        'N': float(max(N, 1)),
        'damage_per_cycle': float(1 / max(N, 1)),
        'stress_amplitude': S,
        'S_f': S_f,
        'b': b,
    }


def strain_life(delta_epsilon: float, sigma_f: float = 1000e6,
                epsilon_f: float = 0.5, b: float = -0.1,
                c: float = -0.6, E: float = 200e9) -> Dict[str, Any]:
    """
    Coffin-Manson strain-life equation.

    Δε/2 = (σ'_f/E)(2N)^b + ε'_f(2N)^c

    Args:
        delta_epsilon: Total strain range
        sigma_f: Fatigue strength coefficient [Pa]
        epsilon_f: Fatigue ductility coefficient
        b: Fatigue strength exponent
        c: Fatigue ductility exponent
        E: Young's modulus [Pa]

    Returns:
        N: Cycles to failure
        elastic_strain: Elastic component
        plastic_strain: Plastic component
    """
    epsilon_a = delta_epsilon / 2

    # Iterative solve (Newton-Raphson)
    N = 1000  # Initial guess

    for _ in range(100):
        two_N = 2 * N
        elastic = (sigma_f / E) * two_N ** b
        plastic = epsilon_f * two_N ** c
        f = elastic + plastic - epsilon_a

        # Derivative
        df = b * elastic / N + c * plastic / N

        if abs(df) < 1e-20:
            break

        N_new = N - f / df
        if abs(N_new - N) / max(N, 1) < 1e-6:
            N = N_new
            break
        N = max(N_new, 1)

    two_N = 2 * N
    elastic = (sigma_f / E) * two_N ** b
    plastic = epsilon_f * two_N ** c

    return {
        'N': float(N),
        'elastic_strain_amplitude': float(elastic),
        'plastic_strain_amplitude': float(plastic),
        'total_strain_amplitude': float(elastic + plastic),
        'transition_life': float((epsilon_f * E / sigma_f) ** (1 / (b - c))),
    }


def rainflow_count(signal: np.ndarray) -> Dict[str, Any]:
    """
    Rainflow counting for fatigue analysis.

    Extracts stress cycles from irregular loading.

    Args:
        signal: Stress or strain time history

    Returns:
        ranges: Cycle ranges
        means: Cycle means
        counts: Cycle counts (0.5 or 1.0)
    """
    signal = np.asarray(signal)

    # Find peaks and valleys
    diff = np.diff(signal)
    peaks_idx = []

    for i in range(1, len(signal) - 1):
        if (diff[i-1] > 0 and diff[i] < 0) or (diff[i-1] < 0 and diff[i] > 0):
            peaks_idx.append(i)

    # Include endpoints
    peaks_idx = [0] + peaks_idx + [len(signal) - 1]
    peaks = signal[peaks_idx]

    # Simple rainflow extraction
    ranges = []
    means = []
    counts = []

    stack = []
    for i, p in enumerate(peaks):
        stack.append(p)

        while len(stack) >= 3:
            s1, s2, s3 = stack[-3], stack[-2], stack[-1]
            r1 = abs(s2 - s1)
            r2 = abs(s3 - s2)

            if r1 <= r2:
                ranges.append(r1)
                means.append((s1 + s2) / 2)
                counts.append(1.0)
                stack.pop(-2)
                stack.pop(-2)
                stack.append(s3)
            else:
                break

    # Remaining half cycles
    for i in range(len(stack) - 1):
        r = abs(stack[i+1] - stack[i])
        m = (stack[i] + stack[i+1]) / 2
        ranges.append(r)
        means.append(m)
        counts.append(0.5)

    return {
        'ranges': ranges,
        'means': means,
        'counts': counts,
        'n_cycles': sum(counts),
        'max_range': max(ranges) if ranges else 0,
        'min_range': min(ranges) if ranges else 0,
    }


def miner_damage(ranges: List[float], counts: List[float],
                 S_f: float = 1000e6, b: float = -0.1) -> Dict[str, Any]:
    """
    Miner's rule for cumulative damage.

    D = Σ(n_i / N_i)
    Failure when D >= 1

    Args:
        ranges: Stress ranges from rainflow
        counts: Cycle counts
        S_f: S-N curve parameter
        b: S-N curve exponent

    Returns:
        damage: Total accumulated damage
        remaining_life: Fraction of life remaining
    """
    total_damage = 0.0
    damage_per_range = []

    for r, n in zip(ranges, counts):
        S = r / 2  # Amplitude
        N_i = 0.5 * (S / S_f) ** (1 / b) if S > 0 else float('inf')
        d_i = n / N_i
        total_damage += d_i
        damage_per_range.append(d_i)

    return {
        'damage': float(total_damage),
        'remaining_life': float(max(0, 1 - total_damage)),
        'failed': total_damage >= 1.0,
        'damage_per_range': damage_per_range,
    }


def paris_law(da_dN: float = None, delta_K: float = None,
              C: float = 1e-11, m: float = 3.0,
              a: float = None, a_c: float = None) -> Dict[str, Any]:
    """
    Paris law for crack growth.

    da/dN = C * (ΔK)^m

    Args:
        da_dN: Crack growth rate [m/cycle]
        delta_K: Stress intensity factor range [Pa√m]
        C: Paris law coefficient
        m: Paris law exponent
        a: Current crack length [m]
        a_c: Critical crack length [m]

    Returns:
        da_dN or delta_K (whichever not provided)
        remaining_cycles: If a and a_c provided
    """
    result = {'C': C, 'm': m}

    if da_dN is not None and delta_K is None:
        delta_K = (da_dN / C) ** (1 / m)
        result['delta_K'] = float(delta_K)
        result['da_dN'] = da_dN

    elif delta_K is not None:
        da_dN = C * delta_K ** m
        result['da_dN'] = float(da_dN)
        result['delta_K'] = delta_K

    if a is not None and a_c is not None and delta_K is not None:
        # Approximate remaining cycles (constant ΔK assumption)
        N_remaining = (a_c - a) / (C * delta_K ** m) if C * delta_K ** m > 0 else float('inf')
        result['remaining_cycles'] = float(max(0, N_remaining))
        result['crack_length'] = a
        result['critical_length'] = a_c

    return result


def compute(signal: np.ndarray = None, S_f: float = 1000e6,
            b: float = -0.1, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for fatigue analysis.

    Args:
        signal: Stress time history
        S_f: S-N curve coefficient
        b: S-N curve exponent
    """
    if signal is not None:
        signal = np.asarray(signal)
        rf = rainflow_count(signal)
        damage = miner_damage(rf['ranges'], rf['counts'], S_f, b)
        return {
            **rf,
            **damage,
        }

    return {'error': 'Provide stress signal for fatigue analysis'}
