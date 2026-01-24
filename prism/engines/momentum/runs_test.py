"""
Runs Test for Momentum/Trend Detection
=======================================

The Wald-Wolfowitz runs test measures randomness in a sequence.
A "run" is a consecutive sequence of similar observations.

For momentum detection:
    - Fewer runs than expected -> trending (signal persists in direction)
    - More runs than expected -> reverting (signal switches direction often)
    - Expected runs -> random walk

Formula:
    Expected runs: E(R) = (2*n_pos*n_neg)/(n_pos + n_neg) + 1
    Variance: Var(R) = (2*n_pos*n_neg*(2*n_pos*n_neg - n))/(n^2*(n-1))
    Z-score: Z = (R - E(R)) / sqrt(Var(R))

Interpretation:
    - Z < -2: Significantly fewer runs -> trending
    - Z > +2: Significantly more runs -> reverting
    - |Z| < 2: No significant pattern

Normalization for characterize.py:
    Output is normalized to 0-1 where:
    - 0 = reverting (many runs, high switching)
    - 1 = trending (few runs, directional persistence)

    normalized = 1 - (runs_observed / runs_max_possible)
    Or using Z-score: normalized = 0.5 - (Z / 6)  # clips Z to [-3, 3]

Supports three computation modes:
    - static: Entire signal -> single value
    - windowed: Rolling windows -> time series
    - point: At time t -> single value

References:
    Wald, A. & Wolfowitz, J. (1940). On a test whether two samples
    are from the same population.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional


def compute(
    series: np.ndarray,
    mode: str = 'static',
    t: Optional[int] = None,
    window_size: int = 200,
    step_size: int = 20,
    threshold: str = 'median',  # 'median', 'mean', or 'zero'
) -> Dict[str, Any]:
    """
    Compute runs test for momentum detection.

    Args:
        series: 1D numpy array of observations
        mode: 'static', 'windowed', or 'point'
        t: Time index for point mode
        window_size: Window size for windowed/point modes
        step_size: Step between windows for windowed mode
        threshold: How to binarize series ('median', 'mean', 'zero', or float)

    Returns:
        mode='static': {'momentum_score': float, 'z_score': float, 'n_runs': int, ...}
        mode='windowed': {'momentum_score': array, 'z_score': array, 't': array, ...}
        mode='point': {'momentum_score': float, 'z_score': float, 't': int, ...}
    """
    series = np.asarray(series).flatten()

    if mode == 'static':
        return _compute_static(series, threshold)
    elif mode == 'windowed':
        return _compute_windowed(series, window_size, step_size, threshold)
    elif mode == 'point':
        return _compute_point(series, t, window_size, threshold)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'static', 'windowed', or 'point'.")


def _compute_static(series: np.ndarray, threshold: str = 'median') -> Dict[str, Any]:
    """Compute runs test on entire signal."""
    n = len(series)

    if n < 10:
        return {
            'momentum_score': 0.5,
            'z_score': 0.0,
            'n_runs': 0,
            'expected_runs': 0.0,
            'n_positive': 0,
            'n_negative': 0,
            'p_value': 1.0,
            'interpretation': 'insufficient data',
        }

    # Binarize the series
    binary = _binarize(series, threshold)

    # Count runs
    n_runs = _count_runs(binary)

    # Count positives and negatives
    n_pos = np.sum(binary == 1)
    n_neg = np.sum(binary == 0)

    if n_pos == 0 or n_neg == 0:
        return {
            'momentum_score': 0.5,
            'z_score': 0.0,
            'n_runs': n_runs,
            'expected_runs': 0.0,
            'n_positive': n_pos,
            'n_negative': n_neg,
            'p_value': 1.0,
            'interpretation': 'all same sign',
        }

    # Expected runs and variance
    expected_runs = (2 * n_pos * n_neg) / (n_pos + n_neg) + 1
    variance = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n**2 * (n - 1))

    if variance <= 0:
        z_score = 0.0
    else:
        z_score = (n_runs - expected_runs) / np.sqrt(variance)

    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    # Interpretation
    if z_score < -2:
        interpretation = 'trending'
    elif z_score > 2:
        interpretation = 'reverting'
    else:
        interpretation = 'random'

    # Normalize to 0-1 (reverting=0, trending=1)
    # Z typically in [-4, 4], we map to [0, 1]
    # Z = -4 -> score = 1 (strong trending)
    # Z = 0 -> score = 0.5 (random)
    # Z = +4 -> score = 0 (strong reverting)
    momentum_score = float(np.clip(0.5 - z_score / 8, 0, 1))

    return {
        'momentum_score': momentum_score,
        'z_score': float(z_score),
        'n_runs': int(n_runs),
        'expected_runs': float(expected_runs),
        'n_positive': int(n_pos),
        'n_negative': int(n_neg),
        'p_value': float(p_value),
        'interpretation': interpretation,
    }


def _compute_windowed(
    series: np.ndarray,
    window_size: int,
    step_size: int,
    threshold: str = 'median',
) -> Dict[str, Any]:
    """Compute runs test over rolling windows."""
    n = len(series)

    if n < window_size:
        return {
            'momentum_score': np.array([]),
            'z_score': np.array([]),
            'n_runs': np.array([]),
            't': np.array([]),
            'window_size': window_size,
            'step_size': step_size,
        }

    t_values = []
    momentum_values = []
    z_values = []
    runs_values = []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window = series[start:end]

        result = _compute_static(window, threshold)

        t_values.append(start + window_size // 2)
        momentum_values.append(result['momentum_score'])
        z_values.append(result['z_score'])
        runs_values.append(result['n_runs'])

    return {
        'momentum_score': np.array(momentum_values),
        'z_score': np.array(z_values),
        'n_runs': np.array(runs_values),
        't': np.array(t_values),
        'window_size': window_size,
        'step_size': step_size,
    }


def _compute_point(
    series: np.ndarray,
    t: int,
    window_size: int,
    threshold: str = 'median',
) -> Dict[str, Any]:
    """Compute runs test at specific time t."""
    if t is None:
        raise ValueError("t is required for point mode")

    n = len(series)

    # Center window on t
    half_window = window_size // 2
    start = max(0, t - half_window)
    end = min(n, start + window_size)

    if end - start < window_size:
        start = max(0, end - window_size)

    window = series[start:end]

    if len(window) < 10:
        return {
            'momentum_score': 0.5,
            'z_score': 0.0,
            'n_runs': 0,
            't': t,
            'window_start': start,
            'window_end': end,
        }

    result = _compute_static(window, threshold)
    result['t'] = t
    result['window_start'] = start
    result['window_end'] = end

    return result


def _binarize(series: np.ndarray, threshold: str = 'median') -> np.ndarray:
    """Convert series to binary based on threshold."""
    if threshold == 'median':
        thresh_val = np.median(series)
    elif threshold == 'mean':
        thresh_val = np.mean(series)
    elif threshold == 'zero':
        thresh_val = 0
    elif isinstance(threshold, (int, float)):
        thresh_val = threshold
    else:
        raise ValueError(f"Unknown threshold: {threshold}")

    return (series > thresh_val).astype(int)


def _count_runs(binary: np.ndarray) -> int:
    """Count the number of runs in a binary sequence."""
    if len(binary) == 0:
        return 0

    # A run starts when the value changes from the previous value
    # First element is always the start of a run
    runs = 1 + np.sum(binary[1:] != binary[:-1])

    return int(runs)


# -----------------------------------------------------------------------------
# Alternative: Runs test on returns/differences
# -----------------------------------------------------------------------------

def compute_on_returns(
    series: np.ndarray,
    mode: str = 'static',
    t: Optional[int] = None,
    window_size: int = 200,
    step_size: int = 20,
) -> Dict[str, Any]:
    """
    Compute runs test on the sign of returns (differences).

    This is often more meaningful for momentum detection:
    - Measures how often the direction of change switches
    - Positive return followed by positive return = same run
    - Sign change = new run

    Args:
        series: 1D numpy array of prices/values
        mode: 'static', 'windowed', or 'point'
        t: Time index for point mode
        window_size: Window size for windowed/point modes
        step_size: Step between windows for windowed mode

    Returns:
        Same as compute() but based on return signs
    """
    series = np.asarray(series).flatten()

    if len(series) < 2:
        return {
            'momentum_score': 0.5,
            'z_score': 0.0,
            'n_runs': 0,
            'interpretation': 'insufficient data',
        }

    # Compute returns
    returns = np.diff(series)

    # Use zero threshold for returns (positive vs negative)
    return compute(returns, mode=mode, t=t, window_size=window_size,
                   step_size=step_size, threshold='zero')
