"""
Signal Typology Orchestrator
============================

Layer 1 of ORTHON framework: Classifies signals along 6 orthogonal axes.

The Six Orthogonal Axes:
    1. Memory        - Temporal persistence (Hurst, ACF decay)
    2. Periodicity   - Cyclical structure (FFT, wavelets)
    3. Volatility    - Variance dynamics (GARCH, rolling std)
    4. Discontinuity - Level shifts / Heaviside (PELT, CUSUM)
    5. Impulsivity   - Shocks / Dirac (derivative spikes, kurtosis)
    6. Complexity    - Predictability (entropy)

Architecture:
    signal_typology.py (entry point)
            │
            ▼
    orchestrator.py (this file - routes + formats)
            │
            ▼
    characterize.py (computes 6 axes)
            │
            ▼
    engine_mapping.py (selects engines)

Usage:
    from prism.typology.orchestrator import run_signal_typology

    results = run_signal_typology({'signal_1': my_array})
    print(results['axes']['signal_1'])
    print(results['engine_recommendations']['signal_1'])
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .engine_mapping import select_engines, get_primary_classification


# Axis names (canonical order)
AXIS_NAMES = [
    'memory',
    'periodicity',
    'volatility',
    'discontinuity',
    'impulsivity',
    'complexity',
]


def run_signal_typology(
    signals: Dict[str, np.ndarray],
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Main orchestrator for Signal Typology.

    Args:
        signals: Dict of {signal_name: np.array}
        config: Optional configuration overrides

    Returns:
        {
            'axes': {signal_name: {axis: score}},
            'classification': {signal_name: primary_type},
            'engine_recommendations': {signal_name: [engines]},
            'metadata': {...}
        }
    """
    # Import here to avoid circular imports
    from prism.engines.characterize import compute_all_axes

    config = config or {}

    results = {
        'axes': {},
        'classification': {},
        'engine_recommendations': {},
        'discontinuity_events': {},
        'impulse_events': {},
        'metadata': {
            'axes_computed': AXIS_NAMES,
            'version': '2.0.0',
            'computed_at': datetime.now().isoformat(),
            'n_signals': len(signals),
        }
    }

    for name, signal in signals.items():
        # Validate input
        if not isinstance(signal, np.ndarray):
            signal = np.asarray(signal, dtype=float)

        if len(signal) < 30:
            # Skip signals that are too short
            results['axes'][name] = {ax: np.nan for ax in AXIS_NAMES}
            results['classification'][name] = 'INSUFFICIENT_DATA'
            results['engine_recommendations'][name] = []
            continue

        # Compute all 6 axes via characterize module
        axis_result = compute_all_axes(signal, config)

        # Extract scores
        axis_scores = {
            'memory': axis_result['memory'],
            'periodicity': axis_result['periodicity'],
            'volatility': axis_result['volatility'],
            'discontinuity': axis_result['discontinuity'],
            'impulsivity': axis_result['impulsivity'],
            'complexity': axis_result['complexity'],
        }
        results['axes'][name] = axis_scores

        # Store event details if present
        if 'discontinuity_events' in axis_result:
            results['discontinuity_events'][name] = axis_result['discontinuity_events']
        if 'impulse_events' in axis_result:
            results['impulse_events'][name] = axis_result['impulse_events']

        # Classify primary signal type
        results['classification'][name] = get_primary_classification(axis_scores)

        # Map to recommended engines
        results['engine_recommendations'][name] = select_engines(axis_scores)

    return results


def analyze_single(
    signal: np.ndarray,
    signal_name: str = 'signal',
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Convenience function for analyzing a single signal.

    Args:
        signal: 1D numpy array
        signal_name: Name for the signal
        config: Optional configuration

    Returns:
        Dict with axes, classification, and recommendations
    """
    results = run_signal_typology({signal_name: signal}, config)

    return {
        'axes': results['axes'][signal_name],
        'classification': results['classification'][signal_name],
        'engines': results['engine_recommendations'][signal_name],
        'discontinuity_events': results['discontinuity_events'].get(signal_name, []),
        'impulse_events': results['impulse_events'].get(signal_name, []),
    }


def get_fingerprint(axis_scores: Dict[str, float]) -> np.ndarray:
    """
    Convert axis scores to a fingerprint vector.

    Args:
        axis_scores: Dict of {axis_name: score}

    Returns:
        numpy array of scores in canonical order
    """
    return np.array([axis_scores.get(ax, 0.0) for ax in AXIS_NAMES])


def fingerprint_distance(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two fingerprints.

    Args:
        fp1: First fingerprint vector
        fp2: Second fingerprint vector

    Returns:
        Distance (0 = identical, sqrt(6) = maximally different)
    """
    return float(np.linalg.norm(fp1 - fp2))


def detect_regime_change(
    previous_axes: Dict[str, float],
    current_axes: Dict[str, float],
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Detect regime change between two typology measurements.

    Args:
        previous_axes: Axis scores from previous window
        current_axes: Axis scores from current window
        threshold: Change threshold for flagging

    Returns:
        Dict with change detection results
    """
    fp_prev = get_fingerprint(previous_axes)
    fp_curr = get_fingerprint(current_axes)

    distance = fingerprint_distance(fp_prev, fp_curr)

    # Find which axes changed most
    changes = {}
    moving_axes = []
    stable_axes = []

    for ax in AXIS_NAMES:
        prev_val = previous_axes.get(ax, 0.0)
        curr_val = current_axes.get(ax, 0.0)
        delta = curr_val - prev_val

        changes[ax] = {
            'previous': prev_val,
            'current': curr_val,
            'delta': delta,
            'abs_delta': abs(delta),
        }

        if abs(delta) >= threshold:
            moving_axes.append(ax)
        else:
            stable_axes.append(ax)

    return {
        'regime_changed': distance >= threshold,
        'distance': distance,
        'threshold': threshold,
        'changes': changes,
        'moving_axes': moving_axes,
        'stable_axes': stable_axes,
    }
