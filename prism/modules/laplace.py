"""
PRISM Laplace Module
====================

Inline Laplace field computation. Called by indicator_vector
to compute field vectors from engine metrics.

This module provides the core Laplace computations that can be
applied to any signal topology of metrics.

Mathematical Foundation:
------------------------
For a metric value series V at times t:

1. GRADIENT: ∇V(t) = (V(t+1) - V(t-1)) / 2
   "How fast is the metric changing?"

2. LAPLACIAN: ∇²V(t) = V(t+1) - 2V(t) + V(t-1)
   "Is change accelerating or decelerating?"

3. DIVERGENCE: Sum of laplacians across all metrics
   SOURCE (>0) = energy injection
   SINK (<0) = energy absorption

Usage:
------
    from prism.modules.laplace import compute_laplace_for_series

    # In indicator_vector after computing metrics:
    field_rows = compute_laplace_for_series(
        indicator_id='SENSOR_01',
        dates=window_dates,
        values=metric_values,
        engine='hurst',
        metric_name='hurst_exponent',
    )
"""

import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime


def compute_gradient(values: np.ndarray) -> np.ndarray:
    """
    Compute first derivative (gradient) with consistent accuracy at boundaries.

    Interior: Central difference O(h²)
        ∇f(t) = (f(t+1) - f(t-1)) / 2

    Boundaries: One-sided second-order O(h²)
        ∇f(0) = (-3f(0) + 4f(1) - f(2)) / 2
        ∇f(n) = (3f(n) - 4f(n-1) + f(n-2)) / 2

    This avoids the ~2x noise variance at boundaries that occurs with
    first-order forward/backward differences, which can cause artificial
    sources/sinks in the Laplace field.

    Args:
        values: 1D array of values

    Returns:
        Array of gradients (same length as input)
    """
    n = len(values)
    gradient = np.full(n, np.nan)

    if n < 2:
        return gradient

    # Interior: central difference O(h²)
    if n >= 3:
        gradient[1:-1] = (values[2:] - values[:-2]) / 2.0

    # Boundaries: one-sided second-order O(h²)
    if n >= 3:
        # Forward second-order at first point
        gradient[0] = (-3*values[0] + 4*values[1] - values[2]) / 2.0
        # Backward second-order at last point
        gradient[-1] = (3*values[-1] - 4*values[-2] + values[-3]) / 2.0
    elif n == 2:
        # Fallback for very short series (first-order, scaled consistently)
        gradient[0] = (values[1] - values[0])
        gradient[-1] = (values[-1] - values[-2])

    return gradient


def compute_laplacian(values: np.ndarray) -> np.ndarray:
    """
    Compute second derivative (Laplacian) with consistent accuracy at boundaries.

    Interior: Central difference O(h²)
        ∇²f(t) = f(t+1) - 2f(t) + f(t-1)

    Boundaries: One-sided second-order O(h²) (requires n >= 4)
        ∇²f(0) = 2f(0) - 5f(1) + 4f(2) - f(3)
        ∇²f(n) = 2f(n) - 5f(n-1) + 4f(n-2) - f(n-3)

    This avoids boundary artifacts that can create false regime signals.

    Args:
        values: 1D array of values

    Returns:
        Array of laplacians (same length as input)
    """
    n = len(values)
    laplacian = np.full(n, np.nan)

    if n < 3:
        return laplacian

    # Interior: central difference O(h²)
    laplacian[1:-1] = values[2:] - 2 * values[1:-1] + values[:-2]

    # Boundaries: one-sided second-order O(h²) (requires 4+ points)
    if n >= 4:
        laplacian[0] = 2*values[0] - 5*values[1] + 4*values[2] - values[3]
        laplacian[-1] = 2*values[-1] - 5*values[-2] + 4*values[-3] - values[-4]

    return laplacian


def compute_laplace_for_series(
    indicator_id: str,
    dates: List[datetime],
    values: np.ndarray,
    engine: str,
    metric_name: str,
) -> List[Dict[str, Any]]:
    """
    Compute Laplace field quantities for a single metric series.

    This is called for each (indicator, engine, metric) combination
    after the engine has computed metrics across all windows.

    Args:
        indicator_id: The indicator identifier
        dates: List of window_end dates (sorted ascending)
        values: Array of metric values corresponding to dates
        engine: Engine name (e.g., 'hurst')
        metric_name: Metric name (e.g., 'hurst_exponent')

    Returns:
        List of dicts, one per date, with field quantities:
            - indicator_id, window_end, engine, metric_name
            - metric_value, gradient, laplacian, gradient_magnitude
    """
    n = len(values)
    if n < 3:
        # Need at least 3 points for laplacian
        return []

    # Compute field quantities
    gradient = compute_gradient(values)
    laplacian = compute_laplacian(values)

    results = []
    for i in range(n):
        # Skip if both gradient and laplacian are NaN
        if np.isnan(gradient[i]) and np.isnan(laplacian[i]):
            continue

        row = {
            'indicator_id': indicator_id,
            'window_end': dates[i],
            'engine': engine,
            'metric_name': metric_name,
            'metric_value': float(values[i]) if not np.isnan(values[i]) else None,
            'gradient': float(gradient[i]) if not np.isnan(gradient[i]) else None,
            'laplacian': float(laplacian[i]) if not np.isnan(laplacian[i]) else None,
            'gradient_magnitude': abs(float(gradient[i])) if not np.isnan(gradient[i]) else None,
        }
        results.append(row)

    return results


def compute_divergence_for_indicator(
    field_rows: List[Dict[str, Any]],
) -> Dict[datetime, Dict[str, float]]:
    """
    Compute divergence (sum of laplacians) per window for an indicator.

    Divergence = Σ ∇²V across all metrics at time t
    - Positive = SOURCE (energy injection)
    - Negative = SINK (energy absorption)

    Args:
        field_rows: List of field row dicts from compute_laplace_for_series

    Returns:
        Dict mapping window_end -> {divergence, total_gradient_mag, n_metrics}
    """
    from collections import defaultdict

    # Group by window_end
    by_window = defaultdict(list)
    for row in field_rows:
        by_window[row['window_end']].append(row)

    results = {}
    for window_end, rows in by_window.items():
        laplacians = [r['laplacian'] for r in rows if r['laplacian'] is not None]
        grad_mags = [r['gradient_magnitude'] for r in rows if r['gradient_magnitude'] is not None]

        results[window_end] = {
            'divergence': sum(laplacians) if laplacians else 0.0,
            'total_gradient_mag': sum(grad_mags) if grad_mags else 0.0,
            'mean_gradient_mag': np.mean(grad_mags) if grad_mags else 0.0,
            'n_metrics': len(rows),
        }

    return results


def add_divergence_to_field_rows(
    field_rows: List[Dict[str, Any]],
    divergence_by_window: Dict[datetime, Dict[str, float]],
    source_threshold: float = 0.1,
    sink_threshold: float = -0.1,
) -> List[Dict[str, Any]]:
    """
    Add divergence and source/sink flags to field rows.

    Args:
        field_rows: List of field row dicts
        divergence_by_window: From compute_divergence_for_indicator
        source_threshold: Divergence above this = source
        sink_threshold: Divergence below this = sink

    Returns:
        Updated field_rows with divergence columns added
    """
    for row in field_rows:
        window_end = row['window_end']
        div_info = divergence_by_window.get(window_end, {})

        row['divergence'] = div_info.get('divergence', 0.0)
        row['total_gradient_mag'] = div_info.get('total_gradient_mag', 0.0)
        row['mean_gradient_mag'] = div_info.get('mean_gradient_mag', 0.0)
        row['n_metrics'] = div_info.get('n_metrics', 0)
        row['is_source'] = row['divergence'] > source_threshold
        row['is_sink'] = row['divergence'] < sink_threshold

    return field_rows
