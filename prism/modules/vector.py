"""
prism/modules/vector.py - Vector layer computation

Pure functions, no I/O. Computes behavioral metrics per signal.

Metrics include:
- Hurst exponent (long-range dependence)
- Sample/permutation/spectral entropy
- Lyapunov exponent (chaos)
- RQA metrics (recurrence)
- GARCH volatility
- Spectral features
- Hilbert transform features
"""

import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import computation engines
try:
    from prism.engines.vector import (
        HurstEngine,
        EntropyEngine,
        LyapunovEngine,
        RQAEngine,
        GARCHEngine,
        SpectralEngine,
        WaveletEngine,
        RealizedVolEngine,
        HilbertEngine,
    )
    ENGINES_AVAILABLE = True
except ImportError:
    ENGINES_AVAILABLE = False


def compute_vector_features(
    df: pl.DataFrame,
    entity_col: str = "entity_id",
    time_col: str = "timestamp",
    signal_col: str = "signal_id",
    value_col: str = "value",
    window_size: int = 252,
    stride: int = 21,
    engines: Optional[List[str]] = None,
    n_workers: int = -1,
) -> pl.DataFrame:
    """
    Compute behavioral metrics for each signal.

    Args:
        df: Input DataFrame with columns [entity_col, time_col, signal_col, value_col]
        entity_col: Entity identifier column
        time_col: Time/cycle column
        signal_col: Signal identifier column
        value_col: Value column
        window_size: Rolling window size
        stride: Window stride
        engines: List of engines to run (None = all)
        n_workers: Parallel workers (-1 = all CPUs)

    Returns:
        DataFrame with vector features per (entity, signal, window)
    """
    if not ENGINES_AVAILABLE:
        return _compute_vector_features_simple(
            df, entity_col, time_col, signal_col, value_col, window_size, stride
        )

    # Get unique (entity, signal) pairs
    pairs = df.select([entity_col, signal_col]).unique().to_dicts()

    results = []

    for pair in pairs:
        entity_id = pair[entity_col]
        signal_id = pair[signal_col]

        # Get signal data
        signal_df = df.filter(
            (pl.col(entity_col) == entity_id) & (pl.col(signal_col) == signal_id)
        ).sort(time_col)

        values = signal_df[value_col].to_numpy()
        times = signal_df[time_col].to_numpy()

        if len(values) < window_size:
            continue

        # Compute windows
        for start in range(0, len(values) - window_size + 1, stride):
            window_values = values[start:start + window_size]
            window_time = times[start + window_size - 1]  # End of window

            metrics = _compute_window_metrics(window_values, engines)

            results.append({
                entity_col: entity_id,
                signal_col: signal_id,
                time_col: window_time,
                "window_start": times[start],
                "window_size": window_size,
                **metrics,
            })

    if not results:
        return pl.DataFrame()

    return pl.DataFrame(results)


def _compute_window_metrics(
    values: np.ndarray,
    engines: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute all metrics for a single window."""
    metrics = {}

    # Default engines
    if engines is None:
        engines = ['hurst', 'entropy', 'spectral', 'garch', 'realized_vol']

    # Hurst exponent
    if 'hurst' in engines:
        try:
            from nolds import hurst_rs
            metrics['hurst_exponent'] = hurst_rs(values)
        except Exception:
            metrics['hurst_exponent'] = np.nan

    # Entropy measures
    if 'entropy' in engines:
        try:
            from antropy import sample_entropy, perm_entropy, spectral_entropy
            metrics['sample_entropy'] = sample_entropy(values)
            metrics['permutation_entropy'] = perm_entropy(values, normalize=True)
            metrics['spectral_entropy'] = spectral_entropy(values, sf=1.0, normalize=True)
        except Exception:
            metrics['sample_entropy'] = np.nan
            metrics['permutation_entropy'] = np.nan
            metrics['spectral_entropy'] = np.nan

    # Spectral features
    if 'spectral' in engines:
        try:
            fft = np.fft.rfft(values)
            freqs = np.fft.rfftfreq(len(values))
            power = np.abs(fft) ** 2

            # Dominant frequency
            metrics['dominant_freq'] = freqs[np.argmax(power[1:])+1] if len(power) > 1 else 0

            # Spectral centroid
            metrics['spectral_centroid'] = np.sum(freqs * power) / (np.sum(power) + 1e-10)

            # Spectral bandwidth
            centroid = metrics['spectral_centroid']
            metrics['spectral_bandwidth'] = np.sqrt(
                np.sum(((freqs - centroid) ** 2) * power) / (np.sum(power) + 1e-10)
            )
        except Exception:
            metrics['dominant_freq'] = np.nan
            metrics['spectral_centroid'] = np.nan
            metrics['spectral_bandwidth'] = np.nan

    # GARCH volatility
    if 'garch' in engines:
        try:
            from arch import arch_model
            returns = np.diff(values) / (values[:-1] + 1e-10)
            if len(returns) > 10 and np.std(returns) > 1e-10:
                model = arch_model(returns * 100, vol='GARCH', p=1, q=1, rescale=False)
                res = model.fit(disp='off', show_warning=False)
                metrics['garch_omega'] = res.params.get('omega', np.nan)
                metrics['garch_alpha'] = res.params.get('alpha[1]', np.nan)
                metrics['garch_beta'] = res.params.get('beta[1]', np.nan)
            else:
                metrics['garch_omega'] = np.nan
                metrics['garch_alpha'] = np.nan
                metrics['garch_beta'] = np.nan
        except Exception:
            metrics['garch_omega'] = np.nan
            metrics['garch_alpha'] = np.nan
            metrics['garch_beta'] = np.nan

    # Realized volatility
    if 'realized_vol' in engines:
        try:
            returns = np.diff(values) / (values[:-1] + 1e-10)
            metrics['realized_volatility'] = np.sqrt(np.sum(returns ** 2))
        except Exception:
            metrics['realized_volatility'] = np.nan

    # Lyapunov exponent
    if 'lyapunov' in engines:
        try:
            from nolds import lyap_r
            metrics['lyapunov_exponent'] = lyap_r(values)
        except Exception:
            metrics['lyapunov_exponent'] = np.nan

    # RQA (computationally expensive)
    if 'rqa' in engines:
        try:
            from pyrqa.time_series import TimeSeries
            from pyrqa.settings import Settings
            from pyrqa.computation import RQAComputation

            ts = TimeSeries(values, embedding_dimension=2, time_delay=1)
            settings = Settings(ts, analysis_type='Classic', similarity_measure='EuclideanMetric',
                               theiler_corrector=1, neighbourhood='FixedRadius', radius=0.1)
            computation = RQAComputation.create(settings)
            result = computation.run()

            metrics['recurrence_rate'] = result.recurrence_rate
            metrics['determinism'] = result.determinism
            metrics['laminarity'] = result.laminarity
        except Exception:
            metrics['recurrence_rate'] = np.nan
            metrics['determinism'] = np.nan
            metrics['laminarity'] = np.nan

    return metrics


def _compute_vector_features_simple(
    df: pl.DataFrame,
    entity_col: str,
    time_col: str,
    signal_col: str,
    value_col: str,
    window_size: int,
    stride: int,
) -> pl.DataFrame:
    """Simplified vector features without full engine stack."""
    pairs = df.select([entity_col, signal_col]).unique().to_dicts()

    results = []

    for pair in pairs:
        entity_id = pair[entity_col]
        signal_id = pair[signal_col]

        signal_df = df.filter(
            (pl.col(entity_col) == entity_id) & (pl.col(signal_col) == signal_id)
        ).sort(time_col)

        values = signal_df[value_col].to_numpy()
        times = signal_df[time_col].to_numpy()

        if len(values) < window_size:
            continue

        for start in range(0, len(values) - window_size + 1, stride):
            window_values = values[start:start + window_size]
            window_time = times[start + window_size - 1]

            # Basic statistics
            results.append({
                entity_col: entity_id,
                signal_col: signal_id,
                time_col: window_time,
                "window_start": times[start],
                "window_size": window_size,
                "mean": float(np.mean(window_values)),
                "std": float(np.std(window_values)),
                "min": float(np.min(window_values)),
                "max": float(np.max(window_values)),
                "skew": float(_skewness(window_values)),
                "kurtosis": float(_kurtosis(window_values)),
            })

    if not results:
        return pl.DataFrame()

    return pl.DataFrame(results)


def _skewness(x: np.ndarray) -> float:
    """Compute skewness."""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-10:
        return 0.0
    return (np.sum((x - mean) ** 3) / n) / (std ** 3)


def _kurtosis(x: np.ndarray) -> float:
    """Compute kurtosis."""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-10:
        return 0.0
    return (np.sum((x - mean) ** 4) / n) / (std ** 4) - 3
