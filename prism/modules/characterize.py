"""
PRISM Characterize Module
=========================

Inline characterization for indicators. Called by indicator_vector
to determine which engines to run.

This module wraps the Characterizer engine for inline use.

Usage:
------
    from prism.modules.characterize import characterize_indicator

    # In indicator_vector processing loop:
    char_result = characterize_indicator(indicator_id, values, dates)
    engines_to_run = char_result.valid_engines
    has_discontinuities = char_result.has_steps or char_result.has_impulses
"""

import numpy as np
from typing import Optional, Set, Dict, Any, Tuple
from datetime import date

# Import the engine
from prism.engines.characterize import Characterizer, CharacterizationResult

# Singleton characterizer instance (stateless, can be reused)
_characterizer = None


def _get_characterizer() -> Characterizer:
    """Get or create singleton Characterizer instance."""
    global _characterizer
    if _characterizer is None:
        _characterizer = Characterizer()
    return _characterizer


def characterize_indicator(
    indicator_id: str,
    values: np.ndarray,
    dates: Optional[np.ndarray] = None,
    window_end: Optional[date] = None,
) -> CharacterizationResult:
    """
    Characterize a single indicator inline.

    This is the primary interface for indicator_vector to characterize
    indicators as they are processed, rather than reading from a
    pre-computed characterization parquet.

    Args:
        indicator_id: The indicator identifier
        values: Signal values (1D array, will be cleaned of NaN)
        dates: Optional observation dates (for frequency detection)
        window_end: Optional window end date (defaults to today)

    Returns:
        CharacterizationResult with:
            - 6 axes (stationarity, memory, periodicity, complexity, determinism, volatility)
            - valid_engines: List of engine names to run
            - metric_weights: Dict of metric weights
            - has_steps, has_impulses: Discontinuity flags
            - dynamical_class: Classification label
    """
    char = _get_characterizer()
    return char.compute(
        values=values,
        indicator_id=indicator_id,
        window_end=window_end,
        dates=dates,
    )


def get_engines_from_characterization(
    char_result: CharacterizationResult,
    core_engines: Set[str],
    conditional_engines: Set[str],
    discontinuity_engines: Set[str],
) -> Tuple[Set[str], bool]:
    """
    Determine which engines to run based on characterization result.

    Args:
        char_result: Result from characterize_indicator()
        core_engines: Set of engines that always run
        conditional_engines: Set of engines that run conditionally
        discontinuity_engines: Set of engines for discontinuity analysis

    Returns:
        Tuple of (engines_to_run, has_discontinuities)
    """
    # Start with core engines
    engines_to_run = core_engines.copy()

    # Add conditional engines that passed characterization
    valid_set = set(char_result.valid_engines)
    for engine in conditional_engines:
        if engine in valid_set:
            engines_to_run.add(engine)

    # Check for discontinuities
    has_discontinuities = char_result.has_steps or char_result.has_impulses

    # Add discontinuity engines if applicable
    if has_discontinuities:
        for engine in discontinuity_engines:
            if engine in valid_set:
                engines_to_run.add(engine)

    return engines_to_run, has_discontinuities


def get_characterization_summary(char_result: CharacterizationResult) -> Dict[str, Any]:
    """
    Get a summary dict from characterization result for logging/storage.

    Args:
        char_result: Result from characterize_indicator()

    Returns:
        Dict with key characterization fields
    """
    return {
        'indicator_id': char_result.indicator_id,
        'dynamical_class': char_result.dynamical_class,
        'ax_stationarity': char_result.ax_stationarity,
        'ax_memory': char_result.ax_memory,
        'ax_periodicity': char_result.ax_periodicity,
        'ax_complexity': char_result.ax_complexity,
        'ax_determinism': char_result.ax_determinism,
        'ax_volatility': char_result.ax_volatility,
        'valid_engines': char_result.valid_engines,
        'has_steps': char_result.has_steps,
        'has_impulses': char_result.has_impulses,
        'n_breaks': char_result.n_breaks,
        'return_method': char_result.return_method,
        'frequency': char_result.frequency,
        'is_step_function': char_result.is_step_function,
    }
