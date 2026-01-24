"""
Ørthon Signal Typology
======================

Signal Typology is Layer 1 of the ORTHON framework:

    Signal Typology    → WHAT is it?     (this package)
    Behavioral Geometry → HOW does it behave?
    Dynamical Systems   → WHEN/HOW does it change?
    Causal Mechanics    → WHY does it change?

The Six Orthogonal Axes:
    1. Memory        - Temporal persistence (Hurst, ACF decay)
    2. Periodicity   - Cyclical structure (FFT, wavelets)
    3. Volatility    - Variance dynamics (GARCH, rolling std)
    4. Discontinuity - Level shifts / Heaviside (PELT, CUSUM)
    5. Impulsivity   - Shocks / Dirac (derivative spikes, kurtosis)
    6. Complexity    - Predictability (entropy)

Key Design:
    CONTINUOUS DYNAMICS (smooth behavior):
        Memory, Periodicity, Volatility, Complexity

    DISCRETE EVENTS (structural behavior):
        Discontinuity (Heaviside - level shifts)
        Impulsivity (Dirac - shocks/spikes)

Usage:
    >>> from prism.typology import run_signal_typology, analyze_single
    >>>
    >>> # Analyze multiple signals
    >>> results = run_signal_typology({'signal_1': my_array})
    >>> print(results['axes']['signal_1'])
    >>> print(results['classification']['signal_1'])
    >>> print(results['engine_recommendations']['signal_1'])
    >>>
    >>> # Quick single analysis
    >>> result = analyze_single(my_array)
    >>> print(result['axes'], result['classification'])

Architecture:
    signal_typology.py (entry point)
            │
            ▼
    orchestrator.py (routes + formats)
            │
            ▼
    characterize.py (computes 6 axes)
            │
            ▼
    engine_mapping.py (selects engines)
"""

__version__ = "2.0.0"
__author__ = "Ørthon Project"

# Orchestrator (main API)
from .orchestrator import (
    run_signal_typology,
    analyze_single,
    get_fingerprint,
    fingerprint_distance,
    detect_regime_change,
    AXIS_NAMES,
)

# Engine mapping
from .engine_mapping import (
    select_engines,
    get_primary_classification,
    get_axis_weights,
    get_engine_priority,
    should_run_engine,
    ENGINE_MAP,
    COMPOUND_ENGINES,
    THRESHOLDS,
)

__all__ = [
    # Version
    "__version__",

    # Orchestrator API
    "run_signal_typology",
    "analyze_single",
    "get_fingerprint",
    "fingerprint_distance",
    "detect_regime_change",
    "AXIS_NAMES",

    # Engine mapping
    "select_engines",
    "get_primary_classification",
    "get_axis_weights",
    "get_engine_priority",
    "should_run_engine",
    "ENGINE_MAP",
    "COMPOUND_ENGINES",
    "THRESHOLDS",
]
