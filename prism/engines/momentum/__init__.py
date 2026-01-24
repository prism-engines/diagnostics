"""
Momentum Engines
================

Engines for measuring directional persistence and trend detection.

Key distinction from Memory:
- Memory (Hurst): Does the signal remember past VALUES? (autocorrelation)
- Momentum (Runs): Does the signal continue in the same DIRECTION? (sequential signs)

A signal can be:
- High memory, low momentum: Smooth, autocorrelated, but direction changes often
- Low memory, high momentum: Noisy values, but trending overall
- High both: Smooth trend
- Low both: Noisy mean-reversion
"""

from prism.engines.momentum.runs_test import compute as runs_test_compute
from prism.engines.momentum.runs_test import compute_on_returns as runs_test_on_returns

__all__ = [
    'runs_test_compute',
    'runs_test_on_returns',
]
