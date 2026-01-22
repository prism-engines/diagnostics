"""
Orthon Report Generation Module

Automated scientific report generation for cohort analysis.
"""

from orthon.report.experimental_comparison import (
    compare_cohorts,
    generate_report,
    ComparisonResult,
    PairwiseDivergence,
    TemporalDivergence,
)

__all__ = [
    "compare_cohorts",
    "generate_report",
    "ComparisonResult",
    "PairwiseDivergence",
    "TemporalDivergence",
]
