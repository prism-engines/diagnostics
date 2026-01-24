"""
Signal characterization - maps normalized scores to labels.

Runtime classification of signal typology scores. Maps 0-1 normalized values
to human-readable labels. No labels stored in data - computed on query.

Usage:
    from prism.signal_typology.characterize import characterize, characterize_signal

    # Single axis
    label = characterize(0.73, 'memory')  # -> 'weak persistent'

    # Full signal profile
    labels = characterize_signal(profile_dict)  # -> {'memory': 'weak persistent', ...}
"""

from typing import Dict, Optional, Any, List
import numpy as np

# -----------------------------------------------------------------------------
# Axis Definitions
# -----------------------------------------------------------------------------

AXES = {
    'memory': {
        'low': 'forgetful',
        'high': 'persistent',
        'description': 'How much the signal remembers its past',
        'engines': ['hurst_rs', 'hurst_dfa', 'acf_decay'],
    },
    'information': {
        'low': 'predictable',
        'high': 'entropic',
        'description': 'How much disorder/randomness in the signal',
        'engines': ['permutation_entropy', 'sample_entropy', 'spectral_entropy'],
    },
    'frequency': {
        'low': 'aperiodic',
        'high': 'periodic',
        'description': 'How much the signal exhibits regular cycles',
        'engines': ['spectral', 'wavelet'],
    },
    'volatility': {
        'low': 'stable',
        'high': 'clustered',
        'description': 'How much variance clusters over time',
        'engines': ['garch', 'realized_vol', 'bipower_variation'],
    },
    'dynamics': {
        'low': 'deterministic',
        'high': 'chaotic',
        'description': 'How sensitive to initial conditions',
        'engines': ['lyapunov', 'embedding', 'phase_space'],
    },
    'recurrence': {
        'low': 'wandering',
        'high': 'returning',
        'description': 'How often the signal revisits previous states',
        'engines': ['rqa'],
    },
    'discontinuity': {
        'low': 'continuous',
        'high': 'discontinuous',
        'description': 'How much the signal exhibits jumps/breaks',
        'engines': ['dirac', 'heaviside', 'structural'],
    },
    'derivatives': {
        'low': 'smooth',
        'high': 'spiky',
        'description': 'How erratic the signal changes are',
        'engines': ['derivatives'],
    },
    'momentum': {
        'low': 'reverting',
        'high': 'trending',
        'description': 'Whether signal tends to continue or reverse direction',
        'engines': ['runs_test'],  # Directional persistence, distinct from memory
    },
}


# -----------------------------------------------------------------------------
# Thresholds
# -----------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    'strong_low': 0.25,
    'weak_low': 0.40,
    'weak_high': 0.60,
    'strong_high': 0.75,
}

# Per-axis threshold overrides (optional)
AXIS_THRESHOLDS: Dict[str, Dict[str, float]] = {
    # 'memory': {'strong_low': 0.30, 'weak_low': 0.45, 'weak_high': 0.55, 'strong_high': 0.70},
    # Add custom thresholds per axis if distributions differ
}


# -----------------------------------------------------------------------------
# Normalization Direction Notes
# -----------------------------------------------------------------------------
#
# IMPORTANT: Engines must normalize so that:
#   0 = low pole label, 1 = high pole label
#
# Axis-specific normalization:
#
# memory (forgetful -> persistent):
#   - Hurst exponent: already 0-1, H > 0.5 = persistent
#   - ACF decay: slow decay = persistent, so use decay_rate directly
#
# information (predictable -> entropic):
#   - Entropy measures: high entropy = entropic
#
# frequency (aperiodic -> periodic):
#   - Spectral entropy: INVERT! High entropy = broadband = aperiodic
#     -> Use: 1 - spectral_entropy
#   - Bandwidth: INVERT! High bandwidth = aperiodic
#     -> Use: 1 - normalized_bandwidth
#   - Peak prominence: high = periodic
#
# volatility (stable -> clustered):
#   - GARCH persistence (alpha + beta): high = clustered
#   - Realized vol ratio (max/min): high = clustered
#
# dynamics (deterministic -> chaotic):
#   - Lyapunov exponent: positive = chaotic
#     -> Normalize: clip and scale to 0-1
#
# recurrence (wandering -> returning):
#   - Recurrence rate: high = returning
#   - Determinism: high = returning
#
# discontinuity (continuous -> discontinuous):
#   - Level shift count: normalize by signal length
#   - CUSUM max: normalize by signal std
#
# derivatives (smooth -> spiky):
#   - Derivative kurtosis: high = spiky
#   - Zero-crossing rate: INVERT! high = smooth oscillation
#
# momentum (reverting -> trending):
#   - Runs test: fewer runs than expected = trending
#     -> Use: 1 - normalized_runs_ratio (so fewer runs = higher score)
#


# -----------------------------------------------------------------------------
# Classification Functions
# -----------------------------------------------------------------------------

def get_thresholds(axis: str) -> Dict[str, float]:
    """Get thresholds for an axis (custom or default)."""
    return AXIS_THRESHOLDS.get(axis, DEFAULT_THRESHOLDS)


def characterize(score: float, axis: str, thresholds: Dict[str, float] = None) -> str:
    """
    Map a 0-1 normalized score to a classification label.

    Args:
        score: Normalized score (0-1)
        axis: Axis name (e.g., 'memory', 'volatility')
        thresholds: Optional custom thresholds dict

    Returns:
        Classification label string

    Examples:
        characterize(0.15, 'memory')  -> 'forgetful'
        characterize(0.35, 'memory')  -> 'weak forgetful'
        characterize(0.50, 'memory')  -> 'indeterminate'
        characterize(0.68, 'memory')  -> 'weak persistent'
        characterize(0.85, 'memory')  -> 'persistent'
    """
    if axis not in AXES:
        raise ValueError(f"Unknown axis: {axis}. Valid: {list(AXES.keys())}")

    if score is None or (isinstance(score, float) and np.isnan(score)):
        return 'insufficient data'

    thresholds = thresholds or get_thresholds(axis)
    low_label = AXES[axis]['low']
    high_label = AXES[axis]['high']

    if score < thresholds['strong_low']:
        return low_label
    elif score < thresholds['weak_low']:
        return f'weak {low_label}'
    elif score < thresholds['weak_high']:
        return 'indeterminate'
    elif score < thresholds['strong_high']:
        return f'weak {high_label}'
    else:
        return high_label


def characterize_signal(profile: Dict[str, float], thresholds: Dict[str, float] = None) -> Dict[str, str]:
    """
    Characterize all axes for a signal profile.

    Args:
        profile: Dict with axis scores {'memory': 0.73, 'volatility': 0.45, ...}
        thresholds: Optional custom thresholds

    Returns:
        Dict with classification labels {'memory': 'weak persistent', ...}
    """
    result = {}
    for axis in AXES:
        if axis in profile:
            result[axis] = characterize(profile[axis], axis, thresholds)
    return result


def characterize_dataframe(df, thresholds: Dict[str, float] = None):
    """
    Add classification columns to a typology dataframe.

    Args:
        df: DataFrame with signal_id and axis score columns
        thresholds: Optional custom thresholds

    Returns:
        DataFrame with added *_class columns
    """
    df = df.copy()
    for axis in AXES:
        if axis in df.columns:
            df[f'{axis}_class'] = df[axis].apply(
                lambda x: characterize(x, axis, thresholds)
            )
    return df


# -----------------------------------------------------------------------------
# Narrative Generation
# -----------------------------------------------------------------------------

def generate_narrative(profile: Dict[str, float], signal_id: str = None) -> str:
    """
    Generate a human-readable narrative for a signal profile.

    Args:
        profile: Dict with axis scores
        signal_id: Optional signal identifier

    Returns:
        Narrative string describing the signal
    """
    labels = characterize_signal(profile)

    # Filter to non-indeterminate characteristics
    notable = {k: v for k, v in labels.items()
               if v not in ['indeterminate', 'insufficient data']}

    if not notable:
        return f"Signal {signal_id or 'unknown'} shows no strong characteristics."

    # Build narrative
    header = f"Signal {signal_id}" if signal_id else "This signal"

    # Group by strength
    strong = []
    weak = []

    for axis, label in notable.items():
        if label.startswith('weak'):
            weak.append((axis, label))
        else:
            strong.append((axis, label))

    parts = [header]

    if strong:
        traits = [f"{label} ({axis})" for axis, label in strong]
        parts.append(f"is {', '.join(traits)}")

    if weak:
        traits = [f"{label.replace('weak ', '')} ({axis})" for axis, label in weak]
        if strong:
            parts.append(f"with weak tendencies toward {', '.join(traits)}")
        else:
            parts.append(f"shows weak tendencies toward {', '.join(traits)}")

    return ' '.join(parts) + '.'


def generate_comparison_narrative(profiles: Dict[str, Dict[str, float]]) -> str:
    """
    Generate narrative comparing multiple signals.

    Args:
        profiles: Dict of signal_id -> profile dict

    Returns:
        Comparison narrative
    """
    all_labels = {sid: characterize_signal(p) for sid, p in profiles.items()}

    lines = []

    # Find shared characteristics
    for axis in AXES:
        axis_labels = {sid: labels.get(axis) for sid, labels in all_labels.items()}
        unique_labels = set(v for v in axis_labels.values() if v and v != 'indeterminate')

        if len(unique_labels) == 1:
            # All signals share this trait
            label = unique_labels.pop()
            lines.append(f"All signals are {label} in {axis}.")
        elif len(unique_labels) > 1:
            # Divergence
            groups: Dict[str, List[str]] = {}
            for sid, label in axis_labels.items():
                if label and label != 'indeterminate':
                    groups.setdefault(label, []).append(sid)

            if len(groups) > 1:
                parts = [f"{', '.join(sids)} are {label}"
                         for label, sids in groups.items()]
                lines.append(f"In {axis}: {'; '.join(parts)}.")

    return '\n'.join(lines) if lines else "Signals show similar characteristics."


# -----------------------------------------------------------------------------
# Query Helpers
# -----------------------------------------------------------------------------

def filter_by_characteristic(df, axis: str, label: str):
    """
    Filter dataframe to signals with a specific characteristic.

    Args:
        df: Typology dataframe
        axis: Axis to filter on
        label: Label to match (e.g., 'persistent', 'weak forgetful')

    Returns:
        Filtered dataframe
    """
    class_col = f'{axis}_class'
    if class_col not in df.columns:
        df = characterize_dataframe(df)

    return df[df[class_col] == label]


def find_similar_signals(df, target_profile: Dict[str, float], n: int = 5):
    """
    Find signals with similar typology profile.

    Args:
        df: Typology dataframe
        target_profile: Reference profile dict
        n: Number of similar signals to return

    Returns:
        DataFrame of n most similar signals
    """
    axes = [a for a in AXES if a in df.columns and a in target_profile]

    if not axes:
        return df.head(0)

    # Compute Euclidean distance
    target_vec = np.array([target_profile[a] for a in axes])

    def distance(row):
        row_vec = np.array([row[a] for a in axes])
        return np.sqrt(np.sum((row_vec - target_vec) ** 2))

    df = df.copy()
    df['_distance'] = df.apply(distance, axis=1)
    result = df.nsmallest(n, '_distance').drop(columns=['_distance'])

    return result


def group_by_typology(df, primary_axis: str, secondary_axis: str = None):
    """
    Group signals by typology classification.

    Args:
        df: Typology dataframe
        primary_axis: Primary grouping axis
        secondary_axis: Optional secondary axis

    Returns:
        Grouped dataframe or dict of groups
    """
    df = characterize_dataframe(df)

    if secondary_axis:
        return df.groupby([f'{primary_axis}_class', f'{secondary_axis}_class'])
    else:
        return df.groupby(f'{primary_axis}_class')


# -----------------------------------------------------------------------------
# SQL Generation (for DuckDB queries)
# -----------------------------------------------------------------------------

def sql_classify_case(axis: str, thresholds: Dict[str, float] = None) -> str:
    """
    Generate SQL CASE statement for classification.

    Args:
        axis: Axis name
        thresholds: Optional custom thresholds

    Returns:
        SQL CASE expression
    """
    thresholds = thresholds or get_thresholds(axis)
    low = AXES[axis]['low']
    high = AXES[axis]['high']

    return f"""
    CASE
        WHEN {axis} IS NULL THEN 'insufficient data'
        WHEN {axis} < {thresholds['strong_low']} THEN '{low}'
        WHEN {axis} < {thresholds['weak_low']} THEN 'weak {low}'
        WHEN {axis} < {thresholds['weak_high']} THEN 'indeterminate'
        WHEN {axis} < {thresholds['strong_high']} THEN 'weak {high}'
        ELSE '{high}'
    END AS {axis}_class
    """


def sql_classify_all() -> str:
    """
    Generate SQL for all axis classifications.

    Returns:
        SQL SELECT clause additions
    """
    cases = [sql_classify_case(axis) for axis in AXES]
    return ',\n    '.join(cases)


# -----------------------------------------------------------------------------
# Axis Information Helpers
# -----------------------------------------------------------------------------

def get_axis_info(axis: str) -> Dict[str, Any]:
    """Get full information about an axis."""
    if axis not in AXES:
        raise ValueError(f"Unknown axis: {axis}")
    return AXES[axis]


def list_axes() -> List[str]:
    """List all available axes."""
    return list(AXES.keys())


def get_axis_description(axis: str) -> str:
    """Get human-readable description of an axis."""
    return AXES[axis]['description']


def get_axis_poles(axis: str) -> tuple:
    """Get (low_label, high_label) for an axis."""
    return (AXES[axis]['low'], AXES[axis]['high'])
