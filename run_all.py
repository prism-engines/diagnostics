#!/usr/bin/env python3
"""
PRISM Pipeline - One Command, Full Analysis

Usage:
    python run_all.py

Input:
    data/observations.parquet   (canonical schema: entity_id, signal_id, I, y)

Output:
    data/primitives.parquet         (per-signal metrics)
    data/primitives_pairs.parquet   (pairwise relationships)
    data/manifold.parquet           (phase space coordinates)

No CLI flags. No prompts. No configuration.
Data determines what engines run.

CANONICAL ENGINE INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame:
        Input:  [entity_id, signal_id, I, y]
        Output: primitives DataFrame
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# Add prism to path
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# CONFIGURATION (internal, not user-facing)
# =============================================================================

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "observations.parquet"
OUTPUT_PRIMITIVES = DATA_DIR / "primitives.parquet"
OUTPUT_PAIRS = DATA_DIR / "primitives_pairs.parquet"
OUTPUT_MANIFOLD = DATA_DIR / "manifold.parquet"

# Limits for expensive engines
MAX_POINTS_EXPENSIVE = 5000
MAX_POINTS_MODERATE = 10000


# =============================================================================
# CANONICAL ENGINE INTERFACE
# All engines: DataFrame in -> DataFrame out
# =============================================================================

def compute_basic_stats(observations: pd.DataFrame) -> pd.DataFrame:
    """Basic statistics for any signal."""
    results = []
    for (entity_id, signal_id), group in observations.groupby(['entity_id', 'signal_id']):
        y = group['y'].values
        results.append({
            'entity_id': entity_id,
            'signal_id': signal_id,
            'n_points': len(y),
            'y_min': float(np.min(y)),
            'y_max': float(np.max(y)),
            'y_mean': float(np.mean(y)),
            'y_std': float(np.std(y)),
            'y_median': float(np.median(y)),
        })
    return pd.DataFrame(results)


def compute_derivatives(observations: pd.DataFrame) -> pd.DataFrame:
    """Compute derivatives."""
    results = []
    for (entity_id, signal_id), group in observations.groupby(['entity_id', 'signal_id']):
        y = group.sort_values('I')['y'].values
        dy = np.diff(y)
        d2y = np.diff(dy)
        results.append({
            'entity_id': entity_id,
            'signal_id': signal_id,
            'dy_mean': float(np.mean(dy)) if len(dy) > 0 else 0.0,
            'dy_std': float(np.std(dy)) if len(dy) > 0 else 0.0,
            'd2y_mean': float(np.mean(d2y)) if len(d2y) > 0 else 0.0,
        })
    return pd.DataFrame(results)


def compute_hurst(observations: pd.DataFrame) -> pd.DataFrame:
    """Hurst exponent via DFA - uses canonical engine interface."""
    try:
        from prism.engines.core.hurst import compute as _compute_hurst
        return _compute_hurst(observations)
    except ImportError:
        # Fallback if engine not available
        results = []
        for (entity_id, signal_id), group in observations.groupby(['entity_id', 'signal_id']):
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                'hurst': np.nan,
                'hurst_r2': np.nan,
            })
        return pd.DataFrame(results)


def compute_entropy(observations: pd.DataFrame) -> pd.DataFrame:
    """Sample entropy - uses canonical engine interface."""
    try:
        from prism.engines.core.entropy import compute as _compute_entropy
        result = _compute_entropy(observations)
        # Rename column to match expected output
        if 'sample_entropy' in result.columns:
            result = result.rename(columns={'sample_entropy': 'entropy'})
        return result[['entity_id', 'signal_id', 'entropy']]
    except ImportError:
        results = []
        for (entity_id, signal_id), group in observations.groupby(['entity_id', 'signal_id']):
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                'entropy': np.nan,
            })
        return pd.DataFrame(results)


def compute_fft(observations: pd.DataFrame) -> pd.DataFrame:
    """FFT dominant frequency - uses canonical engine interface."""
    try:
        from prism.engines.core.fft import compute as _compute_fft
        result = _compute_fft(observations)
        # Select and rename columns to match expected output
        cols = ['entity_id', 'signal_id']
        if 'dominant_frequency' in result.columns:
            result = result.rename(columns={'dominant_frequency': 'dominant_freq'})
        for c in ['dominant_freq', 'spectral_centroid']:
            if c in result.columns:
                cols.append(c)
        return result[cols]
    except ImportError:
        results = []
        for (entity_id, signal_id), group in observations.groupby(['entity_id', 'signal_id']):
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                'dominant_freq': np.nan,
                'spectral_centroid': np.nan,
            })
        return pd.DataFrame(results)


def compute_lyapunov(observations: pd.DataFrame) -> pd.DataFrame:
    """Lyapunov exponent - uses canonical engine interface."""
    try:
        from prism.engines.core.lyapunov import compute as _compute_lyapunov
        result = _compute_lyapunov(observations)
        return result[['entity_id', 'signal_id', 'lyapunov']]
    except ImportError:
        results = []
        for (entity_id, signal_id), group in observations.groupby(['entity_id', 'signal_id']):
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                'lyapunov': np.nan,
            })
        return pd.DataFrame(results)


def compute_garch(observations: pd.DataFrame) -> pd.DataFrame:
    """GARCH volatility model - uses canonical engine interface."""
    try:
        from prism.engines.core.garch import compute as _compute_garch
        result = _compute_garch(observations)
        # Rename columns to match expected output
        rename_map = {'garch_omega': 'garch_omega', 'garch_alpha': 'garch_alpha', 'garch_beta': 'garch_beta'}
        for old, new in rename_map.items():
            if old in result.columns:
                result = result.rename(columns={old: new})
        cols = ['entity_id', 'signal_id']
        for c in ['garch_omega', 'garch_alpha', 'garch_beta']:
            if c in result.columns:
                cols.append(c)
        return result[cols]
    except ImportError:
        results = []
        for (entity_id, signal_id), group in observations.groupby(['entity_id', 'signal_id']):
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                'garch_omega': np.nan,
                'garch_alpha': np.nan,
                'garch_beta': np.nan,
            })
        return pd.DataFrame(results)


def compute_transitions(observations: pd.DataFrame) -> pd.DataFrame:
    """Digital signal transition analysis."""
    results = []
    for (entity_id, signal_id), group in observations.groupby(['entity_id', 'signal_id']):
        y = group.sort_values('I')['y'].values
        n_trans = int(np.sum(np.diff(y) != 0))
        mean_dur = float(len(y) / max(n_trans, 1))
        results.append({
            'entity_id': entity_id,
            'signal_id': signal_id,
            'n_transitions': n_trans,
            'mean_state_duration': mean_dur,
        })
    return pd.DataFrame(results)


# Engine registry: name -> compute function
ENGINES = {
    'basic_stats': compute_basic_stats,
    'derivatives': compute_derivatives,
    'hurst': compute_hurst,
    'entropy': compute_entropy,
    'fft': compute_fft,
    'lyapunov': compute_lyapunov,
    'garch': compute_garch,
    'transitions': compute_transitions,
}


# =============================================================================
# SIGNAL TYPE DETECTION
# =============================================================================

def detect_signal_types(obs: pd.DataFrame) -> dict:
    """
    Automatically detect signal types based on data properties.
    Returns dict: {signal_id: {class, is_constant, ...}}
    """
    typology = {}

    for signal_id, group in obs.groupby('signal_id'):
        y = group['y'].values
        n = len(y)
        std = np.std(y)
        unique_ratio = len(np.unique(y)) / max(n, 1)

        is_constant = std < 1e-10
        is_digital = unique_ratio < 0.05 and not is_constant

        if is_constant:
            signal_class = 'constant'
        elif is_digital:
            signal_class = 'digital'
        else:
            signal_class = 'analog'

        typology[signal_id] = {
            'class': signal_class,
            'is_constant': is_constant,
            'n_points': n,
            'std': std,
        }

    return typology


def map_engines_to_types(typology: dict) -> dict:
    """Map engines to signals based on type."""
    engine_map = {}

    for signal_id, props in typology.items():
        engines = ['basic_stats']

        if props['is_constant']:
            pass  # Only basic stats
        elif props['class'] == 'digital':
            engines.append('transitions')
        else:  # analog
            engines.extend(['derivatives', 'hurst', 'entropy', 'fft'])
            if props['std'] > 1e-6:
                engines.extend(['lyapunov', 'garch'])

        engine_map[signal_id] = engines

    return engine_map


# =============================================================================
# PAIRWISE ANALYSIS
# =============================================================================

def compute_pairwise(obs: pd.DataFrame, typology: dict) -> pd.DataFrame:
    """Compute pairwise correlations between analog signals."""
    from itertools import combinations

    # Filter to analog signals
    analog_signals = [s for s, p in typology.items() if p['class'] == 'analog']

    if len(analog_signals) < 2:
        return pd.DataFrame({'signal_a': [], 'signal_b': [], 'correlation': []})

    pairs = []
    entity = obs['entity_id'].iloc[0]
    entity_obs = obs[obs['entity_id'] == entity]

    for s1, s2 in combinations(analog_signals[:50], 2):
        d1 = entity_obs[entity_obs['signal_id'] == s1].sort_values('I')['y'].values[:2000]
        d2 = entity_obs[entity_obs['signal_id'] == s2].sort_values('I')['y'].values[:2000]

        min_len = min(len(d1), len(d2))
        if min_len < 10:
            continue

        y1, y2 = d1[:min_len], d2[:min_len]
        std1, std2 = np.std(y1), np.std(y2)

        if std1 > 1e-10 and std2 > 1e-10:
            corr = float(np.corrcoef(y1, y2)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0

        pairs.append({'signal_a': s1, 'signal_b': s2, 'correlation': corr})

    return pd.DataFrame(pairs)


# =============================================================================
# MANIFOLD EMBEDDING
# =============================================================================

def compute_manifold(primitives: pd.DataFrame) -> pd.DataFrame:
    """Compute 3D manifold coordinates via PCA."""
    from sklearn.decomposition import PCA

    # Select numeric columns
    exclude = {'entity_id', 'signal_id', 'signal_class'}
    numeric_cols = [c for c in primitives.columns
                    if c not in exclude and primitives[c].dtype in [np.float64, np.float32, np.int64]]

    if not numeric_cols or len(primitives) < 3:
        return primitives[['entity_id', 'signal_id']].assign(
            manifold_x=0.0, manifold_y=0.0, manifold_z=0.0
        )

    # Build matrix, replace NaN with column means
    matrix = primitives[numeric_cols].values
    col_means = np.nanmean(matrix, axis=0)
    for i in range(matrix.shape[1]):
        mask = np.isnan(matrix[:, i])
        matrix[mask, i] = col_means[i] if not np.isnan(col_means[i]) else 0.0

    # PCA
    n_comp = min(3, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(matrix)

    # Pad to 3D
    if coords.shape[1] < 3:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 3 - coords.shape[1]))])

    return primitives[['entity_id', 'signal_id']].assign(
        manifold_x=coords[:, 0],
        manifold_y=coords[:, 1],
        manifold_z=coords[:, 2],
    )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """THE PIPELINE. One command. Everything runs. Results appear."""
    start_time = time.time()

    print("=" * 70)
    print("PRISM PIPELINE")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. LOAD
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading observations...")

    if not INPUT_FILE.exists():
        print(f"  ERROR: {INPUT_FILE} not found")
        print(f"  Required schema: entity_id, signal_id, I, y")
        sys.exit(1)

    obs = pl.read_parquet(INPUT_FILE).to_pandas()
    n_obs = len(obs)
    n_entities = obs['entity_id'].nunique()
    n_signals = obs['signal_id'].nunique()

    print(f"  {n_obs:,} observations")
    print(f"  {n_entities} entities")
    print(f"  {n_signals} signals")

    # -------------------------------------------------------------------------
    # 2. DETECT TYPOLOGY
    # -------------------------------------------------------------------------
    print("\n[2/6] Detecting signal types...")

    typology = detect_signal_types(obs)

    type_counts = {}
    for props in typology.values():
        t = props['class']
        type_counts[t] = type_counts.get(t, 0) + 1

    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")

    # -------------------------------------------------------------------------
    # 3. MAP ENGINES
    # -------------------------------------------------------------------------
    print("\n[3/6] Mapping engines...")

    engine_map = map_engines_to_types(typology)

    all_engines = set()
    for engines in engine_map.values():
        all_engines.update(engines)

    print(f"  {len(all_engines)} engines selected: {sorted(all_engines)}")

    # -------------------------------------------------------------------------
    # 4. COMPUTE PRIMITIVES
    # -------------------------------------------------------------------------
    print("\n[4/6] Computing primitives...")

    # Group observations by signal for efficient processing
    primitives_list = []

    for engine_name in sorted(all_engines):
        print(f"  Running {engine_name}...")

        # Filter to signals that need this engine
        signals_needing = [s for s, engs in engine_map.items() if engine_name in engs]
        obs_subset = obs[obs['signal_id'].isin(signals_needing)]

        if len(obs_subset) == 0:
            continue

        engine_fn = ENGINES.get(engine_name)
        if engine_fn:
            try:
                result_df = engine_fn(obs_subset)
                primitives_list.append(result_df)
            except Exception as e:
                print(f"    {engine_name} failed: {e}")

    # Merge all engine results
    if primitives_list:
        primitives = primitives_list[0]
        for df in primitives_list[1:]:
            primitives = primitives.merge(df, on=['entity_id', 'signal_id'], how='outer')
    else:
        primitives = pd.DataFrame({'entity_id': [], 'signal_id': []})

    # Add signal class
    primitives['signal_class'] = primitives['signal_id'].map(lambda s: typology.get(s, {}).get('class', 'unknown'))

    print(f"  {len(primitives)} primitives computed, {len(primitives.columns)} columns")

    # -------------------------------------------------------------------------
    # 5. COMPUTE PAIRWISE
    # -------------------------------------------------------------------------
    print("\n[5/6] Computing pairwise relationships...")

    pairs = compute_pairwise(obs, typology)

    print(f"  {len(pairs)} pairs computed")

    # -------------------------------------------------------------------------
    # 6. COMPUTE MANIFOLD
    # -------------------------------------------------------------------------
    print("\n[6/6] Computing manifold embedding...")

    manifold = compute_manifold(primitives)

    print(f"  3D coordinates computed")

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Saving results...")

    # Convert to polars for parquet output
    pl.from_pandas(primitives).write_parquet(OUTPUT_PRIMITIVES)
    print(f"  {OUTPUT_PRIMITIVES}")

    pl.from_pandas(pairs).write_parquet(OUTPUT_PAIRS)
    print(f"  {OUTPUT_PAIRS}")

    pl.from_pandas(manifold).write_parquet(OUTPUT_MANIFOLD)
    print(f"  {OUTPUT_MANIFOLD}")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nTime: {elapsed:.1f}s")
    print(f"\nOutputs:")
    print(f"  primitives.parquet       {len(primitives)} rows x {len(primitives.columns)} cols")
    print(f"  primitives_pairs.parquet {len(pairs)} pairs")
    print(f"  manifold.parquet         {len(manifold)} coordinates")

    # Show sample results
    print("\n[SAMPLE: Hurst Exponents]")
    if 'hurst' in primitives.columns:
        hurst_df = primitives[['signal_id', 'hurst']].dropna().sort_values('hurst', ascending=False).head(10)
        for _, row in hurst_df.iterrows():
            h = row['hurst']
            behavior = "TRENDING" if h > 0.6 else "mean-reverting" if h < 0.4 else "random"
            print(f"  {row['signal_id']}: H={h:.3f} ({behavior})")

    print("\n[SAMPLE: Strongest Correlations]")
    if len(pairs) > 0:
        top_pairs = pairs.dropna().nlargest(5, 'correlation')
        for _, row in top_pairs.iterrows():
            print(f"  {row['signal_a']} <-> {row['signal_b']}: r={row['correlation']:.3f}")


if __name__ == '__main__':
    main()
