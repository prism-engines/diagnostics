"""
PRISM HYBRID: Physics-Informed Feature Engineering + Simple Models
===================================================================

Turn PRISM's unsupervised field geometry into supervised predictions.

The Strategy:
    1. PRISM extracts interpretable features (field topology - already done)
    2. Simple models learn optimal weights (this script)
    3. Get deep learning results with interpretable models

Expected Results:
    - LSTM on raw sensors: RMSE ≈ 12-13 (hours to train, black box)
    - PRISM + XGBoost:     RMSE ≈ 13-15 (minutes to train, interpretable)
    - PRISM + Linear:      RMSE ≈ 15-18 (seconds to train, fully transparent)

Usage:
    python -m prism.entry_points.hybrid --domain cmapss
    python -m prism.entry_points.hybrid --domain cmapss --model xgboost
    python -m prism.entry_points.hybrid --domain cmapss --model all --compare-baseline
"""

import argparse
import logging
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ML imports
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from prism.db.parquet_store import (
    ensure_directory,
    get_path,
    get_data_root,
    OBSERVATIONS,
    SIGNALS,
    GEOMETRY,
    STATE,
    COHORTS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PRISMFeatureConfig:
    """Configuration for PRISM feature extraction."""

    # Laplace field features (per signal)
    field_features: List[str] = None

    # Cohort geometry features (per entity)
    geometry_features: List[str] = None

    # Aggregation methods
    agg_methods: List[str] = None

    def __post_init__(self):
        self.field_features = self.field_features or [
            'gradient_mean',
            'gradient_magnitude',
            'laplacian_mean',
            'divergence',
            'field_potential',
            'total_potential',
        ]

        self.geometry_features = self.geometry_features or [
            'cohesion',
            'separation',
            'effective_dim',
            'mst_weight',
        ]

        self.agg_methods = self.agg_methods or [
            'mean',
            'std',
            'min',
            'max',
            'median',
        ]


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_cohort_from_signal(signal_id: str) -> str:
    """Extract cohort from signal_id (e.g., CMAPSS_BPR_FD001_U001 -> FD001_U001)."""
    parts = signal_id.split('_')
    if len(parts) >= 4:
        return '_'.join(parts[-2:])
    return signal_id


def extract_prism_features(
    domain: str,
    config: PRISMFeatureConfig = None,
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Extract PRISM features for supervised learning.

    Returns:
        features_df: DataFrame with entity_id + PRISM features + target
        feature_cols: List of feature column names
    """

    config = config or PRISMFeatureConfig()

    logger.info(f"Extracting PRISM features for domain: {domain}")

    # -------------------------------------------------------------------------
    # Load PRISM outputs
    # -------------------------------------------------------------------------

    # Laplace field (signal-level)
    field_path = get_path(SIGNALS, domain=domain)
    if not field_path.exists():
        raise FileNotFoundError(f"Run laplace.py first: {field_path}")
    field_df = pl.read_parquet(field_path)
    logger.info(f"Loaded field data: {len(field_df):,} rows")

    # Add cohort column
    if 'cohort' not in field_df.columns:
        field_df = field_df.with_columns([
            pl.col('signal_id').map_elements(
                extract_cohort_from_signal, return_dtype=pl.Utf8
            ).alias('cohort')
        ])

    # Cohort geometry (optional)
    geom_path = get_path(GEOMETRY, domain=domain)
    geometry_df = None
    if geom_path.exists():
        geometry_df = pl.read_parquet(geom_path)
        logger.info(f"Loaded geometry data: {len(geometry_df):,} rows")

    # Observations (only load RUL signals for target extraction)
    obs_path = get_path(OBSERVATIONS, domain=domain)
    if not obs_path.exists():
        raise FileNotFoundError(f"Observations not found: {obs_path}")
    # Lazy scan with filter pushdown - only load RUL signals
    obs_df = (
        pl.scan_parquet(obs_path)
        .filter(pl.col('signal_id').str.contains('RUL'))
        .collect()
    )

    # -------------------------------------------------------------------------
    # Extract field features per entity (cohort/engine)
    # -------------------------------------------------------------------------

    entities = field_df['cohort'].unique().to_list()
    logger.info(f"Found {len(entities)} entities")

    field_feature_rows = []

    for entity in entities:
        entity_data = field_df.filter(pl.col('cohort') == entity)

        if entity_data.is_empty():
            continue

        row = {'entity_id': entity}

        # Aggregate each field feature
        for feat in config.field_features:
            if feat not in entity_data.columns:
                continue

            col = entity_data[feat].drop_nulls()
            if len(col) == 0:
                continue

            vals = col.to_numpy()

            # Multiple aggregations per feature
            row[f'{feat}_mean'] = float(np.mean(vals))
            row[f'{feat}_std'] = float(np.std(vals)) if len(vals) > 1 else 0.0
            row[f'{feat}_min'] = float(np.min(vals))
            row[f'{feat}_max'] = float(np.max(vals))
            row[f'{feat}_median'] = float(np.median(vals))
            row[f'{feat}_range'] = float(np.max(vals) - np.min(vals))

            # Trend (late vs early windows)
            if len(vals) >= 6:
                n = len(vals)
                early = np.mean(vals[:n//3])
                late = np.mean(vals[-n//3:])
                row[f'{feat}_trend'] = float(late - early)

        # Source/sink classification ratios
        if 'divergence' in entity_data.columns:
            div = entity_data['divergence'].drop_nulls().to_numpy()
            if len(div) > 0:
                row['source_ratio'] = float(np.mean(div > 0))
                row['sink_ratio'] = float(np.mean(div < 0))
                div_std = np.std(div) if len(div) > 1 else 1.0
                row['bridge_ratio'] = float(np.mean(np.abs(div) < div_std * 0.5))

        field_feature_rows.append(row)

    field_features = pl.DataFrame(field_feature_rows)
    logger.info(f"Extracted field features: {len(field_features.columns)-1} columns")

    # -------------------------------------------------------------------------
    # Extract geometry features per entity (if available)
    # -------------------------------------------------------------------------

    if geometry_df is not None and len(geometry_df) > 0:
        geom_agg = []

        cohort_col = 'cohort_id' if 'cohort_id' in geometry_df.columns else 'cohort'

        for feat in config.geometry_features:
            if feat in geometry_df.columns:
                geom_agg.extend([
                    pl.col(feat).mean().alias(f'geom_{feat}_mean'),
                    pl.col(feat).std().alias(f'geom_{feat}_std'),
                    pl.col(feat).last().alias(f'geom_{feat}_final'),
                ])

        if geom_agg:
            geom_features = (
                geometry_df
                .group_by(cohort_col)
                .agg(geom_agg)
                .rename({cohort_col: 'entity_id'})
            )

            field_features = field_features.join(
                geom_features, on='entity_id', how='left'
            )
            logger.info(f"Added geometry features")

    # -------------------------------------------------------------------------
    # Extract target (RUL) per entity
    # -------------------------------------------------------------------------

    # obs_df already filtered to RUL signals during load
    rul_obs = obs_df

    if len(rul_obs) > 0:
        rul_obs = rul_obs.with_columns([
            pl.col('signal_id').map_elements(
                extract_cohort_from_signal, return_dtype=pl.Utf8
            ).alias('entity_id')
        ])

        # Get RUL stats per entity
        targets = (
            rul_obs
            .group_by('entity_id')
            .agg([
                pl.col('value').max().alias('initial_rul'),
                pl.col('value').min().alias('final_rul'),
                pl.col('value').mean().alias('mean_rul'),
            ])
        )
        logger.info(f"Extracted RUL targets for {len(targets)} entities")

        # Join
        features_with_target = field_features.join(
            targets, on='entity_id', how='inner'
        )
    else:
        features_with_target = field_features
        logger.warning("No RUL target found")

    # Clean up nulls
    features_with_target = features_with_target.fill_null(0.0)

    # Get feature column names
    feature_cols = [c for c in features_with_target.columns
                   if c not in ['entity_id', 'initial_rul', 'final_rul', 'mean_rul']]

    # Exclude break_* features (discontinuity counts - not predictive)
    feature_cols = [c for c in feature_cols if not c.startswith('break_')]

    logger.info(f"Final feature matrix: {features_with_target.shape}")

    return features_with_target, feature_cols


def extract_baseline_features(domain: str) -> Tuple[pl.DataFrame, List[str]]:
    """
    Extract baseline features (raw sensor statistics) for comparison.
    This is what you'd get WITHOUT PRISM.
    """

    obs_path = get_path(OBSERVATIONS, domain=domain)

    # Use two separate lazy queries for non-RUL (baseline) and RUL (target)
    lazy_obs = pl.scan_parquet(obs_path)

    # Basic stats per sensor per entity (non-RUL signals)
    baseline = (
        lazy_obs
        .filter(~pl.col('signal_id').str.contains('RUL'))
        .with_columns([
            pl.col('signal_id').str.extract(r'CMAPSS_([^_]+)_').alias('sensor'),
        ])
        .group_by([
            pl.col('signal_id').map_elements(extract_cohort_from_signal, return_dtype=pl.Utf8).alias('entity_id'),
            'sensor'
        ])
        .agg([
            pl.col('value').mean().alias('mean'),
            pl.col('value').std().alias('std'),
            pl.col('value').min().alias('min'),
            pl.col('value').max().alias('max'),
        ])
        .collect()
    )

    # Pivot to wide format
    baseline_wide = baseline.pivot(
        index='entity_id',
        on='sensor',
        values=['mean', 'std', 'min', 'max'],
    )

    # Get RUL target (separate lazy query with filter pushdown)
    targets = (
        lazy_obs
        .filter(pl.col('signal_id').str.contains('RUL'))
        .with_columns([
            pl.col('signal_id').map_elements(extract_cohort_from_signal, return_dtype=pl.Utf8).alias('entity_id'),
        ])
        .group_by('entity_id')
        .agg([
            pl.col('value').max().alias('initial_rul'),
            pl.col('value').mean().alias('mean_rul'),
        ])
        .collect()
    )

    baseline_with_target = baseline_wide.join(targets, on='entity_id', how='inner')
    baseline_with_target = baseline_with_target.fill_null(0.0)

    feature_cols = [c for c in baseline_with_target.columns
                   if c not in ['entity_id', 'initial_rul', 'mean_rul']]

    return baseline_with_target, feature_cols


# =============================================================================
# MODEL TRAINING
# =============================================================================

def get_models() -> Dict[str, any]:
    """Get dictionary of models to try."""

    models = {
        # Linear models (fastest, most interpretable)
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.1),
        'elasticnet': ElasticNet(alpha=0.1, l1_ratio=0.5),

        # Tree models (fast, good performance)
        'rf': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1,
            random_state=42,
        ),
        'gbm': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        ),
    }

    # Try XGBoost if available
    try:
        from xgboost import XGBRegressor
        models['xgboost'] = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )
    except ImportError:
        pass

    # Try LightGBM if available
    try:
        from lightgbm import LGBMRegressor
        models['lightgbm'] = LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
    except ImportError:
        pass

    return models


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    model: any,
    cv_folds: int = 5,
) -> Tuple[Dict[str, float], any, StandardScaler]:
    """Train model with cross-validation and return metrics."""

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation
    cv_scores = cross_val_score(
        model, X_scaled, y,
        cv=cv_folds,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
    )

    cv_rmse = -cv_scores.mean()
    cv_rmse_std = cv_scores.std()

    # Train/test split for final metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {
        'model': model_name,
        'cv_rmse': cv_rmse,
        'cv_rmse_std': cv_rmse_std,
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'test_mae': mean_absolute_error(y_test, y_pred),
        'test_r2': r2_score(y_test, y_pred),
    }

    return results, model, scaler


def get_feature_importance(
    model: any,
    feature_names: List[str],
) -> Optional[pl.DataFrame]:
    """Extract feature importance from trained model."""

    importance = None

    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)

    if importance is not None:
        return pl.DataFrame({
            'feature': feature_names,
            'importance': importance,
        }).sort('importance', descending=True)

    return None


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_hybrid(
    domain: str,
    model_names: List[str] = None,
    compare_baseline: bool = False,
    target_col: str = 'initial_rul',
) -> Dict[str, any]:
    """
    Run PRISM Hybrid models.

    Args:
        domain: Domain name (e.g., 'cmapss')
        model_names: List of models to try (None = all)
        compare_baseline: Also run on baseline features for comparison
        target_col: Which RUL column to predict

    Returns:
        Dict with results, best model, feature importance
    """

    print("=" * 70)
    print("PRISM HYBRID: Physics-Informed Supervised Learning")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Extract features
    # -------------------------------------------------------------------------

    print("\n[1] EXTRACTING PRISM FEATURES")
    print("-" * 40)

    prism_df, feature_cols = extract_prism_features(domain)

    X_prism = prism_df.select(feature_cols).to_numpy()

    if target_col not in prism_df.columns:
        logger.error(f"Target column {target_col} not found")
        return None

    y = prism_df[target_col].to_numpy()

    print(f"  PRISM features: {X_prism.shape[1]}")
    print(f"  Samples: {X_prism.shape[0]}")
    print(f"  Target: {target_col}")

    # -------------------------------------------------------------------------
    # Extract baseline features (optional)
    # -------------------------------------------------------------------------

    X_baseline = None
    baseline_feature_cols = None

    if compare_baseline:
        print("\n[2] EXTRACTING BASELINE FEATURES")
        print("-" * 40)

        try:
            baseline_df, baseline_feature_cols = extract_baseline_features(domain)
            X_baseline = baseline_df.select(baseline_feature_cols).to_numpy()
            print(f"  Baseline features: {X_baseline.shape[1]}")
        except Exception as e:
            logger.warning(f"Could not extract baseline: {e}")

    # -------------------------------------------------------------------------
    # Train models
    # -------------------------------------------------------------------------

    print("\n[3] TRAINING MODELS")
    print("-" * 40)

    all_models = get_models()

    if model_names:
        all_models = {k: v for k, v in all_models.items() if k in model_names}

    prism_results = []
    baseline_results = []
    best_model = None
    best_rmse = float('inf')
    best_scaler = None
    best_model_name = None

    for name, model in all_models.items():
        print(f"\n  Training {name}...")

        # PRISM features
        try:
            result, trained_model, scaler = train_and_evaluate(
                X_prism, y, name, clone(model)
            )
            result['features'] = 'PRISM'
            prism_results.append(result)

            print(f"    PRISM: RMSE = {result['cv_rmse']:.2f} ± {result['cv_rmse_std']:.2f}, R² = {result['test_r2']:.3f}")

            if result['cv_rmse'] < best_rmse:
                best_rmse = result['cv_rmse']
                best_model = trained_model
                best_scaler = scaler
                best_model_name = name
        except Exception as e:
            logger.warning(f"    PRISM failed: {e}")

        # Baseline features (if requested)
        if X_baseline is not None:
            try:
                result, _, _ = train_and_evaluate(
                    X_baseline, y, name, clone(model)
                )
                result['features'] = 'Baseline'
                baseline_results.append(result)

                print(f"    Baseline: RMSE = {result['cv_rmse']:.2f} ± {result['cv_rmse_std']:.2f}")
            except Exception as e:
                logger.warning(f"    Baseline failed: {e}")

    # -------------------------------------------------------------------------
    # Results summary
    # -------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    all_results = prism_results + baseline_results
    results_df = pl.DataFrame(all_results).sort('cv_rmse')

    print("\n")
    print(results_df.select(['features', 'model', 'cv_rmse', 'cv_rmse_std', 'test_r2']))

    # Feature importance from best model
    importance_df = None
    if best_model is not None:
        importance_df = get_feature_importance(best_model, feature_cols)

        if importance_df is not None:
            print(f"\n[TOP 10 FEATURES - {best_model_name}]")
            print(importance_df.head(10))

    # -------------------------------------------------------------------------
    # Comparison summary
    # -------------------------------------------------------------------------

    if compare_baseline and baseline_results:
        print("\n" + "=" * 70)
        print("PRISM vs BASELINE COMPARISON")
        print("=" * 70)

        for model_name in all_models.keys():
            prism_r = next((r for r in prism_results if r['model'] == model_name), None)
            base_r = next((r for r in baseline_results if r['model'] == model_name), None)

            if prism_r and base_r:
                improvement = (base_r['cv_rmse'] - prism_r['cv_rmse']) / base_r['cv_rmse'] * 100
                print(f"  {model_name}: PRISM {prism_r['cv_rmse']:.2f} vs Baseline {base_r['cv_rmse']:.2f} ({improvement:+.1f}%)")

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------

    # Save to summary directory (not in schema)
    from prism.db.parquet_store import get_data_root
    output_dir = get_data_root(domain) / 'summary'
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.write_parquet(output_dir / 'hybrid_results.parquet')

    if importance_df is not None:
        importance_df.write_parquet(output_dir / 'feature_importance.parquet')

    prism_df.write_parquet(output_dir / 'prism_features.parquet')

    print(f"\nOutputs saved to {output_dir}")

    return {
        'results': results_df,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'best_rmse': best_rmse,
        'feature_importance': importance_df,
        'prism_features': prism_df,
        'scaler': best_scaler,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PRISM Hybrid: Physics-Informed Supervised Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m prism.entry_points.hybrid --domain cmapss
    python -m prism.entry_points.hybrid --domain cmapss --model xgboost
    python -m prism.entry_points.hybrid --domain cmapss --compare-baseline
        """
    )

    parser.add_argument('--domain', type=str, required=True,
                       help='Domain name (e.g., cmapss, climate)')
    parser.add_argument('--model', type=str, default='all',
                       help='Model to train: ridge, rf, xgboost, gbm, all')
    parser.add_argument('--compare-baseline', action='store_true',
                       help='Compare against raw sensor features')
    parser.add_argument('--target', type=str, default='initial_rul',
                       help='Target column: initial_rul, mean_rul, final_rul')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')

    args = parser.parse_args()

    # Select models
    model_names = None
    if args.model != 'all':
        model_names = [args.model]

    # Run
    results = run_hybrid(
        domain=args.domain,
        model_names=model_names,
        compare_baseline=args.compare_baseline,
        target_col=args.target,
    )

    if results is None:
        return 1

    # Final summary
    print("\n" + "=" * 70)
    print("PRISM HYBRID COMPLETE")
    print("=" * 70)
    print(f"\nBest Model: {results['best_model_name']}")
    print(f"Best RMSE:  {results['best_rmse']:.2f}")
    print(f"\nFeatures:   {len(results['prism_features'].columns)} PRISM features")
    print(f"Samples:    {len(results['prism_features'])} entities")

    if results['feature_importance'] is not None:
        print(f"\nTop 3 Features:")
        for row in results['feature_importance'].head(3).iter_rows(named=True):
            print(f"  - {row['feature']}: {row['importance']:.4f}")

    return 0


if __name__ == '__main__':
    exit(main())
