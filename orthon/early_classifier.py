"""
Ørthon Early Classification Module

Classifies entities into risk cohorts using early-life data only.
Domain-agnostic: works with any sensor/signal data.

Two independent detection channels:
1. Structural: Geometry outliers (correlation pattern anomalies)
2. Temporal: Signal derivative rankings (behavioral velocity)

Output: Risk classification with evidence for upstream tracing.
"""

import polars as pl
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from pathlib import Path


class RiskClass(Enum):
    """Risk classification based on combined structural + temporal signals."""
    HIGH = "high"           # Both structural AND temporal anomaly
    TEMPORAL = "temporal"   # Temporal anomaly only (fast degradation signature)
    STRUCTURAL = "structural"  # Structural anomaly only (geometry divergence)
    STANDARD = "standard"   # Neither - normal population


@dataclass
class ClassificationResult:
    """Classification output for a single entity."""
    entity_id: str
    risk_class: RiskClass
    structural_outlier: bool
    temporal_percentile: float  # 0-100, lower = more anomalous
    confidence: float           # 0-1
    evidence_signals: list[str] # Which signals contributed
    evidence_metrics: list[str] # Which geometry metrics flagged


def classify_fleet(
    geometry_path: Path | str,
    vector_path: Path | str,
    life_fraction: float = 0.20,
    temporal_threshold_pct: float = 12.0,
    min_confidence: float = 0.5,
) -> pl.DataFrame:
    """
    Classify all entities in fleet using early-life data only.

    Parameters
    ----------
    geometry_path : Path to geometry.parquet
    vector_path : Path to vector.parquet
    life_fraction : Fraction of life to use (default 0.20 = first 20%)
    temporal_threshold_pct : Percentile threshold for temporal flags (default 12%)
    min_confidence : Minimum confidence to report (default 0.5)

    Returns
    -------
    DataFrame with columns:
        entity_id, risk_class, structural_outlier, temporal_percentile,
        confidence, evidence_signals, evidence_metrics
    """
    # Load data
    geometry = pl.read_parquet(geometry_path)
    vector = pl.read_parquet(vector_path)

    # Get entity list
    entities = geometry.select("entity_id").unique()

    # 1. Structural classification (geometry outliers)
    structural = _detect_structural_outliers(geometry, life_fraction)

    # 2. Temporal classification (derivative rankings per signal)
    temporal = _rank_temporal_signatures(vector, life_fraction)

    # 3. Combine classifications
    classified = _combine_classifications(
        entities=entities,
        structural=structural,
        temporal=temporal,
        temporal_threshold_pct=temporal_threshold_pct,
    )

    return classified.filter(pl.col("confidence") >= min_confidence)


def _detect_structural_outliers(
    geometry: pl.DataFrame,
    life_fraction: float,
) -> pl.DataFrame:
    """
    Detect entities with anomalous correlation structure.

    Uses clustering on geometry features to find outlier groups.
    Returns entity_id, is_outlier, outlier_score, flagged_metrics.
    """
    # Filter to early life
    early_geometry = _filter_early_life(geometry, life_fraction)

    # Aggregate geometry features per entity (early window only)
    geo_features = [
        "pca_var_1", "pca_effective_dim", "clustering_silhouette",
        "mst_total_weight", "mst_avg_degree", "lof_mean",
        "distance_mean", "distance_std", "mi_mean",
        "hull_volume", "hull_centroid_dist"
    ]

    # Use only columns that exist
    available_features = [f for f in geo_features if f in early_geometry.columns]

    if not available_features:
        # No geometry features - return empty structural results
        return pl.DataFrame({
            "entity_id": [],
            "structural_outlier": [],
            "outlier_score": [],
            "flagged_metrics": [],
        })

    # Aggregate per entity
    agg_exprs = [pl.col(f).mean().alias(f) for f in available_features]
    entity_geo = early_geometry.group_by("entity_id").agg(agg_exprs)

    # Compute z-scores for outlier detection
    outlier_scores = _compute_multivariate_outlier_scores(
        entity_geo, available_features
    )

    # Flag outliers (beyond 2.5 sigma on any dimension)
    threshold = 2.5
    outliers = outlier_scores.with_columns([
        (pl.col("max_zscore") > threshold).alias("structural_outlier"),
    ])

    return outliers.select([
        "entity_id", "structural_outlier", "outlier_score", "flagged_metrics"
    ])


def _rank_temporal_signatures(
    vector: pl.DataFrame,
    life_fraction: float,
) -> pl.DataFrame:
    """
    Rank entities by early derivative signatures per signal.

    Returns entity_id, temporal_percentile, top_signals, derivative_scores.
    """
    # Filter to early life
    early_vector = _filter_early_life(vector, life_fraction)

    # Check if we have signal-level data
    if "signal_id" not in early_vector.columns:
        # Fall back to entity-level if no signal breakdown
        return _rank_temporal_entity_level(early_vector)

    # Compute derivative statistics per entity per signal
    # Looking for: elevated derivatives = fast change = potential failure signature

    derivative_stats = (
        early_vector
        .filter(pl.col("signal_id").str.contains("deriv"))
        .group_by(["entity_id", "signal_id"])
        .agg([
            pl.col("value").mean().alias("deriv_mean"),
            pl.col("value").std().alias("deriv_std"),
            pl.col("value").max().alias("deriv_max"),
        ])
    )

    if derivative_stats.height == 0:
        # No derivative signals found - try computing from raw values
        derivative_stats = _compute_derivatives_from_raw(early_vector)

    # Aggregate to entity level - take max across signals
    entity_temporal = (
        derivative_stats
        .group_by("entity_id")
        .agg([
            pl.col("deriv_mean").max().alias("max_deriv_mean"),
            pl.col("deriv_max").max().alias("max_deriv_max"),
            # Track which signals had highest derivatives
            pl.struct(["signal_id", "deriv_mean"])
                .sort_by("deriv_mean", descending=True)
                .first()
                .alias("top_signal_info"),
        ])
    )

    # Rank entities (higher derivative = more anomalous = lower percentile)
    n_entities = entity_temporal.height
    ranked = (
        entity_temporal
        .with_columns([
            pl.col("max_deriv_mean")
                .rank(descending=True)
                .truediv(n_entities)
                .mul(100)
                .alias("temporal_percentile"),
        ])
    )

    return ranked.select([
        "entity_id", "temporal_percentile", "max_deriv_mean", "top_signal_info"
    ])


def _combine_classifications(
    entities: pl.DataFrame,
    structural: pl.DataFrame,
    temporal: pl.DataFrame,
    temporal_threshold_pct: float,
) -> pl.DataFrame:
    """
    Combine structural and temporal classifications into risk classes.

    Risk matrix:
        Structural  | Temporal      | Risk Class
        ------------------------------------
        Outlier     | Top N%        | HIGH
        Outlier     | Normal        | STRUCTURAL
        Normal      | Top N%        | TEMPORAL
        Normal      | Normal        | STANDARD
    """
    # Join all data
    combined = (
        entities
        .join(structural, on="entity_id", how="left")
        .join(temporal, on="entity_id", how="left")
    )

    # Fill nulls for entities without signals in either detector
    combined = combined.with_columns([
        pl.col("structural_outlier").fill_null(False),
        pl.col("temporal_percentile").fill_null(50.0),  # Assume median if unknown
    ])

    # Classify risk
    classified = combined.with_columns([
        pl.when(
            pl.col("structural_outlier") & (pl.col("temporal_percentile") <= temporal_threshold_pct)
        ).then(pl.lit("high"))
        .when(
            pl.col("structural_outlier")
        ).then(pl.lit("structural"))
        .when(
            pl.col("temporal_percentile") <= temporal_threshold_pct
        ).then(pl.lit("temporal"))
        .otherwise(pl.lit("standard"))
        .alias("risk_class"),

        # Confidence based on signal strength
        _compute_confidence(
            pl.col("outlier_score"),
            pl.col("temporal_percentile"),
            temporal_threshold_pct,
        ).alias("confidence"),
    ])

    # Collect evidence
    classified = classified.with_columns([
        pl.col("flagged_metrics").fill_null(pl.lit([])).alias("evidence_metrics"),
        pl.when(pl.col("top_signal_info").is_not_null())
            .then(pl.col("top_signal_info").struct.field("signal_id"))
            .otherwise(pl.lit(None))
            .alias("evidence_signals"),
    ])

    return classified.select([
        "entity_id",
        "risk_class",
        "structural_outlier",
        "temporal_percentile",
        "confidence",
        "evidence_signals",
        "evidence_metrics",
    ])


def _filter_early_life(df: pl.DataFrame, life_fraction: float) -> pl.DataFrame:
    """Filter dataframe to early portion of each entity's life."""
    if "timestamp" not in df.columns:
        return df  # No temporal info, return as-is

    # Get max timestamp per entity
    max_times = df.group_by("entity_id").agg(pl.col("timestamp").max().alias("max_ts"))

    # Join and filter
    return (
        df
        .join(max_times, on="entity_id")
        .filter(pl.col("timestamp") <= pl.col("max_ts") * life_fraction)
        .drop("max_ts")
    )


def _compute_multivariate_outlier_scores(
    df: pl.DataFrame,
    feature_cols: list[str]
) -> pl.DataFrame:
    """Compute outlier scores using z-score method across multiple features."""
    # Compute z-scores for each feature
    zscore_exprs = []
    for col in feature_cols:
        mean = df.select(pl.col(col).mean()).item()
        std = df.select(pl.col(col).std()).item()
        if std > 0:
            zscore_exprs.append(
                ((pl.col(col) - mean) / std).abs().alias(f"{col}_zscore")
            )

    if not zscore_exprs:
        return df.with_columns([
            pl.lit(0.0).alias("outlier_score"),
            pl.lit(0.0).alias("max_zscore"),
            pl.lit([]).alias("flagged_metrics"),
        ])

    with_zscores = df.with_columns(zscore_exprs)

    # Max z-score across all features
    zscore_cols = [f"{col}_zscore" for col in feature_cols if f"{col}_zscore" in with_zscores.columns]

    return with_zscores.with_columns([
        pl.max_horizontal(zscore_cols).alias("max_zscore"),
        pl.mean_horizontal(zscore_cols).alias("outlier_score"),
        # Track which metrics flagged (z > 2)
        pl.concat_list([
            pl.when(pl.col(c) > 2.0).then(pl.lit(c.replace("_zscore", ""))).otherwise(pl.lit(None))
            for c in zscore_cols
        ]).list.drop_nulls().alias("flagged_metrics"),
    ])


def _compute_derivatives_from_raw(vector: pl.DataFrame) -> pl.DataFrame:
    """Compute derivatives from raw signal values if not pre-computed."""
    # Group by entity and signal, compute rate of change
    return (
        vector
        .sort(["entity_id", "signal_id", "timestamp"])
        .with_columns([
            pl.col("value").diff().over(["entity_id", "signal_id"]).alias("derivative")
        ])
        .group_by(["entity_id", "signal_id"])
        .agg([
            pl.col("derivative").mean().alias("deriv_mean"),
            pl.col("derivative").std().alias("deriv_std"),
            pl.col("derivative").max().alias("deriv_max"),
        ])
    )


def _rank_temporal_entity_level(vector: pl.DataFrame) -> pl.DataFrame:
    """Fallback: rank entities when no signal-level breakdown available."""
    entity_stats = (
        vector
        .group_by("entity_id")
        .agg([
            pl.col("value").mean().alias("value_mean"),
            pl.col("value").std().alias("value_std"),
        ])
    )

    n = entity_stats.height
    return entity_stats.with_columns([
        pl.col("value_std").rank(descending=True).truediv(n).mul(100).alias("temporal_percentile"),
        pl.lit(None).alias("max_deriv_mean"),
        pl.lit(None).alias("top_signal_info"),
    ])


def _compute_confidence(
    outlier_score: pl.Expr,
    temporal_percentile: pl.Expr,
    threshold: float,
) -> pl.Expr:
    """Compute confidence score based on signal strength."""
    # Higher confidence when:
    # - Outlier score is very high (far from population)
    # - Temporal percentile is very low (clearly anomalous)

    structural_conf = (outlier_score / 5.0).clip(0, 1)  # Normalize outlier score
    temporal_conf = (1 - temporal_percentile / 100).clip(0, 1)  # Lower percentile = higher conf

    return (structural_conf + temporal_conf) / 2


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Command-line interface for fleet classification."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify fleet entities into risk cohorts using early-life data"
    )
    parser.add_argument("--geometry", required=True, help="Path to geometry.parquet")
    parser.add_argument("--vector", required=True, help="Path to vector.parquet")
    parser.add_argument("--output", required=True, help="Output path for classifications")
    parser.add_argument("--life-fraction", type=float, default=0.20,
                        help="Fraction of life to use (default: 0.20)")
    parser.add_argument("--temporal-threshold", type=float, default=12.0,
                        help="Percentile threshold for temporal flags (default: 12)")

    args = parser.parse_args()

    print(f"Classifying fleet using first {args.life_fraction*100:.0f}% of life...")

    results = classify_fleet(
        geometry_path=args.geometry,
        vector_path=args.vector,
        life_fraction=args.life_fraction,
        temporal_threshold_pct=args.temporal_threshold,
    )

    # Summary
    print("\n" + "="*60)
    print("FLEET CLASSIFICATION SUMMARY")
    print("="*60)

    for risk_class in ["high", "temporal", "structural", "standard"]:
        count = results.filter(pl.col("risk_class") == risk_class).height
        pct = count / results.height * 100 if results.height > 0 else 0
        print(f"  {risk_class.upper():12s}: {count:4d} ({pct:5.1f}%)")

    print("="*60)

    # Show flagged entities
    flagged = results.filter(pl.col("risk_class") != "standard")
    if flagged.height > 0:
        print("\nFLAGGED ENTITIES:")
        print(flagged.select([
            "entity_id", "risk_class", "temporal_percentile", "evidence_signals"
        ]))

    # Save
    results.write_parquet(args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
