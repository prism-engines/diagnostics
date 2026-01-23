"""
Quick test of the summarize module with synthetic data.
"""

import polars as pl
import numpy as np
from pathlib import Path
import sys

# Add parent to path for import
sys.path.insert(0, str(Path(__file__).parent.parent))

from summarize import (
    MomentConfig,
    MomentDetector,
    AdaptiveSampler,
    SnapshotExtractor,
    SummaryGenerator,
)


def create_synthetic_data(n_windows: int = 500) -> pl.DataFrame:
    """
    Create synthetic geometry data that mimics degradation.

    Pattern:
    - Windows 0-100: Healthy (low lof_mean, high coherence)
    - Windows 100-200: Early warning (slight drift)
    - Windows 200-350: Transition (rapid degradation)
    - Windows 350-500: Failed (high lof_mean, low coherence)
    """
    np.random.seed(42)

    timestamps = np.arange(n_windows)

    # Create degradation curve
    # Healthy: flat low
    # Transition: sigmoid rise
    # Failed: flat high

    def sigmoid(x, center, steepness):
        return 1 / (1 + np.exp(-steepness * (x - center)))

    # LOF mean (anomaly score) - starts low, rises
    lof_base = sigmoid(timestamps, center=250, steepness=0.03)
    lof_mean = 0.8 + lof_base * 1.5 + np.random.normal(0, 0.05, n_windows)

    # Healthy distance - similar pattern
    healthy_dist = lof_base * 3 + np.random.normal(0, 0.1, n_windows)
    healthy_dist = np.clip(healthy_dist, 0, None)

    # Mode ID (0=healthy, 1=warning, 2=critical)
    mode_id = np.zeros(n_windows, dtype=int)
    mode_id[timestamps > 150] = 1
    mode_id[timestamps > 300] = 2

    # Coherence proxy (inverse of LOF, normalized)
    coherence = 1 - (lof_mean - lof_mean.min()) / (lof_mean.max() - lof_mean.min())

    # Other geometry metrics
    pca_var_1 = 0.6 + lof_base * 0.3 + np.random.normal(0, 0.02, n_windows)
    clustering_silhouette = 0.7 - lof_base * 0.5 + np.random.normal(0, 0.03, n_windows)
    mst_total_weight = 5 + lof_base * 10 + np.random.normal(0, 0.5, n_windows)
    distance_mean = 2 + lof_base * 4 + np.random.normal(0, 0.2, n_windows)

    df = pl.DataFrame({
        "entity_id": ["unit_001"] * n_windows,
        "timestamp": timestamps,
        "lof_mean": lof_mean,
        "healthy_distance_mean": healthy_dist,
        "mode_id": mode_id,
        "mean_mode_coherence": coherence,
        "pca_var_1": pca_var_1,
        "clustering_silhouette": clustering_silhouette,
        "mst_total_weight": mst_total_weight,
        "distance_mean": distance_mean,
    })

    return df


def test_moment_detection():
    """Test that moments are detected correctly."""
    print("=" * 70)
    print("TEST: Moment Detection")
    print("=" * 70)

    # Create synthetic data
    geom = create_synthetic_data(500)
    print(f"\nSynthetic data: {len(geom)} windows")
    print(f"Columns: {geom.columns}")

    # Detect moments
    config = MomentConfig()
    detector = MomentDetector(config)
    moments = detector.detect_all(geom, entity_id="unit_001")

    print("\nDetected moments:")
    for name, moment in moments.items():
        print(f"\n  {name}:")
        print(f"    Window: {moment.window}")
        if moment.window_range:
            print(f"    Range: {moment.window_range}")
        print(f"    Method: {moment.detection_method}")
        print(f"    Coherence: {moment.coherence:.3f}")
        print(f"    Confidence: {moment.confidence:.2f}")

    # Validate
    t0 = moments["T0_healthy"]
    t1 = moments["T1_uncoupling"]
    t2 = moments["T2_severe"]

    assert t0.window_range[0] < 150, f"T0 should start early, got {t0.window_range}"
    assert t1.window > t0.window_range[1], f"T1 should be after T0"
    assert t2.window > t1.window, f"T2 should be after T1"

    print("\n✓ Moment detection passed!")
    return moments


def test_adaptive_sampling():
    """Test that sampling is dense during transitions."""
    print("\n" + "=" * 70)
    print("TEST: Adaptive Sampling")
    print("=" * 70)

    geom = create_synthetic_data(500)

    config = MomentConfig(n_slider_positions=100)
    detector = MomentDetector(config)
    sampler = AdaptiveSampler(config)

    moments = detector.detect_all(geom, entity_id="unit_001")
    positions = sampler.compute_positions(500, moments)

    print(f"\nGenerated {len(positions)} positions")

    # Count by period
    period_counts = {}
    for _, _, period in positions:
        period_counts[period] = period_counts.get(period, 0) + 1

    print("\nPositions per period:")
    for period, count in sorted(period_counts.items()):
        print(f"  {period}: {count}")

    # Check that transition periods have more samples
    transition_count = period_counts.get("transition_early", 0) + period_counts.get("transition_late", 0)
    healthy_count = period_counts.get("healthy", 0)

    print(f"\nTransition samples: {transition_count}")
    print(f"Healthy samples: {healthy_count}")

    assert transition_count >= healthy_count, "Transition should have >= healthy samples"

    # Show some sample positions
    print("\nSample positions:")
    for i in [0, 25, 50, 75, -1]:
        pos, window, period = positions[i]
        print(f"  Position {pos}: window {window} ({period})")

    print("\n✓ Adaptive sampling passed!")
    return positions


def test_full_pipeline():
    """Test the complete summary generation."""
    print("\n" + "=" * 70)
    print("TEST: Full Pipeline")
    print("=" * 70)

    geom = create_synthetic_data(500)

    config = MomentConfig(n_slider_positions=100)
    generator = SummaryGenerator(config)

    summary_df = generator.generate(
        geom=geom,
        entity_id="unit_001",
        verbose=True,
    )

    print(f"\nSummary DataFrame:")
    print(f"  Rows: {len(summary_df)}")
    print(f"  Columns: {len(summary_df.columns)}")
    print(f"  Columns: {summary_df.columns[:10]}...")

    # Check that we have the expected structure
    assert "position" in summary_df.columns
    assert "window" in summary_df.columns
    assert "period" in summary_df.columns
    assert len(summary_df) <= 100

    # Check that we have geometry columns with deltas
    geom_cols = [c for c in summary_df.columns if c.startswith("geom_")]
    sigma_cols = [c for c in geom_cols if "_sigma" in c]
    delta_cols = [c for c in geom_cols if "_delta_pct" in c]

    print(f"\n  Geometry columns: {len(geom_cols)}")
    print(f"  Sigma columns: {len(sigma_cols)}")
    print(f"  Delta columns: {len(delta_cols)}")

    assert len(sigma_cols) > 0, "Should have sigma columns"
    assert len(delta_cols) > 0, "Should have delta_pct columns"

    # Show sample rows
    print("\nSample rows (position, window, period, lof_mean, lof_sigma):")
    sample = summary_df.select([
        "position", "window", "period",
        "geom_lof_mean", "geom_lof_mean_sigma"
    ]).head(5)
    print(sample)

    # Get detection report
    report = generator.get_detection_report()
    print(f"\nDetection report keys: {list(report.keys())}")

    print("\n✓ Full pipeline passed!")
    return summary_df


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("ORTHON SUMMARIZE MODULE - TESTS")
    print("=" * 70)

    test_moment_detection()
    test_adaptive_sampling()
    test_full_pipeline()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
