"""
Ørthon Summary Generator
========================

Detects critical moments (T₀ healthy, T₁ uncoupling, T₂ severe) and generates
a compact summary dataset for interactive visualization.

Key insight: Adaptive sampling - dense during transitions, sparse during stable periods.

Output:
    summary/slider_summary.parquet  (~100 rows, one per slider position)
    summary/detection_report.json   (how moments were detected)

Usage:
    python -m prism.entry_points.summarize                    # Auto-detect moments
    python -m prism.entry_points.summarize --healthy 0:100    # Manual healthy range
    python -m prism.entry_points.summarize --entity unit_1    # Single entity
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MomentConfig:
    """Configuration for moment detection."""
    # T₀ detection (healthy)
    healthy_coherence_percentile: float = 80.0  # Coherence > this percentile
    healthy_variance_percentile: float = 20.0   # Variance < this percentile
    healthy_fallback_pct: float = 0.20          # First 20% if auto-detect fails

    # T₁ detection (first uncoupling)
    uncoupling_z_threshold: float = 2.0         # Z-score threshold
    uncoupling_min_consecutive: int = 5         # Must persist this many windows
    uncoupling_coherence_drop_pct: float = 15.0 # Alternative: coherence drop

    # T₂ detection (most severe)
    severe_coherence_floor: float = 0.4         # Sustained below this
    severe_sustained_windows: int = 10          # For how long

    # Slider sampling
    n_slider_positions: int = 100

    # Period allocations (should sum to n_slider_positions)
    samples_pre_healthy: int = 5
    samples_healthy: int = 15
    samples_early_warning: int = 15
    samples_transition_early: int = 20
    samples_transition_late: int = 15
    samples_severe: int = 20
    samples_failed: int = 10


@dataclass
class DetectedMoment:
    """A detected critical moment."""
    name: str                    # "T0_healthy", "T1_uncoupling", "T2_severe"
    window: int                  # Window index
    window_range: Optional[Tuple[int, int]] = None  # For ranges (T0)
    label: str = ""              # Human-readable
    detection_method: str = ""   # How it was detected
    trigger_pair: Optional[str] = None  # Which pair broke first (T1)
    trigger_metric: Optional[str] = None
    z_score: Optional[float] = None
    coherence: Optional[float] = None
    confidence: float = 0.0      # 0-1


@dataclass
class SliderPosition:
    """Data for one slider position."""
    position: int               # 0-99
    window: int                 # Actual window in source data
    period: str                 # "healthy", "early_warning", "transition", etc.

    # Will be populated with actual metrics
    signals: Dict[str, Dict[str, float]] = None      # signal_id -> {value, sigma, delta_pct}
    geometry: Dict[str, Dict[str, float]] = None     # pair_id -> {pearson, kendall, te_fwd, etc.}
    state: Dict[str, float] = None                   # coherence, velocity, magnitude, etc.


# =============================================================================
# MOMENT DETECTION
# =============================================================================

class MomentDetector:
    """
    Detect critical moments T₀, T₁, T₂ from geometry/state data.
    """

    def __init__(self, config: MomentConfig = None):
        self.config = config or MomentConfig()
        self.moments: Dict[str, DetectedMoment] = {}

    def detect_all(
        self,
        geom: pl.DataFrame,
        state: Optional[pl.DataFrame] = None,
        entity_id: Optional[str] = None,
    ) -> Dict[str, DetectedMoment]:
        """
        Detect all three moments for an entity.

        Args:
            geom: Geometry data with columns: entity_id, timestamp, lof_mean, mode_id, etc.
            state: Optional state data with speed, acceleration, etc.
            entity_id: If provided, filter to this entity

        Returns:
            Dict with T0_healthy, T1_uncoupling, T2_severe moments
        """
        # Filter to entity if specified
        if entity_id:
            geom = geom.filter(pl.col("entity_id") == entity_id)
            if state is not None:
                state = state.filter(pl.col("entity_id") == entity_id)

        # Sort by timestamp
        geom = geom.sort("timestamp")
        n_windows = len(geom)

        if n_windows < 10:
            raise ValueError(f"Need at least 10 windows, got {n_windows}")

        # Compute coherence proxy from available metrics
        coherence = self._compute_coherence_proxy(geom)

        # Detect each moment
        t0 = self._detect_healthy(geom, coherence)
        t1 = self._detect_uncoupling(geom, coherence, t0)
        t2 = self._detect_severe(geom, coherence, t1)

        self.moments = {
            "T0_healthy": t0,
            "T1_uncoupling": t1,
            "T2_severe": t2,
        }

        return self.moments

    def _compute_coherence_proxy(self, geom: pl.DataFrame) -> np.ndarray:
        """
        Compute coherence proxy from available geometry metrics.

        Uses inverse of lof_mean as primary proxy (lower LOF = more coherent).
        Falls back to mode_id or clustering_silhouette if available.
        """
        n = len(geom)

        # Try different metrics in order of preference
        if "mean_mode_coherence" in geom.columns:
            coherence = geom["mean_mode_coherence"].to_numpy()
            coherence = np.nan_to_num(coherence, nan=0.5)

        elif "lof_mean" in geom.columns:
            # Invert LOF: high LOF = low coherence
            lof = geom["lof_mean"].to_numpy()
            lof = np.nan_to_num(lof, nan=1.0)
            # Normalize to 0-1 range, invert
            lof_min, lof_max = np.percentile(lof, [5, 95])
            if lof_max > lof_min:
                coherence = 1 - (lof - lof_min) / (lof_max - lof_min)
                coherence = np.clip(coherence, 0, 1)
            else:
                coherence = np.ones(n) * 0.5

        elif "clustering_silhouette" in geom.columns:
            # Silhouette: higher = more coherent
            sil = geom["clustering_silhouette"].to_numpy()
            coherence = np.nan_to_num(sil, nan=0.0)
            # Normalize to 0-1
            coherence = (coherence + 1) / 2  # Silhouette is -1 to 1

        elif "healthy_distance_mean" in geom.columns:
            # Healthy distance: lower = more coherent
            hd = geom["healthy_distance_mean"].to_numpy()
            hd = np.nan_to_num(hd, nan=0.0)
            hd_max = np.percentile(hd, 95)
            if hd_max > 0:
                coherence = 1 - np.clip(hd / hd_max, 0, 1)
            else:
                coherence = np.ones(n) * 0.5

        else:
            # Last resort: use mode_id (inverted)
            if "mode_id" in geom.columns:
                mode = geom["mode_id"].to_numpy()
                mode_max = mode.max()
                if mode_max > 0:
                    coherence = 1 - mode / mode_max
                else:
                    coherence = np.ones(n) * 0.5
            else:
                coherence = np.ones(n) * 0.5

        return coherence

    def _detect_healthy(
        self,
        geom: pl.DataFrame,
        coherence: np.ndarray,
    ) -> DetectedMoment:
        """
        Detect T₀: Healthy baseline period.

        Strategy: Find period with highest coherence AND lowest coherence variance.
        """
        n = len(geom)
        cfg = self.config

        # Compute rolling variance of coherence
        window_size = max(10, n // 20)  # 5% of data or at least 10
        coherence_var = np.array([
            np.var(coherence[max(0, i-window_size):i+1])
            for i in range(n)
        ])

        # Find periods with high coherence AND low variance
        coh_threshold = np.percentile(coherence, cfg.healthy_coherence_percentile)
        var_threshold = np.percentile(coherence_var, cfg.healthy_variance_percentile)

        # Mask for healthy candidates
        healthy_mask = (coherence >= coh_threshold) & (coherence_var <= var_threshold)
        healthy_indices = np.where(healthy_mask)[0]

        if len(healthy_indices) > 0:
            # Find longest contiguous healthy period
            diffs = np.diff(healthy_indices)
            breaks = np.where(diffs > 1)[0] + 1
            segments = np.split(healthy_indices, breaks)

            # Pick longest segment
            longest = max(segments, key=len)
            window_start = int(longest[0])
            window_end = int(longest[-1])

            # Confidence based on how clear the separation is
            healthy_coh = coherence[longest].mean()
            other_coh = coherence[~healthy_mask].mean() if (~healthy_mask).sum() > 0 else 0
            confidence = min(1.0, (healthy_coh - other_coh + 0.5))

            method = "high_coherence_low_variance"
        else:
            # Fallback: first N%
            window_start = 0
            window_end = int(n * cfg.healthy_fallback_pct)
            confidence = 0.3
            method = "fallback_first_20pct"

        mean_coherence = float(coherence[window_start:window_end+1].mean())

        return DetectedMoment(
            name="T0_healthy",
            window=window_start,
            window_range=(window_start, window_end),
            label="Healthy baseline",
            detection_method=method,
            coherence=mean_coherence,
            confidence=confidence,
        )

    def _detect_uncoupling(
        self,
        geom: pl.DataFrame,
        coherence: np.ndarray,
        t0: DetectedMoment,
    ) -> DetectedMoment:
        """
        Detect T₁: First significant uncoupling.

        Strategy: First window after T₀ where coherence drops significantly
        and stays dropped.
        """
        n = len(geom)
        cfg = self.config

        # Get healthy baseline stats
        t0_start, t0_end = t0.window_range
        healthy_coh = coherence[t0_start:t0_end+1]
        healthy_mean = healthy_coh.mean()
        healthy_std = healthy_coh.std() + 1e-10

        # Compute z-scores after healthy period
        z_scores = (coherence - healthy_mean) / healthy_std

        # Look for first sustained drop
        search_start = t0_end + 1

        for i in range(search_start, n - cfg.uncoupling_min_consecutive):
            # Check if this starts a sustained drop
            window_z = z_scores[i:i+cfg.uncoupling_min_consecutive]

            if np.all(window_z < -cfg.uncoupling_z_threshold):
                # Found uncoupling
                # Try to identify which metric broke first
                trigger_pair, trigger_metric = self._identify_trigger(geom, i, t0)

                return DetectedMoment(
                    name="T1_uncoupling",
                    window=i,
                    label="First uncoupling detected",
                    detection_method="sustained_z_drop",
                    trigger_pair=trigger_pair,
                    trigger_metric=trigger_metric,
                    z_score=float(z_scores[i]),
                    coherence=float(coherence[i]),
                    confidence=min(1.0, abs(z_scores[i]) / 3),
                )

        # Alternative: look for coherence percentage drop
        for i in range(search_start, n):
            pct_drop = (healthy_mean - coherence[i]) / healthy_mean * 100
            if pct_drop > cfg.uncoupling_coherence_drop_pct:
                return DetectedMoment(
                    name="T1_uncoupling",
                    window=i,
                    label="First uncoupling detected",
                    detection_method="coherence_pct_drop",
                    coherence=float(coherence[i]),
                    confidence=0.6,
                )

        # Fallback: midpoint between healthy end and data end
        fallback_window = (t0_end + n) // 2
        return DetectedMoment(
            name="T1_uncoupling",
            window=fallback_window,
            label="Uncoupling (estimated)",
            detection_method="fallback_midpoint",
            coherence=float(coherence[fallback_window]),
            confidence=0.3,
        )

    def _detect_severe(
        self,
        geom: pl.DataFrame,
        coherence: np.ndarray,
        t1: DetectedMoment,
    ) -> DetectedMoment:
        """
        Detect T₂: Most severe / failure point.

        Strategy: Coherence minimum, or sustained low coherence.
        """
        n = len(geom)
        cfg = self.config

        # Search after T₁
        search_start = t1.window + 1
        if search_start >= n:
            search_start = n - 1

        search_coherence = coherence[search_start:]

        # Strategy A: Find coherence minimum
        min_idx_local = np.argmin(search_coherence)
        min_window = search_start + min_idx_local
        min_coherence = float(search_coherence[min_idx_local])

        # Strategy B: Find first sustained low period
        sustained_window = None
        for i in range(len(search_coherence) - cfg.severe_sustained_windows):
            if np.all(search_coherence[i:i+cfg.severe_sustained_windows] < cfg.severe_coherence_floor):
                sustained_window = search_start + i
                break

        # Pick the earlier of the two (where system first becomes severe)
        if sustained_window is not None and sustained_window < min_window:
            window = sustained_window
            method = "sustained_low_coherence"
            confidence = 0.8
        else:
            window = min_window
            method = "coherence_minimum"
            confidence = 0.7

        # If minimum is at the very end, might be monotonic decline
        if min_idx_local == len(search_coherence) - 1:
            method = "monotonic_decline_endpoint"
            confidence = 0.5

        return DetectedMoment(
            name="T2_severe",
            window=window,
            label="Most severe / failure",
            detection_method=method,
            coherence=float(coherence[window]),
            confidence=confidence,
        )

    def _identify_trigger(
        self,
        geom: pl.DataFrame,
        window: int,
        t0: DetectedMoment,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Try to identify which geometry metric/pair broke first.

        Looks at available pairwise metrics if present.
        """
        # For now, return None - would need pairwise geometry data
        # This is a placeholder for when we have detailed pair metrics
        return None, None


# =============================================================================
# ADAPTIVE SAMPLING
# =============================================================================

class AdaptiveSampler:
    """
    Generate slider window positions with adaptive density.
    Dense during transitions, sparse during stable periods.
    """

    def __init__(self, config: MomentConfig = None):
        self.config = config or MomentConfig()

    def compute_positions(
        self,
        total_windows: int,
        moments: Dict[str, DetectedMoment],
    ) -> List[Tuple[int, int, str]]:
        """
        Compute slider positions based on detected moments.

        Returns:
            List of (position, window, period) tuples
        """
        cfg = self.config
        n = total_windows

        t0 = moments["T0_healthy"]
        t1 = moments["T1_uncoupling"]
        t2 = moments["T2_severe"]

        t0_start, t0_end = t0.window_range
        t1_window = t1.window
        t2_window = t2.window

        # Define periods with boundaries
        periods = [
            ("pre_healthy", 0, t0_start, cfg.samples_pre_healthy),
            ("healthy", t0_start, t0_end, cfg.samples_healthy),
            ("early_warning", t0_end, t1_window, cfg.samples_early_warning),
            ("transition_early", t1_window, (t1_window + t2_window) // 2, cfg.samples_transition_early),
            ("transition_late", (t1_window + t2_window) // 2, t2_window, cfg.samples_transition_late),
            ("severe", t2_window, min(t2_window + (n - t2_window) // 2, n - 1), cfg.samples_severe // 2),
            ("failed", min(t2_window + (n - t2_window) // 2, n - 1), n - 1, cfg.samples_failed),
        ]

        positions = []
        position_idx = 0

        for period_name, start, end, n_samples in periods:
            if end <= start:
                continue
            if n_samples <= 0:
                continue

            # Sample evenly within period
            step = max(1, (end - start) / n_samples)
            for i in range(n_samples):
                window = int(start + i * step)
                window = min(window, n - 1)  # Clamp
                positions.append((position_idx, window, period_name))
                position_idx += 1

        # Ensure we hit exactly the critical moments
        self._ensure_moment_included(positions, t0_start, "healthy")
        self._ensure_moment_included(positions, t0_end, "healthy")
        self._ensure_moment_included(positions, t1_window, "transition_early")
        self._ensure_moment_included(positions, t2_window, "severe")

        # Sort by window and dedupe
        positions = sorted(set(positions), key=lambda x: x[1])

        # Re-index positions
        positions = [(i, w, p) for i, (_, w, p) in enumerate(positions)]

        return positions[:cfg.n_slider_positions]

    def _ensure_moment_included(
        self,
        positions: List[Tuple[int, int, str]],
        window: int,
        period: str,
    ):
        """Ensure a critical moment window is included."""
        windows = [p[1] for p in positions]
        if window not in windows:
            # Find closest position and adjust
            idx = np.argmin(np.abs(np.array(windows) - window))
            positions[idx] = (positions[idx][0], window, period)


# =============================================================================
# SNAPSHOT EXTRACTION
# =============================================================================

class SnapshotExtractor:
    """
    Extract signal, geometry, and state metrics at specific windows.
    """

    def __init__(self, healthy_baseline: Dict[str, Dict[str, float]] = None):
        self.healthy_baseline = healthy_baseline or {}

    def set_healthy_baseline(
        self,
        geom: pl.DataFrame,
        vec: Optional[pl.DataFrame],
        t0: DetectedMoment,
    ):
        """
        Compute healthy baseline statistics from T₀ period.
        """
        t0_start, t0_end = t0.window_range

        # Get geometry metrics during healthy period
        healthy_geom = geom.filter(
            (pl.col("timestamp") >= t0_start) &
            (pl.col("timestamp") <= t0_end)
        )

        # Compute mean/std for each numeric column
        numeric_cols = [c for c in healthy_geom.columns
                       if healthy_geom[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
                       and c not in ["entity_id", "timestamp", "mode_id"]]

        self.healthy_baseline = {}
        for col in numeric_cols:
            values = healthy_geom[col].drop_nulls().to_numpy()
            if len(values) > 0:
                self.healthy_baseline[col] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)) + 1e-10,
                }

    def extract_snapshot(
        self,
        position: int,
        window: int,
        period: str,
        geom: pl.DataFrame,
        vec: Optional[pl.DataFrame] = None,
        state: Optional[pl.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Extract all metrics for a single slider position.
        """
        # Filter to this window
        geom_row = geom.filter(pl.col("timestamp") == window)

        if len(geom_row) == 0:
            # Find nearest window
            all_timestamps = geom["timestamp"].to_numpy()
            nearest_idx = np.argmin(np.abs(all_timestamps - window))
            window = int(all_timestamps[nearest_idx])
            geom_row = geom.filter(pl.col("timestamp") == window)

        snapshot = {
            "position": position,
            "window": window,
            "period": period,
        }

        # Extract geometry metrics
        if len(geom_row) > 0:
            row = geom_row.to_dicts()[0]

            for col, value in row.items():
                if col in ["entity_id", "timestamp", "computed_at", "signal_ids"]:
                    continue

                if value is None:
                    continue

                # Add raw value
                snapshot[f"geom_{col}"] = float(value) if isinstance(value, (int, float)) else value

                # Add delta from healthy baseline
                if col in self.healthy_baseline:
                    baseline = self.healthy_baseline[col]
                    z_score = (value - baseline["mean"]) / baseline["std"]
                    delta_pct = (value - baseline["mean"]) / (baseline["mean"] + 1e-10) * 100
                    snapshot[f"geom_{col}_sigma"] = float(z_score)
                    snapshot[f"geom_{col}_delta_pct"] = float(delta_pct)

        # Extract state metrics if available
        if state is not None:
            state_row = state.filter(pl.col("timestamp") == window)
            if len(state_row) > 0:
                row = state_row.to_dicts()[0]
                for col in ["speed", "acceleration_magnitude", "curvature",
                           "mode_transition", "mode_delta"]:
                    if col in row and row[col] is not None:
                        snapshot[f"state_{col}"] = float(row[col]) if isinstance(row[col], (int, float)) else row[col]

        return snapshot


# =============================================================================
# SUMMARY GENERATOR (MAIN)
# =============================================================================

class SummaryGenerator:
    """
    Main class: Orchestrates detection, sampling, and extraction.
    """

    def __init__(self, config: MomentConfig = None):
        self.config = config or MomentConfig()
        self.detector = MomentDetector(self.config)
        self.sampler = AdaptiveSampler(self.config)
        self.extractor = SnapshotExtractor()

        self.moments: Dict[str, DetectedMoment] = {}
        self.positions: List[Tuple[int, int, str]] = []
        self.snapshots: List[Dict[str, Any]] = []

    def generate(
        self,
        geom: pl.DataFrame,
        vec: Optional[pl.DataFrame] = None,
        state: Optional[pl.DataFrame] = None,
        entity_id: Optional[str] = None,
        verbose: bool = True,
    ) -> pl.DataFrame:
        """
        Generate complete summary dataset.

        Args:
            geom: Geometry data
            vec: Vector data (optional)
            state: State data (optional)
            entity_id: Filter to specific entity
            verbose: Print progress

        Returns:
            DataFrame with ~100 rows, one per slider position
        """
        if verbose:
            print("=" * 70)
            print("ORTHON SUMMARY GENERATOR")
            print("=" * 70)

        # Filter to entity if specified
        if entity_id:
            geom = geom.filter(pl.col("entity_id") == entity_id)
            if vec is not None:
                vec = vec.filter(pl.col("entity_id") == entity_id)
            if state is not None:
                state = state.filter(pl.col("entity_id") == entity_id)

        geom = geom.sort("timestamp")
        n_windows = len(geom)

        if verbose:
            print(f"\nInput: {n_windows} windows")
            if entity_id:
                print(f"Entity: {entity_id}")

        # Step 1: Detect moments
        if verbose:
            print("\n[Step 1] Detecting critical moments...")

        self.moments = self.detector.detect_all(geom, state, entity_id=None)

        if verbose:
            for name, moment in self.moments.items():
                print(f"  {name}: window {moment.window} ({moment.detection_method})")
                print(f"    Coherence: {moment.coherence:.3f}, Confidence: {moment.confidence:.2f}")

        # Step 2: Compute adaptive sampling
        if verbose:
            print("\n[Step 2] Computing adaptive slider positions...")

        self.positions = self.sampler.compute_positions(n_windows, self.moments)

        if verbose:
            period_counts = {}
            for _, _, period in self.positions:
                period_counts[period] = period_counts.get(period, 0) + 1
            print(f"  Total positions: {len(self.positions)}")
            for period, count in period_counts.items():
                print(f"    {period}: {count}")

        # Step 3: Set healthy baseline
        if verbose:
            print("\n[Step 3] Computing healthy baseline...")

        self.extractor.set_healthy_baseline(geom, vec, self.moments["T0_healthy"])

        if verbose:
            print(f"  Baseline metrics: {len(self.extractor.healthy_baseline)}")

        # Step 4: Extract snapshots
        if verbose:
            print("\n[Step 4] Extracting snapshots...")

        self.snapshots = []
        for position, window, period in self.positions:
            snapshot = self.extractor.extract_snapshot(
                position, window, period, geom, vec, state
            )
            self.snapshots.append(snapshot)

        if verbose:
            print(f"  Extracted {len(self.snapshots)} snapshots")

        # Convert to DataFrame
        df = pl.DataFrame(self.snapshots)

        if verbose:
            print(f"\n[Complete] Summary: {len(df)} rows, {len(df.columns)} columns")

        return df

    def get_detection_report(self) -> Dict[str, Any]:
        """
        Get detection report as JSON-serializable dict.
        """
        return {
            "moments": {
                name: asdict(moment) for name, moment in self.moments.items()
            },
            "n_positions": len(self.positions),
            "period_distribution": self._get_period_distribution(),
            "config": asdict(self.config),
        }

    def _get_period_distribution(self) -> Dict[str, int]:
        """Get count of positions per period."""
        dist = {}
        for _, _, period in self.positions:
            dist[period] = dist.get(period, 0) + 1
        return dist

    def save(
        self,
        output_dir: Path,
        df: pl.DataFrame,
    ):
        """
        Save summary data and detection report.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save parquet
        parquet_path = output_dir / "slider_summary.parquet"
        df.write_parquet(parquet_path)
        print(f"Saved: {parquet_path}")

        # Save detection report
        report_path = output_dir / "detection_report.json"
        with open(report_path, "w") as f:
            json.dump(self.get_detection_report(), f, indent=2, default=str)
        print(f"Saved: {report_path}")

        # Save healthy baseline
        baseline_path = output_dir / "healthy_baseline.json"
        with open(baseline_path, "w") as f:
            json.dump(self.extractor.healthy_baseline, f, indent=2)
        print(f"Saved: {baseline_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Orthon Summary Generator - Detect moments and generate slider data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data", type=str, default="data",
                       help="Data directory with geometry.parquet")
    parser.add_argument("--output", type=str, default="summary",
                       help="Output directory")
    parser.add_argument("--entity", type=str, default=None,
                       help="Filter to specific entity_id")
    parser.add_argument("--positions", type=int, default=100,
                       help="Number of slider positions")

    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)

    # Load data
    print(f"Loading data from {data_path}...")

    geom_path = data_path / "geometry.parquet"
    if not geom_path.exists():
        raise FileNotFoundError(f"No geometry data at {geom_path}")
    geom = pl.read_parquet(geom_path)
    print(f"  Geometry: {len(geom)} rows")

    vec_path = data_path / "vector.parquet"
    vec = pl.read_parquet(vec_path) if vec_path.exists() else None
    if vec is not None:
        print(f"  Vector: {len(vec)} rows")

    state_path = data_path / "state.parquet"
    state = pl.read_parquet(state_path) if state_path.exists() else None
    if state is not None:
        print(f"  State: {len(state)} rows")

    # Configure
    config = MomentConfig(n_slider_positions=args.positions)

    # Generate
    generator = SummaryGenerator(config)

    # If multiple entities, process each
    if args.entity:
        entities = [args.entity]
    else:
        entities = geom["entity_id"].unique().sort().to_list()

    print(f"\nProcessing {len(entities)} entities...")

    all_summaries = []
    for entity in entities[:1]:  # Start with first entity for testing
        print(f"\n{'='*70}")
        print(f"ENTITY: {entity}")
        print("="*70)

        summary_df = generator.generate(
            geom=geom,
            vec=vec,
            state=state,
            entity_id=entity,
            verbose=True,
        )

        # Add entity_id column
        summary_df = summary_df.with_columns(pl.lit(entity).alias("entity_id"))
        all_summaries.append(summary_df)

    # Combine all entities
    if all_summaries:
        combined = pl.concat(all_summaries)

        # Save
        generator.save(output_path, combined)

        print("\n" + "="*70)
        print("SUMMARY COMPLETE")
        print("="*70)
        print(f"  Entities: {len(all_summaries)}")
        print(f"  Total rows: {len(combined)}")
        print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
