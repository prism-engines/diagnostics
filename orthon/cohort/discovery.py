"""
Orthon Cohort Discovery

Three-layer hierarchical discovery:
1. SIGNAL TYPES from vector layer (behavioral fingerprints)
2. STRUCTURAL GROUPS from geometry layer (correlation patterns)
3. TEMPORAL COHORTS from state layer (trajectory dynamics)

Key insight: Temporal cohorts must be discovered WITHIN structural groups.
"""

import polars as pl
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from orthon.cohort.confidence import (
    compute_clustering_confidence,
    find_optimal_k,
    ConfidenceMetrics,
)


@dataclass
class CohortResult:
    """Complete cohort discovery result."""

    # Signal types (from vector layer)
    signal_types: Dict[str, List[str]]  # {type_id: [signal_ids]}

    # Structural groups (from geometry layer)
    structural_groups: Dict[str, List[str]]  # {group_id: [entity_ids]}

    # Optional fields with defaults
    signal_type_confidence: Optional[ConfidenceMetrics] = None
    structural_confidence: Optional[ConfidenceMetrics] = None

    # Temporal cohorts (from state layer, within structural groups)
    temporal_cohorts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # {cohort_id: {members: [], trajectory: str, slope: float, group: str}}

    # Metadata
    discovered_at: datetime = field(default_factory=datetime.now)
    data_path: Optional[str] = None

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "COHORT DISCOVERY SUMMARY",
            "=" * 60,
            "",
            f"Signal Types: {len(self.signal_types)}",
        ]

        if self.signal_type_confidence:
            lines.append(f"  Confidence: {self.signal_type_confidence.composite_score:.3f} "
                        f"[{self.signal_type_confidence.interpretation}]")

        for type_id, signals in self.signal_types.items():
            lines.append(f"  {type_id}: {len(signals)} signals")

        lines.append("")
        lines.append(f"Structural Groups: {len(self.structural_groups)}")

        if self.structural_confidence:
            lines.append(f"  Confidence: {self.structural_confidence.composite_score:.3f} "
                        f"[{self.structural_confidence.interpretation}]")

        for group_id, entities in self.structural_groups.items():
            lines.append(f"  {group_id}: {entities}")

        lines.append("")
        lines.append(f"Temporal Cohorts: {len(self.temporal_cohorts)}")

        for cohort_id, profile in self.temporal_cohorts.items():
            traj = profile.get('trajectory', 'unknown')
            members = profile.get('members', [])
            lines.append(f"  {cohort_id}: {members} ({traj})")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dataframe(self) -> pl.DataFrame:
        """Convert to entity-level DataFrame."""
        rows = []

        # Build entity -> group mapping
        entity_to_group = {}
        for group_id, entities in self.structural_groups.items():
            for entity in entities:
                entity_to_group[entity] = group_id

        # Build entity -> cohort mapping
        entity_to_cohort = {}
        for cohort_id, profile in self.temporal_cohorts.items():
            for entity in profile.get('members', []):
                entity_to_cohort[entity] = cohort_id

        # Build rows
        all_entities = set(entity_to_group.keys()) | set(entity_to_cohort.keys())
        for entity in sorted(all_entities):
            rows.append({
                'entity_id': entity,
                'structural_group': entity_to_group.get(entity, 'unknown'),
                'temporal_cohort': entity_to_cohort.get(entity, 'unknown'),
            })

        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def to_signal_types_dataframe(self) -> pl.DataFrame:
        """Convert signal types to DataFrame."""
        rows = []
        for type_id, signals in self.signal_types.items():
            for signal in signals:
                rows.append({'signal_id': signal, 'signal_type': type_id})
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def save(self, path: Path | str):
        """Save cohorts to parquet."""
        path = Path(path)
        self.to_dataframe().write_parquet(path)

    def save_metadata(self, path: Path | str):
        """Save metadata to JSON."""
        path = Path(path)
        metadata = {
            'signal_types': {
                'n_types': len(self.signal_types),
                'confidence': self.signal_type_confidence.composite_score if self.signal_type_confidence else 0,
                'type_members': self.signal_types,
            },
            'structural_groups': {
                'n_groups': len(self.structural_groups),
                'confidence': self.structural_confidence.composite_score if self.structural_confidence else 0,
                'group_members': self.structural_groups,
            },
            'temporal_cohorts': {
                'n_cohorts': len(self.temporal_cohorts),
                'cohort_profiles': {
                    k: {kk: vv for kk, vv in v.items() if kk != 'members'}
                    for k, v in self.temporal_cohorts.items()
                },
                'cohorts_by_group': self._cohorts_by_group(),
            },
            'discovered_at': self.discovered_at.isoformat(),
        }
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def _cohorts_by_group(self) -> Dict[str, List[str]]:
        """Get cohorts organized by structural group."""
        result = {}
        for cohort_id, profile in self.temporal_cohorts.items():
            group = profile.get('group', 'unknown')
            if group not in result:
                result[group] = []
            result[group].append(cohort_id)
        return result


def discover_cohorts(
    vector_path: Optional[Path | str] = None,
    geometry_path: Optional[Path | str] = None,
    state_path: Optional[Path | str] = None,
    data_path: Optional[Path | str] = None,
    max_signal_types: int = 10,
    max_structural_groups: int = 10,
    max_temporal_cohorts: int = 5,
    min_confidence: float = 0.1,
) -> CohortResult:
    """
    Run complete three-layer cohort discovery.

    Parameters
    ----------
    vector_path : Path to vector.parquet (for signal types)
    geometry_path : Path to geometry.parquet (for structural groups)
    state_path : Path to state.parquet (for temporal cohorts)
    data_path : Base data directory (alternative to individual paths)
    max_signal_types : Maximum signal type clusters
    max_structural_groups : Maximum structural groups
    max_temporal_cohorts : Maximum temporal cohorts per structural group
    min_confidence : Minimum confidence to accept clustering (otherwise use 1 cluster)

    Returns
    -------
    CohortResult with all discovered cohorts
    """
    # Resolve paths
    if data_path:
        data_path = Path(data_path)
        vector_path = vector_path or data_path / 'vector.parquet'
        geometry_path = geometry_path or data_path / 'geometry.parquet'
        state_path = state_path or data_path / 'state.parquet'

    # Load data
    vector = pl.read_parquet(vector_path) if vector_path and Path(vector_path).exists() else None
    geometry = pl.read_parquet(geometry_path) if geometry_path and Path(geometry_path).exists() else None
    state = pl.read_parquet(state_path) if state_path and Path(state_path).exists() else None

    # Stage 1: Signal Types
    signal_types, signal_conf = discover_signal_types(vector, max_signal_types, min_confidence)

    # Stage 2: Structural Groups
    structural_groups, structural_conf = discover_structural_groups(geometry, max_structural_groups, min_confidence)

    # Stage 3: Temporal Cohorts (within structural groups)
    temporal_cohorts = discover_temporal_cohorts(state, structural_groups, max_temporal_cohorts, min_confidence)

    return CohortResult(
        signal_types=signal_types,
        signal_type_confidence=signal_conf,
        structural_groups=structural_groups,
        structural_confidence=structural_conf,
        temporal_cohorts=temporal_cohorts,
        data_path=str(data_path) if data_path else None,
    )


def discover_signal_types(
    vector: Optional[pl.DataFrame],
    max_types: int = 10,
    min_confidence: float = 0.1,
) -> tuple[Dict[str, List[str]], Optional[ConfidenceMetrics]]:
    """
    Discover signal types by behavioral fingerprint.

    Clusters signals based on their statistical profiles across all windows.
    Signals with similar behavioral patterns = same type.
    """
    if vector is None:
        return {'type_0': []}, None

    signal_col = 'source_signal' if 'source_signal' in vector.columns else 'signal_id'
    if signal_col not in vector.columns:
        return {'type_0': []}, None

    signals = sorted(vector[signal_col].unique().to_list())

    if 'engine' not in vector.columns:
        return {'type_0': signals}, None

    metrics = sorted(vector['engine'].unique().to_list())

    # Build fingerprint matrix: mean value per metric per signal
    fingerprints = []
    valid_signals = []

    for sig in signals:
        sig_data = vector.filter(pl.col(signal_col) == sig)
        fp = []
        for metric in metrics:
            val = sig_data.filter(pl.col('engine') == metric)['value'].mean()
            fp.append(val if val is not None else 0.0)

        if not all(np.isnan(v) or v == 0 for v in fp):
            fingerprints.append(fp)
            valid_signals.append(sig)

    if len(fingerprints) < 2:
        return {'type_0': signals}, None

    X = np.nan_to_num(np.array(fingerprints))
    X_scaled = StandardScaler().fit_transform(X)

    # Find optimal k with confidence
    best_k, confidence = find_optimal_k(X_scaled, max_k=min(max_types, len(valid_signals) - 1))

    # Check minimum confidence
    if confidence.composite_score < min_confidence:
        return {'type_0': valid_signals}, confidence

    # Final clustering
    labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X_scaled)

    # Build result
    type_members = {}
    for t in range(best_k):
        type_members[f"type_{t}"] = [s for s, l in zip(valid_signals, labels) if l == t]

    return type_members, confidence


def discover_structural_groups(
    geometry: Optional[pl.DataFrame],
    max_groups: int = 10,
    min_confidence: float = 0.1,
) -> tuple[Dict[str, List[str]], Optional[ConfidenceMetrics]]:
    """
    Discover structural groups by correlation pattern similarity.

    Entities with similar pairwise relationship structures = same group.
    """
    if geometry is None:
        return {'group_0': []}, None

    if 'entity_id' not in geometry.columns:
        return {'group_0': []}, None

    entities = sorted(geometry['entity_id'].unique().to_list())

    if len(entities) < 2:
        return {'group_0': entities}, None

    # Build fingerprint per entity from geometry metrics
    numeric_types = [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]
    feature_cols = [c for c in geometry.columns
                   if c not in ['entity_id', 'timestamp', 'window', 'mode_id', 'computed_at', 'signal_ids']
                   and geometry[c].dtype in numeric_types]

    if not feature_cols:
        return {'group_0': entities}, None

    # Aggregate per entity
    agg = geometry.group_by('entity_id').agg([pl.col(c).mean().alias(c) for c in feature_cols])

    fingerprints = []
    for entity in entities:
        row = agg.filter(pl.col('entity_id') == entity)
        fp = []
        for c in feature_cols:
            if len(row) > 0:
                val = row[c].to_list()[0]
                fp.append(val if val is not None else 0.0)
            else:
                fp.append(0.0)
        fingerprints.append(fp)

    X = np.array(fingerprints, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if X.std() == 0:
        return {'group_0': entities}, None

    X_scaled = StandardScaler().fit_transform(X)

    # Find optimal k with confidence
    best_k, confidence = find_optimal_k(X_scaled, max_k=min(max_groups, len(entities) - 1))

    # Check minimum confidence
    if confidence.composite_score < min_confidence:
        return {'group_0': entities}, confidence

    # Final clustering
    labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X_scaled)

    # Build result
    group_members = {}
    for g in range(best_k):
        group_members[f"group_{g}"] = [e for e, l in zip(entities, labels) if l == g]

    return group_members, confidence


def discover_temporal_cohorts(
    state: Optional[pl.DataFrame],
    structural_groups: Dict[str, List[str]],
    max_cohorts_per_group: int = 5,
    min_confidence: float = 0.1,
) -> Dict[str, Dict[str, Any]]:
    """
    Discover temporal cohorts WITHIN each structural group.

    Key insight: Cluster by trajectory dynamics, not structural similarity.
    """
    if state is None or not structural_groups:
        return {}

    result = {}
    global_cohort_id = 0

    for group_name, group_members in structural_groups.items():
        if len(group_members) < 2:
            # Single entity = single cohort
            cohort_name = f"cohort_{global_cohort_id}"
            result[cohort_name] = {
                'trajectory': 'unknown',
                'slope': 0.0,
                'members': group_members,
                'group': group_name,
            }
            global_cohort_id += 1
            continue

        # Build trajectory fingerprint for each entity
        fingerprints = []
        valid_entities = []

        for entity in group_members:
            entity_data = state.filter(pl.col('entity_id') == entity)
            if 'timestamp' in entity_data.columns:
                entity_data = entity_data.sort('timestamp')

            if len(entity_data) < 3:
                continue

            features = []
            trajectory_cols = [
                'speed', 'curvature', 'mean_velocity', 'mean_acceleration',
                'coherence', 'healthy_distance', 'degradation_score',
            ]
            available_cols = [c for c in trajectory_cols if c in entity_data.columns]

            for col in available_cols[:4]:
                values = entity_data[col].drop_nulls().to_numpy()
                if len(values) < 2:
                    features.extend([0, 0, 0, 0, 0])
                    continue

                features.append(float(values[0]))
                features.append(float(values[-1]))
                features.append(float(np.mean(values)))
                features.append(float(values[-1] - values[0]))
                try:
                    features.append(float(np.polyfit(range(len(values)), values, 1)[0]))
                except:
                    features.append(0.0)

            if features:
                fingerprints.append(features)
                valid_entities.append(entity)

        if len(fingerprints) < 2:
            cohort_name = f"cohort_{global_cohort_id}"
            result[cohort_name] = {
                'trajectory': 'unknown',
                'slope': 0.0,
                'members': group_members,
                'group': group_name,
            }
            global_cohort_id += 1
            continue

        X = np.nan_to_num(np.array(fingerprints))
        if X.std() == 0:
            cohort_name = f"cohort_{global_cohort_id}"
            result[cohort_name] = {
                'trajectory': 'stable',
                'slope': 0.0,
                'members': valid_entities,
                'group': group_name,
            }
            global_cohort_id += 1
            continue

        X_scaled = StandardScaler().fit_transform(X)

        # Find optimal k
        best_k, confidence = find_optimal_k(X_scaled, max_k=min(max_cohorts_per_group, len(valid_entities) - 1))

        # Check confidence
        if confidence.composite_score < min_confidence:
            avg_slope = np.mean([fp[4] if len(fp) > 4 else 0 for fp in fingerprints])
            cohort_name = f"cohort_{global_cohort_id}"
            result[cohort_name] = {
                'trajectory': _classify_trajectory(avg_slope),
                'slope': avg_slope,
                'members': valid_entities,
                'group': group_name,
            }
            global_cohort_id += 1
            continue

        # Cluster
        labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X_scaled)

        for c in range(best_k):
            cohort_name = f"cohort_{global_cohort_id}"
            members = [e for e, l in zip(valid_entities, labels) if l == c]

            member_fps = [fingerprints[valid_entities.index(m)] for m in members]
            avg_slope = np.mean([fp[4] if len(fp) > 4 else 0 for fp in member_fps])

            result[cohort_name] = {
                'trajectory': _classify_trajectory(avg_slope),
                'slope': avg_slope,
                'members': members,
                'group': group_name,
            }
            global_cohort_id += 1

    return result


def _classify_trajectory(slope: float) -> str:
    """Classify trajectory based on slope."""
    if slope > 1.0:
        return 'rapid'
    elif slope > 0.1:
        return 'accelerating'
    elif slope > 0.01:
        return 'gradual'
    elif slope < -0.01:
        return 'decelerating'
    else:
        return 'stable'


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface for cohort discovery."""
    import argparse

    parser = argparse.ArgumentParser(description="Orthon Cohort Discovery")
    parser.add_argument('--data-path', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output', type=str, help='Output path for cohorts.parquet')
    args = parser.parse_args()

    data_path = Path(args.data_path)

    print("=" * 70)
    print("ORTHON COHORT DISCOVERY")
    print("=" * 70)
    print(f"Data path: {data_path}")
    print()

    result = discover_cohorts(data_path=data_path)

    print(result.summary())

    # Save outputs
    output_path = args.output or data_path / 'cohorts.parquet'
    result.save(output_path)
    print(f"Saved: {output_path}")

    result.save_metadata(data_path / 'cohorts_metadata.json')
    print(f"Saved: {data_path / 'cohorts_metadata.json'}")

    signal_types_path = data_path / 'signal_types.parquet'
    result.to_signal_types_dataframe().write_parquet(signal_types_path)
    print(f"Saved: {signal_types_path}")


if __name__ == "__main__":
    main()
