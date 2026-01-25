"""
PRISM Layer Dependencies
========================

These are HARD dependencies. Not suggestions. Not optimizations.
If upstream is missing, downstream MUST FAIL.

The dependency chain reflects mathematical necessity:
- You cannot measure motion without knowing the space
- You cannot compute force without knowing the motion
"""

from pathlib import Path
from typing import List


LAYER_DEPENDENCIES = {
    'observations': [],
    'vector': ['observations'],
    'geometry': ['vector'],
    'dynamics': ['vector', 'geometry'],   # BOTH required
    'physics': ['vector', 'geometry', 'dynamics'],  # ALL required
}


def check_dependencies(layer: str, data_dir: Path) -> None:
    """
    Verify all upstream dependencies exist before computing a layer.

    Raises FileNotFoundError if any dependency is missing.

    Args:
        layer: The layer to compute ('vector', 'geometry', 'dynamics', 'physics')
        data_dir: Path to data directory containing parquet files
    """
    required = LAYER_DEPENDENCIES.get(layer, [])

    missing = []
    for dep in required:
        dep_path = data_dir / f'{dep}.parquet'
        if not dep_path.exists():
            missing.append(dep)

    if missing:
        chain = ' → '.join(['observations'] + [l for l in LAYER_DEPENDENCIES.keys() if l != 'observations'][:list(LAYER_DEPENDENCIES.keys()).index(layer)])
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"Cannot compute {layer}: missing required upstream layers\n"
            f"{'='*60}\n\n"
            f"Missing: {', '.join(f'{m}.parquet' for m in missing)}\n\n"
            f"Dependency chain: {chain}\n\n"
            f"Run the upstream layers first:\n"
            + '\n'.join(f"  python -m prism.entry_points.{m}" for m in missing)
            + f"\n{'='*60}"
        )


def get_compute_order() -> List[str]:
    """Return layers in correct computation order."""
    return ['observations', 'vector', 'geometry', 'dynamics', 'physics']


def get_dependencies(layer: str) -> List[str]:
    """Get direct dependencies for a layer."""
    return LAYER_DEPENDENCIES.get(layer, [])


def validate_layer_output(layer: str, n_rows: int, n_entities: int, n_signals: int) -> None:
    """
    Validate that layer output has correct row granularity.

    Args:
        layer: The layer being validated
        n_rows: Actual number of rows in output
        n_entities: Expected number of entities
        n_signals: Expected number of signals per entity
    """
    expected_granularity = {
        'vector': ('entity_signal', n_entities * n_signals),
        'geometry': ('entity', n_entities),
        'dynamics': ('entity', n_entities),
        'physics': ('entity', n_entities),
    }

    if layer not in expected_granularity:
        return

    granularity, expected_rows = expected_granularity[layer]

    if n_rows != expected_rows:
        raise ValueError(
            f"\n{'='*60}\n"
            f"WRONG GRANULARITY for {layer}.parquet\n"
            f"{'='*60}\n\n"
            f"Expected: {expected_rows} rows ({granularity})\n"
            f"Got: {n_rows} rows\n\n"
            f"{'entity' if granularity == 'entity' else 'entity×signal'} granularity required.\n"
            f"{'='*60}"
        )
