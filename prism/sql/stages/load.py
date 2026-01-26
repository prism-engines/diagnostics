"""
Load Stage Orchestrator

PURE: Loads 00_load.sql, creates base views.
NO computation. NO inline SQL.

CANONICAL SCHEMA: observations.parquet MUST have columns:
  entity_id, signal_id, I, y, unit

No mapping. No aliases. I means I. y means y.
Alias mapping happens at INTAKE, not here.
"""

from pathlib import Path
from typing import Optional
import yaml

from .base import StageOrchestrator


# =============================================================================
# CANONICAL SCHEMA - THE RULE
# =============================================================================
# observations.parquet columns: entity_id, signal_id, I, y, unit
#
# I = index (time, space, frequency, scale, cycle)
# y = value (the measurement)
#
# No aliases. No mapping after intake. I and y. Done.
# =============================================================================

CANONICAL_COLUMNS = ['entity_id', 'signal_id', 'I', 'y', 'unit']


class LoadStage(StageOrchestrator):
    """Load observations and create base view.

    EXPECTS canonical schema: entity_id, signal_id, I, y, unit
    Does NOT map column names - that's intake's job.
    """

    SQL_FILE = '00_load.sql'

    VIEWS = [
        'v_base',
        'v_schema_validation',
        'v_signal_inventory',
        'v_data_quality',
    ]

    DEPENDS_ON = []  # First stage, no dependencies

    def __init__(self, conn, domain_config: Optional[dict] = None):
        """
        Initialize LoadStage.

        Args:
            conn: DuckDB connection
            domain_config: Optional domain configuration dict with index settings
        """
        super().__init__(conn)
        self.domain_config = domain_config or {}

    @classmethod
    def from_domain_file(cls, conn, domain_path: str = 'config/domain.yaml'):
        """Create LoadStage with domain config from file."""
        config_path = Path(domain_path)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return cls(conn, domain_config=config)
        return cls(conn)

    def load_observations(self, path: str) -> None:
        """
        Load observations parquet into database.

        EXPECTS canonical schema: entity_id, signal_id, I, y, unit
        Validates schema and loads directly - no column mapping.

        Raises:
            ValueError: If required columns (I, y) are missing
        """
        # Get actual columns from file
        cols = self.conn.execute(f"DESCRIBE SELECT * FROM '{path}'").fetchall()
        col_names = [c[0] for c in cols]

        # Validate canonical schema
        if 'I' not in col_names:
            raise ValueError(
                f"observations.parquet missing required column 'I'. "
                f"Found columns: {col_names}. "
                f"Column mapping should happen at intake, not here."
            )
        if 'y' not in col_names:
            raise ValueError(
                f"observations.parquet missing required column 'y'. "
                f"Found columns: {col_names}. "
                f"Column mapping should happen at intake, not here."
            )

        # Load directly - no mapping needed
        self.conn.execute(f"CREATE OR REPLACE TABLE observations AS SELECT * FROM '{path}'")

    def get_row_count(self) -> int:
        """Return number of rows loaded."""
        return self.conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]

    def get_signal_count(self) -> int:
        """Return number of distinct signals."""
        return self.conn.execute("SELECT COUNT(DISTINCT signal_id) FROM observations").fetchone()[0]

    def get_index_range(self) -> tuple:
        """Return (min_I, max_I) range of index."""
        result = self.conn.execute("SELECT MIN(I), MAX(I) FROM observations").fetchone()
        return result

    def get_index_info(self) -> dict:
        """Return index dimension info from domain config."""
        index_config = self.domain_config.get('index', {})
        return {
            'dimension': index_config.get('dimension', 'unknown'),
            'unit': index_config.get('unit', 'unknown'),
            'sampling_rate': index_config.get('sampling_rate'),
        }
