"""
Base Stage Orchestrator

CANONICAL RULE: Orchestrators are PURE.

This base class provides:
  - load_sql()     : Load SQL from file
  - run()          : Execute the SQL
  - get_views()    : Return list of views created
  - validate()     : Check views exist

NO computation. NO inline SQL. NO business logic.
"""

from pathlib import Path
from typing import List
import duckdb


class StageOrchestrator:
    """
    Base class for pure stage orchestrators.

    Each stage:
    1. Has a SQL file (sql/{stage_name}.sql)
    2. Creates views (v_*)
    3. Has no computation logic
    """

    # Override in subclass
    SQL_FILE: str = None
    VIEWS: List[str] = []
    DEPENDS_ON: List[str] = []

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        """
        Initialize with database connection.

        Args:
            conn: DuckDB connection (shared across all stages)
        """
        self.conn = conn
        self._sql_dir = Path(__file__).parent.parent / 'sql'
        self._loaded = False

    @property
    def sql_path(self) -> Path:
        """Path to this stage's SQL file."""
        if self.SQL_FILE is None:
            raise NotImplementedError(f"{self.__class__.__name__} must define SQL_FILE")
        return self._sql_dir / self.SQL_FILE

    def load_sql(self) -> str:
        """Load SQL from file. No modification."""
        return self.sql_path.read_text()

    def run(self) -> None:
        """
        Execute this stage's SQL.

        PURE: Just loads and executes. No logic.
        """
        sql = self.load_sql()
        self.conn.execute(sql)
        self._loaded = True

    def get_views(self) -> List[str]:
        """Return list of views this stage creates."""
        return self.VIEWS.copy()

    def get_dependencies(self) -> List[str]:
        """Return list of views this stage depends on."""
        return self.DEPENDS_ON.copy()

    def validate(self) -> bool:
        """
        Validate all views exist.

        Returns True if all views are queryable.
        """
        for view in self.VIEWS:
            try:
                self.conn.execute(f"SELECT 1 FROM {view} LIMIT 0")
            except Exception:
                return False
        return True

    def query(self, view_name: str):
        """
        Query a view by name.

        Args:
            view_name: Name of view (must be in self.VIEWS)

        Returns:
            DataFrame
        """
        if view_name not in self.VIEWS:
            raise ValueError(f"View {view_name} not in {self.__class__.__name__}.VIEWS")
        return self.conn.execute(f"SELECT * FROM {view_name}").fetchdf()

    def __repr__(self):
        status = "loaded" if self._loaded else "not loaded"
        return f"<{self.__class__.__name__} [{status}] views={len(self.VIEWS)}>"
