"""
PRISM SQL Pipeline

SQL-first data persistence and analysis.

CANONICAL RULE: Orchestrators are PURE.
All logic lives in SQL files, not in Python.
"""

from .orchestrator import SQLOrchestrator

__all__ = ['SQLOrchestrator']
