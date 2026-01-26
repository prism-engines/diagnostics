"""
Minimal metadata stub for backwards compatibility.
"""

from dataclasses import dataclass, field
from typing import Set, Optional


@dataclass
class EngineMetadata:
    """Engine metadata - stub for compatibility."""
    name: str
    engine_type: str = "core"
    description: str = ""
    domains: Set[str] = field(default_factory=set)
    requires_window: bool = False
    deterministic: bool = True
    min_samples: int = 10
