"""
Minimal engine base stub for backwards compatibility.
The new stream architecture doesn't use BaseEngine, but
some copied engines still import it.
"""

from typing import Dict, Any, Optional
import numpy as np


class BaseEngine:
    """Stub base class - not used in stream architecture."""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def compute(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Use module-level compute() function")


def get_window_dates(*args, **kwargs):
    """Stub function - not used in stream architecture."""
    return None, None
