"""
VSDI Analysis Package
=====================

Core package for modular VSDI data analysis.

This package provides:

- Structured configuration via VSDIConfig
- Reproducible run-folder management
- Excel-based session indexing
- MAT file condition loading
- ROI-based feature extraction
- Downstream analyses (CRF, similarity, day maps)

Design Principles
-----------------
- No global state
- No hidden literals
- IO separated from computation
- Computation separated from visualization
- Reproducible outputs via config hashing
"""

from .config import VSDIConfig
from .paths import RunFolderManager

__all__ = [
    "VSDIConfig",
    "RunFolderManager",
]

__version__ = "0.1.0"
