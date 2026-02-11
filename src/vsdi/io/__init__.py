"""
VSDI IO Submodule
=================

Data input layer for VSDI analysis.

This module is responsible only for loading and parsing external data.
It contains no computation or plotting logic.

Components
----------
- ExcelSessionIndex: Parses session metadata Excel files.
- SessionSpec: Structured representation of a single session.
- MatReader: Loads MATLAB .mat condition arrays safely.
"""

from .excel_sessions import ExcelSessionIndex, SessionSpec
from .mat_reader import MatReader

__all__ = [
    "ExcelSessionIndex",
    "SessionSpec",
    "MatReader",
]
