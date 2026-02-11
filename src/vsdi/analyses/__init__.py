"""
Downstream analyses for VSDI.

These modules consume saved outputs produced by the main overlay pipeline:
- roi_timecourses_{pct}pct.npz
- roi_patterns_{pct}pct.npz
- config.json

Rules:
- No plotting here (see vsdi.viz).
- Prefer functions that return pandas DataFrames / numpy arrays.
- Keep filesystem side-effects explicit and minimal.
"""

from .day_maps import (
    load_patterns_npz,
    compute_day_mean_maps,
)

from .crf import (
    load_timecourses_npz,
    compute_crf_amplitudes,
)

from .similarity import (
    build_patterns_dataframe,
    correlation_matrix,
    block_indices_by_day,
    pairwise_similarity_table,
)

__all__ = [
    # day_maps
    "load_patterns_npz",
    "compute_day_mean_maps",
    # crf
    "load_timecourses_npz",
    "compute_crf_amplitudes",
    # similarity
    "build_patterns_dataframe",
    "correlation_matrix",
    "block_indices_by_day",
    "pairwise_similarity_table",
]
