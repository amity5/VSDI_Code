"""
Visualization layer for VSDI.

Rules:
- Plotting only. No heavy computation; use vsdi.core and vsdi.analyses first.
- Display transforms (rotation/flip) are applied here only.
- Functions should accept data arrays / DataFrames and return (fig, ax).
"""

from .overlays import plot_timecourse_overlays
from .maps import (
    apply_display_transform,
    plot_seed_and_roi_on_map,
    plot_day_map,
)
from .heatmaps import (
    plot_similarity_heatmap,
    plot_day_blocked_similarity_heatmap,
)

__all__ = [
    "plot_timecourse_overlays",
    "apply_display_transform",
    "plot_seed_and_roi_on_map",
    "plot_day_map",
    "plot_similarity_heatmap",
    "plot_day_blocked_similarity_heatmap",
]
