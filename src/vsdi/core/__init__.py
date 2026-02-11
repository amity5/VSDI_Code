"""
Core computation layer for VSDI analysis.

Rules:
- No plotting.
- No hidden literals.
- Prefer stateless functions.
- Keep IO/writing in dedicated modules (manifest/run_folder are exceptions that
  create pipeline artifacts but remain lightweight and explicit).
"""

from .dff import (
    baseline_normalize_dff,
    apply_blank_correction,
    compute_dff,
)

from .roi_masks import (
    circle_mask_2d,
    box_mask_2d,
    mask_2d_to_flat_mask,
)

from .roi_selection import (
    select_seed,
    build_roi_mask_from_seed,
)

from .time_axis import (
    frames_to_ms,
    build_time_axis_ms,
    ms_window_to_frame_slice,
)

from .features import (
    roi_mean_timecourse,
    roi_pattern_vector,
    extract_window_amplitude,
)

from .manifest import (
    build_sessions_manifest,
    build_trial_count_table,
)

from .run_folder import (
    config_tag,
)

__all__ = [
    # dff
    "baseline_normalize_dff",
    "apply_blank_correction",
    "compute_dff",
    # roi masks
    "circle_mask_2d",
    "box_mask_2d",
    "mask_2d_to_flat_mask",
    # roi selection
    "select_seed",
    "build_roi_mask_from_seed",
    # time axis
    "frames_to_ms",
    "build_time_axis_ms",
    "ms_window_to_frame_slice",
    # features
    "roi_mean_timecourse",
    "roi_pattern_vector",
    "extract_window_amplitude",
    # manifest
    "build_sessions_manifest",
    "build_trial_count_table",
    # run folder helpers
    "config_tag",
]
