from __future__ import annotations

from typing import Tuple, Literal, Optional
import numpy as np

from .time_axis import ms_window_to_frame_slice


def _validate_pixels_by_frames(arr: np.ndarray) -> None:
    if not isinstance(arr, np.ndarray):
        raise TypeError("arr must be a numpy array")
    if arr.ndim != 2:
        raise ValueError(f"Expected (n_pixels, n_frames). Got {arr.shape}.")


def roi_mean_timecourse(
    arr: np.ndarray,
    roi_flat_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute mean ROI timecourse.

    Args:
        arr: (n_pixels, n_frames)
        roi_flat_mask: (n_pixels,) boolean

    Returns:
        tc: (n_frames,)
    """
    _validate_pixels_by_frames(arr)
    if roi_flat_mask.ndim != 1:
        raise ValueError("roi_flat_mask must be 1D")
    if roi_flat_mask.shape[0] != arr.shape[0]:
        raise ValueError("roi_flat_mask length must match n_pixels")

    if roi_flat_mask.sum() == 0:
        raise ValueError("ROI mask has zero pixels")

    return arr[roi_flat_mask, :].mean(axis=0)


def roi_pattern_vector(
    arr: np.ndarray,
    roi_flat_mask: np.ndarray,
    t_ms: np.ndarray,
    window_ms: Tuple[int, int],
    agg: Literal["mean", "max"] = "mean",
) -> np.ndarray:
    """
    Compute ROI spatial pattern vector by aggregating over time window.

    Returns:
        vec: (n_roi_pixels,) float
    """
    _validate_pixels_by_frames(arr)
    if t_ms.ndim != 1 or t_ms.shape[0] != arr.shape[1]:
        raise ValueError("t_ms must be 1D and match n_frames")

    if roi_flat_mask.ndim != 1 or roi_flat_mask.shape[0] != arr.shape[0]:
        raise ValueError("roi_flat_mask must be 1D and match n_pixels")

    sl = ms_window_to_frame_slice(t_ms, window_ms)
    roi_data = arr[roi_flat_mask, sl]

    if agg == "mean":
        return roi_data.mean(axis=1)
    if agg == "max":
        return roi_data.max(axis=1)
    raise ValueError(f"Unknown agg={agg}")


def extract_window_amplitude(
    timecourse: np.ndarray,
    t_ms: np.ndarray,
    window_ms: Tuple[int, int],
    agg: Literal["mean", "max"] = "mean",
) -> float:
    """
    Extract scalar amplitude from timecourse in a time window.
    """
    if timecourse.ndim != 1:
        raise ValueError("timecourse must be 1D")
    if t_ms.ndim != 1 or t_ms.shape[0] != timecourse.shape[0]:
        raise ValueError("t_ms must be 1D and match timecourse length")

    sl = ms_window_to_frame_slice(t_ms, window_ms)
    seg = timecourse[sl]

    if agg == "mean":
        return float(np.mean(seg))
    if agg == "max":
        return float(np.max(seg))
    raise ValueError(f"Unknown agg={agg}")
