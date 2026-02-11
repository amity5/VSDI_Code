from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np


DFFMode = Literal["none", "divide_blank", "subtract_blank"]


def _validate_array_2d(arr: np.ndarray) -> None:
    if not isinstance(arr, np.ndarray):
        raise TypeError("arr must be a numpy array")
    if arr.ndim != 2:
        raise ValueError(f"Expected arr.ndim == 2 (pixels x frames). Got {arr.shape}.")


def baseline_normalize_dff(
    arr: np.ndarray,
    baseline_frames: Tuple[int, int],
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute ΔF/F using baseline frames.

    Args:
        arr: shape (n_pixels, n_frames)
        baseline_frames: (start, end) inclusive, 0-based indices
        eps: small constant to avoid division by zero

    Returns:
        dff: same shape as arr
    """
    _validate_array_2d(arr)

    b0, b1 = baseline_frames
    if not (0 <= b0 <= b1 < arr.shape[1]):
        raise ValueError(f"Invalid baseline_frames={baseline_frames} for n_frames={arr.shape[1]}")

    baseline = arr[:, b0 : b1 + 1].mean(axis=1, keepdims=True)
    dff = (arr - baseline) / (baseline + eps)
    return dff


def apply_blank_correction(
    dff_cond: np.ndarray,
    dff_blank: Optional[np.ndarray],
    mode: DFFMode,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Apply blank correction after ΔF/F normalization.

    Args:
        dff_cond: ΔF/F for condition, shape (n_pixels, n_frames)
        dff_blank: ΔF/F for blank condition, same shape
        mode: "none" | "divide_blank" | "subtract_blank"

    Returns:
        corrected ΔF/F array
    """
    _validate_array_2d(dff_cond)
    if mode == "none":
        return dff_cond

    if dff_blank is None:
        raise ValueError("dff_blank must be provided when blank correction is enabled")

    _validate_array_2d(dff_blank)
    if dff_blank.shape != dff_cond.shape:
        raise ValueError(f"Blank shape {dff_blank.shape} != condition shape {dff_cond.shape}")

    if mode == "subtract_blank":
        return dff_cond - dff_blank

    if mode == "divide_blank":
        return dff_cond / (dff_blank + eps)

    raise ValueError(f"Unknown blank correction mode: {mode}")


def compute_dff(
    arr_cond: np.ndarray,
    baseline_frames: Tuple[int, int],
    mode: DFFMode = "none",
    arr_blank: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Full ΔF/F pipeline: baseline normalize, then optional blank correction.

    Args:
        arr_cond: condition array (n_pixels, n_frames)
        baseline_frames: inclusive (start, end)
        mode: blank correction mode
        arr_blank: blank array (n_pixels, n_frames), required if mode != "none"

    Returns:
        corrected ΔF/F for condition
    """
    dff_cond = baseline_normalize_dff(arr_cond, baseline_frames=baseline_frames, eps=eps)
    dff_blank = None if arr_blank is None else baseline_normalize_dff(arr_blank, baseline_frames, eps=eps)
    return apply_blank_correction(dff_cond, dff_blank=dff_blank, mode=mode, eps=eps)
