from __future__ import annotations

from typing import Tuple, Literal
import numpy as np


def circle_mask_2d(
    image_shape: Tuple[int, int],
    seed_xy: Tuple[int, int],
    radius_px: int,
) -> np.ndarray:
    """
    Build a circular ROI mask in image coordinates.

    Conventions:
        - image_shape = (H, W)
        - seed_xy = (x, y) where x is column index, y is row index

    Returns:
        mask2d: shape (H, W) boolean
    """
    if radius_px <= 0:
        raise ValueError("radius_px must be > 0")

    H, W = image_shape
    x0, y0 = seed_xy
    if not (0 <= x0 < W and 0 <= y0 < H):
        raise ValueError(f"seed_xy {seed_xy} out of bounds for image_shape={image_shape}")

    yy, xx = np.ogrid[:H, :W]
    dist2 = (xx - x0) ** 2 + (yy - y0) ** 2
    return dist2 <= (radius_px ** 2)


def box_mask_2d(
    image_shape: Tuple[int, int],
    seed_xy: Tuple[int, int],
    half_width: int,
    half_height: int,
) -> np.ndarray:
    """
    Build a rectangular ROI mask centered at seed.

    Args:
        image_shape: (H, W)
        seed_xy: (x, y)
        half_width: half-width in pixels (x direction)
        half_height: half-height in pixels (y direction)

    Returns:
        mask2d: shape (H, W) boolean
    """
    if half_width < 0 or half_height < 0:
        raise ValueError("half_width/half_height must be >= 0")

    H, W = image_shape
    x0, y0 = seed_xy
    if not (0 <= x0 < W and 0 <= y0 < H):
        raise ValueError(f"seed_xy {seed_xy} out of bounds for image_shape={image_shape}")

    x1 = max(0, x0 - half_width)
    x2 = min(W, x0 + half_width + 1)
    y1 = max(0, y0 - half_height)
    y2 = min(H, y0 + half_height + 1)

    mask = np.zeros((H, W), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def mask_2d_to_flat_mask(mask2d: np.ndarray, order: Literal["C", "F"] = "F") -> np.ndarray:
    """
    Convert a (H, W) boolean mask to flat (H*W,) boolean mask.

    Note: use order="F" if your arrays are stored / reshaped Fortran-style.
    """
    if mask2d.ndim != 2:
        raise ValueError(f"mask2d must be 2D, got {mask2d.shape}")
    return mask2d.reshape(-1, order=order)
