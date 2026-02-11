from __future__ import annotations

from typing import Optional, Tuple, Literal
import numpy as np

from .roi_masks import circle_mask_2d, box_mask_2d


AutoSeedMethod = Literal["peak", "roi_sum"]
ROIShape = Literal["circle", "box"]


def _validate_ref_2d(ref_2d: np.ndarray) -> None:
    if not isinstance(ref_2d, np.ndarray):
        raise TypeError("ref_2d must be a numpy array")
    if ref_2d.ndim != 2:
        raise ValueError(f"ref_2d must be 2D (H, W); got {ref_2d.shape}")


def _crop_box(
    ref_2d: np.ndarray,
    seed_xy: Tuple[int, int],
    half_width: int,
    half_height: int,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Crop ref_2d around seed in a box; returns cropped and top-left offset (x_off, y_off).
    """
    H, W = ref_2d.shape
    x0, y0 = seed_xy

    x1 = max(0, x0 - half_width)
    x2 = min(W, x0 + half_width + 1)
    y1 = max(0, y0 - half_height)
    y2 = min(H, y0 + half_height + 1)

    cropped = ref_2d[y1:y2, x1:x2]
    return cropped, (x1, y1)


def select_seed(
    ref_2d: np.ndarray,
    *,
    manual_seed_xy: Optional[Tuple[int, int]] = None,
    auto_seed: bool = False,
    auto_seed_method: AutoSeedMethod = "peak",
    auto_seed_box_hw: Tuple[int, int] = (10, 10),
    auto_seed_center_xy: Optional[Tuple[int, int]] = None,
    roi_radius_px_for_roi_sum: int = 10,
) -> Tuple[int, int]:
    """
    Choose a seed (x, y) in image coordinates.

    If auto_seed=False: requires manual_seed_xy.
    If auto_seed=True:
      - restrict search to a box around auto_seed_center_xy (or manual_seed_xy if provided),
        else defaults to image center.
      - 'peak': max pixel in cropped region
      - 'roi_sum': pick pixel maximizing sum inside circular ROI of radius roi_radius_px_for_roi_sum

    Args:
        ref_2d: reference image (H, W), typically mean map in a time window
    """
    _validate_ref_2d(ref_2d)
    H, W = ref_2d.shape

    if not auto_seed:
        if manual_seed_xy is None:
            raise ValueError("manual_seed_xy must be provided when auto_seed=False")
        x, y = manual_seed_xy
        if not (0 <= x < W and 0 <= y < H):
            raise ValueError(f"manual_seed_xy {manual_seed_xy} out of bounds for (H,W)=({H},{W})")
        return x, y

    # Determine search center
    if auto_seed_center_xy is not None:
        cx, cy = auto_seed_center_xy
    elif manual_seed_xy is not None:
        cx, cy = manual_seed_xy
    else:
        cx, cy = (W // 2, H // 2)

    hw, hh = auto_seed_box_hw
    cropped, (x_off, y_off) = _crop_box(ref_2d, (cx, cy), hw, hh)

    if cropped.size == 0:
        raise ValueError("Auto-seed crop produced empty region")

    if auto_seed_method == "peak":
        iy, ix = np.unravel_index(np.nanargmax(cropped), cropped.shape)
        return int(ix + x_off), int(iy + y_off)

    if auto_seed_method == "roi_sum":
        # Brute force inside cropped box: evaluate sum in circle around each candidate
        best_val = -np.inf
        best_xy = (cx, cy)
        Hc, Wc = cropped.shape
        for iy in range(Hc):
            for ix in range(Wc):
                x = ix + x_off
                y = iy + y_off
                mask = circle_mask_2d((H, W), (x, y), radius_px=roi_radius_px_for_roi_sum)
                val = float(np.nansum(ref_2d[mask]))
                if val > best_val:
                    best_val = val
                    best_xy = (x, y)
        return int(best_xy[0]), int(best_xy[1])

    raise ValueError(f"Unknown auto_seed_method: {auto_seed_method}")


def build_roi_mask_from_seed(
    image_shape: Tuple[int, int],
    seed_xy: Tuple[int, int],
    roi_shape: ROIShape,
    *,
    radius_px: int = 10,
    box_half_width: int = 10,
    box_half_height: int = 10,
) -> np.ndarray:
    """
    Build a 2D ROI mask (H, W) from seed + ROI parameters.
    """
    if roi_shape == "circle":
        return circle_mask_2d(image_shape, seed_xy, radius_px=radius_px)
    if roi_shape == "box":
        return box_mask_2d(image_shape, seed_xy, half_width=box_half_width, half_height=box_half_height)
    raise ValueError(f"Unknown roi_shape: {roi_shape}")
