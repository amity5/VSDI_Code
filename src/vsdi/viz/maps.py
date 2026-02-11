from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Literal

import numpy as np
import matplotlib.pyplot as plt


def apply_display_transform(
    img2d: np.ndarray,
    *,
    rotate_k: int = 0,
    flip_lr: bool = False,
) -> np.ndarray:
    """
    Apply display-only transforms.

    rotate_k: number of 90-degree rotations counter-clockwise.
    flip_lr: left-right flip after rotation.

    Important: use this ONLY for visualization. Core computations should
    operate on the native coordinate system.
    """
    if img2d.ndim != 2:
        raise ValueError("img2d must be 2D")

    out = np.rot90(img2d, k=int(rotate_k) % 4)
    if flip_lr:
        out = np.fliplr(out)
    return out


def plot_seed_and_roi_on_map(
    base_map: np.ndarray,
    roi_mask_2d: np.ndarray,
    seed_xy: Tuple[int, int],
    *,
    title: str = "",
    rotate_k: int = 0,
    flip_lr: bool = False,
    cmap: str = "viridis",
    alpha_roi: float = 0.35,
    marker_size: float = 60.0,
    figsize: Tuple[float, float] = (5.0, 5.0),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    outpath: Optional[Path] = None,
    dpi: int = 150,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a map with ROI overlay and seed marker.

    Args:
        base_map: (H, W) image for background
        roi_mask_2d: (H, W) boolean
        seed_xy: (x, y) in native coordinates (before display transforms)
    """
    if base_map.shape != roi_mask_2d.shape:
        raise ValueError("base_map and roi_mask_2d must have same shape")
    if base_map.ndim != 2:
        raise ValueError("base_map must be 2D")

    H, W = base_map.shape
    x0, y0 = seed_xy
    if not (0 <= x0 < W and 0 <= y0 < H):
        raise ValueError("seed_xy out of bounds")

    # Apply transforms to base_map and roi for display
    base_disp = apply_display_transform(base_map, rotate_k=rotate_k, flip_lr=flip_lr)
    roi_disp = apply_display_transform(roi_mask_2d.astype(float), rotate_k=rotate_k, flip_lr=flip_lr) > 0.5

    # We must also transform the seed coordinate for correct display.
    # Instead of duplicating tricky geometry, we compute it by transforming
    # a one-hot seed map.
    seed_map = np.zeros_like(base_map, dtype=float)
    seed_map[y0, x0] = 1.0
    seed_disp = apply_display_transform(seed_map, rotate_k=rotate_k, flip_lr=flip_lr)
    sy, sx = np.unravel_index(int(np.argmax(seed_disp)), seed_disp.shape)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(base_disp, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.imshow(np.ma.masked_where(~roi_disp, roi_disp), cmap="gray", alpha=alpha_roi)

    ax.scatter([sx], [sy], s=marker_size, marker="x")

    if title:
        ax.set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plot_day_map(
    day_map: np.ndarray,
    *,
    title: str = "",
    rotate_k: int = 0,
    flip_lr: bool = False,
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (5.0, 5.0),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    outpath: Optional[Path] = None,
    dpi: int = 150,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a single day activation map.

    day_map can be (H, W) or already flattened; this function expects 2D.
    """
    if day_map.ndim != 2:
        raise ValueError("day_map must be 2D (H, W)")

    disp = apply_display_transform(day_map, rotate_k=rotate_k, flip_lr=flip_lr)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(disp, cmap=cmap, vmin=vmin, vmax=vmax)
    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")

    return fig, ax
