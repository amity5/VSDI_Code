from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


def plot_similarity_heatmap(
    R: np.ndarray,
    labels: List[str],
    *,
    title: str = "",
    figsize: Tuple[float, float] = (6.0, 6.0),
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "coolwarm",
    tick_fontsize: int = 7,
    outpath: Optional[Path] = None,
    dpi: int = 150,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a correlation/similarity matrix heatmap.
    """
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be square (n, n)")
    if len(labels) != R.shape[0]:
        raise ValueError("labels length must match R size")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(R, vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=tick_fontsize, rotation=90)
    ax.set_yticklabels(labels, fontsize=tick_fontsize)

    if title:
        ax.set_title(title)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plot_day_blocked_similarity_heatmap(
    R: np.ndarray,
    labels: List[str],
    day_blocks: List[tuple[str, int, int]],
    *,
    title: str = "",
    figsize: Tuple[float, float] = (6.5, 6.5),
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "coolwarm",
    tick_fontsize: int = 7,
    show_block_lines: bool = True,
    outpath: Optional[Path] = None,
    dpi: int = 150,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Heatmap with day-block separators.

    day_blocks: list of (day, start_idx, end_idx_exclusive)
    """
    fig, ax = plot_similarity_heatmap(
        R,
        labels,
        title=title,
        figsize=figsize,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        tick_fontsize=tick_fontsize,
        outpath=None,
    )

    if show_block_lines:
        # Draw separators between blocks
        for _, start, end in day_blocks:
            # horizontal line at end
            ax.axhline(end - 0.5, linewidth=1.0)
            ax.axvline(end - 0.5, linewidth=1.0)

    fig.tight_layout()

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")

    return fig, ax
