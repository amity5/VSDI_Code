from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Literal

import numpy as np
import matplotlib.pyplot as plt


def plot_timecourse_overlays(
    t_ms: np.ndarray,
    Y: np.ndarray,
    labels: Sequence[str],
    *,
    title: str = "",
    stim_window_ms: Optional[Tuple[int, int]] = (0, 300),
    xlim_ms: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    linewidth: float = 1.5,
    alpha: float = 0.9,
    legend: bool = True,
    legend_fontsize: int = 8,
    figsize: Tuple[float, float] = (10.0, 4.0),
    outpath: Optional[Path] = None,
    dpi: int = 150,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot ROI timecourse overlays for a single contrast.

    Args:
        t_ms: (n_frames,)
        Y: (n_sessions, n_frames)
        labels: n_sessions labels (e.g., "YYYYMMDD_A")
        stim_window_ms: optional (start, end) to draw vertical lines
        outpath: if provided, save figure to this path

    Returns:
        (fig, ax)
    """
    if t_ms.ndim != 1:
        raise ValueError("t_ms must be 1D")
    if Y.ndim != 2:
        raise ValueError("Y must be 2D (n_sessions, n_frames)")
    if Y.shape[1] != t_ms.shape[0]:
        raise ValueError("Y.shape[1] must match len(t_ms)")
    if len(labels) != Y.shape[0]:
        raise ValueError("labels length must match Y rows")

    fig, ax = plt.subplots(figsize=figsize)

    for i, lab in enumerate(labels):
        ax.plot(t_ms, Y[i], linewidth=linewidth, alpha=alpha, label=str(lab))

    if stim_window_ms is not None:
        s0, s1 = stim_window_ms
        ax.axvline(s0, linewidth=1.0)
        ax.axvline(s1, linewidth=1.0)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("ΔF/F (ROI mean)")
    if title:
        ax.set_title(title)

    if xlim_ms is not None:
        ax.set_xlim(xlim_ms)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(True, linewidth=0.5, alpha=0.4)

    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=legend_fontsize, frameon=False)

    fig.tight_layout()

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")

    return fig, ax
