from __future__ import annotations

from typing import Tuple
import numpy as np


def frames_to_ms(frame_idx: np.ndarray, frame_rate_hz: float) -> np.ndarray:
    """
    Convert frame indices to milliseconds.
    """
    if frame_rate_hz <= 0:
        raise ValueError("frame_rate_hz must be > 0")
    return (frame_idx / frame_rate_hz) * 1000.0


def build_time_axis_ms(
    n_frames: int,
    frame_rate_hz: float,
    stim_onset_frame_1idx: int,
) -> np.ndarray:
    """
    Build time axis in ms where 0 ms corresponds to stimulus onset.

    Args:
        n_frames: total frames
        frame_rate_hz: frames per second
        stim_onset_frame_1idx: onset frame in 1-based indexing

    Returns:
        t_ms: shape (n_frames,)
    """
    if n_frames <= 0:
        raise ValueError("n_frames must be > 0")
    onset_0idx = stim_onset_frame_1idx - 1
    if not (0 <= onset_0idx < n_frames):
        raise ValueError(f"stim_onset_frame_1idx={stim_onset_frame_1idx} out of bounds for n_frames={n_frames}")

    idx = np.arange(n_frames, dtype=float)
    t_ms = frames_to_ms(idx - onset_0idx, frame_rate_hz=frame_rate_hz)
    return t_ms


def ms_window_to_frame_slice(
    t_ms: np.ndarray,
    window_ms: Tuple[int, int],
) -> slice:
    """
    Convert a time window (ms) into a slice over frames.

    Inclusive bounds:
      start_ms <= t_ms <= end_ms

    Returns:
        slice(a, b) usable as arr[:, a:b]
    """
    if t_ms.ndim != 1:
        raise ValueError("t_ms must be 1D")
    start_ms, end_ms = window_ms
    if start_ms > end_ms:
        raise ValueError("window_ms start must be <= end")

    idx = np.where((t_ms >= start_ms) & (t_ms <= end_ms))[0]
    if idx.size == 0:
        raise ValueError(f"No frames found in window_ms={window_ms}")
    return slice(int(idx.min()), int(idx.max()) + 1)
