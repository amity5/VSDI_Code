"""This provides a robust “CRF amplitude table” (contrast → amplitude per session).
 Actual fitting (Naka–Rushton) can be added later"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Literal, Dict

import numpy as np
import pandas as pd

from vsdi.core.time_axis import ms_window_to_frame_slice


@dataclass(frozen=True)
class TimecoursesBundle:
    pct: int
    labels: list[str]
    t_ms: np.ndarray               # (n_frames,)
    Y: np.ndarray                  # (n_sessions, n_frames)
    n_trials: Optional[np.ndarray] = None


def _parse_day_letter(labels: list[str]) -> tuple[list[str], list[str]]:
    days = []
    letters = []
    for lab in labels:
        s = str(lab)
        if "_" in s:
            d, l = s.split("_", 1)
            days.append(d)
            letters.append(l)
        else:
            days.append(s)
            letters.append("")
    return days, letters


def load_timecourses_npz(npz_path: Path, pct: int) -> TimecoursesBundle:
    """
    Load roi_timecourses_{pct}pct.npz.

    Expected keys:
      - labels: (n_sessions,)
      - t_ms: (n_frames,)
      - Y: (n_sessions, n_frames)

    Optional:
      - n_trials: (n_sessions,)
    """
    data = np.load(npz_path, allow_pickle=True)

    labels = [str(x) for x in data["labels"].tolist()] if "labels" in data else []
    if not labels:
        raise ValueError(f"{npz_path} missing 'labels'")

    t_ms = np.asarray(data["t_ms"])
    Y = np.asarray(data["Y"])
    n_trials = np.asarray(data["n_trials"]) if "n_trials" in data else None

    if t_ms.ndim != 1:
        raise ValueError("t_ms must be 1D")
    if Y.ndim != 2:
        raise ValueError("Y must be 2D (n_sessions, n_frames)")
    if Y.shape[1] != t_ms.shape[0]:
        raise ValueError("Y.shape[1] must match len(t_ms)")

    return TimecoursesBundle(pct=pct, labels=labels, t_ms=t_ms, Y=Y, n_trials=n_trials)


def compute_crf_amplitudes(
    bundles: list[TimecoursesBundle],
    *,
    window_ms: Tuple[int, int],
    agg: Literal["mean", "max"] = "mean",
) -> pd.DataFrame:
    """
    Build a tidy table of amplitudes per (session, contrast).

    Returns columns:
      session, day, letter, pct, amplitude, n_trials (if available)
    """
    rows = []

    for b in bundles:
        sl = ms_window_to_frame_slice(b.t_ms, window_ms)
        seg = b.Y[:, sl]  # (n_sessions, n_window_frames)

        if agg == "mean":
            amp = seg.mean(axis=1)
        elif agg == "max":
            amp = seg.max(axis=1)
        else:
            raise ValueError(f"Unknown agg={agg}")

        day, letter = _parse_day_letter(b.labels)

        for i, lab in enumerate(b.labels):
            rows.append({
                "session": lab,
                "day": day[i],
                "letter": letter[i],
                "pct": int(b.pct),
                "amplitude": float(amp[i]),
                "n_trials": (int(b.n_trials[i]) if b.n_trials is not None else None),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["pct", "day", "letter", "session"]).reset_index(drop=True)
    return df
