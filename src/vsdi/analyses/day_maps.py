"""This analysis assumes your pattern NPZ includes ROI vectors + metadata needed to reshape full-FOV maps if you saved them. Since many pipelines only save ROI vectors, this code supports both:

If NPZ contains fov_maps (n_sessions, H, W) or (n_sessions, n_pixels)

Otherwise it can still compute day-wise averages over ROI vectors (not full maps)"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PatternsBundle:
    """
    Container for pattern outputs loaded from roi_patterns_{pct}pct.npz.
    """
    pct: int
    labels: list[str]
    day: list[str]
    letter: list[str]
    vecs: np.ndarray  # (n_sessions, n_features)
    n_trials: Optional[np.ndarray] = None

    # Optional FOV data if present in NPZ
    fov: Optional[np.ndarray] = None          # (n_sessions, H, W) or (n_sessions, n_pixels)
    image_shape: Optional[Tuple[int, int]] = None
    flatten_order: Literal["C", "F"] = "F"


def _parse_day_letter(labels: list[str]) -> tuple[list[str], list[str]]:
    """
    Best-effort parser for session labels.

    Expected common formats:
    - "YYYYMMDD_A"
    - "2023-01-01_A"
    - "dayX_A"
    If parsing fails, day is full label and letter is empty.
    """
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


def load_patterns_npz(npz_path: Path, pct: int, *, flatten_order: Literal["C", "F"] = "F") -> PatternsBundle:
    """
    Load a patterns NPZ produced by the overlay pipeline.

    Expected keys (minimum):
      - labels: (n_sessions,)
      - vecs OR patterns: (n_sessions, n_features)

    Optional:
      - n_trials: (n_sessions,)
      - fov or fov_maps: (n_sessions, H, W) or (n_sessions, n_pixels)
      - image_shape: (2,)
      - flatten_order: "F"/"C"
    """
    data = np.load(npz_path, allow_pickle=True)

    labels = [str(x) for x in data["labels"].tolist()] if "labels" in data else []
    if not labels:
        raise ValueError(f"{npz_path} missing 'labels'")

    if "vecs" in data:
        vecs = np.asarray(data["vecs"])
    elif "patterns" in data:
        vecs = np.asarray(data["patterns"])
    else:
        raise ValueError(f"{npz_path} missing 'vecs' or_
