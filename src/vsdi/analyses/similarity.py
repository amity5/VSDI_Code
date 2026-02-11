"""This provides the “patterns_df” table and correlation matrices. No plotting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Literal
import numpy as np
import pandas as pd


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


def build_patterns_dataframe(
    pct: int,
    labels: List[str],
    vecs: np.ndarray,
) -> pd.DataFrame:
    """
    Create a tidy patterns dataframe:
      contrast pct, session label, day, letter, vec (object dtype)
    """
    if vecs.ndim != 2:
        raise ValueError("vecs must be 2D (n_sessions, n_features)")
    if len(labels) != vecs.shape[0]:
        raise ValueError("labels length must match vecs rows")

    day, letter = _parse_day_letter(labels)

    return pd.DataFrame({
        "pct": int(pct),
        "session": labels,
        "day": day,
        "letter": letter,
        "vec": list(vecs),
    })


def correlation_matrix(A: np.ndarray) -> np.ndarray:
    """
    Pearson correlation matrix for rows of A.
    A: (n_samples, n_features)
    Returns: (n_samples, n_samples)
    """
    if A.ndim != 2:
        raise ValueError("A must be 2D")

    A = A.astype(float, copy=False)
    A = A - A.mean(axis=1, keepdims=True)

    denom = np.linalg.norm(A, axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    A = A / denom

    return A @ A.T


def block_indices_by_day(days: List[str]) -> List[Tuple[str, int, int]]:
    """
    Given a list of day labels in sorted order, return contiguous blocks:
    [(day, start_idx, end_idx_exclusive), ...]
    """
    if not days:
        return []

    blocks = []
    start = 0
    for i in range(1, len(days)):
        if days[i] != days[i - 1]:
            blocks.append((days[start], start, i))
            start = i
    blocks.append((days[start], start, len(days)))
    return blocks


def pairwise_similarity_table(
    patterns_df: pd.DataFrame,
    *,
    method: Literal["pearson"] = "pearson",
) -> pd.DataFrame:
    """
    Build a long-form table of pairwise similarities for one contrast.

    patterns_df must contain columns: session, day, letter, vec
    """
    if "vec" not in patterns_df.columns:
        raise ValueError("patterns_df must contain 'vec' column")

    labels = patterns_df["session"].tolist()
    vecs = np.vstack(patterns_df["vec"].to_list())

    if method != "pearson":
        raise ValueError(f"Unsupported method: {method}")

    R = correlation_matrix(vecs)

    rows = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            rows.append({
                "session_i": labels[i],
                "session_j": labels[j],
                "sim": float(R[i, j]),
                "day_i": patterns_df.loc[i, "day"],
                "day_j": patterns_df.loc[j, "day"],
                "letter_i": patterns_df.loc[i, "letter"],
                "letter_j": patterns_df.loc[j, "letter"],
            })

    return pd.DataFrame(rows)
