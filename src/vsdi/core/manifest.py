from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Iterable, Tuple, Optional
import pandas as pd

from vsdi.io.excel_sessions import SessionSpec


def build_sessions_manifest(sessions: List[SessionSpec]) -> pd.DataFrame:
    """
    Build a tidy manifest with one row per session and contrast mappings.

    Columns:
      - date_code
      - letter
      - mat_path
      - contrasts (sorted list as string)
      - plus one column per contrast like 'pct_10' -> condition key

    This manifest is designed to be written as CSV by an IO/output layer.
    """
    rows = []
    all_pcts = sorted({pct for s in sessions for pct in s.pct_to_condition.keys()})

    for s in sessions:
        row = {
            "date_code": s.date_code,
            "letter": s.letter,
            "mat_path": str(s.mat_path),
            "contrasts": ",".join(str(p) for p in sorted(s.pct_to_condition.keys())),
        }
        for pct in all_pcts:
            row[f"pct_{pct}"] = s.pct_to_condition.get(pct, "")
        rows.append(row)

    return pd.DataFrame(rows)


def build_trial_count_table(
    session_labels: List[str],
    pct_to_ntrials: Dict[int, List[int]],
) -> pd.DataFrame:
    """
    Build a wide trial-count table:
      rows = sessions
      cols = pct_{X}
    """
    if any(len(v) != len(session_labels) for v in pct_to_ntrials.values()):
        raise ValueError("Each pct list must match session_labels length")

    data = {"session": session_labels}
    for pct in sorted(pct_to_ntrials.keys()):
        data[f"pct_{pct}"] = pct_to_ntrials[pct]
    return pd.DataFrame(data)
