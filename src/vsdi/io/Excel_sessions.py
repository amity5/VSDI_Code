import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd


@dataclass
class SessionSpec:
    """
    Represents one experimental session (as defined by the Excel sheet).

    In your setup, the session's data lives in:
        <data_root>/<animal>_<date_code><letter>/condsAN.mat

    The Excel sheet defines which condition index corresponds to each contrast.
    """
    animal: str
    date_code: str               # e.g. "011221" (ddmmyy)
    letter: str                  # e.g. "a"
    pct_to_cond_index: Dict[int, int]  # e.g. {100: 3, 32: 4, 16: 5, 10: 6, 8: 7}

    # Optional: commonly present "special" conditions
    blank_true_idx: Optional[int] = None
    blank_choice_idx: Optional[int] = None
    error_idx: Optional[int] = None


class ExcelSessionIndex:
    """
    Parses Used_Sessions_details.xlsx.

    Your Excel layout is typically:
      - A header row somewhere containing "Date" and "Session"
      - "Date" column is filled only for the first row of each block (needs fill-down)
      - Condition columns are numbered 1..8 (or similar)
      - Cells contain labels like:
          "100% Contrast"
          0.32, 0.16, 0.10, 0.08 (fractions -> convert to percent)
          "True blank", "Blank w/ choice", "Error"
    """

    def __init__(self, excel_path: Path, sheet_name: str = "Sheet1", *, default_animal: str = "boromir"):
        self.excel_path = Path(excel_path)
        self.sheet_name = sheet_name
        self.default_animal = default_animal

    def build_sessions(self) -> List[SessionSpec]:
        header_row = self._find_header_row()
        df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name, header=header_row)

        # Normalize column names (strip whitespace)
        df = df.rename(columns={c: str(c).strip() for c in df.columns})

        date_col, session_col = self._find_date_and_session_cols(df.columns)

        # Drop rows without session letter
        df = df[df[session_col].notna()].copy()

        # Fill down date
        df[date_col] = df[date_col].ffill()

        # Identify condition index columns: "1", "2", ... OR int 1,2,... that became "1","2"
        cond_cols: List[Tuple[int, str]] = []
        for c in df.columns:
            cs = str(c).strip()
            if cs.isdigit():
                cond_cols.append((int(cs), c))
        cond_cols = sorted(cond_cols, key=lambda x: x[0])

        if not cond_cols:
            raise ValueError(
                "Could not find condition index columns (expected columns like 1..8). "
                f"Columns found: {list(df.columns)}"
            )

        sessions: List[SessionSpec] = []

        for _, row in df.iterrows():
            date_code = self._format_date_code(row[date_col])
            letter = str(row[session_col]).strip()

            pct_to_cond_index: Dict[int, int] = {}
            blank_true_idx: Optional[int] = None
            blank_choice_idx: Optional[int] = None
            error_idx: Optional[int] = None

            for cond_idx, col_name in cond_cols:
                cell = row[col_name]
                if pd.isna(cell):
                    continue

                # Track special conditions if present
                s = str(cell).strip()
                s_low = s.lower()

                if "blank" in s_low and "true" in s_low:
                    blank_true_idx = cond_idx
                if "blank" in s_low and ("choice" in s_low or "w/ choice" in s_low or "with choice" in s_low):
                    blank_choice_idx = cond_idx
                if "error" in s_low:
                    error_idx = cond_idx

                # Try to parse contrast percent from this cell
                pct = self._parse_contrast_pct(cell)
                if pct is not None:
                    pct_to_cond_index[pct] = cond_idx

            sessions.append(
                SessionSpec(
                    animal=self.default_animal,
                    date_code=date_code,
                    letter=letter,
                    pct_to_cond_index=pct_to_cond_index,
                    blank_true_idx=blank_true_idx,
                    blank_choice_idx=blank_choice_idx,
                    error_idx=error_idx,
                )
            )

        return sessions

    # -------------------------
    # Helpers
    # -------------------------

    def _find_header_row(self) -> int:
        """
        Find the row index that contains the header (must include 'Date' and 'Session').
        """
        preview = pd.read_excel(self.excel_path, sheet_name=self.sheet_name, header=None, nrows=80)
        for i in range(preview.shape[0]):
            row = preview.iloc[i].astype(str).str.strip().str.lower().tolist()
            if any(x == "date" for x in row) and any(x == "session" for x in row):
                return i
        # fallback: assume first row is header
        return 0

    @staticmethod
    def _find_date_and_session_cols(cols: Any) -> Tuple[str, str]:
        cols_norm = [str(c).strip() for c in cols]
        date_candidates = [c for c in cols_norm if c.lower() == "date"]
        session_candidates = [c for c in cols_norm if c.lower() == "session"]

        if not date_candidates or not session_candidates:
            raise ValueError(f"Could not find 'Date'/'Session' columns. Columns: {cols_norm}")

        return date_candidates[0], session_candidates[0]

    @staticmethod
    def _format_date_code(date_val: Any) -> str:
        """
        Convert Excel 'Date' into ddmmyy string (e.g., 01/12/2021 -> '011221').
        """
        dt = pd.to_datetime(date_val, dayfirst=True, errors="coerce")
        if pd.isna(dt):
            # If already like "011221", keep only digits
            s = str(date_val)
            digits = re.sub(r"\D", "", s)
            if len(digits) >= 6:
                return digits[-6:]
            raise ValueError(f"Could not parse date value: {date_val}")
        return dt.strftime("%d%m%y")

    @staticmethod
    def _parse_contrast_pct(cell: Any) -> Optional[int]:
        """
        Parse contrast percent from a cell value.

        Supports:
          - "100% Contrast" -> 100
          - numeric 0.32 -> 32 (fraction)
          - numeric 32 -> 32 (already percent)
          - strings like "0.32" -> 32
        Ignores:
          - "Error", blanks, non-numeric non-percent strings
        """
        # Numeric cell
        if isinstance(cell, (int, float)) and not pd.isna(cell):
            v = float(cell)
            if 0 < v <= 1:
                return int(round(v * 100))
            if 1 < v <= 1000:
                # assume already percent if plausible
                return int(round(v))
            return None

        s = str(cell).strip()
        if not s:
            return None

        s_low = s.lower()
        if "error" in s_low or "blank" in s_low:
            return None

        # Look for "xx%"
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
        if m:
            return int(round(float(m.group(1))))

        # Otherwise try parse float like "0.32" or "32"
        try:
            v = float(s)
        except ValueError:
            return None

        if 0 < v <= 1:
            return int(round(v * 100))
        if 1 < v <= 1000:
            return int(round(v))
        return None
