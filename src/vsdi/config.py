from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Optional, Literal, Dict, Any
import json
import hashlib


@dataclass(frozen=True)
class VSDIConfig:
    """
    Global configuration object for VSDI pipeline.

    All parameters must live here.
    No hard-coded literals elsewhere.

    This object can be:
    - Instantiated directly
    - Loaded from JSON via VSDIConfig.from_json()
    """

    # -----------------
    # Dataset / Paths
    # -----------------
    excel_path: Path
    data_root: Path
    output_root: Path

    sheet_name: str = "Sheet1"

    # -----------------
    # Timing
    # -----------------
    frame_rate_hz: float = 100.0
    stim_onset_frame_1idx: int = 27
    stim_duration_ms: int = 300

    # Window used for ROI pattern extraction or amplitude
    time_window_ms: Tuple[int, int] = (0, 500)

    # -----------------
    # Baseline / DFF
    # -----------------
    baseline_frames: Tuple[int, int] = (25, 27)
    dff_mode: Literal["none", "divide_blank", "subtract_blank"] = "none"

    # -----------------
    # ROI Selection
    # -----------------
    roi_shape: Literal["circle", "box"] = "circle"
    radius_px: int = 10

    seed_xy: Optional[Tuple[int, int]] = None

    auto_seed: bool = False
    auto_seed_method: Literal["peak", "roi_sum"] = "peak"
    auto_seed_box_hw: Tuple[int, int] = (10, 10)
    auto_seed_contrast_pct: Optional[int] = None
    auto_seed_time_ms: Tuple[int, int] = (0, 300)

    # -----------------
    # Display (plotting only)
    # -----------------
    display_rotate_k: int = 0
    display_flip_lr: bool = False

    # -----------------
    # Output
    # -----------------
    save_npz: bool = True
    save_figures: bool = True

    # ==========================================================
    # Serialization
    # ==========================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to a JSON-serializable dict.
        """
        d = asdict(self)

        # Convert Paths to strings
        d["excel_path"] = str(self.excel_path)
        d["data_root"] = str(self.data_root)
        d["output_root"] = str(self.output_root)

        return d

    def hash_tag(self) -> str:
        """
        Create short deterministic hash of config.
        Used to build reproducible run folder names.
        """
        config_json = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_json.encode()).hexdigest()[:8]

    # ==========================================================
    # JSON Loader
    # ==========================================================

    @staticmethod
    def from_json(path: Path) -> "VSDIConfig":
        """
        Load configuration from JSON file.

        Handles:
        - Path conversion
        - list -> tuple conversion
        - Optional tuple fields
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config JSON not found: {path}")

        with open(path, "r") as f:
            raw: Dict[str, Any] = json.load(f)

        # -----------------
        # Convert required Paths
        # -----------------
        raw["excel_path"] = Path(raw["excel_path"])
        raw["data_root"] = Path(raw["data_root"])
        raw["output_root"] = Path(raw["output_root"])

        # -----------------
        # Convert list fields -> tuples
        # -----------------
        tuple_fields = [
            "time_window_ms",
            "baseline_frames",
            "auto_seed_box_hw",
            "auto_seed_time_ms",
        ]

        for field in tuple_fields:
            if field in raw and raw[field] is not None:
                raw[field] = tuple(raw[field])

        # -----------------
        # Convert seed_xy
        # -----------------
        if "seed_xy" in raw and raw["seed_xy"] is not None:
            raw["seed_xy"] = tuple(raw["seed_xy"])

        return VSDIConfig(**raw)
