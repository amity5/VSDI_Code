from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from vsdi.config import VSDIConfig
from vsdi.paths import RunFolderManager
from vsdi.io import ExcelSessionIndex, MatReader

from vsdi.core.time_axis import build_time_axis_ms, ms_window_to_frame_slice
from vsdi.core.dff import compute_dff
from vsdi.core.roi_masks import mask_2d_to_flat_mask
from vsdi.core.roi_selection import select_seed, build_roi_mask_from_seed
from vsdi.core.features import roi_mean_timecourse, roi_pattern_vector


def _npz_timecourses_name(pct: int) -> str:
    return f"roi_timecourses_{pct}pct.npz"


def _npz_patterns_name(pct: int) -> str:
    return f"roi_patterns_{pct}pct.npz"


def _save_timecourses_npz(
    outdir: Path,
    pct: int,
    labels: List[str],
    t_ms: np.ndarray,
    Y: np.ndarray,
    n_trials: np.ndarray,
) -> None:
    np.savez_compressed(
        outdir / _npz_timecourses_name(pct),
        labels=np.asarray(labels, dtype=object),
        t_ms=t_ms,
        Y=Y,
        n_trials=n_trials,
    )


def _save_patterns_npz(
    outdir: Path,
    pct: int,
    labels: List[str],
    vecs: np.ndarray,
    n_trials: np.ndarray,
    image_shape: Tuple[int, int],
    flatten_order: str,
) -> None:
    np.savez_compressed(
        outdir / _npz_patterns_name(pct),
        labels=np.asarray(labels, dtype=object),
        vecs=vecs,
        n_trials=n_trials,
        image_shape=np.asarray(image_shape, dtype=int),
        flatten_order=np.asarray(flatten_order, dtype=object),
    )


def _infer_image_shape(n_pixels: int) -> Tuple[int, int]:
    """
    Best-effort inference for square images.
    You can replace this with explicit cfg.image_shape later if desired.
    """
    s = int(round(np.sqrt(n_pixels)))
    if s * s != n_pixels:
        raise ValueError(
            f"Cannot infer square image_shape from n_pixels={n_pixels}. "
            f"Provide explicit shape or extend loader to return it."
        )
    return (s, s)  # (H, W)


def run(cfg: VSDIConfig, *, label: str | None = None) -> Path:
    """
    Main overlay pipeline:
    - read sessions from Excel
    - load condition arrays from MAT
    - preprocess (ΔF/F + optional blank correction)
    - choose ROI (manual or auto-seed)
    - compute per-session ROI timecourses and spatial pattern vectors
    - write NPZs per contrast inside a run folder

    Returns:
        outdir
    """
    outdir = RunFolderManager(cfg.output_root).create_run_folder(cfg, label=label)

    sessions = ExcelSessionIndex(cfg.excel_path, sheet_name=cfg.sheet_name).build_sessions()
    if len(sessions) == 0:
        raise RuntimeError("No sessions found in Excel index")

    mat = MatReader(cfg.data_root)

    # Gather all contrasts present
    all_pcts = sorted({pct for s in sessions for pct in s.pct_to_condition.keys()})
    if len(all_pcts) == 0:
        raise RuntimeError("No contrast columns found in Excel (expected columns like '10%')")

    # Choose a contrast for auto-seed reference if requested
    seed_pct = cfg.auto_seed_contrast_pct if cfg.auto_seed_contrast_pct is not None else all_pcts[0]

    # Load first available session’s data for seed reference + shape inference
    ref_arr = None
    ref_shape = None
    ref_t_ms = None

    for s in sessions:
        if seed_pct in s.pct_to_condition:
            cond_key = s.pct_to_condition[seed_pct]
            arr = mat.load_condition(s.mat_path, cond_key)  # (n_pixels, n_frames)
            if arr.ndim != 2:
                raise ValueError(f"Loaded condition array must be 2D, got {arr.shape}")
            ref_shape = _infer_image_shape(arr.shape[0])
            ref_t_ms = build_time_axis_ms(arr.shape[1], cfg.frame_rate_hz, cfg.stim_onset_frame_1idx)
            # Use a mean map in configured window for seed selection reference
            sl = ms_window_to_frame_slice(ref_t_ms, cfg.auto_seed_time_ms)
            ref_map_flat = np.mean(arr[:, sl], axis=1)
            ref_arr = ref_map_flat.reshape(ref_shape, order="F")
            break

    if ref_arr is None or ref_shape is None or ref_t_ms is None:
        raise RuntimeError(f"Could not find any session containing seed_pct={seed_pct}")

    # Determine seed
    seed_xy = select_seed(
        ref_arr,
        manual_seed_xy=cfg.seed_xy,
        auto_seed=cfg.auto_seed,
        auto_seed_method=cfg.auto_seed_method,
        auto_seed_box_hw=cfg.auto_seed_box_hw,
        auto_seed_center_xy=cfg.seed_xy,
        roi_radius_px_for_roi_sum=cfg.radius_px,
    )

    # Build ROI mask
    roi_mask_2d = build_roi_mask_from_seed(
        ref_shape,
        seed_xy,
        cfg.roi_shape,
        radius_px=cfg.radius_px,
        box_half_width=cfg.auto_seed_box_hw[0],
        box_half_height=cfg.auto_seed_box_hw[1],
    )
    roi_flat = mask_2d_to_flat_mask(roi_mask_2d, order="F")

    # Save ROI/seed metadata in config.json already; additionally dump quick metadata
    meta = {
        "chosen_seed_xy": list(seed_xy),
        "image_shape": list(ref_shape),
        "roi_n_pixels": int(roi_flat.sum()),
        "flatten_order": "F",
        "contrasts": all_pcts,
    }
    with open(outdir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # For each contrast: compute timecourses + patterns for all sessions that have it
    for pct in all_pcts:
        labels: List[str] = []
        t_ms: np.ndarray | None = None
        Y_list: List[np.ndarray] = []
        vecs_list: List[np.ndarray] = []
        n_trials_list: List[int] = []

        for s in sessions:
            if pct not in s.pct_to_condition:
                continue

            cond_key = s.pct_to_condition[pct]
            arr = mat.load_condition(s.mat_path, cond_key)  # (n_pixels, n_frames)

            # Optionally load blank if present and configured
            arr_blank = None
            if cfg.dff_mode != "none":
                blank_key = s.pct_to_condition.get(0, None)  # common: 0% column used as blank
                if blank_key is not None:
                    arr_blank = mat.load_condition(s.mat_path, blank_key)

            arr_dff = compute_dff(
                arr,
                baseline_frames=cfg.baseline_frames,
                mode=cfg.dff_mode,
                arr_blank=arr_blank,
            )

            if t_ms is None:
                t_ms = build_time_axis_ms(arr_dff.shape[1], cfg.frame_rate_hz, cfg.stim_onset_frame_1idx)

            # Extract features
            tc = roi_mean_timecourse(arr_dff, roi_flat)
            vec = roi_pattern_vector(arr_dff, roi_flat, t_ms, cfg.time_window_ms, agg="mean")

            # Label format: date_letter
            label_s = f"{s.date_code}_{s.letter}"
            labels.append(label_s)
            Y_list.append(tc)
            vecs_list.append(vec)
            # If you have trial counts per condition, wire it here; for now 1 per session as placeholder
            n_trials_list.append(1)

        if t_ms is None or len(labels) == 0:
            continue

        Y = np.vstack(Y_list)
        vecs = np.vstack(vecs_list)
        n_trials = np.asarray(n_trials_list, dtype=int)

        _save_timecourses_npz(outdir, pct, labels, t_ms, Y, n_trials)
        _save_patterns_npz(outdir, pct, labels, vecs, n_trials, ref_shape, "F")

    return outdir


def main() -> None:
    p = argparse.ArgumentParser(description="Run VSDI overlay pipeline (timecourses + patterns).")

    p.add_argument("--excel", required=True, type=Path, help="Path to Used_Sessions_details.xlsx")
    p.add_argument("--data-root", required=True, type=Path, help="Root folder for MAT files")
    p.add_argument("--out-root", required=True, type=Path, help="Output root folder")
    p.add_argument("--sheet", default="Sheet1", help="Excel sheet name")

    p.add_argument("--frame-rate", type=float, default=100.0, help="Frame rate (Hz)")
    p.add_argument("--onset-1idx", type=int, default=27, help="Stimulus onset frame (1-based)")
    p.add_argument("--stim-ms", type=int, default=300, help="Stimulus duration (ms)")
    p.add_argument("--window-ms", type=int, nargs=2, default=(0, 500), help="Pattern/timecourse window in ms")
    p.add_argument("--baseline-frames", type=int, nargs=2, default=(25, 27), help="Baseline frames inclusive (0-based)")

    p.add_argument("--roi-shape", choices=["circle", "box"], default="circle")
    p.add_argument("--radius-px", type=int, default=10)

    p.add_argument("--seed", type=int, nargs=2, default=None, metavar=("X", "Y"), help="Manual seed xy (x y)")
    p.add_argument("--auto-seed", action="store_true")
    p.add_argument("--auto-seed-method", choices=["peak", "roi_sum"], default="peak")
    p.add_argument("--auto-seed-box-hw", type=int, nargs=2, default=(10, 10), metavar=("HW", "HH"))
    p.add_argument("--auto-seed-contrast", type=int, default=None, help="Contrast pct used for auto-seed reference")
    p.add_argument("--auto-seed-time-ms", type=int, nargs=2, default=(0, 300), help="Window used to build seed ref map")

    p.add_argument("--dff-mode", choices=["none", "divide_blank", "subtract_blank"], default="none")
    p.add_argument("--label", default=None, help="Optional label appended to run folder name")

    args = p.parse_args()

    cfg = VSDIConfig(
        excel_path=args.excel,
        data_root=args.data_root,
        output_root=args.out_root,
        sheet_name=args.sheet,
        frame_rate_hz=args.frame_rate,
        stim_onset_frame_1idx=args.onset_1idx,
        stim_duration_ms=args.stim_ms,
        time_window_ms=(int(args.window_ms[0]), int(args.window_ms[1])),
        baseline_frames=(int(args.baseline_frames[0]), int(args.baseline_frames[1])),
        dff_mode=args.dff_mode,
        roi_shape=args.roi_shape,
        radius_px=args.radius_px,
        seed_xy=(int(args.seed[0]), int(args.seed[1])) if args.seed is not None else None,
        auto_seed=args.auto_seed,
        auto_seed_method=args.auto_seed_method,
        auto_seed_box_hw=(int(args.auto_seed_box_hw[0]), int(args.auto_seed_box_hw[1])),
        auto_seed_contrast_pct=args.auto_seed_contrast,
        auto_seed_time_ms=(int(args.auto_seed_time_ms[0]), int(args.auto_seed_time_ms[1])),
    )

    outdir = run(cfg, label=args.label)
    print(str(outdir))


if __name__ == "__main__":
    main()
