from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np

from vsdi.analyses.day_maps import load_patterns_npz, compute_day_mean_maps
from vsdi.viz.maps import plot_day_map


def _find_pattern_npzs(run_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in run_dir.glob("roi_patterns_*pct.npz"):
        m = re.search(r"roi_patterns_(\d+)pct\.npz$", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out, key=lambda x: x[0])


def main() -> None:
    p = argparse.ArgumentParser(description="Compute and plot day maps from roi_patterns NPZs.")
    p.add_argument("--run-dir", required=True, type=Path, help="Run folder containing NPZ outputs")
    p.add_argument("--out-subdir", default="daymaps", help="Subfolder inside run-dir for outputs")
    p.add_argument("--weighted", action="store_true", help="Weight day means by n_trials if present")

    p.add_argument("--rotate-k", type=int, default=0)
    p.add_argument("--flip-lr", action="store_true")

    args = p.parse_args()

    run_dir = args.run_dir
    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    npzs = _find_pattern_npzs(run_dir)
    if not npzs:
        raise RuntimeError("No roi_patterns_{pct}pct.npz files found in run-dir")

    for pct, npz_path in npzs:
        bundle = load_patterns_npz(npz_path, pct=pct)
        day_to_map = compute_day_mean_maps(bundle, weighted=args.weighted)

        # Plot if the map is 2D; otherwise skip plotting (ROI vector)
        for day, arr in day_to_map.items():
            if arr.ndim == 2:
                plot_day_map(
                    arr,
                    title=f"Day map — {pct}% — {day}",
                    rotate_k=args.rotate_k,
                    flip_lr=args.flip_lr,
                    outpath=out_dir / f"daymap_{pct}pct_{day}.png",
                )

    print(str(out_dir))


if __name__ == "__main__":
    main()
