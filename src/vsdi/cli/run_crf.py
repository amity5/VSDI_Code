from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

from vsdi.analyses.crf import load_timecourses_npz, compute_crf_amplitudes


def _find_timecourse_npzs(run_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in run_dir.glob("roi_timecourses_*pct.npz"):
        m = re.search(r"roi_timecourses_(\d+)pct\.npz$", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out, key=lambda x: x[0])


def main() -> None:
    p = argparse.ArgumentParser(description="Compute CRF amplitudes table from timecourse NPZs.")
    p.add_argument("--run-dir", required=True, type=Path, help="Run folder containing NPZ outputs")
    p.add_argument("--out-csv", default="crf_amplitudes.csv", help="Output CSV filename (inside run-dir)")
    p.add_argument("--window-ms", type=int, nargs=2, default=(0, 300), help="Window for amplitude extraction")
    p.add_argument("--agg", choices=["mean", "max"], default="mean")

    args = p.parse_args()

    run_dir = args.run_dir
    npzs = _find_timecourse_npzs(run_dir)
    if not npzs:
        raise RuntimeError("No roi_timecourses_{pct}pct.npz files found in run-dir")

    bundles = [load_timecourses_npz(path, pct=pct) for pct, path in npzs]
    df = compute_crf_amplitudes(bundles, window_ms=(args.window_ms[0], args.window_ms[1]), agg=args.agg)

    out_path = run_dir / args.out_csv
    df.to_csv(out_path, index=False)

    print(str(out_path))


if __name__ == "__main__":
    main()
