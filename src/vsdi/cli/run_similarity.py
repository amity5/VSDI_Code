from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

from vsdi.analyses.day_maps import load_patterns_npz
from vsdi.analyses.similarity import (
    build_patterns_dataframe,
    correlation_matrix,
    block_indices_by_day,
    pairwise_similarity_table,
)
from vsdi.viz.heatmaps import plot_day_blocked_similarity_heatmap


def _find_pattern_npzs(run_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in run_dir.glob("roi_patterns_*pct.npz"):
        m = re.search(r"roi_patterns_(\d+)pct\.npz$", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out, key=lambda x: x[0])


def main() -> None:
    p = argparse.ArgumentParser(description="Compute similarity matrices/tables from roi_patterns NPZs.")
    p.add_argument("--run-dir", required=True, type=Path, help="Run folder containing NPZ outputs")
    p.add_argument("--out-subdir", default="similarity", help="Subfolder inside run-dir for outputs")

    p.add_argument("--save-heatmaps", action="store_true", help="Save day-blocked heatmap PNGs")
    p.add_argument("--save-pairwise", action="store_true", help="Save pairwise similarity CSVs")

    args = p.parse_args()

    run_dir = args.run_dir
    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    npzs = _find_pattern_npzs(run_dir)
    if not npzs:
        raise RuntimeError("No roi_patterns_{pct}pct.npz files found in run-dir")

    for pct, npz_path in npzs:
        bundle = load_patterns_npz(npz_path, pct=pct)
        df = build_patterns_dataframe(pct=pct, labels=bundle.labels, vecs=bundle.vecs)

        # Sort by day then letter then session
        df = df.sort_values(["day", "letter", "session"]).reset_index(drop=True)

        A = np.vstack(df["vec"].to_list())
        R = correlation_matrix(A)

        labels = df["session"].tolist()
        day_blocks = block_indices_by_day(df["day"].tolist())

        # Save matrix as NPZ
        np.savez_compressed(out_dir / f"corr_{pct}pct.npz", labels=np.asarray(labels, dtype=object), R=R)

        if args.save_pairwise:
            pair_df = pairwise_similarity_table(df)
            pair_df.to_csv(out_dir / f"pairwise_{pct}pct.csv", index=False)

        if args.save_heatmaps:
            plot_day_blocked_similarity_heatmap(
                R,
                labels,
                day_blocks,
                title=f"Similarity — {pct}%",
                outpath=out_dir / f"heatmap_{pct}pct.png",
            )

    print(str(out_dir))


if __name__ == "__main__":
    main()
