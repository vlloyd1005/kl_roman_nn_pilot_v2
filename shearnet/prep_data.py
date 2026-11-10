#!/usr/bin/env python3
"""Convert the v3 parquet into fast-loading .npy arrays.

Writes to <outdir>:
  noisy.npy  (N, 6, 32, 32) float32   noisy stamps (F129, F158, roll0..3)
  clean.npy  (N, 6, 32, 32) float32   noiseless stamps
  var.npy    (N, 6, 32, 32) float32   variance maps
  table.parquet                       every scalar column (truths, pop.*, snr.*, ...)
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CHANNELS = ["bb_F129", "bb_F158", "gr_roll0", "gr_roll1", "gr_roll2", "gr_roll3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(a.parquet)
    cols = pf.schema_arrow.names
    img_cols = [c for c in cols if c.startswith("img.")]
    scal_cols = [c for c in cols if not c.startswith("img.")]
    pq.write_table(pq.read_table(a.parquet, columns=scal_cols), out / "table.parquet")
    n = pf.metadata.num_rows

    for suffix, name in [("", "noisy"), ("_clean", "clean"), ("_var", "var")]:
        arr = np.lib.format.open_memmap(out / f"{name}.npy", mode="w+",
                                        dtype=np.float32, shape=(n, 6, 32, 32))
        for k, ch in enumerate(CHANNELS):
            col = pq.read_table(a.parquet, columns=[f"img.{ch}{suffix}"]).column(0)
            flat = col.combine_chunks().flatten().to_numpy(zero_copy_only=False)
            arr[:, k] = flat.reshape(n, 32, 32).astype(np.float32)
        arr.flush()
        del arr
        print("wrote", name)


if __name__ == "__main__":
    main()
