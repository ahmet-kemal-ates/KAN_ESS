"""
build_dataset.py -- One-time preprocessing of raw ePLB RW*.mat files.

Run this script ONCE, pointing it at the raw dataset directory.
Outputs preprocessed  dataset/RW{9,10,11,12}.mat  files (~5-10 MB each)
that are committed to the repository so anyone can clone and train
without the original raw .mat files.

Saves as .mat for consistency with the ENNC workflow and so files
can be opened directly in MATLAB for inspection.

Usage
-----
    python build_dataset.py --raw "path/to/Matlab"

    # Or just run without args to use a file dialog:
    python build_dataset.py

What it does
------------
  1. Loads each raw RW*.mat via training.preprocess.load_rw_file()
  2. Decimates from 1 s to 10 s (keeps every 10th sample)
  3. Saves dataset/RW{N}.mat with variables: V, I, T, t, SoC, Q, Cn, Ts

Decimation rationale
--------------------
  Raw data: ~7.7 M samples per file at 1 s (~154 MB uncompressed)
  After x10 decimate: ~770 K samples (~5-10 MB per file)
  Battery dynamics at 10 s resolution are fully preserved for SoC
  estimation; the KAN sees ample variety in (V, I, Q, SoC) space.
"""

import os
import sys
import argparse
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training.preprocess import load_rw_file

DECIMATE = 10
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
RW_NAMES = ["RW9.mat", "RW10.mat", "RW11.mat", "RW12.mat"]


def build(raw_dir, decimate=DECIMATE):
    os.makedirs(OUT_DIR, exist_ok=True)

    for fname in RW_NAMES:
        mat_path = os.path.join(raw_dir, fname)
        if not os.path.isfile(mat_path):
            print(f"  [SKIP] {fname} not found in {raw_dir}")
            continue

        print(f"Processing {fname} ...")
        ds = load_rw_file(mat_path)
        N  = len(ds["V"])

        idx = np.arange(0, N, decimate)
        out = {
            "V":   ds["V"][idx].reshape(-1, 1).astype(np.float32),
            "I":   ds["I"][idx].reshape(-1, 1).astype(np.float32),
            "T":   ds["T"][idx].reshape(-1, 1).astype(np.float32),
            "t":   ds["t"][idx].reshape(-1, 1).astype(np.float32),
            "SoC": ds["SoC"][idx].reshape(-1, 1).astype(np.float32),
            "Q":   ds["Q"][idx].reshape(-1, 1).astype(np.float32),
            "Cn":  np.array([[float(ds["Cn"])]], dtype=np.float32),
            "Ts":  np.array([[float(ds["Ts"]) * decimate]], dtype=np.float32),
        }

        out_path = os.path.join(OUT_DIR, fname)
        sio.savemat(out_path, out, do_compression=True)

        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  {N:>9,} -> {len(idx):>7,} samples  |  {out_path}  ({size_mb:.1f} MB)")
        print(f"  SoC=[{out['SoC'].min():.3f},{out['SoC'].max():.3f}]  "
              f"V=[{out['V'].min():.3f},{out['V'].max():.3f}]  "
              f"Cn={out['Cn'].item():.3f} Ah")

    print(f"\nDone. Dataset directory: {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default=None,
                        help="Directory containing raw RW*.mat files")
    parser.add_argument("--decimate", type=int, default=DECIMATE,
                        help=f"Keep every N-th sample (default: {DECIMATE})")
    args = parser.parse_args()

    raw_dir = args.raw
    if raw_dir is None:
        try:
            import tkinter as tk
            from tkinter.filedialog import askdirectory
            root = tk.Tk(); root.withdraw()
            raw_dir = askdirectory(title="Select directory containing RW*.mat files")
            root.destroy()
        except Exception:
            pass

    if not raw_dir or not os.path.isdir(raw_dir):
        print("Usage: python build_dataset.py --raw <path/to/Matlab>")
        sys.exit(1)

    build(raw_dir, decimate=args.decimate)
