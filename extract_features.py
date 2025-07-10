#!/usr/bin/env python3
# extract_features.py

import pandas as pd
import numpy as np
from scipy.signal import welch
import argparse

parser = argparse.ArgumentParser(
    description="Extract sliding-window features from synced watch data"
)
parser.add_argument(
    "--input", "-i",
    default="watch_data_sync.csv",
    help="Synchronized CSV with columns: ts, acc_g, SpO2, pulse, hr_bpm"
)
parser.add_argument(
    "--output", "-o",
    default="watch_features.csv",
    help="Output CSV for extracted features"
)
parser.add_argument(
    "--fs", type=float, default=10.0,
    help="Sampling rate in Hz (must match sync freq)"
)
parser.add_argument(
    "--window", type=float, default=10.0,
    help="Window length in seconds"
)
parser.add_argument(
    "--step", type=float, default=5.0,
    help="Step size in seconds (overlap = window–step)"
)
args = parser.parse_args()

# 1) Load synchronized data
df = pd.read_csv(args.input)
n_samples = len(df)
duration = n_samples / args.fs

# 2) Window parameters
win_n  = int(args.window * args.fs)
step_n = int(args.step   * args.fs)

print(f"[+] Loaded {n_samples} samples ({duration:.1f} s)")
print(f"[+] Window: {args.window}s → {win_n} samples")
print(f"[+] Step:   {args.step}s → {step_n} samples")

if n_samples < win_n:
    print(f"[!] ERROR: only {n_samples} samples (<{win_n}).\n"
          "    • Run logging longer or reduce --window duration.")
    exit(1)

# 3) Slide windows and extract features
features = []
for start in range(0, n_samples - win_n + 1, step_n):
    win = df.iloc[start : start + win_n]
    t0  = win["ts"].iloc[0]
    feats = {"window_start": t0}

    # Time-domain features for each channel
    for ch in ["acc_g", "SpO2", "pulse", "hr_bpm"]:
        x = win[ch].values
        feats[f"{ch}_mean"]  = x.mean()
        feats[f"{ch}_std"]   = x.std()
        feats[f"{ch}_min"]   = x.min()
        feats[f"{ch}_max"]   = x.max()
        feats[f"{ch}_range"] = x.max() - x.min()

    # Frequency-domain features on accelerometer
    f_axis, Pxx = welch(win["acc_g"].values, fs=args.fs, nperseg=min(win_n, 256))
    feats["acc_spec_energy"] = Pxx.sum()
    feats["acc_dom_freq"]    = float(f_axis[np.argmax(Pxx)])

    features.append(feats)

print(f"[+] Extracted {len(features)} windows of features")

# 4) Save to CSV
pd.DataFrame(features).to_csv(args.output, index=False)
print(f"[+] Features written to {args.output}")
