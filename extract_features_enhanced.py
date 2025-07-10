#!/usr/bin/env python3
# extract_features_enhanced.py

import argparse
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy
from sklearn.linear_model import LinearRegression

def safe_slope(x):
    s = pd.Series(x).interpolate(limit_direction="both")
    if s.isna().any():
        return np.nan
    idx = np.arange(len(s)).reshape(-1,1)
    try:
        lr = LinearRegression().fit(idx, s.values)
        return float(lr.coef_[0])
    except:
        return np.nan

def extract_window_features(window, fs):
    req = ["acc_g","SpO2","pulse","hr_bpm","ts"]
    if any(col not in window.columns for col in req):
        return None
    if window["acc_g"].count() < 2 or window["hr_bpm"].count() < 2:
        return None

    feats = {"window_start": window["ts"].iloc[0]}
    # time-domain & higher-order
    for col in ["acc_g","SpO2","pulse","hr_bpm"]:
        x = window[col].values
        feats[f"{col}_mean"]     = np.nanmean(x)
        feats[f"{col}_std"]      = np.nanstd(x)
        feats[f"{col}_min"]      = np.nanmin(x)
        feats[f"{col}_max"]      = np.nanmax(x)
        feats[f"{col}_range"]    = feats[f"{col}_max"] - feats[f"{col}_min"]
        feats[f"{col}_skew"]     = skew(x) if np.nanstd(x)>0 else 0.0
        feats[f"{col}_kurtosis"] = kurtosis(x) if np.nanstd(x)>0 else 0.0
        feats[f"{col}_median"]   = np.nanmedian(x)
        feats[f"{col}_p25"]      = np.nanpercentile(x,25)
        feats[f"{col}_p75"]      = np.nanpercentile(x,75)
        feats[f"{col}_slope"]    = safe_slope(x)

    # frequency-domain on acc_g
    acc_vals = window["acc_g"].values
    if len(np.unique(acc_vals)) > 1:
        f_ax, Pxx = welch(acc_vals, fs=fs, nperseg=min(len(acc_vals),256))
        feats["acc_spec_energy"] = np.nansum(Pxx)
        # band powers
        for lo, hi in [(0,1),(1,3),(3,5)]:
            mask = (f_ax>=lo)&(f_ax<hi)
            feats[f"acc_pow_{lo}_{hi}"] = np.nansum(Pxx[mask])
        feats["acc_spec_entropy"] = entropy(Pxx + 1e-8)
    else:
        feats["acc_spec_energy"]  = np.nan
        feats["acc_pow_0_1"]      = np.nan
        feats["acc_pow_1_3"]      = np.nan
        feats["acc_pow_3_5"]      = np.nan
        feats["acc_spec_entropy"] = np.nan

    # HRV from hr_bpm
    hr = window["hr_bpm"].values
    if len(hr)>1 and np.all(hr>0):
        rr    = 60.0 / hr
        diffs = np.diff(rr)
        feats["hrv_rmssd"] = np.sqrt(np.nanmean(diffs**2))
        feats["hrv_pnn50"] = np.sum(np.abs(diffs)>0.05) / len(diffs)
    else:
        feats["hrv_rmssd"] = np.nan
        feats["hrv_pnn50"] = np.nan

    return feats

def main():
    p = argparse.ArgumentParser(
        description="Extract enhanced features from synced watch data"
    )
    p.add_argument("-i","--input",  default="watch_data_sync.csv",
                   help="CSV with ts, acc_g, SpO2, pulse, hr_bpm")
    p.add_argument("-o","--output", default="watch_features_enhanced.csv",
                   help="Where to save enhanced features")
    p.add_argument("--fs",     type=float, default=10.0,
                   help="Sampling rate (Hz)")
    p.add_argument("--window", type=float, default=10.0,
                   help="Window length (s)")
    p.add_argument("--step",   type=float, default=5.0,
                   help="Step size (s)")
    args = p.parse_args()

    # 1) Load & index by datetime for time-based interpolate
    df = pd.read_csv(args.input)
    df["datetime"] = pd.to_datetime(df["ts"], unit="s")
    df = df.set_index("datetime").sort_index()

    # 2) Interpolate missing values in time, then back/forward-fill
    df = df.interpolate(method="time", limit_direction="both")
    df = df.bfill().ffill()

    # 3) Reset index, keep ts for windowing
    df = df.reset_index(drop=True)

    n = len(df)
    win_n  = int(args.window * args.fs)
    step_n = int(args.step   * args.fs)

    if n < win_n:
        print(f"ERROR: only {n} samples (<{win_n}) for one window")
        return

    features = []
    for start in range(0, n - win_n + 1, step_n):
        window = df.iloc[start : start + win_n]
        feats  = extract_window_features(window, args.fs)
        if feats is not None:
            features.append(feats)

    out = pd.DataFrame(features)
    out.to_csv(args.output, index=False)
    print(f"[+] Extracted {len(out)} windows → {args.output}")

if __name__=="__main__":
    main()
