#!/usr/bin/env python3
"""
extract_features_final.py

Load raw_data_clean.csv, slide an 8 s window every 4 s,
compute 20 features per window, and write features_final.csv.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, linregress
from scipy.signal import welch, find_peaks
from scipy.fft import rfft, rfftfreq

# ─── CONFIG ────────────────────────────────────────────────────────────────
IN_CSV     = "raw_data_clean.csv"
OUT_CSV    = "features_final.csv"
WINDOW_SEC = 8.0
STEP_SEC   = 4.0

# UUIDs for sensor streams
ACC_UUID  = "0000fea1-0000-1000-8000-00805f9b34fb"
PPG_UUID  = "00002a37-0000-1000-8000-00805f9b34fb"
SPO2_UUID = "0000fee3-0000-1000-8000-00805f9b34fb"

def step_freq(arr, dt):
    if len(arr) < 4:
        return 0.0
    yf = np.abs(rfft(arr - arr.mean()))
    xf = rfftfreq(len(arr), dt)
    return float(xf[np.argmax(yf)])

def zcr(arr):
    if len(arr) < 2:
        return 0.0
    return float(((arr[:-1] * arr[1:]) < 0).sum()) / len(arr)

def extract_window_features(df_win):
    # Acc features
    acc = df_win[df_win.sensor_uuid == ACC_UUID].value.to_numpy()
    ts_acc = df_win[df_win.sensor_uuid == ACC_UUID].timestamp.to_numpy()
    n = len(acc)
    mean = acc.mean() if n else 0.0
    std  = acc.std()  if n else 0.0
    mn   = acc.min()  if n else 0.0
    mx   = acc.max()  if n else 0.0
    med  = float(np.median(acc)) if n else 0.0
    skw  = float(skew(acc)) if n>2 else 0.0
    krt  = float(kurtosis(acc)) if n>3 else 0.0

    dt_acc = np.median(np.diff(ts_acc)) if len(ts_acc)>1 else 0.1
    jerk = np.diff(acc)/dt_acc if n>1 else np.array([0.0])
    jm   = float(np.max(np.abs(jerk)))
    peaks,_ = find_peaks(np.abs(acc),
                        height=0.5*np.max(np.abs(acc)) if n else 1)
    pc   = len(peaks)
    f,Pxx = welch(acc, fs=1/dt_acc, nperseg=min(256,n))
    bp   = float(np.trapz(Pxx[(f>=0.5)&(f<=3.0)], f[(f>=0.5)&(f<=3.0)]))
    sf   = step_freq(acc, dt_acc)
    zr   = zcr(acc)

    # PPG features
    ppg = df_win[df_win.sensor_uuid == PPG_UUID].value.to_numpy()
    pm = ppg.mean() if len(ppg) else 0.0
    ps = ppg.std()  if len(ppg) else 0.0
    diffs = np.diff(ppg) if len(ppg)>1 else np.array([0.0])
    rmssd = float(np.sqrt(np.mean(diffs**2)))
    slope = float(linregress(range(len(ppg)), ppg).slope) if len(ppg)>1 else 0.0

    # SpO₂ features
    spo = df_win[df_win.sensor_uuid == SPO2_UUID].value.to_numpy()
    sm = spo.mean() if len(spo) else 0.0
    ss = spo.std()  if len(spo) else 0.0
    roc = (spo[-1]-spo[0])/WINDOW_SEC if len(spo)>1 else 0.0

    # Label
    label = df_win.label.iloc[0]

    return {
        "acc_count": n, "acc_mean": mean, "acc_std": std,
        "acc_min": mn, "acc_max": mx, "acc_med": med,
        "acc_skew": skw, "acc_kurt": krt,
        "acc_jerk_max": jm, "acc_peak_count": pc,
        "acc_bandpower": bp, "acc_step_freq": sf, "acc_zcr": zr,
        "ppg_mean": pm, "ppg_std": ps, "ppg_rmssd": rmssd, "ppg_slope": slope,
        "spo2_mean": sm, "spo2_std": ss, "spo2_roc": roc,
        "label": label
    }

def main():
    df = pd.read_csv(IN_CSV)
    rows = []
    for lbl in df.label.unique():
        sub = df[df.label == lbl]
        t0, t1 = sub.timestamp.min(), sub.timestamp.max()
        t = t0
        while t + WINDOW_SEC <= t1:
            win = sub[(sub.timestamp >= t) & (sub.timestamp < t+WINDOW_SEC)]
            if not win.empty:
                rows.append(extract_window_features(win))
            t += STEP_SEC

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"✅ Extracted {len(out)} windows → {OUT_CSV}")

if __name__ == "__main__":
    main()
