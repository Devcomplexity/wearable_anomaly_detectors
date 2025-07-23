#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, linregress
from scipy.signal import welch, find_peaks

RAW_CSV    = "raw_data_labeled.csv"
LABELS_CSV = "labels.csv"
OUT_CSV    = "features_final.csv"
WINDOW     = 10
STEP       = 5

ACC_UUID  = "0000fea1-0000-1000-8000-00805f9b34fb"
PPG_UUID  = "00002a37-0000-1000-8000-00805f9b34fb"
SPO2_UUID = "0000fee3-0000-1000-8000-00805f9b34fb"

def parse_val(u,h):
    b=bytes.fromhex(h)
    if u==ACC_UUID:
        x=int.from_bytes(b[:2],"little",signed=True)
        y=int.from_bytes(b[2:4],"little",signed=True)
        return float((x*x+y*y)**0.5)
    if u==PPG_UUID:  return float(b[1])
    if u==SPO2_UUID: return float(b[2])
    return np.nan

def bandpower(x, fs=50, fmin=0.5, fmax=3):
    if len(x)<2: return 0.0
    f,P = welch(x, fs=fs, nperseg=min(256,len(x)))
    mask = (f>=fmin)&(f<=fmax)
    return float(np.trapz(P[mask], f[mask]))

raw    = pd.read_csv(RAW_CSV)
labels = pd.read_csv(LABELS_CSV)
raw["value"] = raw.apply(lambda r: parse_val(r["sensor_uuid"], r["hex"]), axis=1)

rows = []
for _, lbl in labels.iterrows():
    s,e,lab = lbl.start_ts, lbl.end_ts, lbl.label
    seg = raw[(raw.timestamp>=s)&(raw.timestamp<=e)]
    t = s
    while t+WINDOW <= e:
        win = seg[(seg.timestamp>=t)&(seg.timestamp<t+WINDOW)]
        if not win.empty:
            feat = {"label":lab,"start_ts":t,"end_ts":t+WINDOW}
            acc = win[win.sensor_uuid==ACC_UUID].value.to_numpy()
            # basic ACC
            feat.update({
                "acc_count": len(acc),
                "acc_mean":  acc.mean()  if acc.size else 0.0,
                "acc_std":   acc.std()   if acc.size else 0.0,
                "acc_min":   acc.min()   if acc.size else 0.0,
                "acc_max":   acc.max()   if acc.size else 0.0,
                "acc_med":   float(np.median(acc)) if acc.size else 0.0,
                "acc_skew":  float(skew(acc))  if acc.size>2 else 0.0,
                "acc_kurt":  float(kurtosis(acc)) if acc.size>3 else 0.0
            })
            # jerk & peaks
            ts = win[win.sensor_uuid==ACC_UUID].timestamp.to_numpy()
            dt = np.median(np.diff(ts)) if ts.size>1 else 0.1
            jerk = np.diff(acc)/dt if acc.size>1 else np.array([0.0])
            feat["acc_jerk_max"]     = float(np.max(np.abs(jerk)))
            peaks,_ = find_peaks(np.abs(acc), height=0.5*np.max(np.abs(acc)) if acc.size else 1)
            feat["acc_peak_count"]   = len(peaks)
            feat["acc_bandpower"]    = bandpower(acc, fs=1/dt)
            # PPG features
            ppg = win[win.sensor_uuid==PPG_UUID].value.to_numpy()
            diffs = np.diff(ppg) if ppg.size>1 else np.array([0.0])
            feat.update({
                "ppg_mean":  ppg.mean()  if ppg.size else 0.0,
                "ppg_std":   ppg.std()   if ppg.size else 0.0,
                "ppg_rmssd": float(np.sqrt(np.mean(diffs**2))),
                "ppg_slope": float(linregress(range(ppg.size),ppg).slope) if ppg.size>1 else 0.0
            })
            # SpO₂ features
            spo = win[win.sensor_uuid==SPO2_UUID].value.to_numpy()
            roc = (spo[-1]-spo[0])/WINDOW if spo.size>1 else 0.0
            feat.update({
                "spo2_mean": spo.mean() if spo.size else 0.0,
                "spo2_std":  spo.std()  if spo.size else 0.0,
                "spo2_roc":  roc
            })
            rows.append(feat)
        t += STEP

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f"✅ Saved {len(rows)} windows → '{OUT_CSV}'")
