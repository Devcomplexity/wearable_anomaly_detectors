#!/usr/bin/env python3
"""
extract_features_enriched.py

- Reads raw_data_labeled.csv & labels.csv
- Slides WINDOW-sec windows every STEP-sec
- Extracts 15 enriched features per window, padding zeros when no data
- Writes features_enriched.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, linregress

# ── CONFIG ─────────────────────────────────────────────────────────────
RAW_CSV    = "raw_data_labeled.csv"
LABELS_CSV = "labels.csv"
OUT_CSV    = "features_enriched.csv"

WINDOW_SEC = 10    # window length in seconds
STEP_SEC   = 5     # window stride

# Sensor UUIDs
ACC_UUID  = "0000fea1-0000-1000-8000-00805f9b34fb"
PPG_UUID  = "00002a37-0000-1000-8000-00805f9b34fb"
SPO2_UUID = "0000fee3-0000-1000-8000-00805f9b34fb"
# ────────────────────────────────────────────────────────────────────────

def parse_val(uuid: str, h: str) -> float:
    """Convert a hex payload to a numeric value."""
    b = bytes.fromhex(h)
    if uuid == ACC_UUID:
        x = int.from_bytes(b[0:2], "little", signed=True)
        y = int.from_bytes(b[2:4], "little", signed=True)
        return float((x*x + y*y) ** 0.5)
    if uuid == PPG_UUID:
        return float(b[1])
    if uuid == SPO2_UUID:
        return float(b[2])
    return np.nan

# ── LOAD RAW & LABELS ────────────────────────────────────────────────────
raw    = pd.read_csv(RAW_CSV)
labels = pd.read_csv(LABELS_CSV)

# Append numeric values
raw["value"] = raw.apply(lambda r: parse_val(r["sensor_uuid"], r["hex"]), axis=1)

# ── SLIDING WINDOW FEATURE EXTRACTION ───────────────────────────────────
features = []
for _, lbl in labels.iterrows():
    start_ts, end_ts, label = lbl["start_ts"], lbl["end_ts"], lbl["label"]
    seg = raw[(raw.timestamp >= start_ts) & (raw.timestamp <= end_ts)]

    curr = start_ts
    while curr + WINDOW_SEC <= end_ts:
        win = seg[(seg.timestamp >= curr) & (seg.timestamp < curr + WINDOW_SEC)]
        if not win.empty:
            feat = {"label": label, "start_ts": curr, "end_ts": curr + WINDOW_SEC}

            # ACC features
            acc = win[win.sensor_uuid == ACC_UUID]["value"].to_numpy()
            feat["acc_count"] = acc.size
            feat["acc_mean"]  = acc.mean() if acc.size else 0.0
            feat["acc_std"]   = acc.std()  if acc.size else 0.0
            feat["acc_min"]   = acc.min()  if acc.size else 0.0
            feat["acc_max"]   = acc.max()  if acc.size else 0.0
            feat["acc_med"]   = float(np.median(acc)) if acc.size else 0.0
            feat["acc_skew"]  = float(skew(acc))  if acc.size > 2 else 0.0
            feat["acc_kurt"]  = float(kurtosis(acc)) if acc.size > 3 else 0.0

            # PPG features
            ppg = win[win.sensor_uuid == PPG_UUID]["value"].to_numpy()
            diffs = np.diff(ppg) if ppg.size > 1 else np.array([0.0])
            feat["ppg_mean"] = ppg.mean() if ppg.size else 0.0
            feat["ppg_std"]  = ppg.std()  if ppg.size else 0.0
            feat["ppg_rmssd"]= float(np.sqrt(np.mean(diffs**2)))
            feat["ppg_slope"]= float(linregress(np.arange(ppg.size), ppg).slope) if ppg.size > 1 else 0.0

            # SpO₂ features
            spo2 = win[win.sensor_uuid == SPO2_UUID]["value"].to_numpy()
            feat["spo2_mean"] = spo2.mean() if spo2.size else 0.0
            feat["spo2_std"]  = spo2.std()  if spo2.size else 0.0
            feat["spo2_roc"]  = (spo2[-1] - spo2[0]) / WINDOW_SEC if spo2.size > 1 else 0.0

            features.append(feat)

        curr += STEP_SEC

# ── SAVE TO CSV ─────────────────────────────────────────────────────────
df_feat = pd.DataFrame(features)
df_feat.to_csv(OUT_CSV, index=False)
print(f"✅ Saved {len(df_feat)} windows → '{OUT_CSV}'")
