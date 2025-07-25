import pandas as pd
import numpy as np
from scipy.stats import linregress

RAW_CSV    = "raw_data_labeled.csv"
OUT_CSV    = "features_sliding.csv"
ACC_UUID   = "0000fea1-0000-1000-8000-00805f9b34fb"
PPG_UUID   = "00002a37-0000-1000-8000-00805f9b34fb"
SPO2_UUID  = "0000fee3-0000-1000-8000-00805f9b34fb"

WINDOW_SEC = 10   # 10 s windows
STEP_SEC   = 5    # slide by 5 s

def parse_hex(uuid, h):
    b = bytes.fromhex(h)
    if uuid == ACC_UUID:
        x = int.from_bytes(b[:2], "little", signed=True)
        y = int.from_bytes(b[2:4], "little", signed=True)
        return float((x*x + y*y)**0.5)
    if uuid == PPG_UUID:
        return float(b[1])
    if uuid == SPO2_UUID:
        return float(b[2])
    return np.nan

# load & prep
raw = pd.read_csv(RAW_CSV)
raw["ts"]   = pd.to_datetime(raw.timestamp, unit="s")
raw          = raw.set_index("ts").sort_index()
raw["label"] = raw.label.ffill()
raw          = raw[raw.label!="None"]
raw["value"] = raw.apply(lambda r: parse_hex(r["sensor_uuid"], r["hex"]), axis=1)

# sliding windows
rows = []
start = raw.index.min()
end   = raw.index.max()
curr  = start

while curr + pd.Timedelta(seconds=WINDOW_SEC) <= end:
    win = raw[curr: curr+pd.Timedelta(seconds=WINDOW_SEC)]
    if not win.empty:
        feats = {"start_ts": curr.timestamp(),
                 "end_ts": (curr+pd.Timedelta(seconds=WINDOW_SEC)).timestamp(),
                 "label": win.label.mode()[0]}
        # ACC
        acc = win[win.sensor_uuid==ACC_UUID].value
        feats.update({
            "acc_count": acc.count(),
            "acc_mean":  acc.mean() if not acc.empty else np.nan,
            "acc_std":   acc.std()  if not acc.empty else np.nan
        })
        # PPG
        ppg = win[win.sensor_uuid==PPG_UUID].value
        feats.update({
            "ppg_mean": ppg.mean() if not ppg.empty else np.nan,
            "ppg_std":  ppg.std()  if not ppg.empty else np.nan,
        })
        if len(ppg)>=2:
            slope,_,_,_,_ = linregress(np.arange(len(ppg)), ppg.values)
            feats["ppg_slope"] = slope
        else:
            feats["ppg_slope"] = np.nan
        # SpO₂
        spo = win[win.sensor_uuid==SPO2_UUID].value
        feats.update({
            "spo2_mean": spo.mean() if not spo.empty else np.nan,
            "spo2_std":  spo.std()  if not spo.empty else np.nan
        })
        rows.append(feats)
    curr += pd.Timedelta(seconds=STEP_SEC)

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f"Extracted {len(rows)} windows → {OUT_CSV}")
