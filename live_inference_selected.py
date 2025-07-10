#!/usr/bin/env python3
# live_inference_selected.py

import asyncio
import time
import struct
import os
from collections import deque, defaultdict

import numpy as np
import pandas as pd
import joblib
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy
from sklearn.linear_model import LinearRegression
from bleak import BleakScanner, BleakClient

# Desktop notification (fallback print)
try:
    from plyer import notification
    def send_notification(t,m):
        notification.notify(title=t, message=m, timeout=5)
except ImportError:
    def send_notification(t,m):
        print('\a', f"{t}: {m}")

# — CONFIGURATION —
WATCH_ADDR     = "F4:E2:45:58:19:36"
FS             = 10.0    # sampling Hz
WINDOW_SEC     = 30.0    # sliding window length
STEP_SEC       = 10.0    # inference step

# BLE UUIDs unchanged
PPG_UUID           = "00002a37-0000-1000-8000-00805f9b34fb"
SPO2_NOTIFY_UUID   = "0000fee3-0000-1000-8000-00805f9b34fb"
SPO2_CONTROL_UUIDS = ["0000fee2-0000-1000-8000-00805f9b34fb"]
SPO2_CMDS          = [bytes([0x01,0x01])]
ACC_NOTIFY_UUID    = "0000fea1-0000-1000-8000-00805f9b34fb"
ACC_CONTROL_UUIDS  = ["0000fea2-0000-1000-8000-00805f9b34fb"]
ACC_CMDS           = [bytes([0x01])]

# Load your 11 features + trained scaler/model
FEATURES = pd.read_csv("features_kept.csv", header=None)[0].tolist()
SCALER   = joblib.load("scaler_selected.joblib")
MODEL    = joblib.load("iso_forest_selected.joblib")
THRESH   = MODEL.offset_
print(f"[+] Anomaly threshold (offset_): {THRESH:.3f}")

# In‐memory buffer for raw notifications
raw_buffer = deque(maxlen=int((WINDOW_SEC + STEP_SEC)*FS*3))
counters   = defaultdict(int)

# — PARSERS — convert hex payload → numeric
def parse_acc(h):
    b=bytes.fromhex(h)
    return struct.unpack("<i",b)[0]/1000.0 if len(b)>=4 else None

def parse_spo2(h):
    b=bytes.fromhex(h)
    return (b[2],b[3]) if len(b)>=4 else (None,None)

def parse_ppg(h):
    b=bytes.fromhex(h)
    if len(b)<2: return None
    flags,hr=b[0],b[1]
    return hr, bool(flags&0x04)

PARSERS = {"ACC":parse_acc, "SpO2":parse_spo2, "PPG":parse_ppg}

# — SAFE SLOPE helper —
def safe_slope(x):
    s=pd.Series(x).interpolate(limit_direction="both")
    if s.isna().any(): return np.nan
    idx=np.arange(len(s)).reshape(-1,1)
    return float(LinearRegression().fit(idx,s.values).coef_[0])

# — CALLBACK when notif arrives —
def make_notify_cb(sensor):
    def _cb(_,data):
        ts=time.time(); raw=data.hex()
        raw_buffer.append((ts,sensor,raw))
        n=counters[sensor]+1; counters[sensor]=n
        if sensor=="ACC":
            print(f"[DATA] ACC#{n}: {parse_acc(raw):.3f} g")
        elif sensor=="PPG":
            out=parse_ppg(raw)
            if out: hr,ct=out; print(f"[DATA] PPG#{n}: {hr} bpm  contact={ct}")
        else:
            s,p=parse_spo2(raw)
            print(f"[DATA] SpO2#{n}: {s}% pulse={p} bpm")
    return _cb

# — AUTO‐DETECT ACC & SpO2 control commands —
async def detect_cmd(client, uuid_notify, ctrls, cmds, sensor):
    best=(None,None,-1)
    await client.start_notify(uuid_notify, make_notify_cb(sensor))
    for cu in ctrls:
        for cmd in cmds:
            raw_buffer.clear()
            await client.write_gatt_char(cu,cmd)
            await asyncio.sleep(2.0)
            cnt=sum(1 for _,s,_ in raw_buffer if s==sensor)
            print(f"  Tried {cu}+{cmd.hex()} → {cnt}")
            if cnt>best[2]: best=(cu,cmd,cnt)
    await client.stop_notify(uuid_notify)
    return best[0],best[1]

# — EXTRACT the 11 selected features from one window_df —
def extract_selected(win_df):
    df=win_df.rename(columns={"ACC":"acc_g","PPG":"hr_bpm"})
    x_acc,x_hr=df["acc_g"].values, df["hr_bpm"].values
    feat={
      "acc_g_mean":x_acc.mean(), "acc_g_std":x_acc.std(),
      "acc_g_skew":skew(x_acc) if x_acc.std()>0 else 0.0,
      "hr_bpm_mean":x_hr.mean(), "hr_bpm_std":x_hr.std(),
      "hr_bpm_skew":skew(x_hr) if x_hr.std()>0 else 0.0,
      "hr_bpm_kurtosis":kurtosis(x_hr) if x_hr.std()>0 else 0.0,
      "hr_bpm_slope":safe_slope(x_hr)
    }
    f,Pxx=welch(x_acc,fs=FS,nperseg=min(len(x_acc),256))
    feat["acc_spec_energy"],feat["acc_spec_entropy"]=Pxx.sum(),entropy(Pxx+1e-8)
    if len(x_hr)>1:
        rr=60.0/x_hr; diffs=np.diff(rr)
        feat["hrv_rmssd"]=np.sqrt(np.nanmean(diffs**2))
    else:
        feat["hrv_rmssd"]=np.nan
    return np.array([feat[f] for f in FEATURES]).reshape(1,-1)

# — LOG SCORES (creates scores_log.csv) —
LOG_PATH=os.path.join(os.path.dirname(__file__),"scores_log.csv")
def log_score(ts,score,is_anom):
    header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH,"a") as f:
        if header: f.write("timestamp,score,anomaly\n")
        f.write(f"{ts:.3f},{score:.5f},{int(is_anom)}\n")
    print(f"[LOG] {time.strftime('%H:%M:%S')} score={score:.3f} anom={is_anom}")

# — KEEP ACC & SpO₂ streams alive every WINDOW_SEC —
async def keep_streams(client, spo2_cu,spo2_cmd, acc_cu,acc_cmd):
    while True:
        await asyncio.sleep(WINDOW_SEC)
        await client.write_gatt_char(spo2_cu,spo2_cmd)
        await client.write_gatt_char(acc_cu, acc_cmd)

# — MAIN INFERENCE LOOP —  
async def inference_loop():
    await asyncio.sleep(WINDOW_SEC)
    print(f"\n[+] Detecting every {STEP_SEC}s over last {WINDOW_SEC}s\n")
    while True:
        # build DataFrame of raw notifications
        raws=pd.DataFrame(raw_buffer,columns=["ts","sensor","hex"])
        if raws.empty:
            await asyncio.sleep(STEP_SEC); continue

        # parse into numeric rows
        rows=[]
        for _,r in raws.iterrows():
            out=PARSERS[r.sensor](r.hex)
            if out is None: continue
            if r.sensor=="SpO2":
                s,p=out; rows.append({"dt":pd.to_datetime(r.ts,unit="s"),"SpO2":s,"pulse":p})
            else:
                val=out if r.sensor!="PPG" else out[0]
                rows.append({"dt":pd.to_datetime(r.ts,unit="s"), r.sensor:val})

        df=pd.DataFrame(rows).set_index("dt")
        # resample to uniform 10 Hz
        dfs=(df.sort_index()
               .resample(f"{int(1000/FS)}ms")
               .ffill().bfill())
        # slice last WINDOW_SEC via datetime index
        cutoff=pd.Timestamp.now() - pd.Timedelta(seconds=WINDOW_SEC)
        win=dfs[dfs.index >= cutoff]

        if len(win) < WINDOW_SEC*FS:
            print(f"[!] Only {len(win)} samples—waiting…")
            await asyncio.sleep(STEP_SEC); continue

        # extract, scale, score
        X=extract_selected(win)
        Xs=SCALER.transform(X)
        score=MODEL.decision_function(Xs)[0]
        is_anom=score < THRESH
        tag="🚨 ANOMALY" if is_anom else "OK"
        print(f"[{time.ctime()}] {tag} score={score:.3f}")

        # log + notify
        log_score(time.time(), score, is_anom)
        if is_anom:
            send_notification("Anomaly Detected", f"score={score:.3f}")

        await asyncio.sleep(STEP_SEC)

# — CONNECT & START —  
async def main():
    print(f"[+] Scanning for {WATCH_ADDR}…")
    dev=await BleakScanner.find_device_by_address(WATCH_ADDR,timeout=10.0)
    if not dev: 
        print("[!] Watch not found"); return

    async with BleakClient(dev) as client:
        print("[+] Connected; enabling notifications…")
        for uuid in (PPG_UUID, SPO2_NOTIFY_UUID, ACC_NOTIFY_UUID):
            ch=client.services.get_characteristic(uuid)
            for d in ch.descriptors:
                if d.uuid.endswith("2902"):
                    await client.write_gatt_descriptor(d.handle,bytes([1,0]))

        # detect SpO2 & ACC commands
        spo2_cu,spo2_cmd = await detect_cmd(
            client, SPO2_NOTIFY_UUID, SPO2_CONTROL_UUIDS, SPO2_CMDS, "SpO2"
        )
        acc_cu,acc_cmd   = await detect_cmd(
            client, ACC_NOTIFY_UUID,  ACC_CONTROL_UUIDS,  ACC_CMDS,  "ACC"
        )
        print("→ SPO₂:",spo2_cu,spo2_cmd.hex(), "→ ACC:",acc_cu,acc_cmd.hex())

        # subscribe & start
        await client.start_notify(PPG_UUID,     make_notify_cb("PPG"))
        await client.start_notify(SPO2_NOTIFY_UUID, make_notify_cb("SpO2"))
        await client.start_notify(ACC_NOTIFY_UUID,  make_notify_cb("ACC"))

        print("\n[+] Starting streams…")
        await client.write_gatt_char(spo2_cu,spo2_cmd); await asyncio.sleep(0.1)
        await client.write_gatt_char(acc_cu, acc_cmd); await asyncio.sleep(0.1)

        # keep streams alive
        asyncio.create_task(keep_streams(client, spo2_cu,spo2_cmd, acc_cu,acc_cmd))
        # begin inference
        await inference_loop()

if __name__=="__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[+] Exiting")
