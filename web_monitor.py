#!/usr/bin/env python3
"""
web_monitor.py

FastAPI app that streams BLE watch data, computes enriched features every 2s,
runs your stacking model, and pushes live SSE updates immediately.
"""

import asyncio
import time
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from collections import deque
from bleak import BleakScanner, BleakClient
from scipy.stats import skew, kurtosis, linregress
from scipy.signal import welch, find_peaks
from scipy.fft import rfft, rfftfreq

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

# ─── CONFIG ────────────────────────────────────────────────────────────────
WATCH_ADDR     = "F4:E2:45:58:19:36"

ACC_UUID       = "0000fea1-0000-1000-8000-00805f9b34fb"
ACC_CTRL_UUIDS = [
    "0000fea2-0000-1000-8000-00805f9b34fb",
    "0000fec7-0000-1000-8000-00805f9b34fb"
]

PPG_UUID       = "00002a37-0000-1000-8000-00805f9b34fb"
SPO2_UUID      = "0000fee3-0000-1000-8000-00805f9b34fb"

WINDOW_SEC     = 8    # smaller window for faster response
STEP_SEC       = 2    # compute every 2 seconds

warnings.filterwarnings("ignore", category=UserWarning)

# Load your trained model pipeline & label encoder
PIPE = joblib.load("best_pipeline.pkl")
LE   = joblib.load("label_encoder.pkl")

# Buffers to hold recent sensor readings
buffers = {
    ACC_UUID:  deque(),
    PPG_UUID:  deque(),
    SPO2_UUID: deque()
}

# Shared state for SSE
latest_event = {"timestamp": "", "label": "", "probs": {}}
event_queue  = asyncio.Queue()  # push new JSON strings here

# ─── BLE CALLBACKS & CONNECT ────────────────────────────────────────────────
def hex_to_value(uuid: str, h: str) -> float:
    b = bytes.fromhex(h)
    if uuid == ACC_UUID:
        x = int.from_bytes(b[0:2], "little", signed=True)
        y = int.from_bytes(b[2:4], "little", signed=True)
        return float((x*x + y*y)**0.5)
    if uuid == PPG_UUID:
        return float(b[1])
    if uuid == SPO2_UUID:
        return float(b[2])
    return np.nan

def make_cb(uuid: str):
    def callback(_, data):
        ts  = time.time()
        val = hex_to_value(uuid, data.hex())
        buf = buffers[uuid]
        buf.append((ts, val))
        cutoff = ts - WINDOW_SEC
        while buf and buf[0][0] < cutoff:
            buf.popleft()
    return callback

async def ble_connect():
    dev = await BleakScanner.find_device_by_address(WATCH_ADDR, timeout=10.0)
    if not dev:
        raise RuntimeError("Watch not found; ensure it’s advertising.")
    client = BleakClient(dev)
    await client.connect()

    # ACC notifications
    await client.start_notify(ACC_UUID, make_cb(ACC_UUID))
    for ctrl in ACC_CTRL_UUIDS:
        await client.write_gatt_char(ctrl, b"\x01")
        await asyncio.sleep(0.1)

    # PPG & SpO₂ notifications
    await client.start_notify(PPG_UUID,  make_cb(PPG_UUID))
    await client.start_notify(SPO2_UUID, make_cb(SPO2_UUID))

    print("✅ BLE connected and notifications started.")
    return client

# ─── FEATURE ENGINEERING ────────────────────────────────────────────────────
def step_freq(acc: np.ndarray, dt: float) -> float:
    if acc.size < 4:
        return 0.0
    yf = np.abs(rfft(acc - acc.mean()))
    xf = rfftfreq(len(acc), dt)
    return float(xf[np.argmax(yf)])

def zcr(acc: np.ndarray) -> float:
    if acc.size < 2:
        return 0.0
    return float(((acc[:-1] * acc[1:]) < 0).sum()) / acc.size

def compute_df() -> pd.DataFrame:
    now = time.time()

    # ACC features
    acc_list = [v for t,v in buffers[ACC_UUID] if now-WINDOW_SEC <= t <= now]
    acc      = np.array(acc_list, dtype=float)
    n        = acc.size
    mean     = acc.mean()  if n else 0.0
    std      = acc.std()   if n else 0.0
    mn       = acc.min()   if n else 0.0
    mx       = acc.max()   if n else 0.0
    med      = float(np.median(acc)) if n else 0.0
    skw      = float(skew(acc))       if n>2 else 0.0
    krt      = float(kurtosis(acc))   if n>3 else 0.0

    ts_acc = np.array([t for t,_ in buffers[ACC_UUID] if now-WINDOW_SEC <= t <= now])
    dt     = np.median(np.diff(ts_acc)) if ts_acc.size>1 else 0.1
    jerk   = np.diff(acc)/dt           if n>1 else np.array([0.0])
    jm     = float(np.max(np.abs(jerk)))
    peaks,_= find_peaks(np.abs(acc), height=0.5*np.max(np.abs(acc)) if n else 1)
    pc     = len(peaks)
    f,Pxx  = welch(acc, fs=1/dt, nperseg=min(256, n))
    bp     = float(np.trapz(Pxx[(f>=0.5)&(f<=3.0)], f[(f>=0.5)&(f<=3.0)]))

    # extra ACC features
    sf = step_freq(acc, dt)
    zr = zcr(acc)

    # PPG features
    ppg_list = [v for t,v in buffers[PPG_UUID] if now-WINDOW_SEC <= t <= now]
    ppg      = np.array(ppg_list, dtype=float)
    pm       = ppg.mean() if ppg.size else 0.0
    ps       = ppg.std()  if ppg.size else 0.0
    diffs    = np.diff(ppg) if ppg.size>1 else np.array([0.0])
    pr       = float(np.sqrt(np.mean(diffs**2)))
    pslope   = float(linregress(range(ppg.size), ppg).slope) if ppg.size>1 else 0.0

    # SpO₂ features
    spo_list = [v for t,v in buffers[SPO2_UUID] if now-WINDOW_SEC <= t <= now]
    spo      = np.array(spo_list, dtype=float)
    sm       = spo.mean() if spo.size else 0.0
    ss       = spo.std()  if spo.size else 0.0
    sroc     = (spo[-1]-spo[0])/WINDOW_SEC if spo.size>1 else 0.0

    # Combine into DataFrame
    vals = [
        n, mean, std, mn, mx, med, skw, krt,
        jm, pc, bp, sf, zr,
        pm, ps, pr, pslope,
        sm, ss, sroc
    ]
    cols = [
        "acc_count","acc_mean","acc_std","acc_min","acc_max","acc_med","acc_skew","acc_kurt",
        "acc_jerk_max","acc_peak_count","acc_bandpower","acc_step_freq","acc_zcr",
        "ppg_mean","ppg_std","ppg_rmssd","ppg_slope",
        "spo2_mean","spo2_std","spo2_roc"
    ]
    return pd.DataFrame([vals], columns=cols)

# ─── BACKGROUND MONITOR & SSE QUEUE ──────────────────────────────────────────
async def monitor_loop():
    client = await ble_connect()
    try:
        while True:
            await asyncio.sleep(STEP_SEC)
            df = compute_df()

            # ML prediction
            idx  = PIPE.predict(df)[0]
            lbl  = LE.inverse_transform([int(idx)])[0]

            # JSON-friendly probs
            proba       = PIPE.predict_proba(df)[0]
            class_idx   = PIPE.classes_.astype(int)
            class_names = LE.inverse_transform(class_idx)
            probs = {class_names[i]: float(proba[i])
                     for i in range(len(class_names))}

            # build event payload
            latest_event.update({
                "timestamp": time.strftime("%H:%M:%S"),
                "label":     lbl,
                "probs":     probs
            })
            await event_queue.put(json.dumps(latest_event))

    except asyncio.CancelledError:
        await client.disconnect()

# ─── FASTAPI APP & ENDPOINTS ─────────────────────────────────────────────────
app = FastAPI()

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(monitor_loop())

@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html><head><title>Live Monitor</title></head><body>
<h1>Real-Time Activity Monitor</h1>
<div id="lbl">Waiting…</div>
<ul id="pr"></ul>
<script>
 const es = new EventSource('/stream');
 es.onmessage = e => {
   const d = JSON.parse(e.data);
   document.getElementById('lbl').textContent = 
     `${d.timestamp} → ${d.label}`;
   const pr = document.getElementById('pr');
   pr.innerHTML = '';
   for(const [c,p] of Object.entries(d.probs)){
     const li = document.createElement('li');
     li.textContent = `${c}: ${(p*100).toFixed(1)}%`;
     pr.appendChild(li);
   }
 };
</script>
</body></html>"""

@app.get("/stream")
def stream():
    async def event_gen():
        while True:
            data = await event_queue.get()
            yield f"data: {data}\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")
