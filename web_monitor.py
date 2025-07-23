#!/usr/bin/env python3
"""
web_monitor.py

Minimal real-time monitor:
 - Connects to BLE watch (ACC, PPG/HR, SpO₂)
 - Buffers last 8s of data
 - Every 2s computes 20 features (13 ACC, 4 PPG, 3 SpO₂)
 - Runs your saved pipeline (best_pipeline.pkl)
 - Streams label + probabilities via SSE to a Bootstrap UI
"""

import os, asyncio, time, json
from collections import deque

import joblib, numpy as np, pandas as pd
from bleak import BleakScanner, BleakClient
from scipy.stats import skew, kurtosis, linregress
from scipy.signal import welch, find_peaks
from scipy.fft import rfft, rfftfreq
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

# ─── CONFIG ────────────────────────────────────────────────────────────────
WATCH_ADDR   = os.getenv("WATCH_ADDR", "F4:E2:45:58:19:36")

ACC_UUID     = "0000fea1-0000-1000-8000-00805f9b34fb"
ACC_CTL_UUIDS= [
    "0000fea2-0000-1000-8000-00805f9b34fb",
    "0000fec7-0000-1000-8000-00805f9b34fb"
]
PPG_UUID     = "00002a37-0000-1000-8000-00805f9b34fb"
SPO2_UUID    = "0000fee3-0000-1000-8000-00805f9b34fb"

WINDOW_SEC   = 8.0
STEP_SEC     = 2.0

# ─── LOAD MODEL & ENCODER ───────────────────────────────────────────────────
PIPE = joblib.load("best_pipeline.pkl")
LE   = joblib.load("label_encoder.pkl")

# ─── BUFFERS & EVENT QUEUE ─────────────────────────────────────────────────
buffers     = {ACC_UUID: deque(), PPG_UUID: deque(), SPO2_UUID: deque()}
event_queue = asyncio.Queue()

# ─── SIMPLE HTML DASHBOARD ──────────────────────────────────────────────────
html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Live Monitor</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
        rel="stylesheet"/>
  <style>
    body { background:#f8f9fa; }
    .status { font-size:2.5rem; font-weight:600; }
    .prob { margin-bottom:.5rem; }
  </style>
</head>
<body>
  <div class="container py-4">
    <div class="card mb-4">
      <div class="card-body text-center">
        <div id="time" class="text-secondary">--:--:--</div>
        <div id="status" class="status">Waiting for data…</div>
      </div>
    </div>
    <div id="probs" class="card">
      <div class="card-body"></div>
    </div>
  </div>
  <script>
    const es = new EventSource('/stream');
    es.onmessage = e => {
      const d = JSON.parse(e.data);
      document.getElementById('time').textContent   = d.timestamp;
      document.getElementById('status').textContent = d.label.toUpperCase();
      let html = '';
      for (const [cls,p] of Object.entries(d.probs)) {
        const pct = (p*100).toFixed(1);
        html += `<div class="prob d-flex justify-content-between">
                   <span>${cls}</span><span>${pct}%</span>
                 </div>`;
      }
      document.querySelector('#probs .card-body').innerHTML = html;
    };
  </script>
</body>
</html>
"""

# ─── BLE CALLBACKS ──────────────────────────────────────────────────────────
def hex_to_val(uuid, hx: str) -> float:
    b = bytes.fromhex(hx)
    if uuid == ACC_UUID:
        # signed 16-bit little-endian
        x = int.from_bytes(b[0:2], byteorder='little', signed=True)
        y = int.from_bytes(b[2:4], byteorder='little', signed=True)
        return float((x*x + y*y)**0.5)
    if uuid == PPG_UUID:
        # heart rate in byte[1]
        return float(b[1])
    if uuid == SPO2_UUID:
        # SpO₂ in byte[2]
        return float(b[2])
    return float('nan')

def make_cb(uuid):
    def callback(_, data: bytearray):
        ts  = time.time()
        val = hex_to_val(uuid, data.hex())
        buf = buffers[uuid]
        buf.append((ts, val))
        # purge older than WINDOW_SEC
        while buf and buf[0][0] < ts - WINDOW_SEC:
            buf.popleft()
    return callback

async def ble_connect():
    dev = await BleakScanner.find_device_by_address(WATCH_ADDR, timeout=10)
    if not dev:
        raise RuntimeError("Watch not found.")
    client = BleakClient(dev)
    await client.connect()

    # start ACC notifications
    await client.start_notify(ACC_UUID, make_cb(ACC_UUID))
    for ctl in ACC_CTL_UUIDS:
        await client.write_gatt_char(ctl, b'\x01')
        await asyncio.sleep(0.1)

    # start PPG & SpO₂
    await client.start_notify(PPG_UUID,  make_cb(PPG_UUID))
    await client.start_notify(SPO2_UUID, make_cb(SPO2_UUID))

    print("✅ BLE connected")
    return client

# ─── FEATURE ENGINEERING ────────────────────────────────────────────────────
def step_freq(arr, dt):
    if arr.size < 4:
        return 0.0
    yf = np.abs(rfft(arr - arr.mean()))
    xf = rfftfreq(arr.size, dt)
    return float(xf[np.argmax(yf)])

def zcr(arr):
    if arr.size < 2:
        return 0.0
    return float(((arr[:-1] * arr[1:]) < 0).sum()) / arr.size

def compute_features():
    now = time.time()

    # ACC features
    acc = np.array([v for t,v in buffers[ACC_UUID] if now-WINDOW_SEC <= t <= now])
    n = acc.size
    mean,std = (acc.mean(), acc.std()) if n else (0.0, 0.0)
    mn, mx, med = (acc.min(), acc.max(), float(np.median(acc))) if n else (0.0,0.0,0.0)
    sk = float(skew(acc))    if n>2 else 0.0
    kt = float(kurtosis(acc))if n>3 else 0.0

    ts_arr = np.array([t for t,_ in buffers[ACC_UUID] if now-WINDOW_SEC <= t <= now])
    dt = np.median(np.diff(ts_arr)) if ts_arr.size>1 else 0.1
    jr = np.diff(acc)/dt if n>1 else np.array([0.0])
    jm = float(np.max(np.abs(jr)))
    peaks,_ = find_peaks(np.abs(acc), height=0.5*np.max(np.abs(acc)) if n else 1)
    pc = len(peaks)
    if n>1:
        f,Pxx = welch(acc, fs=1/dt, nperseg=min(256,n))
        bp = float(np.trapz(Pxx[(f>=0.5)&(f<=3.0)], f[(f>=0.5)&(f<=3.0)]))
    else:
        bp = 0.0
    sf = step_freq(acc, dt)
    zr = zcr(acc)

    # PPG features
    ppg = np.array([v for t,v in buffers[PPG_UUID] if now-WINDOW_SEC <= t <= now])
    pm, ps = (ppg.mean(), ppg.std()) if ppg.size else (0.0, 0.0)
    diffs = np.diff(ppg) if ppg.size>1 else np.array([0.0])
    rmssd = float(np.sqrt(np.mean(diffs**2)))
    slope = float(linregress(range(ppg.size), ppg).slope) if ppg.size>1 else 0.0

    # SpO₂ features
    spo = np.array([v for t,v in buffers[SPO2_UUID] if now-WINDOW_SEC <= t <= now])
    sm, ss = (spo.mean(), spo.std()) if spo.size else (0.0, 0.0)
    roc = (spo[-1]-spo[0])/WINDOW_SEC if spo.size>1 else 0.0

    feats = [
        n, mean, std, mn, mx, med, sk, kt,
        jm, pc, bp, sf, zr,
        pm, ps, rmssd, slope,
        sm, ss, roc
    ]
    cols = [
        "acc_count","acc_mean","acc_std","acc_min","acc_max","acc_med",
        "acc_skew","acc_kurt","acc_jerk_max","acc_peak_count",
        "acc_bandpower","acc_step_freq","acc_zcr",
        "ppg_mean","ppg_std","ppg_rmssd","ppg_slope",
        "spo2_mean","spo2_std","spo2_roc"
    ]
    return pd.DataFrame([feats], columns=cols)

# ─── MONITOR LOOP ───────────────────────────────────────────────────────────
async def monitor_loop():
    client = await ble_connect()
    try:
        while True:
            await asyncio.sleep(STEP_SEC)
            df = compute_features()

            proba = PIPE.predict_proba(df)[0]
            idx   = int(np.argmax(proba))
            lbl   = LE.inverse_transform([idx])[0]
            classes = LE.inverse_transform(PIPE.classes_.astype(int))
            probs = {classes[i]: float(proba[i]) for i in range(len(classes))}

            payload = {
                "timestamp": time.strftime("%H:%M:%S"),
                "label":     lbl,
                "probs":     probs
            }
            await event_queue.put(json.dumps(payload))
    except asyncio.CancelledError:
        await client.disconnect()

# ─── FASTAPI APP ───────────────────────────────────────────────────────────
app = FastAPI()

@app.on_event("startup")
async def startup():
    asyncio.create_task(monitor_loop())

@app.get("/", response_class=HTMLResponse)
def index():
    return html

@app.get("/stream")
def stream():
    async def event_gen():
        while True:
            data = await event_queue.get()
            yield f"data: {data}\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")
