#!/usr/bin/env python3
# append_collect_and_label.py

import os, asyncio, time, csv
from bleak import BleakScanner, BleakClient

WATCH_ADDR = "F4:E2:45:58:19:36"
RAW_CSV    = "raw_data_labeled.csv"
LABELS_CSV = "labels.csv"

# Adjust these durations if you like
RECOMMENDED = {
    "rest":        420,   # 7 min
    "walk":        420,
    "run":         420,
    "high_motion": 300,   # 5 min
    "fall":        180    # 3 min (repeat falls back-to-back)
}
DEFAULT_DURATION = 300

ACC_UUID   = "0000fea1-0000-1000-8000-00805f9b34fb"
ACC_CTRLS  = ["0000fea2-0000-1000-8000-00805f9b34fb",
              "0000fec7-0000-1000-8000-00805f9b34fb"]
OTHER_UUIDS= ["00002a19-0000-1000-8000-00805f9b34fb",
              "0000fee1-0000-1000-8000-00805f9b34fb",
              "0000fee3-0000-1000-8000-00805f9b34fb",
              "00002a37-0000-1000-8000-00805f9b34fb"]

def ensure(path, header):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

def prompt():
    n = int(input("How many NEW sessions? "))
    labs = [input(f" Label #{i+1}: ") for i in range(n)]
    return labs

def make_cb(uuid):
    def cb(_, data):
        ts = time.time()
        with open(RAW_CSV, "a", newline="") as f:
            csv.writer(f).writerow([f"{ts:.6f}", uuid, data.hex(), make_cb.cur_label])
    return cb

async def record(labels):
    dev = await BleakScanner.find_device_by_address(WATCH_ADDR, timeout=10)
    if not dev: 
        print("Watch not found."); return
    async with BleakClient(dev) as c:
        # ACC
        cb = make_cb(ACC_UUID)
        await c.start_notify(ACC_UUID, cb)
        for u in ACC_CTRLS:
            await c.write_gatt_char(u, b"\x01")
            await asyncio.sleep(0.1)

        # other streams
        for u in OTHER_UUIDS:
            try: await c.start_notify(u, make_cb(u))
            except: pass

        for lab in labels:
            dur = RECOMMENDED.get(lab, DEFAULT_DURATION)
            input(f"Press ENTER to START '{lab}' ({dur}s)…")
            start = time.time()
            make_cb.cur_label = lab
            print(f"▶️  Recording {lab}")
            await asyncio.sleep(dur)
            end = time.time()
            make_cb.cur_label = "None"
            print(f"◼ Done {lab}\n")
            with open(LABELS_CSV, "a", newline="") as f:
                csv.writer(f).writerow([f"{start:.6f}", f"{end:.6f}", lab])

def main():
    ensure(RAW_CSV, ["timestamp","sensor_uuid","hex","label"])
    ensure(LABELS_CSV, ["start_ts","end_ts","label"])
    labs = prompt()
    try: asyncio.run(record(labs))
    except KeyboardInterrupt: print("Interrupted.")
    print("✅ Appended new data.")

if __name__=="__main__":
    main()
