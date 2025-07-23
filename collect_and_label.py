#!/usr/bin/env python3
import asyncio, time, csv
from bleak import BleakScanner, BleakClient

WATCH_ADDR = "F4:E2:45:58:19:36"

RAW_CSV    = "raw_data_labeled.csv"
LABELS_CSV = "labels.csv"

# Realistic recording durations (seconds)
RECOMMENDED = {
    "rest":        300,   # 5 min
    "walk":        300,   # 1 min
    "run":         300,   # 1 min
    "fall":        120,   # simulate a fall
    "high_motion": 180    # e.g. jumping
}
DEFAULT_DURATION = 180

# ACC characteristic
ACC_NOTIFY_UUID = "0000fea1-0000-1000-8000-00805f9b34fb"
ACC_CTRL_UUIDS  = [
    "0000fea2-0000-1000-8000-00805f9b34fb",
    "0000fec7-0000-1000-8000-00805f9b34fb"
]
CTRL_CMD = b"\x01"

# Other notify streams
OTHER_NOTIFY_UUIDS = [
    "00002a19-0000-1000-8000-00805f9b34fb",
    "0000fee1-0000-1000-8000-00805f9b34fb",
    "0000fee3-0000-1000-8000-00805f9b34fb",
    "00002a37-0000-1000-8000-00805f9b34fb"
]

current_label = "None"
events        = []

def prompt_labels():
    n = int(input("How many sessions will you record? "))
    labs = [input(f"  Label #{i+1}: ").strip() for i in range(n)]
    print(f"\n→ Sessions: {labs}\n")
    return labs

def init_csv():
    with open(RAW_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp","sensor_uuid","hex","label"])

def save_labels():
    with open(LABELS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_ts","end_ts","label"])
        for s,e,lab in events:
            w.writerow([f"{s:.6f}", f"{e:.6f}", lab])
    print(f"\n→ Labels saved to '{LABELS_CSV}'.")

def make_callback(uuid):
    def cb(_, data):
        ts = time.time()
        with open(RAW_CSV, "a", newline="") as f:
            csv.writer(f).writerow([f"{ts:.6f}", uuid, data.hex(), current_label])
    return cb

async def record_sessions(labels):
    print("🔍 Scanning for device…")
    dev = await BleakScanner.find_device_by_address(WATCH_ADDR, timeout=10.0)
    if not dev:
        print("❌ Device not found."); return

    async with BleakClient(dev) as client:
        print("✅ Connected\n")

        # 1) Subscribe ACC first
        await client.start_notify(ACC_NOTIFY_UUID, make_callback(ACC_NOTIFY_UUID))
        print(f"→ Subscribed ACC")

        # 2) Enable ACC controls
        for uuid in ACC_CTRL_UUIDS:
            await client.write_gatt_char(uuid, CTRL_CMD)
            print(f"→ Wrote 0x01 to {uuid}")
            await asyncio.sleep(0.1)

        # 3) Subscribe other streams
        for uuid in OTHER_NOTIFY_UUIDS:
            await client.start_notify(uuid, make_callback(uuid))
            print(f"→ Subscribed {uuid}")

        print("\n▶️  Streaming started.\n")

        global current_label
        for lab in labels:
            dur = RECOMMENDED.get(lab, DEFAULT_DURATION)
            input(f"Press ENTER to START '{lab}' ({dur}s)…")
            start_ts = time.time()
            current_label = lab

            print(f"● Recording '{lab}'…")
            await asyncio.sleep(dur)

            end_ts = time.time()
            current_label = "None"
            events.append((start_ts, end_ts, lab))
            print(f"◼ Finished '{lab}'.\n")

    save_labels()

def main():
    init_csv()
    labels = prompt_labels()
    try:
        asyncio.run(record_sessions(labels))
    except KeyboardInterrupt:
        print("\nInterrupted; saving labels…")
        save_labels()

if __name__ == "__main__":
    main()
