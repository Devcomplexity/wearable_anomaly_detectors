#!/usr/bin/env python3
# append_collect_and_label.py
#
# Prompts you for N new activity sessions, records each via BLE,
# and APPENDS both the raw hex readings and the start/end label to CSVs.

import os
import asyncio
import time
import csv
from bleak import BleakScanner, BleakClient

# ===== CONFIGURATION =====
WATCH_ADDR = "F4:E2:45:58:19:36"
RAW_CSV    = "raw_data_labeled.csv"
LABELS_CSV = "labels.csv"

# Recommended durations per activity (seconds)
RECOMMENDED = {
    "rest":        420,   # 7 minutes
    "walk":        300,
    "run":         300,
    "high_motion": 300,   # 5 minutes
    "fall":        180    # 3 minutes
}
DEFAULT_DURATION = 300     # fallback duration

# BLE characteristic UUIDs
ACC_UUID   = "0000fea1-0000-1000-8000-00805f9b34fb"
ACC_CTRLS  = [
    "0000fea2-0000-1000-8000-00805f9b34fb",
    "0000fec7-0000-1000-8000-00805f9b34fb"
]
OTHER_UUIDS = [
    "00002a19-0000-1000-8000-00805f9b34fb",
    "0000fee1-0000-1000-8000-00805f9b34fb",
    "0000fee3-0000-1000-8000-00805f9b34fb",
    "00002a37-0000-1000-8000-00805f9b34fb"
]


# ===== HELPERS =====
def ensure(path, header):
    """Create file with header row if it doesn't exist."""
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def prompt_labels():
    """Ask user how many sessions to record and their labels."""
    n = int(input("How many NEW sessions do you want to record? "))
    labels = []
    for i in range(n):
        lab = input(f" Label #{i+1} (e.g. run, fall, high_motion): ").strip()
        labels.append(lab)
    return labels


def make_cb(uuid):
    """
    Returns a callback that appends each incoming hex reading to RAW_CSV.
    It stamps each row with timestamp, UUID, hex data, and the current label.
    """
    def callback(_, data):
        ts = time.time()
        row = [f"{ts:.6f}", uuid, data.hex(), callback.cur_label]
        with open(RAW_CSV, "a", newline="") as f:
            csv.writer(f).writerow(row)
    callback.cur_label = "None"
    return callback


async def record_sessions(labels):
    """Connect to watch, then for each label record for its duration."""
    dev = await BleakScanner.find_device_by_address(WATCH_ADDR, timeout=10.0)
    if not dev:
        print("⚠️  Watch not found; ensure it's advertising.")
        return

    async with BleakClient(dev) as client:
        # Set up ACC notifications
        acc_cb = make_cb(ACC_UUID)
        await client.start_notify(ACC_UUID, acc_cb)
        for ctrl in ACC_CTRLS:
            await client.write_gatt_char(ctrl, b"\x01")
            await asyncio.sleep(0.1)

        # Set up the other sensors
        for uuid in OTHER_UUIDS:
            try:
                await client.start_notify(uuid, make_cb(uuid))
            except Exception:
                pass  # ignore any that fail

        # Loop through each requested label
        for lab in labels:
            duration = RECOMMENDED.get(lab, DEFAULT_DURATION)
            input(f"\nPress ENTER to START '{lab}' recording ({duration} seconds)…")
            start_ts = time.time()

            # Set the callback's current label
            acc_cb.cur_label = lab
            for cb_uuid in OTHER_UUIDS:
                # update label for other callbacks, too
                # they share the same callback function closure
                make_cb(cb_uuid).cur_label = lab

            print(f"▶️  Recording '{lab}'…")
            await asyncio.sleep(duration)
            end_ts = time.time()

            # Reset label to avoid stray writes
            acc_cb.cur_label = "None"
            for cb_uuid in OTHER_UUIDS:
                make_cb(cb_uuid).cur_label = "None"

            print(f"◼ Finished '{lab}'")

            # Append label interval
            with open(LABELS_CSV, "a", newline="") as f:
                csv.writer(f).writerow([f"{start_ts:.6f}", f"{end_ts:.6f}", lab])

    print("\n✅ Successfully appended new data.")


def main():
    # 1) Ensure CSVs exist w/ headers
    ensure(RAW_CSV,    ["timestamp","sensor_uuid","hex","label"])
    ensure(LABELS_CSV, ["start_ts","end_ts","label"])

    # 2) Prompt user
    labs = prompt_labels()

    # 3) Record & append
    try:
        asyncio.run(record_sessions(labs))
    except KeyboardInterrupt:
        print("\n🛑 Recording interrupted; partial data may have been saved.")


if __name__ == "__main__":
    main()
