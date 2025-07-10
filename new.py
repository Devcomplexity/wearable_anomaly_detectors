#!/usr/bin/env python3
# new.py

import asyncio
import csv
import time
import argparse
from bleak import BleakScanner, BleakClient, BleakError

# ————— Argument Parsing —————
parser = argparse.ArgumentParser(
    description="BLE service lister & raw‐data logger"
)
parser.add_argument("-a","--address",
                    default="F4:E2:45:58:19:36",
                    help="Watch BLE MAC address")
parser.add_argument("--ppg",
                    default="00002a37-0000-1000-8000-00805f9b34fb",
                    help="PPG notify UUID")
parser.add_argument("--spo2",
                    default="0000fee3-0000-1000-8000-00805f9b34fb",
                    help="SpO₂ notify UUID")
parser.add_argument("--acc",
                    default="0000fea1-0000-1000-8000-00805f9b34fb",
                    help="ACC notify UUID")
parser.add_argument("-o","--csv",
                    default="watch_data.csv",
                    help="Output CSV file")
parser.add_argument("-l","--list", action="store_true",
                    help="Only list GATT services & characteristics")
args = parser.parse_args()

TARGET_ADDRESS = args.address
CHAR_UUID_PPG  = args.ppg
CHAR_UUID_SPO2 = args.spo2
CHAR_UUID_ACC  = args.acc
CSV_FILE       = args.csv

# ————— List GATT Services —————
async def list_services():
    print(f"[+] Scanning for {TARGET_ADDRESS}…")
    device = await BleakScanner.find_device_by_address(
        TARGET_ADDRESS, timeout=8.0
    )
    if not device:
        print(f"[!] {TARGET_ADDRESS} not found. Enable Discoverable mode.")
        return

    print(f"[+] Found {device.name!r} @ {device.address}")
    try:
        async with BleakClient(device) as client:
            print("[+] Available services & characteristics:")
            for svc in client.services:
                print(f"\n Service {svc.uuid}:")
                for ch in svc.characteristics:
                    props = ",".join(ch.properties)
                    print(f"   {ch.uuid} — {props}")
    except BleakError as e:
        print(f"[!] Connection error: {e}")

# ————— Log Notifications —————
async def log_data():
    print(f"[+] Looking up {TARGET_ADDRESS}…")
    device = await BleakScanner.find_device_by_address(
        TARGET_ADDRESS, timeout=8.0
    )
    if not device:
        print(f"[!] {TARGET_ADDRESS} not found. Enable Discoverable mode.")
        return

    try:
        async with BleakClient(device) as client:
            print(f"[+] Connected to {device.name!r} @ {device.address}")
            with open(CSV_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp","sensor","raw_hex"])
                f.flush()

                def make_cb(sensor):
                    def cb(_, data: bytearray):
                        writer.writerow([time.time(), sensor, data.hex()])
                        f.flush()
                    return cb

                await client.start_notify(CHAR_UUID_PPG,  make_cb("PPG"))
                await client.start_notify(CHAR_UUID_SPO2, make_cb("SpO2"))
                await client.start_notify(CHAR_UUID_ACC,  make_cb("ACC"))

                print("[+] Logging… press Ctrl+C to stop.")
                try:
                    while True:
                        await asyncio.sleep(1)
                except (KeyboardInterrupt, asyncio.CancelledError):
                    print("\n[+] Stopping notifications…")

                # Unsubscribe
                await client.stop_notify(CHAR_UUID_PPG)
                await client.stop_notify(CHAR_UUID_SPO2)
                await client.stop_notify(CHAR_UUID_ACC)
                print(f"[+] Data saved to {CSV_FILE}")

    except BleakError as e:
        print(f"[!] BLE error: {e}")

# ————— Main Entry —————
def main():
    if args.list:
        asyncio.run(list_services())
    else:
        try:
            asyncio.run(log_data())
        except KeyboardInterrupt:
            # Catch any leftover interrupt in the event loop
            print("\n[+] Interrupted by user; exiting gracefully.")

if __name__ == "__main__":
    main()
