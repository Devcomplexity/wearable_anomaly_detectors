# new.py

import asyncio
import csv
import time
import argparse
from bleak import BleakScanner, BleakClient, BleakError

# 1) Your watch’s MAC address:
TARGET_ADDRESS = "F4:E2:45:58:19:36"

# 2) After discovery, fill in these UUIDs:
CHAR_UUID_PPG  = "00002a37-0000-1000-8000-00805f9b34fb"
CHAR_UUID_SPO2 = "ffea0002-0000-1000-8000-00805f9b34fb"
CHAR_UUID_ACC  = "fee70002-0000-1000-8000-00805f9b34fb"

CSV_FILE = "watch_data.csv"


async def list_services():
    print(f"Scanning for device {TARGET_ADDRESS}…")
    # find_device_by_address combines scan + match
    device = await BleakScanner.find_device_by_address(
        TARGET_ADDRESS, timeout=8.0
    )
    if not device:
        print(f"❌ Device {TARGET_ADDRESS} not found. Make sure it's in discoverable mode.")
        return

    print(f"✅ Found {device.name!r} @ {device.address}\nListing services…")
    try:
        async with BleakClient(device) as client:
            # services are cached in client.services
            for svc in client.services:
                print(f"\nService {svc.uuid}:")
                for ch in svc.characteristics:
                    props = ",".join(ch.properties)
                    print(f"  {ch.uuid} — {props}")
    except BleakError as e:
        print("❌ Failed to connect:", e)


async def log_data():
    print(f"Looking up {TARGET_ADDRESS}…")
    device = await BleakScanner.find_device_by_address(
        TARGET_ADDRESS, timeout=8.0
    )
    if not device:
        print(f"❌ Device {TARGET_ADDRESS} not found. Ensure discoverable.")
        return

    try:
        async with BleakClient(device) as client:
            print(f"✅ Connected to {device.name!r} @ {device.address}")

            # prepare CSV
            with open(CSV_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "sensor", "raw_hex"])
                f.flush()

                def _cb(sensor):
                    def callback(_, data: bytearray):
                        writer.writerow([time.time(), sensor, data.hex()])
                        f.flush()
                    return callback

                # start notifications
                await client.start_notify(CHAR_UUID_PPG,  _cb("PPG"))
                await client.start_notify(CHAR_UUID_SPO2, _cb("SpO2"))
                await client.start_notify(CHAR_UUID_ACC,  _cb("ACC"))

                print("Logging… press Ctrl+C to stop.")
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    print("\nStopping notifications…")

                # clean up
                await client.stop_notify(CHAR_UUID_PPG)
                await client.stop_notify(CHAR_UUID_SPO2)
                await client.stop_notify(CHAR_UUID_ACC)
                print(f"✅ Data saved to {CSV_FILE}")

    except BleakError as e:
        print("❌ BLE error:", e)


def main():
    parser = argparse.ArgumentParser(description="BLE discovery & data logger")
    parser.add_argument(
        "--list", action="store_true",
        help="Scan & list all services/characteristics"
    )
    args = parser.parse_args()

    if args.list:
        asyncio.run(list_services())
    else:
        asyncio.run(log_data())


if __name__ == "__main__":
    main()
