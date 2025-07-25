# data_stream.py

import asyncio, time
from collections import deque
from bleak import BleakScanner, BleakClient
from scan import WATCH_ADDR, FS, WINDOW_SEC, STEP_SEC, UUIDS
from scripts.utils import log

# Raw buffer holds (timestamp, sensor, raw_hex)
raw_buffer = deque(maxlen=int((WINDOW_SEC + STEP_SEC) * FS * 3))

def _make_cb(sensor):
    def callback(_, data):
        ts = time.time()
        raw = data.hex()
        raw_buffer.append((ts, sensor, raw))
        log(f"RAW {sensor}: {raw}")
    return callback

async def connect_and_subscribe():
    log(f"Scanning for {WATCH_ADDR}…")
    dev = await BleakScanner.find_device_by_address(WATCH_ADDR, timeout=10.0)
    if not dev:
        log("❌ Watch not found")
        return None

    async with BleakClient(dev) as client:
        log("✅ Connected; enabling notifications…")

        # Enable CCCDs
        for char in (UUIDS["PPG_CHAR"], UUIDS["SPO2_NOTIFY"], UUIDS["ACC_NOTIFY"]):
            await client.start_notify(char, _make_cb(char))

        # Auto‐detect control commands
        for notify_uuid, ctrls in [("SPO2_NOTIFY","SPO2_CONTROL_CMDS"), ("ACC_NOTIFY","ACC_CONTROL_CMDS")]:
            for uuid_ctrl, cmd in UUIDS[ctrls]:
                await client.write_gatt_char(uuid_ctrl, cmd)
                await asyncio.sleep(0.1)

        log("🎉 Streaming started")
        # Keep alive in background
        asyncio.create_task(_keep_alive(client))
        # Keep raw_buffer globally accessible
        while True:
            await asyncio.sleep(1)

async def _keep_alive(client):
    """Re-send control commands every WINDOW_SEC to prevent stream timeout."""
    while True:
        await asyncio.sleep(WINDOW_SEC)
        for uuid_ctrl, cmd in UUIDS["SPO2_CONTROL_CMDS"] + UUIDS["ACC_CONTROL_CMDS"]:
            await client.write_gatt_char(uuid_ctrl, cmd)
