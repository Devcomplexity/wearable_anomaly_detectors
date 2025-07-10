# parse_watch_data.py

import csv
import struct
import pandas as pd

INPUT_CSV  = "watch_data.csv"
OUTPUT_CSV = "watch_data_parsed.csv"

def parse_acc(raw_hex):
    b = bytes.fromhex(raw_hex)
    # little-endian signed int32 → g
    val = struct.unpack("<i", b)[0] / 1000.0
    return {"acc_g": val}

def parse_spo2(raw_hex):
    b = bytes.fromhex(raw_hex)
    # Our minimal decoder: byte[2]=SpO₂%, byte[3]=pulse
    spo2 = pulse = None
    if len(b) >= 4:
        spo2 = b[2]
        pulse = b[3]
    return {"SpO2": spo2, "pulse": pulse}

def parse_ppg(raw_hex):
    b = bytes.fromhex(raw_hex)
    # BLE Heart Rate Measurement (0x2A37):
    #   flags = b[0], HR in uint8 at b[1]
    #   bit2 of flags = sensor-contact detected
    hr = contact = None
    if len(b) >= 2:
        flags = b[0]
        hr    = b[1]
        contact = bool(flags & 0x04)
    return {"hr_bpm": hr, "contact": contact}

# map sensor names to parsers
PARSERS = {
    "ACC":  parse_acc,
    "SpO2": parse_spo2,
    "PPG":  parse_ppg,
}

def main():
    rows = []
    with open(INPUT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ts     = float(r["timestamp"])
            sensor = r["sensor"]
            raw    = r["raw_hex"]
            parser = PARSERS.get(sensor)
            if not parser:
                continue
            parsed = parser(raw)
            parsed.update({"timestamp": ts, "sensor": sensor})
            rows.append(parsed)

    df = pd.DataFrame(rows)
    # Pivot so each timestamp yields one row
    df_parsed = (df
        .drop(columns=["sensor"])
        .groupby("timestamp", as_index=False)
        .first()
    )
    df_parsed.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df_parsed)} rows → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
