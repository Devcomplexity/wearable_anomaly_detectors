#!/usr/bin/env python3
"""
preprocess.py

Load raw BLE hex readings and session labels,
convert hex to numeric sensor values, strip label whitespace,
drop invalid rows, and produce a clean CSV ready for feature extraction.
"""

import os
import pandas as pd
import numpy as np

# ─── CONFIG ────────────────────────────────────────────────────────────────
RAW_CSV   = "raw_data_labeled.csv"
CLEAN_CSV = "raw_data_clean.csv"

# Only keep these three streams
ACC_UUID  = "0000fea1-0000-1000-8000-00805f9b34fb"
PPG_UUID  = "00002a37-0000-1000-8000-00805f9b34fb"
SPO2_UUID = "0000fee3-0000-1000-8000-00805f9b34fb"
KEEP_UUIDS = {ACC_UUID, PPG_UUID, SPO2_UUID}


def hex_to_value(uuid: str, hexstr: str) -> float:
    """
    Convert the 4-byte hex string from each notification
    into a meaningful numeric reading.
    """
    b = bytes.fromhex(hexstr)
    if uuid == ACC_UUID:
        x = int.from_bytes(b[0:2], "little", signed=True)
        y = int.from_bytes(b[2:4], "little", signed=True)
        return float(np.hypot(x, y))
    if uuid == PPG_UUID:
        return float(b[1])
    if uuid == SPO2_UUID:
        return float(b[2])
    return np.nan


def main():
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"{RAW_CSV} not found")

    # 1) Load raw hex‐encoded readings
    df = pd.read_csv(RAW_CSV)

    # 2) Drop rows without a valid label or outside our known UUIDs
    df = df[df.label.notna() & df.label.str.lower().ne("none")]
    df = df[df.sensor_uuid.isin(KEEP_UUIDS)]

    # 3) Convert timestamp to float
    df["timestamp"] = df["timestamp"].astype(float)

    # 4) Decode hex → numeric value
    df["value"] = df.apply(
        lambda row: hex_to_value(row.sensor_uuid, row.hex),
        axis=1
    )

    # 5) Drop rows where conversion failed
    df = df.dropna(subset=["value"])

    # 5b) Strip whitespace from labels
    df["label"] = df["label"].str.strip()

    # 6) Select & reorder columns
    df_clean = df[["timestamp", "sensor_uuid", "value", "label"]].copy()

    # 7) Sort by time
    df_clean = df_clean.sort_values("timestamp").reset_index(drop=True)

    # 8) Write out the clean CSV
    df_clean.to_csv(CLEAN_CSV, index=False)
    print(f"✅ Written cleaned data to {CLEAN_CSV} ({len(df_clean)} rows)")


if __name__ == "__main__":
    main()
