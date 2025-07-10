#!/usr/bin/env python3
# sync_and_resample.py

import pandas as pd
import argparse

parser = argparse.ArgumentParser(
    description="Synchronize & resample parsed watch data"
)
parser.add_argument(
    "--input", "-i",
    default="watch_data_parsed.csv",
    help="CSV with parsed columns: timestamp, acc_g, SpO2, pulse, hr_bpm, contact"
)
parser.add_argument(
    "--output", "-o",
    default="watch_data_sync.csv",
    help="Output CSV on a uniform time grid"
)
parser.add_argument(
    "--freq", "-f",
    default="100ms",
    help="Resampling frequency (e.g. '100ms' for 10 Hz)"
)
args = parser.parse_args()

# 1) Load parsed data
df = pd.read_csv(args.input)

# 2) Build a datetime index from timestamp (seconds since epoch)
df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
df = df.set_index("datetime").sort_index()

# 3) Select only the numeric sensor columns you care about
cols = ["acc_g", "SpO2", "pulse", "hr_bpm"]
df = df[cols]

# 4) Resample at the desired frequency, forward-fill then backward-fill
print(f"[+] Resampling to {args.freq} grid…")
df_sync = df.resample(args.freq).ffill().bfill()

# 5) Add back a float-seconds column if downstream expects it
df_sync["ts"] = df_sync.index.view("int64") / 1e9

# 6) Save synchronized CSV
df_sync.reset_index(drop=True).to_csv(args.output, index=False)
print(f"[+] Synchronized data saved to {args.output} ({len(df_sync)} rows)")
