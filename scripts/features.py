# preprocess.py

import pandas as pd
import numpy as np
from scan import FS, WINDOW_SEC, raw_buffer
from scripts.utils import log

# Hex → numeric parsers
def parse_acc(hexstr):
    return int.from_bytes(bytes.fromhex(hexstr), 'little', signed=True) / 1000.0

def parse_ppg(hexstr):
    b = bytes.fromhex(hexstr)
    flags, hr = b[0], b[1]
    return hr, bool(flags & 0x04)

def parse_spo2(hexstr):
    b = bytes.fromhex(hexstr)
    return b[2], b[3]

def build_window_df():
    """Convert raw_buffer into a resampled DataFrame indexed by datetime."""
    rows = []
    for ts, sensor, raw in list(raw_buffer):
        dt = pd.to_datetime(ts, unit="s")
        if sensor == UUIDS["ACC_NOTIFY"]:
            rows.append({"dt": dt, "ACC": parse_acc(raw)})
        elif sensor == UUIDS["PPG_CHAR"]:
            hr, _ = parse_ppg(raw)
            rows.append({"dt": dt, "PPG": hr})
        elif sensor == UUIDS["SPO2_NOTIFY"]:
            sp, _ = parse_spo2(raw)
            rows.append({"dt": dt, "SpO2": sp})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("dt").sort_index()
    # Resample at 100 ms intervals
    freq = f"{int(1000/FS)}ms"
    dfs = df.resample(freq).ffill().bfill()
    # Slice last WINDOW_SEC
    cutoff = pd.Timestamp.now() - pd.Timedelta(seconds=WINDOW_SEC)
    window = dfs[dfs.index >= cutoff]
    return window
