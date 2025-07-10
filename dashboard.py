#!/usr/bin/env python3
# dashboard.py

import streamlit as st
import pandas as pd
import time
from plyer import notification

st.set_page_config(page_title="Anomaly Dashboard", layout="wide")
st.title("Real-Time Anomaly Dashboard")

# Sidebar: threshold slider (default to your model.offset_)
threshold = st.sidebar.slider(
    "Anomaly Threshold", -2.0, 2.0, -0.599, 0.01
)

chart = st.line_chart()
anom_chart = st.area_chart()
last_notified = 0.0
log_path = "scores_log.csv"

def send_desktop_notification(ts, score):
    notification.notify(
        title="🚨 Anomaly Detected",
        message=f"{time.strftime('%H:%M:%S', time.localtime(ts))} → score={score:.3f}",
        timeout=5
    )

while True:
    # Attempt to load the log
    try:
        df = pd.read_csv(log_path)
    except FileNotFoundError:
        st.warning("Waiting for `scores_log.csv`…")
        time.sleep(1)
        continue

    # Prepare DataFrame
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("timestamp").sort_index().tail(200)

    # Update score plot
    chart.add_rows(df[["score"]])

    # Identify new anomalies
    df["is_anom"] = df["score"] < threshold
    anoms = df[df.is_anom]

    if not anoms.empty:
        anom_chart.add_rows(anoms[["score"]])
        newest_ts = anoms.index[-1].timestamp()
        newest_score = anoms["score"].iloc[-1]

        if newest_ts > last_notified:
            send_desktop_notification(newest_ts, newest_score)
            last_notified = newest_ts

        st.warning(f"New anomaly @ {anoms.index[-1].strftime('%H:%M:%S')}  score={newest_score:.3f}")

    time.sleep(1)
