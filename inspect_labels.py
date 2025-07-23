#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv("features_final.csv")
print("Counts before strip():")
print(df.label.value_counts(), "\n")

df.label = df.label.str.strip()
print("Counts after strip():")
print(df.label.value_counts())

df.to_csv("features_final_clean.csv", index=False)
