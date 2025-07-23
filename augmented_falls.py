#!/usr/bin/env python3
import pandas as pd, numpy as np

df = pd.read_csv("features_final_clean.csv")
falls = df[df.label=="fall"].copy()
for col in ["acc_jerk_max","acc_peak_count","acc_bandpower"]:
    sigma = falls[col].std()*0.1
    falls[col] += np.random.normal(0, sigma, len(falls))
aug = pd.concat([df, falls], ignore_index=True)
aug.to_csv("features_augmented.csv", index=False)
print("Original:", len(df), "Augmented:", len(aug))
