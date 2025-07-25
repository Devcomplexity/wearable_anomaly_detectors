#!/usr/bin/env python3
"""
augment_falls.py

Read features_final_clean.csv, oversample the 'fall' rows by adding small Gaussian
noise until it matches the largest class, then write features_augmented.csv.
"""

import pandas as pd
import numpy as np

IN_CSV = "features_final_clean.csv"
OUT_CSV = "features_augmented.csv"

def main():
    # 1) Load clean features
    df = pd.read_csv(IN_CSV)

    # 2) Separate falls and others
    falls  = df[df.label == "fall"]
    others = df[df.label != "fall"]

    # 3) Compute target and current fall count
    counts   = df.label.value_counts()
    target_n = counts.max()   # largest class size
    current  = len(falls)
    to_add   = target_n - current

    print(f"Fall windows: {current}, target: {target_n}, augmenting by {to_add}")

    # 4) Generate synthetic fall samples
    augmented = []
    numeric_cols = df.columns.drop("label")
    for _ in range(to_add):
        # sample one existing fall
        sample = falls.sample(1, replace=True).iloc[0]
        values = sample[numeric_cols].values.astype(float)
        # add tiny noise
        noise = np.random.normal(scale=0.01, size=values.shape)
        new_vals = values + noise
        row = dict(zip(numeric_cols, new_vals))
        row["label"] = "fall"
        augmented.append(row)

    # 5) Combine and shuffle
    df_aug = pd.concat([others, falls, pd.DataFrame(augmented)], ignore_index=True)
    df_aug = df_aug.sample(frac=1, random_state=42).reset_index(drop=True)

    # 6) Save
    df_aug.to_csv(OUT_CSV, index=False)
    print(f"✅ Saved {len(df_aug)} windows to {OUT_CSV}")

if __name__ == "__main__":
    main()
