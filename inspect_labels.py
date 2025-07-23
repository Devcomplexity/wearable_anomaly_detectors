#!/usr/bin/env python3
"""
inspect_labels.py

Load features_final.csv, display label counts to verify classes,
and write out features_final_clean.csv for the next stages.
"""

import pandas as pd

IN_CSV  = "features_final.csv"
OUT_CSV = "features_final_clean.csv"

def main():
    # 1) Load the features you just extracted
    df = pd.read_csv(IN_CSV)

    # 2) Show the counts for each activity label
    print("\nClass counts in features_final.csv:")
    print(df.label.value_counts())

    # 3) Save a clean copy (we’re not dropping anything here)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Saved clean features to {OUT_CSV}")

if __name__ == "__main__":
    main()
