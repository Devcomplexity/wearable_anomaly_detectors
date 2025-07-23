import pandas as pd
df = pd.read_csv("features_sliding.csv")   # or "features.csv"
print(df.shape)   # should be (n_windows, n_features+3)
print(df.head())
