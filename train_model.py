#!/usr/bin/env python3
# train_model.py

import pandas as pd, numpy as np, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# 1) Load & split
df = pd.read_csv("watch_features.csv")
X  = df.drop(columns=["window_start"])
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# 2) Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 3) Train
model = IsolationForest(
    n_estimators=100, contamination="auto", random_state=42
)
model.fit(X_train_s)

# 4) Score
scores = model.decision_function(X_test_s)
thresh = np.percentile(scores, 5)
labels = (scores < thresh).astype(int)

print(f"Anomalies in test: {labels.sum()}/{len(labels)}")

# 5) Save
joblib.dump(scaler, "scaler.joblib")
joblib.dump(model,  "iso_forest.joblib")
print("Artifacts saved: scaler.joblib, iso_forest.joblib")
