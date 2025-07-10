#!/usr/bin/env python3
# retrain_selected_model.py

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing    import StandardScaler
from sklearn.ensemble         import IsolationForest
from sklearn.model_selection import train_test_split

def main():
    # 1) Load enhanced feature matrix
    df = pd.read_csv("watch_features_enhanced.csv")

    # 2) Load kept‐features list robustly (first column)
    feat_df = pd.read_csv("features_kept.csv")
    features = feat_df.iloc[:, 0].astype(str).tolist()
    # Drop any literal "0" if it snuck in
    features = [f for f in features if f != "0"]
    print("Using features:", features)

    # 3) Subset & fill missing
    X = df[features].fillna(0).values

    # 4) Train/test split
    X_train, X_test = train_test_split(
        X, test_size=0.2, random_state=42
    )

    # 5) Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 6) Train IsolationForest
    model = IsolationForest(
        n_estimators=200,
        max_samples=0.8,
        contamination=0.05,
        random_state=42
    )
    model.fit(X_train_s)

    # 7) Score test set & flag bottom 5%
    scores = model.decision_function(X_test_s)
    thresh = np.percentile(scores, 5)
    y_pred = (scores < thresh).astype(int)

    print(f"Test samples: {len(X_test_s)}")
    print(f"Anomalies flagged: {y_pred.sum()} ({100*y_pred.mean():.1f}%)")

    # 8) Save artifacts
    joblib.dump(scaler, "scaler_selected.joblib")
    joblib.dump(model,  "iso_forest_selected.joblib")
    print("Saved: scaler_selected.joblib, iso_forest_selected.joblib")

if __name__ == "__main__":
    main()
