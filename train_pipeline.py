#!/usr/bin/env python3
"""
train_pipeline.py

Load features_augmented.csv,
drop any tiny classes (<2 samples),
clean out infinities/NaNs,
stratified split,
train a stacking pipeline (median‐impute → scale → SMOTE → RF+LR stack),
evaluate on hold‐out set,
and save best_pipeline.pkl + label_encoder.pkl.
"""

import joblib
import pandas as pd
import numpy as np

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ─── LOAD & CLEAN ───────────────────────────────────────────────────────────
df = pd.read_csv("features_augmented.csv")

# 1) Drop any class with fewer than 2 samples
counts = df.label.value_counts()
small  = counts[counts < 2].index.tolist()
if small:
    print(f"Dropping classes with <2 samples: {small}")
    df = df[~df.label.isin(small)]

# 2) Separate X & y
X = df.drop("label", axis=1)
y = df["label"]

# 3) Replace infinities and drop rows with non‐finite values
X = X.replace([np.inf, -np.inf], np.nan)
mask = X.notna().all(axis=1)
if (~mask).any():
    print(f"Dropping {mask.size - mask.sum()} rows with non‐finite features")
    X = X[mask]
    y = y[mask]

# ─── ENCODE & SPLIT ──────────────────────────────────────────────────────────
le    = LabelEncoder().fit(y)
y_enc = le.transform(y)

Xtr, Xte, ytr, yte = train_test_split(
    X, y_enc,
    stratify=y_enc,
    test_size=0.2,
    random_state=42
)

# ─── BUILD PIPELINE ─────────────────────────────────────────────────────────
estimators = [
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
]
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1,
    passthrough=True
)
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
    ("smote",  SMOTE(random_state=42)),
    ("stack",  stack)
])

# ─── TRAIN & EVALUATE ───────────────────────────────────────────────────────
pipe.fit(Xtr, ytr)
ypr = pipe.predict(Xte)

print("\nTest Set Classification Report:")
print(classification_report(yte, ypr, target_names=le.classes_))

# ─── SAVE ARTIFACTS ─────────────────────────────────────────────────────────
joblib.dump(pipe, "best_pipeline.pkl")
joblib.dump(le,   "label_encoder.pkl")
print("\n✅ Saved best_pipeline.pkl and label_encoder.pkl")
