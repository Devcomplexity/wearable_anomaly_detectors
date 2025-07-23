#!/usr/bin/env python3
# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

FEATURES_FILE = "features_sliding.csv"  # or "features.csv"

def load_and_prepare():
    # 1) Load features
    df = pd.read_csv(FEATURES_FILE)
    print("Loaded features:", df.shape)

    # 2) Fill all NaNs with 0 so no rows are dropped
    df = df.fillna(0)
    print("After fillna     :", df.shape)

    # 3) Split into X and y
    X = df.drop(["label","start_ts","end_ts"], axis=1)
    y = df["label"]
    return X, y

def main():
    # Load and prepare data
    X, y = load_and_prepare()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}\n")

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=clf.classes_,
                yticklabels=clf.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # Save model
    joblib.dump(clf, "activity_classifier.pkl")
    print("\nModel saved to activity_classifier.pkl")

if __name__ == "__main__":
    main()
