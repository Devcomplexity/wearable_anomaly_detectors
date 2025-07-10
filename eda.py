#!/usr/bin/env python3
# eda_summary.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def main():
    # 1) Load features
    df = pd.read_csv("watch_features_enhanced.csv")

    # 2) Drop window_start and constant columns
    if "window_start" in df:
        df = df.drop(columns=["window_start"])
    # Remove zero-variance features
    var = df.var()
    df = df.loc[:, var > 0]

    # 3) Print descriptive statistics
    print("\nDescriptive Statistics:\n")
    print(df.describe().T)

    # 4) Histograms of dynamic features
    plt.figure(figsize=(10,6))
    df.hist(bins=25, figsize=(10,6))
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    plt.savefig("summary_histograms.png")
    plt.show()

    # 5) Correlation heatmap
    corr = df.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("summary_correlation.png")
    plt.show()

    # 6) PCA projection (2D)
    Xz = (df - df.mean()) / df.std()
    proj = PCA(2).fit_transform(Xz.fillna(0))
    plt.figure(figsize=(5,4))
    plt.scatter(proj[:,0], proj[:,1], s=30, alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA (2D)")
    plt.tight_layout()
    plt.savefig("summary_pca.png")
    plt.show()

    print("\nSaved: summary_histograms.png, summary_correlation.png, summary_pca.png\n")

if __name__ == "__main__":
    main()
