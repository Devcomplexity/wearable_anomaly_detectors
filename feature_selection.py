# feature_selection.py

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

# 1. Load and standardize
df = pd.read_csv("watch_features_enhanced.csv")
X  = df.drop(columns=["window_start"])
Xz = (X - X.mean()) / X.std()

# 2. Remove near‐constant features
vt = VarianceThreshold(threshold=0.01)  
vt.fit(Xz.fillna(0))
keep = Xz.columns[vt.get_support()]
print("Retaining features:", list(keep))

X_sel = Xz[keep]

# 3. Remove highly correlated
corr = X_sel.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
print("Dropping highly correlated:", to_drop)

X_fs = X_sel.drop(columns=to_drop)

# 4. Optional: PCA to reduce to K components
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)  # keep 95% variance
Xp = pca.fit_transform(X_fs.fillna(0))
print("Reduced shape:", Xp.shape)

# Save selected feature list
pd.Series(list(X_fs.columns)).to_csv("features_kept.csv", index=False)
