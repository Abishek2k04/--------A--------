
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

def main():
    iris = load_iris()
    X = iris.data
    feature_names = iris.feature_names

    df = pd.DataFrame(X, columns=feature_names)
    print("First 5 rows (original):\n", df.head(), "\n")

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Means after scaling (approx):", np.round(X_scaled.mean(axis=0), 4))
    print("Stdev after scaling (approx):", np.round(X_scaled.std(axis=0), 4), "\n")

    # Variance thresholding (remove features with very low variance)
    selector = VarianceThreshold(threshold=0.2)  # threshold chosen after scaling
    X_sel = selector.fit_transform(X_scaled)
    kept = selector.get_support()
    kept_features = [f for f, k in zip(feature_names, kept) if k]

    print(f"Shape before selection: {X_scaled.shape}")
    print(f"Shape after selection:  {X_sel.shape}")
    print("Kept features:", kept_features, "\n")

    # PCA for quick visualization (optional)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print("Explained variance ratio by first 2 PCA components:", np.round(pca.explained_variance_ratio_, 4))

if __name__ == "__main__":
    main()