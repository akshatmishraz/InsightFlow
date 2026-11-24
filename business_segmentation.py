"""K-Means and hierarchical segmentation with optional PCA projection."""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def segment_dataset(df, feature_columns, n_clusters=3, method="kmeans", use_pca=False, random_state=42):
    if not feature_columns or any(c not in df.columns for c in feature_columns):
        raise ValueError("Select existing feature columns.")
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")
    if method not in {"kmeans", "hierarchical"}:
        raise ValueError("method must be 'kmeans' or 'hierarchical'.")
    numeric = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    valid_mask = numeric.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    clean = numeric.loc[valid_mask]
    if len(clean) <= n_clusters:
        raise ValueError("The number of complete rows must exceed n_clusters.")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean)
    if use_pca:
        components = min(2, scaled.shape[1], scaled.shape[0])
        if components < 1:
            raise ValueError("PCA requires at least one usable feature.")
        pca = PCA(n_components=components, random_state=random_state)
        transformed = pca.fit_transform(scaled)
        explained = pca.explained_variance_ratio_.tolist()
    else:
        pca = None
        transformed = scaled
        explained = None
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        labels = model.fit_predict(transformed)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(transformed)
    output = df.copy()
    output["segment"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    output.loc[clean.index, "segment"] = labels
    profile = clean.assign(segment=labels).groupby("segment").agg(["count", "mean", "median"])
    sizes = pd.Series(labels).value_counts().sort_index().rename("rows")
    return {"data": output, "cluster_sizes": sizes, "profile": profile,
            "method": method, "features": list(feature_columns),
            "pca_explained_variance_ratio": explained,
            "model": model, "scaler": scaler, "pca": pca,
            "note": "Clusters are exploratory groupings, not naturally occurring truths. Review profiles and business meaning before acting."}
