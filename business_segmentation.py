"""Validated, interpretable K-Means and hierarchical business segmentation."""

from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def segment_dataset(df, feature_columns, n_clusters=3, method="kmeans", use_pca=False, random_state=42):
    """Cluster complete numeric records, retaining excluded rows without a segment.

    PCA, when enabled, changes the clustering feature space (not just the plot).
    Segment labels are arbitrary identifiers, not ordinal business rankings.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not df.columns.is_unique:
        raise ValueError("Duplicate DataFrame column names are not supported.")
    if "segment" in df.columns:
        raise ValueError("Reserved output column 'segment' already exists.")
    if not isinstance(feature_columns, (list, tuple)) or not feature_columns:
        raise ValueError("Select existing feature columns as a nonempty list or tuple.")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Feature columns must be distinct.")
    if any(c not in df.columns for c in feature_columns):
        raise ValueError("Select existing feature columns.")
    if isinstance(n_clusters, (bool, np.bool_)) or not isinstance(n_clusters, Integral) or n_clusters < 2:
        raise ValueError("n_clusters must be an integer of at least 2.")
    if method not in {"kmeans", "hierarchical"}:
        raise ValueError("method must be 'kmeans' or 'hierarchical'.")
    if not isinstance(use_pca, (bool, np.bool_)):
        raise ValueError("use_pca must be boolean.")
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, Integral) or random_state < 0:
        raise ValueError("random_state must be a nonnegative integer.")

    numeric = df.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid_mask = numeric.notna().all(axis=1).to_numpy(dtype=bool)
    clean = numeric.iloc[np.flatnonzero(valid_mask)].copy()
    n = len(clean)
    if n <= n_clusters:
        raise ValueError("The number of complete rows must exceed n_clusters.")
    if not np.isfinite(clean.to_numpy(dtype=float)).all():
        raise ValueError("Numeric features must be finite.")
    if not any(clean[c].nunique() > 1 for c in feature_columns):
        raise ValueError("At least one feature must have variation.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean)
    if not np.isfinite(scaled).all():
        raise ValueError("Feature scaling produced nonfinite values.")
    constant_features = [c for c in feature_columns if clean[c].nunique() <= 1]
    warnings = []
    if constant_features:
        warnings.append("Constant features provide no clustering information: " + ", ".join(map(str, constant_features)))
    if n_clusters > max(2, n // 3):
        warnings.append("Many clusters relative to the number of records; interpret small segments cautiously.")

    pca = None
    explained = None
    transformed = scaled
    if use_pca:
        components = min(2, scaled.shape[1], scaled.shape[0])
        pca = PCA(n_components=components, random_state=int(random_state))
        transformed = pca.fit_transform(scaled)
        explained = [float(v) for v in pca.explained_variance_ratio_]
        if components < scaled.shape[1]:
            warnings.append("PCA reduces dimensions before clustering and can change segment assignments.")

    if method == "kmeans":
        model = KMeans(n_clusters=int(n_clusters), n_init=10, random_state=int(random_state))
    else:
        model = AgglomerativeClustering(n_clusters=int(n_clusters))
    labels = model.fit_predict(transformed)
    observed = len(np.unique(labels))
    if observed != n_clusters:
        warnings.append(f"Requested {n_clusters} clusters, but the model produced {observed} distinct labels.")

    output = df.copy()
    # Positional assignment avoids ambiguous .loc writes on duplicate indexes.
    segment_values = pd.array([pd.NA] * len(df), dtype="Int64")
    segment_values[valid_mask] = labels.astype("int64")
    output["segment"] = segment_values

    profiled = clean.reset_index(drop=True).copy()
    profiled["segment"] = labels
    profile = profiled.groupby("segment").agg({c: ["count", "mean", "median"] for c in feature_columns})
    sizes = pd.Series(labels).value_counts().sort_index().rename("rows")
    sizes.index.name = "segment"
    summary = profiled.groupby("segment")[feature_columns].mean()
    overall = clean.mean()
    overall_sd = clean.std(ddof=0).replace(0, np.nan)
    standardized_profile = (summary - overall) / overall_sd
    standardized_profile = standardized_profile.fillna(0.0)
    top_distinguishing_features = {}
    for segment_id, row in standardized_profile.iterrows():
        top_distinguishing_features[int(segment_id)] = [
            {"feature": col, "standardized_difference": float(row[col])}
            for col in row.abs().sort_values(ascending=False, kind="stable").index[:min(3, len(feature_columns))]
        ]

    silhouette = None
    db_index = None
    if 2 <= observed < n:
        # Evaluate in the same feature space in which clustering was performed.
        silhouette = float(silhouette_score(transformed, labels))
        db_index = float(davies_bouldin_score(transformed, labels))
    else:
        warnings.append("Cluster quality metrics are unavailable when fewer than two distinct clusters are formed.")

    return {
        "data": output, "cluster_sizes": sizes, "profile": profile,
        "method": method, "features": list(feature_columns),
        "pca_explained_variance_ratio": explained,
        "model": model, "scaler": scaler, "pca": pca,
        "rows_scored": n, "excluded_rows": int(len(df) - n),
        "requested_clusters": int(n_clusters), "observed_clusters": observed,
        "silhouette_score": silhouette, "davies_bouldin_score": db_index,
        "segment_mean_profile": summary,
        "standardized_segment_profile": standardized_profile,
        "top_distinguishing_features": top_distinguishing_features,
        "constant_features": constant_features, "warnings": warnings,
        "clustering_space": "pca" if use_pca else "standardized_features",
        "note": "Clusters are exploratory groupings, not naturally occurring truths. "
                "Labels are arbitrary; distinguishing features describe differences from the dataset mean, "
                "not causal explanations. Review profiles and business meaning before acting.",
    }
