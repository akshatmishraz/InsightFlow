"""Univariate and multivariate anomaly detection utilities."""

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest


def detect_univariate_anomalies(df, column, method="iqr", threshold=1.5):
    if column not in df:
        raise ValueError(f"Column not found: {column}")
    if method not in {"iqr", "zscore"}:
        raise ValueError("method must be 'iqr' or 'zscore'.")
    values = pd.to_numeric(df[column], errors="coerce")
    valid = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3:
        raise ValueError("At least three valid numeric observations are required.")
    if method == "iqr":
        q1, q3 = valid.quantile([0.25, 0.75])
        spread = q3 - q1
        lower, upper = q1-threshold*spread, q3+threshold*spread
        mask = (values < lower) | (values > upper)
        details = {"q1": float(q1), "q3": float(q3), "iqr": float(spread),
                   "lower_bound": float(lower), "upper_bound": float(upper)}
    else:
        if threshold <= 0:
            raise ValueError("Z-score threshold must be positive.")
        sd = valid.std(ddof=0)
        if sd == 0:
            raise ValueError("Z-scores cannot be calculated for a constant column.")
        scores = (values - valid.mean()) / sd
        mask = scores.abs() > threshold
        details = {"threshold": float(threshold), "z_scores": scores}
    result = df.copy()
    result["anomaly_score"] = np.nan
    if method == "zscore":
        result["anomaly_score"] = details.pop("z_scores")
    result["is_anomaly"] = mask.fillna(False)
    return {"method": method, "anomaly_count": int(result["is_anomaly"].sum()),
            "anomaly_percentage": float(result["is_anomaly"].mean()*100),
            "details": details, "data": result}


def detect_multivariate_anomalies(df, columns, contamination=0.05, random_state=42):
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5.")
    if len(columns) < 2 or any(c not in df.columns for c in columns):
        raise ValueError("Select at least two existing numeric columns.")
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    valid_mask = numeric.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    clean = numeric.loc[valid_mask]
    if len(clean) < 10:
        raise ValueError("At least 10 complete rows are required for Isolation Forest.")
    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(clean)
    scores = model.decision_function(clean)
    output = df.copy()
    output["is_anomaly"] = False
    output["anomaly_score"] = np.nan
    output.loc[clean.index, "is_anomaly"] = labels == -1
    output.loc[clean.index, "anomaly_score"] = -scores
    return {"method": "Isolation Forest", "rows_scored": len(clean),
            "anomaly_count": int((labels == -1).sum()),
            "anomaly_percentage_of_scored": float((labels == -1).mean()*100),
            "data": output, "model": model}
