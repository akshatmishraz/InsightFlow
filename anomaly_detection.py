"""Validated, interpretable univariate and multivariate anomaly detection.

Anomaly scores are method-specific and MUST NOT be compared across methods.
Isolation Forest feature diagnostics are descriptive, not model attributions.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

_OUTPUT_COLUMNS = {"is_anomaly", "anomaly_score", "anomaly_severity", "top_deviating_feature", "top_feature_deviation"}


def _frame(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("DataFrame must not be empty.")
    collisions = _OUTPUT_COLUMNS.intersection(df.columns)
    if collisions:
        raise ValueError(f"Reserved output columns already exist: {sorted(collisions)}")


def _positive_finite(value, name):
    if isinstance(value, (bool, str)):
        raise ValueError(f"{name} must be a positive finite number.")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError(f"{name} must be a positive finite number.") from None
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return number


def _numeric(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _severity(flag, score, baseline):
    """Relative priority tiers for flagged records, not calibrated risk levels."""
    labels = np.full(len(flag), "none", dtype=object)
    labels[flag & (score < 2 * baseline)] = "low"
    labels[flag & (score >= 2 * baseline) & (score < 3 * baseline)] = "medium"
    labels[flag & (score >= 3 * baseline)] = "high"
    return labels


def detect_univariate_anomalies(df, column, method="iqr", threshold=1.5):
    """Detect IQR or absolute Z-score anomalies; preserve input row order/index.

    IQR anomaly_score is distance beyond the nearest fence divided by IQR.
    If IQR is zero, the score is raw distance beyond the fence instead.
    Z-score anomaly_score is the signed population Z-score (legacy behavior).
    """
    _frame(df)
    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")
    if not df.columns.is_unique:
        raise ValueError("DataFrame column names must be unique.")
    if method not in {"iqr", "zscore"}:
        raise ValueError("method must be 'iqr' or 'zscore'.")
    threshold = _positive_finite(threshold, "threshold")
    values = _numeric(df[column])
    clean = values.dropna()
    if len(clean) < 3:
        raise ValueError("At least three valid numeric observations are required.")
    array = values.to_numpy(dtype=float)
    valid = np.isfinite(array)
    flag = np.zeros(len(df), dtype=bool)
    score = np.full(len(df), np.nan)
    warnings = []
    if method == "iqr":
        q1, q3 = clean.quantile([0.25, 0.75])
        spread = float(q3 - q1)
        lower, upper = float(q1 - threshold * spread), float(q3 + threshold * spread)
        flag[valid] = (array[valid] < lower) | (array[valid] > upper)
        distance = np.maximum(np.maximum(lower - array[valid], array[valid] - upper), 0)
        score[valid] = distance / spread if spread > 0 else distance
        if spread == 0:
            warnings.append("IQR is zero; scores use raw distance from the fence instead of IQR units.")
        details = {"q1": float(q1), "q3": float(q3), "iqr": spread,
                   "lower_bound": lower, "upper_bound": upper,
                   "threshold": threshold,
                   "score_unit": "IQR multiples beyond fence" if spread > 0 else "raw distance beyond fence"}
        # Severity is relative to fence exceedance, not the detection threshold.
        baseline = threshold if spread > 0 else max(float(clean.std(ddof=0)), 1e-12)
        severity = _severity(flag, score, baseline)
    else:
        sd = float(clean.std(ddof=0))
        if sd == 0:
            raise ValueError("Z-scores cannot be calculated for a constant column.")
        mean = float(clean.mean())
        score[valid] = (array[valid] - mean) / sd
        flag[valid] = np.abs(score[valid]) > threshold
        severity = _severity(flag, np.abs(score), threshold)
        details = {"threshold": threshold, "mean": mean, "standard_deviation": sd,
                   "score_unit": "signed population Z-score"}
    output = df.copy()
    output["anomaly_score"] = score
    output["is_anomaly"] = flag
    output["anomaly_severity"] = severity
    return {"method": method, "anomaly_count": int(flag.sum()),
            "anomaly_percentage": float(flag.mean() * 100), "details": details,
            "rows_scored": int(valid.sum()), "excluded_rows": int((~valid).sum()),
            "warnings": warnings, "data": output}


def detect_multivariate_anomalies(df, columns, contamination=0.05, random_state=42):
    """Detect unusual feature combinations with Isolation Forest.

    Higher anomaly_score means more anomalous; >0 corresponds to flagged rows.
    top_deviating_feature is a robust marginal deviation diagnostic, NOT an
    explanation of the Isolation Forest decision or a causal attribution.
    """
    _frame(df)
    if not df.columns.is_unique:
        raise ValueError("DataFrame column names must be unique.")
    if isinstance(columns, (str, bytes)):
        raise ValueError("Select at least two distinct numeric columns.")
    try:
        columns = list(columns)
    except TypeError:
        raise ValueError("Select at least two distinct numeric columns.") from None
    if len(columns) < 2 or len(set(columns)) != len(columns) or any(c not in df.columns for c in columns):
        raise ValueError("Select at least two distinct existing numeric columns.")
    if isinstance(contamination, (bool, str)):
        raise ValueError("contamination must be between 0 and 0.5.")
    try:
        contamination = float(contamination)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("contamination must be between 0 and 0.5.") from None
    if not np.isfinite(contamination) or not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5.")
    numeric = df[columns].apply(_numeric)
    valid_mask = numeric.notna().all(axis=1).to_numpy(dtype=bool)
    clean = numeric.iloc[np.flatnonzero(valid_mask)]
    if len(clean) < 10:
        raise ValueError("At least 10 complete rows are required for Isolation Forest.")
    if (clean.nunique(dropna=True) <= 1).all():
        raise ValueError("Isolation Forest requires variation in at least one feature.")
    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(clean)
    scores = -model.decision_function(clean)
    flag = labels == -1
    # Descriptive diagnostics based on robust standardized distance from median.
    medians = clean.median(axis=0)
    mad = (clean - medians).abs().median(axis=0)
    std = clean.std(axis=0, ddof=0)
    scales = (1.4826 * mad).where(mad > 0, std).replace(0, 1.0)
    deviations = ((clean - medians) / scales).abs()
    max_deviation = deviations.max(axis=1).to_numpy(dtype=float)
    feature = deviations.idxmax(axis=1).to_numpy(dtype=object)
    severity = np.full(len(clean), "none", dtype=object)
    # Rank only flagged records by anomaly score; tiers are relative to this run.
    if flag.any():
        ranked = pd.Series(scores[flag]).rank(pct=True, method="average").to_numpy()
        severity[flag] = np.where(ranked > 2 / 3, "high", np.where(ranked > 1 / 3, "medium", "low"))
    output = df.copy()
    output["is_anomaly"] = False
    output["anomaly_score"] = np.nan
    output["anomaly_severity"] = "not_scored"
    output["top_deviating_feature"] = pd.Series([None] * len(df), index=df.index, dtype=object).to_numpy()
    output["top_feature_deviation"] = np.nan
    positions = np.flatnonzero(valid_mask)
    for name, data in (("is_anomaly", flag), ("anomaly_score", scores),
                       ("anomaly_severity", severity), ("top_deviating_feature", feature),
                       ("top_feature_deviation", max_deviation)):
        output.iloc[positions, output.columns.get_loc(name)] = data
    return {"method": "Isolation Forest", "rows_scored": len(clean),
            "excluded_rows": int((~valid_mask).sum()),
            "anomaly_count": int(flag.sum()),
            "anomaly_percentage_of_scored": float(flag.mean() * 100),
            "data": output, "model": model,
            "diagnostic_note": "Top deviating feature is a marginal deviation, not a model attribution.",
            "warnings": (["Some rows were excluded due to missing or non-finite feature values."]
                         if (~valid_mask).any() else [])}
