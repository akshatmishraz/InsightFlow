"""Key-based dataset reconciliation and exploratory numeric distribution drift."""

from collections import Counter
from numbers import Real

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _columns(df, label):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame.")
    if not df.columns.is_unique:
        raise ValueError(f"{label} contains duplicate column names.")
    if not all(isinstance(c, str) for c in df.columns):
        raise ValueError(f"{label} must have string column names.")


def _equal(a, b):
    """Compare scalar values, treating two missing values as equal."""
    if pd.isna(a) and pd.isna(b):
        return True
    try:
        result = a == b
        return bool(result) if not pd.isna(result) else False
    except (TypeError, ValueError):
        return False


def _finite(series):
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)


def compare_datasets(df_old, df_new, key_columns, numeric_columns=None, drift_alpha=0.05):
    """Compare snapshots using non-null unique keys.

    Changes are assessed only for columns present in both snapshots. New or
    removed columns are reported separately, not as field-level modifications.
    Numeric drift compares *all* valid values in each snapshot, not just shared
    records. KS p-values are exploratory and do not imply causality.
    """
    _columns(df_old, "df_old")
    _columns(df_new, "df_new")
    if isinstance(key_columns, (str, bytes)) or key_columns is None:
        raise ValueError("key_columns must be a non-empty sequence of distinct column names.")
    try:
        keys = list(key_columns)
    except TypeError as exc:
        raise ValueError("key_columns must be a non-empty sequence.") from exc
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("key_columns must be non-empty and distinct.")
    if any(c not in df_old.columns or c not in df_new.columns for c in keys):
        raise ValueError("Key columns must exist in both datasets.")
    if df_old[keys].isna().any().any() or df_new[keys].isna().any().any():
        raise ValueError("Key columns cannot contain missing values.")
    if df_old.duplicated(keys).any() or df_new.duplicated(keys).any():
        raise ValueError("Key columns must uniquely identify rows in both datasets.")
    if isinstance(drift_alpha, (bool, np.bool_)) or not isinstance(drift_alpha, Real) or not np.isfinite(drift_alpha) or not 0 < drift_alpha < 1:
        raise ValueError("drift_alpha must be a finite number between 0 and 1.")
    if numeric_columns is None:
        numeric = []
    elif isinstance(numeric_columns, (str, bytes)):
        raise ValueError("numeric_columns must be a sequence, not a string.")
    else:
        numeric = list(numeric_columns)
    if len(numeric) != len(set(numeric)):
        raise ValueError("numeric_columns must be distinct.")
    if any(c in keys for c in numeric):
        raise ValueError("numeric_columns cannot include key columns.")
    if any(c not in df_old.columns or c not in df_new.columns for c in numeric):
        raise ValueError("numeric_columns must exist in both datasets.")

    old_cols, new_cols = set(df_old.columns), set(df_new.columns)
    common = [c for c in df_old.columns if c in new_cols]
    compare_cols = [c for c in common if c not in keys]
    old = df_old.set_index(keys, drop=False)
    new = df_new.set_index(keys, drop=False)
    # Preserve input order instead of sorting keys (which may have mixed types).
    added_idx = new.index[~new.index.isin(old.index)]
    deleted_idx = old.index[~old.index.isin(new.index)]
    shared_idx = old.index[old.index.isin(new.index)]
    changes = []
    counts = Counter()
    for idx in shared_idx:
        old_row, new_row = old.loc[idx], new.loc[idx]
        changed_fields = []
        for col in compare_cols:
            a, b = old_row[col], new_row[col]
            if not _equal(a, b):
                changed_fields.append({"column": col, "old_value": a, "new_value": b})
                counts[col] += 1
        if changed_fields:
            key_values = idx if isinstance(idx, tuple) else (idx,)
            changes.append({"key": dict(zip(keys, key_values)), "changed_fields": changed_fields})

    dtype_changes = {
        c: {"old_dtype": str(df_old[c].dtype), "new_dtype": str(df_new[c].dtype)}
        for c in common if str(df_old[c].dtype) != str(df_new[c].dtype)
    }
    drift = {}
    warnings = []
    for col in numeric:
        a, b = _finite(df_old[col]), _finite(df_new[col])
        if len(a) < 2 or len(b) < 2:
            warnings.append(f"Skipped drift test for '{col}': fewer than two finite numeric values in a snapshot.")
            continue
        test = ks_2samp(a.to_numpy(), b.to_numpy())
        old_mean, new_mean = float(a.mean()), float(b.mean())
        drift[col] = {
            "ks_statistic": float(test.statistic),
            "p_value": float(test.pvalue),
            "distribution_shift_detected": bool(test.pvalue < drift_alpha),
            "old_mean": old_mean, "new_mean": new_mean,
            "mean_difference": new_mean - old_mean,
            "old_median": float(a.median()), "new_median": float(b.median()),
            "old_count": len(a), "new_count": len(b),
            "old_excluded_rows": len(df_old) - len(a),
            "new_excluded_rows": len(df_new) - len(b),
            "ks_effect_size": float(test.statistic),
        }
    summary = {
        "added": len(added_idx), "deleted": len(deleted_idx),
        "modified": len(changes), "unchanged_shared": len(shared_idx) - len(changes),
        "shared": len(shared_idx),
    }
    return {
        "old_rows": len(df_old), "new_rows": len(df_new),
        "schema_changes": {
            "added_columns": sorted(new_cols - old_cols),
            "removed_columns": sorted(old_cols - new_cols),
            "dtype_changes": dtype_changes,
        },
        "added_records": new.loc[added_idx].reset_index(drop=True),
        "deleted_records": old.loc[deleted_idx].reset_index(drop=True),
        "modified_records": changes,
        "unchanged_shared_records": summary["unchanged_shared"],
        "change_summary": summary,
        "field_change_counts": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        "numeric_distribution_drift": drift,
        "warnings": warnings,
        "note": "Distribution tests compare full snapshots and are exploratory; assess KS effect size, sample coverage, multiple testing, and business context alongside p-values.",
    }
