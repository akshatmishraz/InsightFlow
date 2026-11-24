"""Dataset schema, key-based reconciliation, and numerical drift comparison."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def compare_datasets(df_old, df_new, key_columns, numeric_columns=None, drift_alpha=0.05):
    if not isinstance(df_old, pd.DataFrame) or not isinstance(df_new, pd.DataFrame):
        raise TypeError("Both inputs must be pandas DataFrames.")
    if not key_columns or any(c not in df_old or c not in df_new for c in key_columns):
        raise ValueError("Key columns must be supplied and exist in both datasets.")
    if df_old.duplicated(key_columns).any() or df_new.duplicated(key_columns).any():
        raise ValueError("Key columns must uniquely identify rows in both datasets.")
    old_cols, new_cols = set(df_old.columns), set(df_new.columns)
    common = sorted(old_cols & new_cols)
    old = df_old.set_index(key_columns, drop=False)
    new = df_new.set_index(key_columns, drop=False)
    old_keys, new_keys = old.index, new.index
    added_idx = new_keys.difference(old_keys)
    deleted_idx = old_keys.difference(new_keys)
    shared_idx = old_keys.intersection(new_keys)
    compare_cols = [c for c in common if c not in key_columns]
    changes = []
    for idx in shared_idx:
        old_row, new_row = old.loc[idx], new.loc[idx]
        changed_fields = []
        for col in compare_cols:
            a, b = old_row[col], new_row[col]
            equal = (pd.isna(a) and pd.isna(b)) or a == b
            if not equal:
                changed_fields.append({"column": col, "old_value": a, "new_value": b})
        if changed_fields:
            key_value = idx if isinstance(idx, tuple) else (idx,)
            changes.append({"key": dict(zip(key_columns, key_value)), "changed_fields": changed_fields})
    schema = {"added_columns": sorted(new_cols-old_cols), "removed_columns": sorted(old_cols-new_cols)}
    drift = {}
    for col in (numeric_columns or []):
        if col not in common:
            continue
        a = pd.to_numeric(df_old[col], errors="coerce").dropna()
        b = pd.to_numeric(df_new[col], errors="coerce").dropna()
        if len(a) >= 2 and len(b) >= 2:
            test = ks_2samp(a, b)
            drift[col] = {"ks_statistic": float(test.statistic), "p_value": float(test.pvalue),
                          "distribution_shift_detected": bool(test.pvalue < drift_alpha),
                          "old_mean": float(a.mean()), "new_mean": float(b.mean()),
                          "old_count": len(a), "new_count": len(b)}
    return {
        "old_rows": len(df_old), "new_rows": len(df_new),
        "schema_changes": schema,
        "added_records": new.loc[added_idx].reset_index(drop=True),
        "deleted_records": old.loc[deleted_idx].reset_index(drop=True),
        "modified_records": changes,
        "unchanged_shared_records": len(shared_idx)-len(changes),
        "numeric_distribution_drift": drift,
        "note": "Distribution tests are exploratory; interpret p-values with sample size and business context."
    }
