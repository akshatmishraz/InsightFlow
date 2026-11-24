"""Statistical analysis and experimentation utilities for InsightFlow."""

import numpy as np
import pandas as pd
from scipy import stats


def _numeric_groups(df, group_column, value_column, groups):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    for col in (group_column, value_column):
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")
    if group_column == value_column:
        raise ValueError("Group and value columns must be different.")
    selected = df.loc[df[group_column].isin(groups), [group_column, value_column]].copy()
    selected[value_column] = pd.to_numeric(selected[value_column], errors="coerce")
    selected[value_column] = selected[value_column].replace([np.inf, -np.inf], np.nan)
    excluded = {
        str(group): int(((selected[group_column] == group) & selected[value_column].isna()).sum())
        for group in groups
    }
    selected = selected.dropna(subset=[group_column, value_column])
    arrays = {
        group: selected.loc[selected[group_column] == group, value_column].to_numpy(float)
        for group in groups
    }
    return arrays, excluded


def compare_two_groups(df, group_column, value_column, group_a, group_b, alpha=0.05):
    """Welch independent t-test; difference and CI are group_b minus group_a."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if group_a == group_b:
        raise ValueError("Choose two different groups.")
    arrays, excluded = _numeric_groups(df, group_column, value_column, [group_a, group_b])
    a, b = arrays[group_a], arrays[group_b]
    if min(len(a), len(b)) < 2:
        raise ValueError("Each group needs at least two valid numeric observations.")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    if va == 0 and vb == 0:
        raise ValueError("Both groups have zero variability; Welch's t-test is undefined.")
    test = stats.ttest_ind(a, b, equal_var=False)
    se2 = va / len(a) + vb / len(b)
    df_welch = se2**2 / ((va / len(a))**2 / (len(a)-1) + (vb / len(b))**2 / (len(b)-1))
    diff = float(np.mean(b) - np.mean(a))
    se = float(np.sqrt(se2))
    critical = stats.t.ppf(1-alpha/2, df_welch)
    pooled = np.sqrt(((len(a)-1)*va + (len(b)-1)*vb) / (len(a)+len(b)-2))
    d = float(diff / pooled) if pooled > 0 else None
    correction = 1 - 3 / (4*(len(a)+len(b)-2)-1)
    return {
        "test": "Welch's independent t-test",
        "group_a": group_a, "group_b": group_b,
        "n_a": len(a), "n_b": len(b),
        "excluded_non_numeric_or_missing": excluded,
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "mean_difference_b_minus_a": diff,
        "t_statistic": float(test.statistic), "degrees_of_freedom": float(df_welch),
        "p_value": float(test.pvalue), "alpha": alpha,
        "confidence_interval": (float(diff-critical*se), float(diff+critical*se)),
        "cohens_d_pooled": d,
        "hedges_g": float(d*correction) if d is not None else None,
        "statistically_significant": bool(test.pvalue < alpha),
        "interpretation": (
            "Evidence of a difference in group means at the selected significance level."
            if test.pvalue < alpha else
            "Insufficient evidence of a difference in group means at the selected significance level."
        ),
        "caveat": "Assumes independent observations; inspect distributions and outliers. Statistical significance does not establish causality."
    }


def correlation_analysis(df, columns, method="pearson"):
    """Pairwise correlation matrix and p-value matrix for numeric columns."""
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'.")
    if len(columns) < 2 or any(c not in df.columns for c in columns):
        raise ValueError("Select at least two existing columns.")
    data = df[columns].apply(pd.to_numeric, errors="coerce")
    corr = data.corr(method=method)
    pvals = pd.DataFrame(np.nan, index=columns, columns=columns)
    for i, x in enumerate(columns):
        for y in columns[i+1:]:
            pair = data[[x, y]].dropna()
            if len(pair) >= 3 and pair[x].nunique() > 1 and pair[y].nunique() > 1:
                test = stats.pearsonr(pair[x], pair[y]) if method == "pearson" else stats.spearmanr(pair[x], pair[y])
                pvals.loc[x, y] = pvals.loc[y, x] = float(test.pvalue)
    return {"method": method, "correlation": corr, "p_values": pvals}


def chi_square_test(df, column_a, column_b):
    """Chi-square test of independence for two categorical columns."""
    if column_a not in df or column_b not in df:
        raise ValueError("Both selected columns must exist.")
    table = pd.crosstab(df[column_a], df[column_b])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Each categorical variable must have at least two observed categories.")
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    denom = min(table.shape[0]-1, table.shape[1]-1)
    cramers_v = float(np.sqrt(chi2 / (n*denom))) if n and denom else np.nan
    return {"contingency_table": table, "chi2": float(chi2), "p_value": float(p),
            "degrees_of_freedom": int(dof), "expected_frequencies": pd.DataFrame(expected, index=table.index, columns=table.columns),
            "cramers_v": cramers_v}


def one_way_anova(df, group_column, value_column):
    """One-way ANOVA across 3+ independent groups."""
    if group_column not in df or value_column not in df:
        raise ValueError("Selected columns must exist.")
    data = df[[group_column, value_column]].copy()
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    groups = [g[value_column].to_numpy(float) for _, g in data.groupby(group_column)]
    if len(groups) < 3 or any(len(g) < 2 for g in groups):
        raise ValueError("ANOVA requires at least three groups with two valid observations each.")
    f, p = stats.f_oneway(*groups)
    return {"test": "One-way ANOVA", "group_count": len(groups), "f_statistic": float(f), "p_value": float(p),
            "interpretation": "At least one group mean may differ." if p < 0.05 else "Insufficient evidence that group means differ."}
