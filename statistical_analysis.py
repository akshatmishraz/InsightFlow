"""Statistical analysis and experimentation utilities for InsightFlow.

The functions in this module are UI-independent and return Python-native
summaries plus pandas objects that can be rendered by Streamlit.
"""

import numpy as np
import pandas as pd
from scipy import stats


def _validate_dataframe(df):
    """Raise a clear error unless df is a pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("The dataset has no rows.")


def _validate_columns(df, columns):
    """Validate that requested column names exist and are unambiguous."""
    if not isinstance(columns, (list, tuple)) or not columns:
        raise ValueError("Provide at least one column name.")
    if len(set(columns)) != len(columns):
        raise ValueError("Duplicate column selections are not allowed.")
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Column not found: {missing[0]}")


def _finite_numeric(series):
    """Convert values to numeric and treat infinities as missing."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def _numeric_groups(df, group_column, value_column, groups):
    """Return finite numeric observations and exclusion counts for each group."""
    _validate_dataframe(df)
    _validate_columns(df, [group_column, value_column])

    if group_column == value_column:
        raise ValueError("Group and value columns must be different.")
    if any(pd.isna(group) for group in groups):
        raise ValueError("Group labels cannot be missing.")
    if len(set(groups)) != len(groups):
        raise ValueError("Group labels must be different.")

    selected = df.loc[
        df[group_column].isin(groups), [group_column, value_column]
    ].copy()

    original_count = {
        group: int((selected[group_column] == group).sum()) for group in groups
    }
    selected[value_column] = _finite_numeric(selected[value_column])

    excluded = {
        str(group): original_count[group]
        - int(
            (
                (selected[group_column] == group)
                & selected[value_column].notna()
            ).sum()
        )
        for group in groups
    }

    selected = selected.dropna(subset=[group_column, value_column])
    arrays = {
        group: selected.loc[
            selected[group_column] == group, value_column
        ].to_numpy(dtype=float)
        for group in groups
    }
    return arrays, excluded


def compare_two_groups(
    df, group_column, value_column, group_a, group_b, alpha=0.05
):
    """Compare two independent groups using Welch's t-test.

    The estimated mean difference and confidence interval are group_b - group_a.
    """
    if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite number strictly between 0 and 1.")
    if group_a == group_b:
        raise ValueError("Choose two different groups.")

    arrays, excluded = _numeric_groups(
        df, group_column, value_column, [group_a, group_b]
    )
    a, b = arrays[group_a], arrays[group_b]

    if min(len(a), len(b)) < 2:
        raise ValueError(
            "Each group needs at least two valid numeric observations."
        )

    variance_a = float(np.var(a, ddof=1))
    variance_b = float(np.var(b, ddof=1))
    if variance_a == 0 and variance_b == 0:
        raise ValueError(
            "Both groups have zero variability; Welch's t-test is undefined."
        )

    test = stats.ttest_ind(a, b, equal_var=False)
    if not np.isfinite(test.statistic) or not np.isfinite(test.pvalue):
        raise ValueError(
            "Welch's t-test could not produce a finite result for these data."
        )

    n_a, n_b = len(a), len(b)
    variance_term_a = variance_a / n_a
    variance_term_b = variance_b / n_b
    standard_error_squared = variance_term_a + variance_term_b

    denominator = (
        variance_term_a**2 / (n_a - 1)
        + variance_term_b**2 / (n_b - 1)
    )
    if denominator <= 0 or standard_error_squared <= 0:
        raise ValueError("The selected data do not support a valid t-test.")

    degrees_freedom = standard_error_squared**2 / denominator
    difference = float(np.mean(b) - np.mean(a))
    standard_error = float(np.sqrt(standard_error_squared))
    critical_value = stats.t.ppf(1 - alpha / 2, degrees_freedom)

    pooled_variance = (
        (n_a - 1) * variance_a + (n_b - 1) * variance_b
    ) / (n_a + n_b - 2)
    pooled_standard_deviation = float(np.sqrt(pooled_variance))
    cohens_d = (
        float(difference / pooled_standard_deviation)
        if pooled_standard_deviation > 0
        else None
    )

    correction_denominator = 4 * (n_a + n_b - 2) - 1
    hedges_correction = (
        1 - 3 / correction_denominator
        if correction_denominator > 0
        else None
    )
    hedges_g = (
        float(cohens_d * hedges_correction)
        if cohens_d is not None and hedges_correction is not None
        else None
    )

    p_value = float(test.pvalue)
    significant = bool(p_value < alpha)

    return {
        "test": "Welch's independent t-test",
        "group_a": group_a,
        "group_b": group_b,
        "n_a": n_a,
        "n_b": n_b,
        "excluded_non_numeric_or_missing": excluded,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "std_a": float(np.std(a, ddof=1)),
        "std_b": float(np.std(b, ddof=1)),
        "mean_difference_b_minus_a": difference,
        "t_statistic": float(test.statistic),
        "degrees_of_freedom": float(degrees_freedom),
        "p_value": p_value,
        "alpha": float(alpha),
        "confidence_interval": (
            float(difference - critical_value * standard_error),
            float(difference + critical_value * standard_error),
        ),
        "cohens_d_pooled": cohens_d,
        "hedges_g": hedges_g,
        "statistically_significant": significant,
        "interpretation": (
            "Evidence of a difference in group means at the selected "
            "significance level."
            if significant
            else "Insufficient evidence of a difference in group means at "
            "the selected significance level."
        ),
        "caveat": (
            "Assumes independent observations. Inspect distributions and "
            "outliers; statistical significance does not establish causality."
        ),
    }


def correlation_analysis(df, columns, method="pearson"):
    """Return pairwise correlation, p-value, and valid-pair-count matrices.

    P-values are calculated only when at least three finite paired observations
    exist and both variables have non-zero variation.
    """
    _validate_dataframe(df)
    _validate_columns(df, columns)

    if len(columns) < 2:
        raise ValueError("Select at least two columns.")
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'.")

    data = pd.DataFrame(
        {column: _finite_numeric(df[column]) for column in columns},
        index=df.index,
    )
    correlation = data.corr(method=method, min_periods=2)
    p_values = pd.DataFrame(np.nan, index=columns, columns=columns)
    pair_counts = pd.DataFrame(0, index=columns, columns=columns, dtype=int)

    for column in columns:
        valid_count = int(data[column].notna().sum())
        pair_counts.loc[column, column] = valid_count

    for index, column_x in enumerate(columns):
        for column_y in columns[index + 1:]:
            pair = data[[column_x, column_y]].dropna()
            count = len(pair)
            pair_counts.loc[column_x, column_y] = count
            pair_counts.loc[column_y, column_x] = count

            if (
                count >= 3
                and pair[column_x].nunique() > 1
                and pair[column_y].nunique() > 1
            ):
                if method == "pearson":
                    test = stats.pearsonr(pair[column_x], pair[column_y])
                else:
                    test = stats.spearmanr(pair[column_x], pair[column_y])
                p_values.loc[column_x, column_y] = float(test.pvalue)
                p_values.loc[column_y, column_x] = float(test.pvalue)

    return {
        "method": method,
        "correlation": correlation,
        "p_values": p_values,
        "pair_counts": pair_counts,
    }


def chi_square_test(df, column_a, column_b):
    """Test independence between two categorical columns.

    Missing rows are excluded. A warning is returned when expected cell counts
    are small, because the chi-square approximation may then be unreliable.
    """
    _validate_dataframe(df)
    _validate_columns(df, [column_a, column_b])

    if column_a == column_b:
        raise ValueError("Choose two different categorical columns.")

    valid = df[[column_a, column_b]].dropna()
    excluded_rows = int(len(df) - len(valid))
    table = pd.crosstab(valid[column_a], valid[column_b])

    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError(
            "Each categorical variable must have at least two observed "
            "categories after excluding missing rows."
        )

    chi2, p_value, degrees_freedom, expected = stats.chi2_contingency(table)
    expected_frame = pd.DataFrame(
        expected, index=table.index, columns=table.columns
    )

    sample_size = int(table.to_numpy().sum())
    denominator = min(table.shape[0] - 1, table.shape[1] - 1)
    cramers_v = (
        float(np.sqrt(chi2 / (sample_size * denominator)))
        if sample_size > 0 and denominator > 0
        else np.nan
    )

    small_expected = int((expected < 5).sum())
    total_cells = int(expected.size)
    small_expected_share = (
        float(small_expected / total_cells) if total_cells else 0.0
    )
    warnings = []
    if (expected < 1).any():
        warnings.append(
            "At least one expected cell count is below 1; the chi-square "
            "approximation may be unreliable."
        )
    elif small_expected_share > 0.20:
        warnings.append(
            "More than 20% of expected cell counts are below 5; consider "
            "combining justified categories or using an exact test."
        )

    return {
        "contingency_table": table,
        "chi2": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(degrees_freedom),
        "expected_frequencies": expected_frame,
        "cramers_v": cramers_v,
        "sample_size": sample_size,
        "excluded_rows": excluded_rows,
        "expected_cells_below_5": small_expected,
        "expected_cells_below_5_share": small_expected_share,
        "warnings": warnings,
    }


def one_way_anova(df, group_column, value_column, alpha=0.05):
    """Run classical one-way ANOVA across at least three independent groups.

    This is the equal-variance (classical) ANOVA, not Welch's ANOVA. It tests
    whether all group means are equal; a significant result does not identify
    which groups differ.
    """
    _validate_dataframe(df)
    _validate_columns(df, [group_column, value_column])

    if group_column == value_column:
        raise ValueError("Group and value columns must be different.")
    if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite number strictly between 0 and 1.")

    data = df[[group_column, value_column]].copy()
    data[value_column] = _finite_numeric(data[value_column])
    data = data.dropna(subset=[group_column, value_column])

    grouped = list(data.groupby(group_column, sort=False, observed=True))
    groups = [group[value_column].to_numpy(dtype=float) for _, group in grouped]

    if len(groups) < 3 or any(len(group) < 2 for group in groups):
        raise ValueError(
            "ANOVA requires at least three groups with two valid numeric "
            "observations each."
        )

    within_group_sum_squares = sum(
        float(np.sum((group - np.mean(group)) ** 2)) for group in groups
    )
    if within_group_sum_squares == 0:
        raise ValueError(
            "There is no within-group variability; the ANOVA F-statistic "
            "is undefined."
        )

    f_statistic, p_value = stats.f_oneway(*groups)
    if not np.isfinite(f_statistic) or not np.isfinite(p_value):
        raise ValueError(
            "ANOVA could not produce a finite result for these data."
        )

    significant = bool(p_value < alpha)
    group_summary = pd.DataFrame(
        [
            {
                "group": label,
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
            }
            for (label, _), values in zip(grouped, groups)
        ]
    )

    return {
        "test": "One-way ANOVA",
        "group_count": len(groups),
        "sample_size": int(sum(len(group) for group in groups)),
        "f_statistic": float(f_statistic),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "degrees_of_freedom_between": len(groups) - 1,
        "degrees_of_freedom_within": int(sum(len(group) for group in groups) - len(groups)),
        "statistically_significant": significant,
        "group_summary": group_summary,
        "interpretation": (
            "Evidence that at least one group mean differs at the selected "
            "significance level."
            if significant
            else "Insufficient evidence that group means differ at the "
            "selected significance level."
        ),
        "caveat": (
            "Classical ANOVA assumes independent observations, approximately "
            "normal residuals within groups, and similar group variances. "
            "A significant result does not identify which groups differ."
        ),
    }
