import numpy as np
import pandas as pd
import pytest

from statistical_analysis import (
    compare_two_groups,
    correlation_analysis,
    chi_square_test,
    one_way_anova,
)
from anomaly_detection import detect_univariate_anomalies, detect_multivariate_anomalies
from dataset_comparison import compare_datasets
from predictive_analytics import regression_analysis, forecast_time_series
from business_segmentation import segment_dataset


# ---------------------------------------------------------------------------
# Statistical Analysis
# ---------------------------------------------------------------------------

def test_welch_ttest():
    df = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 4, "y": [1, 2, 2, 3, 6, 7, 8, 9]})
    result = compare_two_groups(df, "g", "y", "a", "b")
    assert result["mean_difference_b_minus_a"] == pytest.approx(5.5)
    assert 0 <= result["p_value"] <= 1
    assert result["confidence_interval"][0] < result["confidence_interval"][1]
    assert result["n_a"] == 4
    assert result["n_b"] == 4


def test_welch_excludes_missing_nonnumeric_and_infinite_values():
    df = pd.DataFrame({
        "g": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "y": [1, 2, None, "bad", 5, 6, np.inf, 7],
    })
    result = compare_two_groups(df, "g", "y", "a", "b")
    assert result["n_a"] == 2
    assert result["n_b"] == 3
    assert result["excluded_non_numeric_or_missing"] == {"a": 2, "b": 1}


def test_welch_rejects_insufficient_group_size():
    df = pd.DataFrame({"g": ["a", "a", "b"], "y": [1, 2, 3]})
    with pytest.raises(ValueError, match="two valid numeric"):
        compare_two_groups(df, "g", "y", "a", "b")


def test_welch_rejects_two_constant_groups():
    df = pd.DataFrame({"g": ["a"] * 3 + ["b"] * 3, "y": [2] * 3 + [5] * 3})
    with pytest.raises(ValueError, match="zero variability"):
        compare_two_groups(df, "g", "y", "a", "b")


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1, np.nan])
def test_welch_rejects_invalid_alpha(alpha):
    df = pd.DataFrame({"g": ["a", "a", "b", "b"], "y": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="alpha"):
        compare_two_groups(df, "g", "y", "a", "b", alpha=alpha)


def test_correlations_and_pair_counts():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
    result = correlation_analysis(df, ["x", "y"])
    assert result["correlation"].loc["x", "y"] == pytest.approx(1)
    assert result["pair_counts"].loc["x", "y"] == 4
    assert result["p_values"].loc["x", "y"] == pytest.approx(0)


def test_correlation_handles_missing_and_constant_values():
    df = pd.DataFrame({
        "x": [1, 2, 3, np.nan, 5],
        "y": [2, 4, 6, 8, np.inf],
        "constant": [7, 7, 7, 7, 7],
    })
    result = correlation_analysis(df, ["x", "y", "constant"])
    assert result["pair_counts"].loc["x", "y"] == 3
    assert result["correlation"].loc["x", "y"] == pytest.approx(1)
    assert pd.isna(result["p_values"].loc["x", "constant"])


def test_correlation_rejects_duplicate_columns():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    with pytest.raises(ValueError, match="Duplicate"):
        correlation_analysis(df, ["x", "x", "y"])


def test_chi_square_reports_missing_rows_and_small_expected_counts():
    df = pd.DataFrame({
        "a": ["x", "x", "y", "y", None],
        "b": ["m", "m", "n", "n", "m"],
    })
    result = chi_square_test(df, "a", "b")
    assert result["chi2"] >= 0
    assert result["sample_size"] == 4
    assert result["excluded_rows"] == 1
    assert result["expected_cells_below_5"] == 4
    assert result["warnings"]


def test_chi_square_rejects_single_category():
    df = pd.DataFrame({"a": ["x", "x", "y"], "b": ["m", "m", "m"]})
    with pytest.raises(ValueError, match="two observed categories"):
        chi_square_test(df, "a", "b")


def test_anova_reports_group_summary_and_degrees_of_freedom():
    df = pd.DataFrame({
        "g": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
        "v": [1, 2, 3, 5, 6, 7, 9, 10, 11],
    })
    result = one_way_anova(df, "g", "v")
    assert result["p_value"] < 0.05
    assert result["group_count"] == 3
    assert result["degrees_of_freedom_between"] == 2
    assert result["degrees_of_freedom_within"] == 6
    assert len(result["group_summary"]) == 3


def test_anova_excludes_invalid_numeric_values():
    df = pd.DataFrame({
        "g": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "c"],
        "v": [1, 2, "bad", 5, 6, 7, 9, 10, 11, np.inf],
    })
    result = one_way_anova(df, "g", "v")
    assert result["sample_size"] == 8


def test_anova_rejects_zero_within_group_variability():
    df = pd.DataFrame({
        "g": ["a", "a", "b", "b", "c", "c"],
        "v": [1, 1, 2, 2, 3, 3],
    })
    with pytest.raises(ValueError, match="within-group variability"):
        one_way_anova(df, "g", "v")


def test_statistical_functions_validate_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        correlation_analysis([1, 2, 3], ["x", "y"])


# ---------------------------------------------------------------------------
# Existing tests for the other four analytics modules — retained unchanged
# ---------------------------------------------------------------------------

def test_iqr_anomaly():
    df = pd.DataFrame({"v": [1, 2, 2, 3, 4, 100]})
    result = detect_univariate_anomalies(df, "v")
    assert result["anomaly_count"] >= 1


def test_isolation_forest():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(60, 2))
    x[-1] = [9, 9]
    df = pd.DataFrame(x, columns=["x", "y"])
    result = detect_multivariate_anomalies(df, ["x", "y"], contamination=0.05)
    assert result["anomaly_count"] >= 1


def test_dataset_reconciliation():
    old = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
    new = pd.DataFrame({"id": [2, 3], "value": [25, 30], "extra": [1, 1]})
    result = compare_datasets(old, new, ["id"], ["value"])
    assert len(result["added_records"]) == 1
    assert len(result["deleted_records"]) == 1
    assert len(result["modified_records"]) == 1
    assert result["schema_changes"]["added_columns"] == ["extra"]


def test_regression():
    x = np.arange(30)
    df = pd.DataFrame({"x": x, "y": 2 * x + 3})
    result = regression_analysis(df, ["x"], "y")
    assert result["rmse"] < 1e-8


def test_forecast():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({"date": dates, "value": np.arange(30, dtype=float)})
    result = forecast_time_series(df, "date", "value", periods=5, frequency="D")
    assert len(result["forecast"]) == 5


def test_segmentation():
    df = pd.DataFrame({
        "x": [1, 1.1, 1.2, 9, 9.1, 9.2],
        "y": [1, 1.2, 0.9, 9, 8.9, 9.1],
    })
    result = segment_dataset(df, ["x", "y"], n_clusters=2)
    assert result["data"]["segment"].nunique() == 2

# ---------------------------------------------------------------------------
# Anomaly Detection — additional quality and regression tests
# ---------------------------------------------------------------------------

def test_iqr_scores_and_severity():
    df = pd.DataFrame({"v": [1, 2, 2, 3, 4, 100]})
    result = detect_univariate_anomalies(df, "v")
    assert result["anomaly_count"] >= 1
    assert result["data"].loc[5, "anomaly_score"] > 0
    assert result["data"].loc[5, "anomaly_severity"] in {"low", "medium", "high"}
    assert result["rows_scored"] == 6


def test_iqr_zero_spread_is_handled():
    df = pd.DataFrame({"v": [2, 2, 2, 2, 2, 50]})
    result = detect_univariate_anomalies(df, "v")
    assert result["anomaly_count"] == 1
    assert result["warnings"]
    assert result["data"].iloc[-1]["is_anomaly"]


def test_zscore_signed_score_and_missing_exclusions():
    df = pd.DataFrame({"v": [-100, 1, 2, 3, 4, 5, "bad", np.inf]})
    result = detect_univariate_anomalies(df, "v", method="zscore", threshold=1.5)
    assert result["rows_scored"] == 6
    assert result["excluded_rows"] == 2
    assert result["data"].iloc[0]["anomaly_score"] < 0
    assert not result["data"].iloc[6]["is_anomaly"]
    assert pd.isna(result["data"].iloc[7]["anomaly_score"])


@pytest.mark.parametrize("threshold", [0, -1, np.nan, np.inf, "oops", True])
def test_univariate_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        detect_univariate_anomalies(pd.DataFrame({"v": [1, 2, 3]}), "v", threshold=threshold)


def test_zscore_constant_column_rejected():
    with pytest.raises(ValueError, match="constant"):
        detect_univariate_anomalies(pd.DataFrame({"v": [2, 2, 2]}), "v", method="zscore")


def test_univariate_preserves_duplicate_index():
    df = pd.DataFrame({"v": [1, 1, 1, 2, 100]}, index=[1, 1, 1, 2, 2])
    result = detect_univariate_anomalies(df, "v")
    assert result["data"].index.tolist() == df.index.tolist()
    assert result["data"]["is_anomaly"].tolist()[-1]


def test_multivariate_exclusions_and_diagnostics():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.normal(size=(60, 2)), columns=["x", "y"])
    df.loc[59] = [12, 12]
    df.loc[58, "x"] = np.inf
    result = detect_multivariate_anomalies(df, ["x", "y"], contamination=0.05)
    assert result["rows_scored"] == 59
    assert result["excluded_rows"] == 1
    assert result["data"].loc[59, "is_anomaly"]
    assert result["data"].loc[59, "top_deviating_feature"] in {"x", "y"}
    assert result["data"].loc[58, "anomaly_severity"] == "not_scored"
    assert pd.isna(result["data"].loc[58, "anomaly_score"])
    assert "not a model attribution" in result["diagnostic_note"]


def test_multivariate_preserves_duplicate_index():
    rng = np.random.default_rng(4)
    df = pd.DataFrame(rng.normal(size=(30, 2)), columns=["x", "y"], index=[0] * 30)
    result = detect_multivariate_anomalies(df, ["x", "y"])
    assert result["data"].index.tolist() == df.index.tolist()
    assert result["data"]["anomaly_score"].notna().all()


@pytest.mark.parametrize("contamination", [0, 0.5, -1, np.nan, np.inf, "bad", True])
def test_multivariate_rejects_invalid_contamination(contamination):
    df = pd.DataFrame({"x": range(15), "y": range(15)})
    with pytest.raises(ValueError, match="contamination"):
        detect_multivariate_anomalies(df, ["x", "y"], contamination=contamination)


def test_multivariate_rejects_duplicate_features():
    df = pd.DataFrame({"x": range(15), "y": range(15)})
    with pytest.raises(ValueError, match="distinct"):
        detect_multivariate_anomalies(df, ["x", "x"])


def test_multivariate_rejects_all_constant_features():
    df = pd.DataFrame({"x": [1] * 15, "y": [2] * 15})
    with pytest.raises(ValueError, match="variation"):
        detect_multivariate_anomalies(df, ["x", "y"])


def test_anomaly_functions_reject_non_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        detect_univariate_anomalies([1, 2, 3], "v")
    with pytest.raises(TypeError, match="DataFrame"):
        detect_multivariate_anomalies([1, 2, 3], ["x", "y"])


def test_anomaly_functions_reject_reserved_output_columns():
    df = pd.DataFrame({"v": [1, 2, 3], "is_anomaly": [False] * 3})
    with pytest.raises(ValueError, match="Reserved"):
        detect_univariate_anomalies(df, "v")

# ---------------------------------------------------------------------------
# Business Segmentation — quality and regression tests
# ---------------------------------------------------------------------------

def test_segmentation_reports_quality_and_profiles():
    df = pd.DataFrame({"revenue": [1, 1.2, 1.1, 10, 10.2, 10.1],
                       "visits": [2, 2.1, 1.9, 20, 20.2, 19.8]})
    result = segment_dataset(df, ["revenue", "visits"], n_clusters=2)
    assert result["rows_scored"] == 6
    assert result["excluded_rows"] == 0
    assert result["observed_clusters"] == 2
    assert -1 <= result["silhouette_score"] <= 1
    assert result["davies_bouldin_score"] >= 0
    assert result["cluster_sizes"].sum() == 6
    assert set(result["top_distinguishing_features"]) == {0, 1}
    assert set(result["segment_mean_profile"].columns) == {"revenue", "visits"}


def test_segmentation_duplicate_index_and_missing_rows():
    df = pd.DataFrame({"x": [1, 1.2, 9, 9.2, np.inf, 1.1],
                       "y": [1, 1.1, 9, 9.1, 2, None]}, index=[1, 1, 2, 2, 2, 2])
    result = segment_dataset(df, ["x", "y"], n_clusters=2)
    assert result["rows_scored"] == 4
    assert result["excluded_rows"] == 2
    assert result["data"].index.tolist() == df.index.tolist()
    assert result["data"]["segment"].notna().tolist() == [True] * 4 + [False] * 2


def test_segmentation_hierarchical_and_pca():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(np.vstack([rng.normal(0, .1, (12, 3)), rng.normal(5, .1, (12, 3))]),
                      columns=["a", "b", "c"])
    result = segment_dataset(df, ["a", "b", "c"], n_clusters=2, method="hierarchical", use_pca=True)
    assert result["clustering_space"] == "pca"
    assert len(result["pca_explained_variance_ratio"]) == 2
    assert result["observed_clusters"] == 2
    assert result["pca"] is not None
    assert result["warnings"]


def test_segmentation_reproducible():
    rng = np.random.default_rng(10)
    df = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    a = segment_dataset(df, ["a", "b", "c"], n_clusters=3)
    b = segment_dataset(df, ["a", "b", "c"], n_clusters=3)
    assert a["data"]["segment"].tolist() == b["data"]["segment"].tolist()


def test_segmentation_rejects_all_constant_features():
    df = pd.DataFrame({"x": [1] * 15, "y": [2] * 15})
    with pytest.raises(ValueError, match="variation"):
        segment_dataset(df, ["x", "y"], n_clusters=2)


def test_segmentation_reports_constant_feature():
    df = pd.DataFrame({"x": [1, 1.1, 1.2, 8, 8.1, 8.2], "y": [5] * 6})
    result = segment_dataset(df, ["x", "y"], n_clusters=2)
    assert result["constant_features"] == ["y"]
    assert result["warnings"]


@pytest.mark.parametrize("n_clusters", [0, 1, 2.5, True, "3", np.nan])
def test_segmentation_invalid_cluster_count(n_clusters):
    df = pd.DataFrame({"x": range(15), "y": range(15)})
    with pytest.raises(ValueError, match="n_clusters"):
        segment_dataset(df, ["x", "y"], n_clusters=n_clusters)


@pytest.mark.parametrize("features", [[], ["x", "x"], ["x", "missing"], "x"])
def test_segmentation_invalid_features(features):
    df = pd.DataFrame({"x": range(15), "y": range(15)})
    with pytest.raises(ValueError):
        segment_dataset(df, features, n_clusters=2)


def test_segmentation_rejects_reserved_output():
    df = pd.DataFrame({"x": range(15), "y": range(15), "segment": range(15)})
    with pytest.raises(ValueError, match="Reserved"):
        segment_dataset(df, ["x", "y"], n_clusters=2)


def test_segmentation_rejects_duplicate_dataframe_columns():
    df = pd.DataFrame(np.arange(30).reshape(15, 2), columns=["x", "x"])
    with pytest.raises(ValueError, match="Duplicate"):
        segment_dataset(df, ["x"], n_clusters=2)


def test_segmentation_rejects_non_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        segment_dataset([1, 2, 3], ["x"], n_clusters=2)


def test_segmentation_does_not_mutate_input():
    df = pd.DataFrame({"x": [1, 1.1, 1.2, 9, 9.1, 9.2], "y": [1, 1.2, 1.1, 9, 9.2, 9.1]})
    original = df.copy(deep=True)
    segment_dataset(df, ["x", "y"], n_clusters=2)
    pd.testing.assert_frame_equal(df, original)
