import numpy as np
import pandas as pd
import pytest

from statistical_analysis import compare_two_groups, correlation_analysis, chi_square_test, one_way_anova
from anomaly_detection import detect_univariate_anomalies, detect_multivariate_anomalies
from dataset_comparison import compare_datasets
from predictive_analytics import regression_analysis, forecast_time_series
from business_segmentation import segment_dataset


def test_welch_ttest():
    df = pd.DataFrame({"g": ["a"]*4+["b"]*4, "y": [1, 2, 2, 3, 6, 7, 8, 9]})
    r = compare_two_groups(df, "g", "y", "a", "b")
    assert r["mean_difference_b_minus_a"] == pytest.approx(5.5)
    assert 0 <= r["p_value"] <= 1


def test_correlations():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
    assert correlation_analysis(df, ["x", "y"])["correlation"].loc["x", "y"] == pytest.approx(1)


def test_chi_square():
    df = pd.DataFrame({"a": ["x", "x", "y", "y"], "b": ["m", "m", "n", "n"]})
    assert chi_square_test(df, "a", "b")["chi2"] >= 0


def test_anova():
    df = pd.DataFrame({"g": ["a"]*3+["b"]*3+["c"]*3, "v": [1, 2, 3, 5, 6, 7, 9, 10, 11]})
    assert one_way_anova(df, "g", "v")["p_value"] < 0.05


def test_iqr_anomaly():
    df = pd.DataFrame({"v": [1, 2, 2, 3, 4, 100]})
    r = detect_univariate_anomalies(df, "v")
    assert r["anomaly_count"] >= 1


def test_isolation_forest():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(60, 2))
    x[-1] = [9, 9]
    df = pd.DataFrame(x, columns=["x", "y"])
    r = detect_multivariate_anomalies(df, ["x", "y"], contamination=0.05)
    assert r["anomaly_count"] >= 1


def test_dataset_reconciliation():
    old = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
    new = pd.DataFrame({"id": [2, 3], "value": [25, 30], "extra": [1, 1]})
    r = compare_datasets(old, new, ["id"], ["value"])
    assert len(r["added_records"]) == 1
    assert len(r["deleted_records"]) == 1
    assert len(r["modified_records"]) == 1
    assert r["schema_changes"]["added_columns"] == ["extra"]


def test_regression():
    x = np.arange(30)
    df = pd.DataFrame({"x": x, "y": 2*x+3})
    r = regression_analysis(df, ["x"], "y")
    assert r["rmse"] < 1e-8


def test_forecast():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({"date": dates, "value": np.arange(30, dtype=float)})
    r = forecast_time_series(df, "date", "value", periods=5, frequency="D")
    assert len(r["forecast"]) == 5


def test_segmentation():
    df = pd.DataFrame({"x": [1, 1.1, 1.2, 9, 9.1, 9.2], "y": [1, 1.2, 0.9, 9, 8.9, 9.1]})
    r = segment_dataset(df, ["x", "y"], n_clusters=2)
    assert r["data"]["segment"].nunique() == 2
