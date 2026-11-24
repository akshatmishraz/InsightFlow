# InsightFlow Advanced Analytics Extension

This extension adds five independently callable analytics capabilities while leaving the existing `app.py` unchanged:

1. Statistical analysis: Welch independent t-test, Pearson/Spearman correlation, chi-square, one-way ANOVA.
2. Anomaly detection: IQR, Z-score, Isolation Forest.
3. Dataset reconciliation: schema changes, key-based added/deleted/modified rows, numeric distribution drift.
4. Predictive analytics: multiple linear regression and a simple lag-based time-series baseline forecast.
5. Business segmentation: K-Means, hierarchical clustering, standardized features, optional PCA.

## Install

From the project root, merge the dependencies in `requirements-advanced.txt` into your existing `requirements.txt` (do not overwrite existing dependencies blindly), then run:

```bash
pip install -r requirements.txt
```

## Copy files into your existing InsightFlow repository

Copy these five `.py` files into the same folder as `app.py`:
- `statistical_analysis.py`
- `anomaly_detection.py`
- `dataset_comparison.py`
- `predictive_analytics.py`
- `business_segmentation.py`

Copy the `tests` folder into the project root. If you already have a tests folder, merge its contents.

## Test

From the project root:

```bash
pytest -v
```

## Important limitations

- Welch's t-test assumes independent observations. It is not for paired/repeated measurements.
- Correlation and significance tests do not prove causation.
- IQR/Z-score are univariate approaches; Isolation Forest is multivariate.
- Key-based reconciliation requires unique key columns in both datasets.
- The KS test is a useful exploratory numeric distribution comparison, not a universal drift detector.
- Regression uses a chronological row split as currently ordered; sort time-dependent data before calling it.
- The forecasting function is a simple autoregressive baseline. It is not a production seasonal forecasting system.
- Clustering is exploratory and requires human review of cluster profiles.
- These are analytical building blocks. They are not yet wired into Streamlit navigation; integrate only after tests pass.

## Deployment

After copying files and confirming tests pass, commit the new modules and tests to GitHub. Streamlit Community Cloud deploys the application through the existing `app.py`; the modules are imported by the app when the UI integration is added.
