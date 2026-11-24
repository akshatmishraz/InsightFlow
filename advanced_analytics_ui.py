"""Streamlit presentation layer for InsightFlow's five analytics engines.

Never mutates the source dataset. Import engines lazily so the Home/Cleaner pages
still load even if an optional analytics dependency is missing.
"""
from io import BytesIO
import json
import numpy as np
import pandas as pd
import hashlib
import streamlit as st


def _csv_download(data, name, label="Download CSV"):
    if isinstance(data, pd.Series):
        data = data.rename_axis("feature").reset_index(name="value")
    st.download_button(label, data.to_csv(index=False).encode("utf-8-sig"), name,
                       mime="text/csv")


def _table(data, label=None):
    if label:
        st.subheader(label)
    if isinstance(data, pd.Series):
        data = data.rename_axis("name").reset_index(name="value")
    st.dataframe(data, use_container_width=True)


def _metrics(result, names):
    cols = st.columns(len(names))
    for col, (key, title) in zip(cols, names):
        value = result.get(key)
        if isinstance(value, (float, np.floating)):
            value = f"{value:,.4g}"
        col.metric(title, str(value) if value is not None else "N/A")


def _numeric(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _dataset_fingerprint(df):
    """Hash content and schema to invalidate reports after cleaning changes."""
    try:
        digest = hashlib.sha256()
        digest.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
        digest.update(repr(list(zip(map(str, df.columns), map(str, df.dtypes)))).encode())
        return digest.hexdigest()
    except (TypeError, ValueError):
        return hashlib.sha256(df.to_csv(index=True).encode('utf-8')).hexdigest()


def _record(kind, result):
    """Store only the latest analysis for reporting; invalidate on dataset changes."""
    st.session_state["insightflow_latest_analysis"] = {
        "kind": kind, "result": result,
        "dataset_signature": st.session_state.get("dataset_signature"),
        "source": st.session_state.get("insightflow_analysis_source"),
        "data_fingerprint": st.session_state.get("insightflow_analysis_fingerprint"),
    }


def _run(action):
    try:
        action()
    except (ValueError, TypeError, KeyError, ImportError, RuntimeError) as exc:
        st.error(f"Analysis could not be completed: {exc}")


def _statistics(df):
    from statistical_analysis import compare_two_groups, correlation_analysis, chi_square_test, one_way_anova
    kind = st.selectbox("Analysis", ["Correlation", "Welch t-test", "Chi-square", "One-way ANOVA"])
    cols = list(df.columns)
    if kind == "Correlation":
        selected = st.multiselect("Numeric columns", _numeric(df), default=_numeric(df)[:3])
        method = st.selectbox("Correlation method", ["pearson", "spearman"])
        if st.button("Run correlation", type="primary"):
            def task():
                r = correlation_analysis(df, selected, method=method)
                _record("Correlation", r)
                _table(r["correlation"], "Correlation matrix")
                if "p_values" in r: _table(r["p_values"], "P-values")
                if "pair_counts" in r: _table(r["pair_counts"], "Pairwise sample counts")
                _csv_download(r["correlation"].reset_index(), "correlation.csv")
            _run(task)
    elif kind in ("Welch t-test", "One-way ANOVA"):
        group = st.selectbox("Group column", cols)
        value = st.selectbox("Numeric outcome", _numeric(df)) if _numeric(df) else None
        if kind == "Welch t-test":
            groups = df[group].dropna().unique().tolist()
            if len(groups) > 100:
                st.warning("Group selection is limited to 100 distinct values.")
                groups = groups[:100]
            a = st.selectbox("Group A", groups, key="stat_group_a") if groups else None
            b = st.selectbox("Group B", groups, index=min(1, len(groups)-1), key="stat_group_b") if groups else None
            alpha = st.number_input("Significance level", min_value=0.001, max_value=0.2, value=0.05)
            if st.button("Run t-test", type="primary"):
                def task():
                    if a == b: raise ValueError("Choose two different groups.")
                    r = compare_two_groups(df, group, value, a, b, alpha=alpha)
                    _record("Welch t-test", r)
                    st.json({k:v for k,v in r.items() if isinstance(v,(str,int,float,bool,list,tuple,dict,type(None)))}, expanded=True)
                _run(task)
        elif st.button("Run ANOVA", type="primary"):
            def task():
                r = one_way_anova(df, group, value)
                _record("One-way ANOVA", r)
                st.json({k:v for k,v in r.items() if isinstance(v,(str,int,float,bool,list,tuple,dict,type(None)))})
                if "group_summary" in r: _table(r["group_summary"], "Group summary")
            _run(task)
    else:
        a = st.selectbox("First category", cols)
        b = st.selectbox("Second category", cols, index=min(1,len(cols)-1))
        if st.button("Run chi-square", type="primary"):
            def task():
                if a == b: raise ValueError("Choose different columns.")
                r = chi_square_test(df,a,b)
                _record("Chi-square", r)
                st.json({k:v for k,v in r.items() if isinstance(v,(str,int,float,bool,list,tuple,dict,type(None)))})
                for k in ("observed", "expected", "contingency_table"):
                    if k in r: _table(r[k], k.replace('_',' ').title())
            _run(task)


def _anomalies(df):
    from anomaly_detection import detect_univariate_anomalies, detect_multivariate_anomalies
    mode = st.radio("Detection", ["Single column", "Multiple columns"], horizontal=True)
    nums = _numeric(df)
    if not nums:
        st.warning("Select a dataset with numeric columns.")
        return
    if mode == "Single column":
        col = st.selectbox("Column", nums)
        method = st.selectbox("Method", ["iqr", "zscore"])
        threshold = st.number_input("Threshold", min_value=0.1, value=1.5 if method == "iqr" else 3.0)
        if st.button("Detect anomalies", type="primary"):
            def task():
                r = detect_univariate_anomalies(df, col, method=method, threshold=threshold)
                _record("Anomaly Detection", r)
                _show_anomalies(r)
            _run(task)
    else:
        features = st.multiselect("Numeric features", nums, default=nums[:2])
        contamination = st.slider("Expected anomaly fraction", 0.01, 0.49, 0.05, 0.01)
        if st.button("Detect anomalies", type="primary"):
            def task():
                r = detect_multivariate_anomalies(df, features, contamination=contamination)
                _record("Anomaly Detection", r)
                _show_anomalies(r)
            _run(task)


def _show_anomalies(r):
    _metrics(r, [("anomaly_count","Anomalies"),("rows_scored","Rows scored")])
    for warning in r.get("warnings",[]): st.warning(str(warning))
    result = r["data"]
    _table(result[result["is_anomaly"].fillna(False)], "Flagged records")
    _csv_download(result, "anomaly_results.csv", "Download scored dataset")
    if r.get("diagnostic_note"): st.caption(r["diagnostic_note"])


def _compare(df, loader):
    from dataset_comparison import compare_datasets
    st.info("The active dataset is the OLD snapshot. Upload the NEW snapshot below.")
    uploaded = st.file_uploader("New dataset", type=["csv","xlsx","xls"], key="comparison_upload")
    if uploaded is None: return
    try:
        newer = loader(uploaded.getvalue(), uploaded.name)
    except (ValueError, TypeError, OSError) as exc:
        st.error(f"Cannot load comparison file: {exc}")
        return
    common = [c for c in df.columns if c in newer.columns]
    keys = st.multiselect("Unique key column(s)", common)
    numeric = st.multiselect("Numeric columns for drift", [c for c in common if c in _numeric(df) and c in _numeric(newer)])
    if st.button("Compare datasets", type="primary"):
        def task():
            r = compare_datasets(df, newer, keys, numeric)
            _record("Dataset Comparison", r)
            st.json(r["schema_changes"])
            _metrics({"added":len(r["added_records"]),"deleted":len(r["deleted_records"]),
                      "modified":len(r["modified_records"]),"unchanged":r["unchanged_shared_records"]},
                     [("added","Added"),("deleted","Deleted"),("modified","Modified"),("unchanged","Unchanged")])
            for key, title in [("added_records","Added rows"),("deleted_records","Deleted rows")]:
                _table(r[key], title)
                _csv_download(r[key], key+".csv")
            changes = []
            for item in r["modified_records"]:
                for change in item["changed_fields"]:
                    changes.append({**item["key"], **change})
            if changes:
                changed = pd.DataFrame(changes)
                _table(changed, "Modified fields")
                _csv_download(changed,"modified_fields.csv")
            if r.get("numeric_distribution_drift"):
                _table(pd.DataFrame.from_dict(r["numeric_distribution_drift"], orient="index"), "Numerical drift")
        _run(task)


def _predict(df):
    from predictive_analytics import regression_analysis, forecast_time_series
    mode = st.radio("Task", ["Regression", "Time-series forecast"], horizontal=True)
    nums = _numeric(df)
    if mode == "Regression":
        target = st.selectbox("Target", nums) if nums else None
        features = st.multiselect("Features", [c for c in nums if c != target])
        test_size = st.slider("Test fraction", 0.1, 0.4, 0.2, 0.05)
        if st.button("Train regression", type="primary"):
            def task():
                r = regression_analysis(df, features, target, test_size=test_size)
                _record("Regression", r)
                _metrics(r, [("mae","MAE"),("rmse","RMSE"),("r2","R²")])
                _table(r["predictions"], "Holdout predictions")
                _csv_download(r["predictions"], "regression_predictions.csv")
                _table(r["coefficients"], "Coefficients")
                st.caption(r.get("note", ""))
            _run(task)
    else:
        date_col = st.selectbox("Date column", list(df.columns))
        value_col = st.selectbox("Value column", nums) if nums else None
        periods = st.number_input("Periods ahead", min_value=1, max_value=365, value=7)
        frequency = st.selectbox("Frequency", ["Infer automatically","D","W","MS","h"])
        if st.button("Generate forecast", type="primary"):
            def task():
                r = forecast_time_series(df,date_col,value_col,periods=int(periods),
                                         frequency=None if frequency=="Infer automatically" else frequency)
                _record("Forecast", r)
                _metrics(r, [("mae","Backtest MAE"),("rmse","Backtest RMSE")])
                _table(r["forecast"], "Future forecast")
                st.line_chart(r["forecast"].set_index(date_col)["forecast"])
                _csv_download(r["forecast"], "forecast.csv")
                st.caption(r.get("note", ""))
            _run(task)


def _segments(df):
    from business_segmentation import segment_dataset
    nums = _numeric(df)
    features = st.multiselect("Numeric segmentation features", nums, default=nums[:2])
    clusters = st.number_input("Number of segments", min_value=2, max_value=30, value=3)
    method = st.selectbox("Algorithm", ["kmeans","hierarchical"])
    pca = st.checkbox("Use PCA for clustering", value=False,
                      help="Changes the clustering feature space; use only when appropriate.")
    if st.button("Build segments", type="primary"):
        def task():
            r = segment_dataset(df,features,n_clusters=int(clusters),method=method,use_pca=pca)
            _record("Business Segmentation", r)
            _table(r["cluster_sizes"], "Segment sizes")
            _table(r["profile"], "Segment profiles")
            for metric in ("silhouette_score","davies_bouldin_score"):
                if metric in r: st.metric(metric.replace('_',' ').title(), f"{r[metric]:.3f}" if r[metric] is not None else "N/A")
            _table(r["data"].head(100), "Segment assignments (preview)")
            _csv_download(r["data"], "segmented_data.csv", "Download all assignments")
            st.caption(r.get("note", ""))
        _run(task)


def render_advanced_analytics(load_file):
    """Render analytics without modifying st.session_state's source datasets."""
    st.title("📈 Advanced Analytics")
    if "df" not in st.session_state:
        st.info("First load a dataset in Smart Data Cleaner, then return here.")
        return
    source = st.radio("Dataset to analyze", ["Cleaned dataset", "Original dataset"], horizontal=True)
    key = "cleaned_df" if source == "Cleaned dataset" else "df"
    st.session_state["insightflow_analysis_source"] = source
    df = st.session_state.get(key)
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.warning("The selected dataset has no rows.")
        return
    st.session_state["insightflow_analysis_fingerprint"] = _dataset_fingerprint(df)
    st.caption(f"{st.session_state.get('file_name','Dataset')} · {len(df):,} rows · {len(df.columns):,} columns · read-only analysis")
    if not df.columns.is_unique:
        st.error("Duplicate column names must be resolved before analytics.")
        return
    module = st.selectbox("Choose an analytics module", ["Statistical Analysis", "Anomaly Detection",
                          "Dataset Comparison", "Predictive Analytics", "Business Segmentation"])
    if module == "Statistical Analysis": _statistics(df)
    elif module == "Anomaly Detection": _anomalies(df)
    elif module == "Dataset Comparison": _compare(df, load_file)
    elif module == "Predictive Analytics": _predict(df)
    else: _segments(df)
