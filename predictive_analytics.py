"""Validated regression and baseline autoregressive forecasting utilities."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def _fraction(value, name):
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or not 0 < value < 0.5:
        raise ValueError(f"{name} must be a finite number between 0 and 0.5.")


def _columns(df, names):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if not df.columns.is_unique:
        raise ValueError("Duplicate DataFrame column names are unsupported.")
    if len(names) != len(set(names)) or any(c not in df.columns for c in names):
        raise ValueError("Columns must be distinct and exist in the DataFrame.")


def regression_analysis(df, feature_columns, target_column, test_size=0.2, random_state=42, split_strategy="random"):
    """Fit numeric linear regression; random split by default, ordered split when requested."""
    features = list(feature_columns) if feature_columns is not None else []
    _columns(df, features + [target_column])
    if not features:
        raise ValueError("At least one feature column is required.")
    _fraction(test_size, "test_size")
    if split_strategy not in ("random", "ordered"):
        raise ValueError("split_strategy must be 'random' or 'ordered'.")
    numeric = df[features + [target_column]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = numeric.notna().all(axis=1).to_numpy()
    clean = numeric.iloc[np.flatnonzero(valid)]
    if len(clean) < 10:
        raise ValueError("At least 10 complete rows are required for regression.")
    X = clean[features].to_numpy(dtype=float)
    y = clean[target_column].to_numpy(dtype=float)
    positions = np.flatnonzero(valid)
    if split_strategy == "random":
        train_idx, test_idx = train_test_split(np.arange(len(clean)), test_size=test_size, random_state=random_state)
    else:
        split = int(len(clean) * (1 - test_size))
        train_idx, test_idx = np.arange(split), np.arange(split, len(clean))
    if len(train_idx) < 2 or len(test_idx) < 2:
        raise ValueError("Not enough rows for train/test split.")
    model = LinearRegression().fit(X[train_idx], y[train_idx])
    predicted = model.predict(X[test_idx]); actual = y[test_idx]
    baseline = np.full(len(test_idx), y[train_idx].mean())
    nz = np.abs(actual) > 1e-12
    mape = float(np.mean(np.abs((actual[nz] - predicted[nz]) / actual[nz])) * 100) if nz.any() else None
    pred_df = pd.DataFrame({"actual": actual, "predicted": predicted}, index=df.index[positions[test_idx]])
    pred_df["residual"] = pred_df["actual"] - pred_df["predicted"]
    return {"model": model, "feature_columns": features, "target_column": target_column,
            "train_rows": len(train_idx), "test_rows": len(test_idx), "rows_excluded": len(df)-len(clean),
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted)), "mape_percent": mape,
            "baseline_mae": float(mean_absolute_error(actual, baseline)),
            "baseline_rmse": float(np.sqrt(mean_squared_error(actual, baseline))),
            "predictions": pred_df, "coefficients": pd.Series(model.coef_, index=features),
            "intercept": float(model.intercept_), "split_strategy": split_strategy,
            "note": "Random split is appropriate only for independent observations. For time-dependent data, sort chronologically and use split_strategy='ordered'. Avoid leakage from features derived using future or target information."}


def forecast_time_series(df, date_column, value_column, periods=7, frequency=None, test_size=0.2):
    """Three-lag linear autoregression with recursive holdout backtest."""
    _columns(df, [date_column, value_column])
    if isinstance(periods, (bool, str)) or not isinstance(periods, (int, np.integer)) or periods < 1:
        raise ValueError("periods must be a positive integer.")
    _fraction(test_size, "test_size")
    data = df[[date_column, value_column]].copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    data = data.dropna().sort_values(date_column)
    if data.empty:
        raise ValueError("No valid dated numeric observations.")
    if data[date_column].duplicated().any():
        data = data.groupby(date_column, as_index=False)[value_column].mean()
    if len(data) < 12:
        raise ValueError("At least 12 dated observations are required.")
    if frequency is None:
        frequency = pd.infer_freq(data[date_column])
        if frequency is None:
            raise ValueError("Cannot infer a regular frequency; supply frequency explicitly.")
    try:
        series = data.set_index(date_column)[value_column].asfreq(frequency)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid or incompatible frequency.") from exc
    if len(series) < 12:
        raise ValueError("At least 12 regularly spaced observations are required.")
    # Interpolation is fitted only to training history, never using holdout values.
    split = int(len(series) * (1 - test_size))
    if split < 8 or len(series) - split < 2:
        raise ValueError("Insufficient training or holdout observations.")
    train = series.iloc[:split].interpolate(method="time").ffill().bfill()
    if train.isna().any():
        raise ValueError("Training history contains no usable numeric values.")
    lag = 3
    def fit(values):
        X = np.array([values[i-lag:i] for i in range(lag, len(values))])
        y = np.asarray(values[lag:])
        return LinearRegression().fit(X, y)
    def recursive(model, history, count):
        history = list(history)
        future = []
        for _ in range(count):
            val = float(model.predict(np.array(history[-lag:]).reshape(1, -1))[0])
            future.append(val); history.append(val)
        return np.array(future)
    eval_model = fit(train.to_numpy(float))
    test_pred = recursive(eval_model, train.to_numpy(float), len(series)-split)
    observed = series.iloc[split:]
    valid_test = observed.notna().to_numpy()
    if valid_test.sum() < 2:
        raise ValueError("At least two observed holdout values are required.")
    actual = observed.to_numpy(float)[valid_test]
    predicted = test_pred[valid_test]
    baseline = np.full(len(actual), train.iloc[-1])
    # Fit final model on observed history with explicit interpolation for gaps.
    full = series.interpolate(method="time").ffill().bfill()
    final_model = fit(full.to_numpy(float))
    future = recursive(final_model, full.to_numpy(float), periods)
    future_dates = pd.date_range(start=series.index[-1], periods=periods+1, freq=frequency)[1:]
    return {"forecast": pd.DataFrame({date_column: future_dates, "forecast": future}),
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "baseline_mae": float(mean_absolute_error(actual, baseline)),
            "baseline_rmse": float(np.sqrt(mean_squared_error(actual, baseline))),
            "backtest": pd.DataFrame({date_column: observed.index[valid_test], "actual": actual, "predicted": predicted}),
            "train_rows": len(train), "test_rows": int(valid_test.sum()),
            "missing_periods": int(series.isna().sum()), "frequency": frequency,
            "model": "3-lag autoregressive linear regression",
            "note": "Recursive holdout backtest, not one-step teacher-forced evaluation. Baseline carries the last training value forward. Interpolation in the final fit is exploratory; check missing dates and seasonal effects before business use."}
