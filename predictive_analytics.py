"""Regression and univariate time-series forecasting."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_analysis(df, feature_columns, target_column, test_size=0.2, random_state=42):
    if target_column not in df or not feature_columns or any(c not in df for c in feature_columns):
        raise ValueError("Target and feature columns must exist.")
    if target_column in feature_columns:
        raise ValueError("Target column cannot also be a feature.")
    if not 0 < test_size < 0.5:
        raise ValueError("test_size must be between 0 and 0.5.")
    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target_column], errors="coerce")
    valid = X.replace([np.inf, -np.inf], np.nan).notna().all(axis=1) & y.replace([np.inf, -np.inf], np.nan).notna()
    X, y = X.loc[valid], y.loc[valid]
    if len(X) < 10:
        raise ValueError("At least 10 complete rows are required for regression.")
    split = int(len(X)*(1-test_size))
    if split < 2 or len(X)-split < 2:
        raise ValueError("Not enough rows for train/test split.")
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = LinearRegression().fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    denominator = np.abs(y_test.to_numpy())
    nonzero = denominator > 1e-12
    mape = float(np.mean(np.abs((y_test.to_numpy()[nonzero]-predictions[nonzero])/denominator[nonzero]))*100) if nonzero.any() else None
    return {"model": model, "feature_columns": list(feature_columns), "target_column": target_column,
            "train_rows": len(X_train), "test_rows": len(X_test),
            "mae": float(mean_absolute_error(y_test, predictions)), "rmse": rmse,
            "r2": float(r2_score(y_test, predictions)), "mape_percent": mape,
            "predictions": pd.DataFrame({"actual": y_test, "predicted": predictions}, index=y_test.index),
            "coefficients": pd.Series(model.coef_, index=feature_columns),
            "intercept": float(model.intercept_),
            "note": "Rows are split in their existing order. Sort chronologically before use on time-dependent data."}


def forecast_time_series(df, date_column, value_column, periods=7, frequency=None, test_size=0.2):
    if date_column not in df or value_column not in df:
        raise ValueError("Date and value columns must exist.")
    if periods < 1:
        raise ValueError("periods must be at least 1.")
    data = df[[date_column, value_column]].copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data = data.dropna().sort_values(date_column)
    if data[date_column].duplicated().any():
        data = data.groupby(date_column, as_index=False)[value_column].mean()
    if len(data) < 12:
        raise ValueError("At least 12 dated observations are required for this baseline forecast.")
    if frequency is None:
        inferred = pd.infer_freq(data[date_column])
        frequency = inferred or pd.tseries.frequencies.to_offset(data[date_column].diff().median()).freqstr
    series = data.set_index(date_column)[value_column].asfreq(frequency)
    series = series.interpolate(method="time").dropna()
    if len(series) < 12:
        raise ValueError("Not enough observations after regularizing the time series.")
    values = series.to_numpy(float)
    # One-step-ahead autoregression using the previous 3 values as features.
    lag = 3
    X = np.array([values[i-lag:i] for i in range(lag, len(values))])
    y = values[lag:]
    split = max(lag+2, int(len(X)*(1-test_size)))
    if len(X)-split < 1:
        split = len(X)-1
    model_eval = LinearRegression().fit(X[:split], y[:split])
    test_pred = model_eval.predict(X[split:])
    actual = y[split:]
    mae = float(mean_absolute_error(actual, test_pred))
    rmse = float(np.sqrt(mean_squared_error(actual, test_pred)))
    final_model = LinearRegression().fit(X, y)
    history = list(values[-lag:])
    future = []
    for _ in range(periods):
        prediction = float(final_model.predict(np.array(history[-lag:]).reshape(1, -1))[0])
        future.append(prediction)
        history.append(prediction)
    future_dates = pd.date_range(start=series.index[-1], periods=periods+1, freq=frequency)[1:]
    return {"forecast": pd.DataFrame({date_column: future_dates, "forecast": future}),
            "mae": mae, "rmse": rmse, "frequency": frequency,
            "model": "3-lag autoregressive linear regression",
            "note": "Baseline forecast only; it does not model seasonality or external drivers. Validate frequency and backtest horizon for your use case."}
