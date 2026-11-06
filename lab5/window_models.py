import requests
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

LATITUDE = 50.45
LONGITUDE = 30.52
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"


def fetch_weather_series(
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.Series:
    print("[INFO] Завантаження даних Open-Meteo...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "timezone": "Europe/Kyiv",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    daily = response.json()["daily"]

    df = pd.DataFrame({
        "Date": pd.to_datetime(daily["time"]),
        "Temperature": pd.to_numeric(daily["temperature_2m_mean"], errors="coerce"),
    })
    df = df.sort_values("Date").set_index("Date")
    series = df["Temperature"].interpolate(method="linear", limit_direction="both")
    series.index.freq = "D"
    print(f"[INFO] Отримано спостережень: N = {len(series)} діб ({start_date} .. {end_date})")
    return series


def check_stationarity_properties(series: pd.Series) -> dict:
    vals = series.dropna().to_numpy()
    diff_vals = np.diff(vals)

    adf_orig = adfuller(vals, autolag="AIC")
    adf_diff = adfuller(diff_vals, autolag="AIC")

    return {
        "N": len(vals),
        "Mean": float(np.mean(vals)),
        "Std": float(np.std(vals)),
        "ADF_Orig_Stat": float(adf_orig[0]),
        "ADF_Orig_p": float(adf_orig[1]),
        "Orig_Stationary": bool(adf_orig[1] < 0.05),
        "ADF_Diff_Stat": float(adf_diff[0]),
        "ADF_Diff_p": float(adf_diff[1]),
        "Diff_Stationary": bool(adf_diff[1] < 0.05),
    }


# АЛГОРИТМИ КОВЗНОГО ВІКНА

def simple_moving_average(series: pd.Series, window: int = 7) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def weighted_moving_average(series: pd.Series, window: int = 7) -> pd.Series:
    weights = np.arange(1, window + 1, dtype=float)

    def calc_wma(x):
        w = weights[-len(x):]
        return np.dot(x, w) / w.sum()

    return series.rolling(window=window, min_periods=1).apply(calc_wma, raw=True)


def exponential_moving_average(series: pd.Series, span: int = 7) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# МЕТОДИ СТАТИСТИЧНОГО НАВЧАННЯ

def fit_harmonic_regression(series: pd.Series, n_harmonics: int = 2) -> dict:

    y = series.to_numpy()
    n = len(y)
    t = np.arange(n, dtype=float)
    omega = 2.0 * np.pi / 365.25

    columns = [np.ones(n), t]
    for k in range(1, n_harmonics + 1):
        columns.append(np.sin(k * omega * t))
        columns.append(np.cos(k * omega * t))

    X = np.column_stack(columns)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    residuals = y - fitted

    return {
        "beta": beta,
        "n_harmonics": n_harmonics,
        "fitted": pd.Series(fitted, index=series.index),
        "residuals": pd.Series(residuals, index=series.index),
    }


def predict_harmonic_regression(model_dict: dict, t_indices: np.ndarray, index: pd.Index) -> pd.Series:
    beta = model_dict["beta"]
    n_harmonics = model_dict["n_harmonics"]
    omega = 2.0 * np.pi / 365.25

    columns = [np.ones(len(t_indices)), t_indices]
    for k in range(1, n_harmonics + 1):
        columns.append(np.sin(k * omega * t_indices))
        columns.append(np.cos(k * omega * t_indices))

    X_future = np.column_stack(columns)
    pred = X_future @ beta
    return pd.Series(pred, index=index)


def adaptive_alpha_beta_filter(
    series: pd.Series,
    dt: float = 1.0,
    alpha_min: float = 0.12,
    alpha_max: float = 0.40,
    beta_min: float = 0.005,
    beta_max: float = 0.04,
    history_window: int = 21,
) -> tuple[pd.Series, pd.Series]:
    y = series.to_numpy(dtype=float)
    n = len(y)
    x_est = np.zeros(n, dtype=float)
    x_pred_arr = np.zeros(n, dtype=float)
    v_est = np.zeros(n, dtype=float)
    innovations = []

    x_est[0] = y[0]
    x_pred_arr[0] = y[0]

    for i in range(1, n):
        x_pred = x_est[i - 1] + v_est[i - 1] * dt
        v_pred = v_est[i - 1]
        x_pred_arr[i] = x_pred

        innov = y[i] - x_pred
        recent = np.asarray(innovations[-history_window:])
        if len(recent) >= 3:
            med = np.median(recent)
            mad = np.median(np.abs(recent - med))
            scale = max(1.4826 * mad, 0.3)
        else:
            scale = 0.3

        norm_innov = abs(innov) / scale
        adapt = np.clip((norm_innov - 1.0) / 3.0, 0.0, 1.0)
        alpha = alpha_min + adapt * (alpha_max - alpha_min)
        beta = beta_min + adapt * (beta_max - beta_min)

        x_est[i] = x_pred + alpha * innov
        v_est[i] = v_pred + (beta * innov) / dt
        innovations.append(float(innov))

    return pd.Series(x_est, index=series.index), pd.Series(x_pred_arr, index=series.index)