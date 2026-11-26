import warnings
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf

warnings.filterwarnings("ignore")

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


def analyze_and_decompose(series: pd.Series, period: int = 365) -> dict:
    vals = series.dropna().to_numpy(dtype=float)
    adf_res = adfuller(vals, autolag="AIC")
    acf_vals = acf(vals, nlags=30, fft=True)

    decomp = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
    var_resid = np.var(decomp.resid)
    var_detrend = np.var(decomp.seasonal + decomp.resid)
    var_deseas = np.var(decomp.trend + decomp.resid)

    ft = float(max(0.0, 1.0 - var_resid / max(var_detrend, 1e-9)))
    fs = float(max(0.0, 1.0 - var_resid / max(var_deseas, 1e-9)))

    return {
        "N": len(vals),
        "Mean": float(np.mean(vals)),
        "Std": float(np.std(vals)),
        "Min": float(np.min(vals)),
        "Max": float(np.max(vals)),
        "ADF_Stat": float(adf_res[0]),
        "ADF_p": float(adf_res[1]),
        "Is_Stationary": bool(adf_res[1] < 0.05),
        "Trend_Strength": ft,
        "Seasonal_Strength": fs,
        "ACF_Lag1": float(acf_vals[1]),
        "ACF_Lag7": float(acf_vals[7]),
        "decomp": decomp,
    }


def compute_time_features(day_indices: np.ndarray, period: float = 365.25) -> np.ndarray:
    omega = 2.0 * np.pi / period
    return np.column_stack([
        np.sin(omega * day_indices),
        np.cos(omega * day_indices),
        np.sin(2.0 * omega * day_indices),
        np.cos(2.0 * omega * day_indices),
    ])


def prepare_supervised_tensors(
    series: pd.Series,
    window_size: int = 21,
    val_size: int = 90,
    test_size: int = 180,
) -> dict:
    raw_vals = series.to_numpy(dtype=float).reshape(-1, 1)
    n_total = len(raw_vals)
    train_cutoff = n_total - val_size - test_size

    # Масштабування виключно на навчальній частині (запобігання витоку даних Data Leakage)
    scaler = StandardScaler()
    scaler.fit(raw_vals[:train_cutoff])
    scaled_temp = scaler.transform(raw_vals).flatten()

    t_idx = np.arange(n_total, dtype=float)
    fourier_feats = compute_time_features(t_idx)
    all_features = np.column_stack([scaled_temp, fourier_feats])

    X_list, y_list, target_dates, target_t = [], [], [], []
    for i in range(window_size, n_total):
        X_list.append(all_features[i - window_size : i, :])
        y_list.append(scaled_temp[i])
        target_dates.append(series.index[i])
        target_t.append(i)

    X_all = np.asarray(X_list, dtype=np.float32)
    y_all = np.asarray(y_list, dtype=np.float32).reshape(-1, 1)
    dates_all = pd.DatetimeIndex(target_dates)

    # Розбиття вікон за часовими межами
    n_samples = len(X_all)
    idx_train_end = n_samples - val_size - test_size
    idx_val_end = n_samples - test_size

    return {
        "X_all": X_all,
        "y_all": y_all,
        "dates_all": dates_all,
        "X_train": X_all[:idx_train_end],
        "y_train": y_all[:idx_train_end],
        "dates_train": dates_all[:idx_train_end],
        "X_val": X_all[idx_train_end:idx_val_end],
        "y_val": y_all[idx_train_end:idx_val_end],
        "dates_val": dates_all[idx_train_end:idx_val_end],
        "X_test": X_all[idx_val_end:],
        "y_test": y_all[idx_val_end:],
        "dates_test": dates_all[idx_val_end:],
        "scaler": scaler,
        "scaled_temp": scaled_temp,
        "all_features": all_features,
        "window_size": window_size,
        "val_size": val_size,
        "test_size": test_size,
    }