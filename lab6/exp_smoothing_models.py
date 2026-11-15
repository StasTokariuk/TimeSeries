import warnings
import requests
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing

warnings.filterwarnings("ignore")

LATITUDE = 50.45
LONGITUDE = 30.52
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"


def fetch_multivariate_weather_data(
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    print("[INFO] Завантаження даних Open-Meteo...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "surface_pressure_mean",
            "wind_speed_10m_max",
        ],
        "timezone": "Europe/Kyiv",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    daily = response.json()["daily"]

    df = pd.DataFrame({
        "Date": pd.to_datetime(daily["time"]),
        "Temperature": pd.to_numeric(daily["temperature_2m_mean"], errors="coerce"),
        "Humidity": pd.to_numeric(daily["relative_humidity_2m_mean"], errors="coerce"),
        "Pressure": pd.to_numeric(daily["surface_pressure_mean"], errors="coerce"),
        "WindSpeed": pd.to_numeric(daily["wind_speed_10m_max"], errors="coerce"),
    })
    df = df.sort_values("Date").set_index("Date")
    df = df.interpolate(method="linear", limit_direction="both")
    df.index.freq = "D"
    print(f"[INFO] Отримано спостережень: N = {len(df)} діб, показників: {df.shape[1]}")
    return df


def analyze_ts_properties(series: pd.Series, period: int = 365) -> dict:
    vals = series.dropna().to_numpy()
    adf_res = adfuller(vals, autolag="AIC")
    acf_vals = acf(vals, nlags=30, fft=True)

    decomp = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
    var_resid = np.var(decomp.resid)
    var_detrend = np.var(decomp.seasonal + decomp.resid)
    var_deseas = np.var(decomp.trend + decomp.resid)

    trend_strength = float(max(0.0, 1.0 - var_resid / max(var_detrend, 1e-9)))
    seasonal_strength = float(max(0.0, 1.0 - var_resid / max(var_deseas, 1e-9)))

    # Оцінка лінійного нахилу тренду
    t = np.arange(len(vals), dtype=float)
    slope = float(np.polyfit(t, vals, 1)[0])

    # Показник Херста
    lags = range(2, min(60, len(vals) // 4))
    tau = [np.std(np.subtract(vals[lag:], vals[:-lag])) for lag in lags]
    hurst = float(np.polyfit(np.log(lags), np.log(tau), 1)[0])

    return {
        "Mean": float(np.mean(vals)),
        "Std": float(np.std(vals)),
        "ADF_Stat": float(adf_res[0]),
        "ADF_p": float(adf_res[1]),
        "Is_Stationary": bool(adf_res[1] < 0.05),
        "Trend_Strength": trend_strength,
        "Seasonal_Strength": seasonal_strength,
        "ACF_Lag1": float(acf_vals[1]),
        "ACF_Lag7": float(acf_vals[7]),
        "Hurst": hurst,
        "Linear_Slope": slope,
    }


def _extract_seasonal_profile_short_sample(series: pd.Series, forecast_steps: int, period: float = 365.25):
    y = series.to_numpy(dtype=float)
    n = len(y)
    t_train = np.arange(n, dtype=float)
    t_future = np.arange(n, n + forecast_steps, dtype=float)
    omega = 2.0 * np.pi / period

    # Гармонічний базис (2 гармоніки) без константи для чистої сезонної складової з нульовим середнім
    X_s_train = np.column_stack([
        np.sin(omega * t_train), np.cos(omega * t_train),
        np.sin(2 * omega * t_train), np.cos(2 * omega * t_train),
    ])
    X_s_future = np.column_stack([
        np.sin(omega * t_future), np.cos(omega * t_future),
        np.sin(2 * omega * t_future), np.cos(2 * omega * t_future),
    ])

    y_centered = y - np.mean(y)
    coeffs, _, _, _ = np.linalg.lstsq(X_s_train, y_centered, rcond=None)
    seasonal_train = pd.Series(X_s_train @ coeffs, index=series.index)
    seasonal_future = X_s_future @ coeffs
    return seasonal_train, seasonal_future


def fit_exponential_smoothing_suite(
    train_series: pd.Series,
    forecast_steps: int,
    seasonal_period: int = 365,
) -> dict:
    future_index = pd.date_range(
        start=train_series.index[-1] + pd.Timedelta(days=1),
        periods=forecast_steps,
        freq="D",
    )

    # 1. Просте експоненціальне згладжування (SES)
    ses_fit = SimpleExpSmoothing(train_series, initialization_method="estimated").fit(optimized=True)
    ses_fitted = pd.Series(ses_fit.fittedvalues, index=train_series.index)
    ses_forecast = pd.Series(ses_fit.forecast(forecast_steps).to_numpy(), index=future_index)

    # 2. Лінійний метод Хольта (Подвійне експоненціальне згладжування)
    holt_fit = Holt(train_series, initialization_method="estimated").fit(optimized=True)
    holt_fitted = pd.Series(holt_fit.fittedvalues, index=train_series.index)
    holt_forecast = pd.Series(holt_fit.forecast(forecast_steps).to_numpy(), index=future_index)

    # 3. Метод Хольта з демпфованим трендом
    damped_fit = Holt(train_series, damped_trend=True, initialization_method="estimated").fit(optimized=True)
    damped_fitted = pd.Series(damped_fit.fittedvalues, index=train_series.index)
    damped_forecast = pd.Series(damped_fit.forecast(forecast_steps).to_numpy(), index=future_index)

    # 4. Потрійне експоненціальне згладжування Хольта-Вінтерса (Holt-Winters)
    if len(train_series) >= 2 * seasonal_period:
        hw_fit = ExponentialSmoothing(
            train_series,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=seasonal_period,
            initialization_method="estimated",
        ).fit(optimized=True)
        hw_fitted = pd.Series(hw_fit.fittedvalues, index=train_series.index)
        hw_forecast = pd.Series(hw_fit.forecast(forecast_steps).to_numpy(), index=future_index)
        hw_aic = float(hw_fit.aic)
        hw_params = {
            "alpha": float(hw_fit.params.get("smoothing_level", np.nan)),
            "beta": float(hw_fit.params.get("smoothing_trend", np.nan)),
            "gamma": float(hw_fit.params.get("smoothing_seasonal", np.nan)),
            "phi": float(hw_fit.params.get("damping_trend", np.nan)),
        }
    else:
        # Якщо навчальна вибірка менша за 2 повні річні цикли (< 730 діб під час train/test split),
        # оцінюємо сезонний профіль і застосовуємо експоненціальне згладжування Хольта до десезоналізованого ряду
        seasonal_train, seasonal_future = _extract_seasonal_profile_short_sample(
            train_series, forecast_steps, period=float(seasonal_period)
        )
        deseasonal = train_series - seasonal_train
        hw_base = Holt(deseasonal, damped_trend=True, initialization_method="estimated").fit(optimized=True)
        hw_fitted = pd.Series(hw_base.fittedvalues + seasonal_train, index=train_series.index)
        hw_forecast = pd.Series(hw_base.forecast(forecast_steps).to_numpy() + seasonal_future, index=future_index)
        hw_aic = float(hw_base.aic)
        hw_params = {
            "alpha": float(hw_base.params.get("smoothing_level", np.nan)),
            "beta": float(hw_base.params.get("smoothing_trend", np.nan)),
            "gamma": 0.05,
            "phi": float(hw_base.params.get("damping_trend", np.nan)),
        }

    return {
        "SES": {
            "fitted": ses_fitted,
            "forecast": ses_forecast,
            "aic": float(ses_fit.aic),
            "params": {"alpha": float(ses_fit.params["smoothing_level"])},
        },
        "Holt_Linear": {
            "fitted": holt_fitted,
            "forecast": holt_forecast,
            "aic": float(holt_fit.aic),
            "params": {
                "alpha": float(holt_fit.params["smoothing_level"]),
                "beta": float(holt_fit.params["smoothing_trend"]),
            },
        },
        "Holt_Damped": {
            "fitted": damped_fitted,
            "forecast": damped_forecast,
            "aic": float(damped_fit.aic),
            "params": {
                "alpha": float(damped_fit.params["smoothing_level"]),
                "beta": float(damped_fit.params["smoothing_trend"]),
                "phi": float(damped_fit.params["damping_trend"]),
            },
        },
        "Holt_Winters": {
            "fitted": hw_fitted,
            "forecast": hw_forecast,
            "aic": hw_aic,
            "params": hw_params,
        },
    }