import requests
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller, acf
from  sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
    print("[INFO] Завантаження багатовимірних даних Open-Meteo...")
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
    print(f"[INFO] Отримано спостережень: {len(df)}, показників: {df.shape[1]}")
    return df


def decompose_series(series: pd.Series, period: int = 365) -> dict:
    # Класична адитивна декомпозиція
    classical = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")

    # STL декомпозиція
    stl_Period = period if period % 2 == 1 else period + 1
    stl = STL(series, period=period, seasonal=stl_Period, robust=True).fit()

    return {
        "classical": classical,
        "stl": stl,
    }


def calculate_hurst_exponent(ts: np.ndarray, max_lag: int = 100) -> float:
    lags = range(2, min(max_lag, len(ts) // 4))
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return float(poly[0])


def analyze_properties(series: pd.Series, stl_res) -> dict:
    values = series.dropna().to_numpy()

    # ADF-тест Дікі-Фуллера на стаціонарність
    adf_result = adfuller(values, autolag="AIC")

    # Сила тренду та сезонності за формулою Hyndman & Athanasopoulos
    var_resid = np.var(stl_res.resid)
    var_detrended = np.var(stl_res.seasonal + stl_res.resid)
    var_deseasonal = np.var(stl_res.trend + stl_res.resid)

    trend_strength = max(0.0, 1.0 - var_resid / var_detrended)
    seasonal_strength = max(0.0, 1.0 - var_resid / var_deseasonal)

    # Показник Херста
    hurst = calculate_hurst_exponent(values)

    # Автокореляція
    acf_vals = acf(values, nlags=30, fft=True)

    return {
        "Mean": float(np.mean(values)),
        "Std": float(np.std(values)),
        "Median": float(np.median(values)),
        "IQR": float(stats.iqr(values)),
        "Skewness": float(stats.skew(values)),
        "Kurtosis": float(stats.kurtosis(values)),
        "ADF_Statistic": float(adf_result[0]),
        "ADF_p_value": float(adf_result[1]),
        "Is_Stationary": bool(adf_result[1] < 0.05),
        "Hurst_Exponent": hurst,
        "Trend_Strength": float(trend_strength),
        "Seasonal_Strength": float(seasonal_strength),
        "ACF_Lag1": float(acf_vals[1]),
        "ACF_Lag7": float(acf_vals[7]),
        "ACF_Lag30": float(acf_vals[30]),
    }


def correlation_analysis(df: pd.DataFrame, max_lag: int = 30) -> dict:
    pearson_corr = df.corr(method="pearson")
    spearman_corr = df.corr(method="spearman")

    # Крос-кореляція між Температурою та Вологістю при зсувах у часі
    x = (df["Temperature"] - df["Temperature"].mean()) / df["Temperature"].std()
    y = (df["Humidity"] - df["Humidity"].mean()) / df["Humidity"].std()
    lags = np.arange(-max_lag, max_lag + 1)
    ccf = [x.corr(y.shift(lag)) for lag in lags]

    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "ccf_lags": lags,
        "ccf_temp_humidity": np.array(ccf),
    }


def cluster_time_series(df: pd.DataFrame, n_clusters: int = 4) -> dict:
    # 1. Багатофакторна кластеризація станів (K-Means)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[["Temperature", "Humidity", "Pressure", "WindSpeed"]])

    kmeans_multi = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_multi = kmeans_multi.fit_predict(features_scaled)
    sil_score_multi = silhouette_score(features_scaled, labels_multi)

    df_clustered = df.copy()
    df_clustered["Cluster"] = labels_multi
    centroids_original = df_clustered.groupby("Cluster")[["Temperature", "Humidity", "Pressure", "WindSpeed"]].mean()

    # 2. Кластеризація часових підрядів (7-денні вікна температури, Z-нормалізовані за формою)
    temp_vals = df["Temperature"].to_numpy()
    n_weeks = len(temp_vals) // 7
    weekly_segments = temp_vals[: n_weeks * 7].reshape(n_weeks, 7)

    # Нормалізація кожного тижня
    kmeans_weekly = KMeans(n_clusters=3, random_state=42, n_init=10)
    weekly_labels = kmeans_weekly.fit_predict(weekly_segments)

    return {
        "df_clustered": df_clustered,
        "centroids": centroids_original,
        "silhouette_multi": float(sil_score_multi),
        "weekly_segments": weekly_segments,
        "weekly_labels": weekly_labels,
        "weekly_centers": kmeans_weekly.cluster_centers_,
    }