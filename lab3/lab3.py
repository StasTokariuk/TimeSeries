from __future__ import annotations

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# НАЛАШТУВАННЯ

LATITUDE = 50.45
LONGITUDE = 30.52
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

RANDOM_SEED = 42
EPSILON = 1e-9

plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["font.size"] = 10



# 1. ОТРИМАННЯ РЕАЛЬНИХ ДАНИХ

def fetch_weather_data(
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:


    print("[INFO] Завантаження метеорологічних даних Open-Meteo...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "timezone": "Europe/Kyiv",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise ConnectionError(
            f"Не вдалося отримати дані з Open-Meteo: {exc}"
        ) from exc

    if "daily" not in payload:
        raise ValueError(f"API не повернув блок daily: {payload}")

    daily = payload["daily"]
    if "time" not in daily or "temperature_2m_mean" not in daily:
        raise ValueError("У відповіді API відсутні очікувані поля.")

    df = pd.DataFrame({
        "Date": pd.to_datetime(daily["time"]),
        "Temperature": pd.to_numeric(
            daily["temperature_2m_mean"], errors="coerce"
        ),
    })

    df = df.sort_values("Date").reset_index(drop=True)

    missing_count = int(df["Temperature"].isna().sum())
    if df["Temperature"].notna().sum() == 0:
        raise ValueError("API повернув ряд без жодного коректного вимірювання.")

    df["Temperature"] = (
        df["Temperature"]
        .interpolate(method="linear", limit_direction="both")
    )

    print(f"[INFO] Отримано спостережень: {len(df)}")
    print(f"[INFO] Початкових пропусків: {missing_count}")

    return df


# 2. ВИЯВЛЕННЯ ТА КОМПЕНСАЦІЯ АНОМАЛІЙ

def detect_and_compensate_anomalies(
    values: np.ndarray,
    window: int = 7,
    threshold: float = 3.5,
) -> tuple[np.ndarray, np.ndarray]:


    y = np.asarray(values, dtype=float)

    if y.ndim != 1 or len(y) < 3:
        raise ValueError("Потрібен одновимірний ряд щонайменше з 3 значень.")
    if not np.all(np.isfinite(y)):
        raise ValueError("Перед пошуком аномалій ряд не повинен містити NaN/inf.")

    local_median = (
        pd.Series(y)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy()
    )
    residuals = y - local_median

    residual_median = np.median(residuals)
    mad = np.median(np.abs(residuals - residual_median))
    robust_sigma = 1.4826 * mad

    if robust_sigma < EPSILON:
        robust_sigma = np.std(residuals)

    if robust_sigma < EPSILON:
        anomalies = np.array([], dtype=int)
    else:
        robust_z = (
            np.abs(residuals - residual_median) / (robust_sigma + EPSILON)
        )
        anomalies = np.flatnonzero(robust_z > threshold)

    cleaned = y.copy()

    if anomalies.size:
        valid_mask = np.ones(len(y), dtype=bool)
        valid_mask[anomalies] = False
        valid_indices = np.flatnonzero(valid_mask)

        if valid_indices.size >= 2:
            cleaned[anomalies] = np.interp(
                anomalies, valid_indices, y[valid_mask]
            )
        else:
            raise ValueError(
                "Недостатньо нормальних точок для компенсації аномалій."
            )

    print(f"[INFO] Виявлено аномалій: {len(anomalies)}")
    return cleaned, anomalies

# 3. КЛАСИЧНИЙ ALPHA-BETA ФІЛЬТР

def alpha_beta_filter(
    values: np.ndarray,
    dt: float = 1.0,
    alpha: float = 0.15,
    beta: float = 0.01,
) -> np.ndarray:

    y = np.asarray(values, dtype=float)

    if y.ndim != 1 or len(y) == 0:
        raise ValueError("Вхідний ряд має бути непорожнім одновимірним масивом.")
    if dt <= 0:
        raise ValueError("dt має бути додатним.")
    if not (0 <= alpha <= 1 and 0 <= beta <= 1):
        raise ValueError("Коефіцієнти alpha та beta мають належати [0, 1].")

    x_est = np.zeros(len(y), dtype=float)
    v_est = np.zeros(len(y), dtype=float)

    x_est[0] = y[0]

    for i in range(1, len(y)):
        x_pred = x_est[i - 1] + v_est[i - 1] * dt
        v_pred = v_est[i - 1]

        innovation = y[i] - x_pred

        x_est[i] = x_pred + alpha * innovation
        v_est[i] = v_pred + (beta * innovation) / dt

    return x_est

# 4. АДАПТИВНИЙ ALPHA-BETA ФІЛЬТР

def adaptive_alpha_beta_filter(
    values: np.ndarray,
    dt: float = 1.0,
    alpha_min: float = 0.10,
    alpha_max: float = 0.35,
    beta_min: float = 0.005,
    beta_max: float = 0.03,
    history_window: int = 21,
    scale_floor: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    y = np.asarray(values, dtype=float)

    if y.ndim != 1 or len(y) == 0:
        raise ValueError("Вхідний ряд має бути непорожнім одновимірним масивом.")
    if dt <= 0:
        raise ValueError("dt має бути додатним.")
    if history_window < 3:
        raise ValueError("history_window має бути не меншим за 3.")

    n = len(y)
    x_est = np.zeros(n, dtype=float)
    v_est = np.zeros(n, dtype=float)
    alpha_history = np.zeros(n, dtype=float)
    beta_history = np.zeros(n, dtype=float)
    innovation_history: list[float] = []

    x_est[0] = y[0]
    alpha_history[0] = alpha_min
    beta_history[0] = beta_min

    for i in range(1, n):
        x_pred = x_est[i - 1] + v_est[i - 1] * dt
        v_pred = v_est[i - 1]
        innovation = y[i] - x_pred

        recent = np.asarray(innovation_history[-history_window:])

        if len(recent) >= 3:
            med = np.median(recent)
            mad = np.median(np.abs(recent - med))
            innovation_scale = max(1.4826 * mad, scale_floor)
        else:
            innovation_scale = scale_floor

        normalized_innovation = abs(innovation) / innovation_scale

        adaptation = np.clip(
            (normalized_innovation - 1.0) / 3.0, 0.0, 1.0
        )

        alpha = alpha_min + adaptation * (alpha_max - alpha_min)
        beta = beta_min + adaptation * (beta_max - beta_min)

        x_est[i] = x_pred + alpha * innovation
        v_est[i] = v_pred + (beta * innovation) / dt

        alpha_history[i] = alpha
        beta_history[i] = beta
        innovation_history.append(float(innovation))

    return x_est, alpha_history, beta_history, v_est


# 5. МЕТРИКИ ЯКОСТІ

def calculate_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> dict[str, float]:

    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    if reference.shape != estimate.shape:
        raise ValueError("Еталонний та оцінений сигнали мають бути однакової форми.")

    error = estimate - reference

    return {
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "Bias": float(np.mean(error)),
    }


# 6. СИНТЕТИЧНИЙ ЕКСПЕРИМЕНТ З ВІДОМИМ ЕТАЛОНОМ

def create_synthetic_experiment(
    n: int = 500,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)

    true_signal = (
        10.0
        + 7.0 * np.sin(2 * np.pi * t / 180)
        + 0.008 * t
    )
    true_signal[250:] += 4.0 + 0.035 * (t[250:] - 250)

    noisy_signal = true_signal + rng.normal(0, 1.2, n)

    anomaly_indices = np.array([65, 140, 141, 310, 420])
    noisy_signal[anomaly_indices] += np.array([9, -11, 10, -12, 13])

    return t, true_signal, noisy_signal, anomaly_indices


# 7. ГРАФІКИ ДЛЯ РЕАЛЬНИХ ДАНИХ

def plot_weather_results(
    dates: pd.Series,
    raw: np.ndarray,
    cleaned: np.ndarray,
    classic: np.ndarray,
    adaptive: np.ndarray,
    anomalies: np.ndarray,
) -> None:

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

    axes[0].plot(dates, raw, color="gray", linewidth=1, label="Початкові дані")
    if len(anomalies):
        axes[0].scatter(
            dates.iloc[anomalies],
            raw[anomalies],
            color="red",
            s=35,
            label="Виявлені аномалії",
            zorder=3,
        )
    axes[0].set_title("1. Початкові метеодані та виявлені аномалії")
    axes[0].set_ylabel("Температура, °C")
    axes[0].grid(True, linestyle=":")
    axes[0].legend()

    axes[1].plot(dates, cleaned, color="black", alpha=0.45, label="Очищені дані")
    axes[1].plot(dates, classic, color="royalblue", label="Класичний alpha-beta")
    axes[1].set_title("2. Класичний alpha-beta фільтр")
    axes[1].set_ylabel("Температура, °C")
    axes[1].grid(True, linestyle=":")
    axes[1].legend()

    axes[2].plot(dates, cleaned, color="black", alpha=0.35, label="Очищені дані")
    axes[2].plot(dates, adaptive, color="darkorange", label="Адаптивний alpha-beta")
    axes[2].set_title("3. Адаптивний alpha-beta фільтр")
    axes[2].set_xlabel("Дата")
    axes[2].set_ylabel("Температура, °C")
    axes[2].grid(True, linestyle=":")
    axes[2].legend()

    fig.tight_layout()
    plt.show()


# 8. ГРАФІКИ ТА МЕТРИКИ СИНТЕТИЧНОГО ЕКСПЕРИМЕНТУ

def run_synthetic_validation() -> None:

    t, truth, noisy, injected_anomalies = create_synthetic_experiment()

    cleaned, detected = detect_and_compensate_anomalies(
        noisy, window=9, threshold=5.0
    )

    classic = alpha_beta_filter(cleaned, alpha=0.15, beta=0.01)
    adaptive, alpha_values, beta_values, _ = adaptive_alpha_beta_filter(cleaned)

    classic_metrics = calculate_metrics(truth, classic)
    adaptive_metrics = calculate_metrics(truth, adaptive)

    print("\n[VALIDATION] Синтетичний експеримент")
    print("Еталонний сигнал відомий, тому метрики оцінюють похибку відновлення.")
    print(f"Внесені штучні аномалії: {len(injected_anomalies)}")
    print(f"Виявлені алгоритмом аномалії: {len(detected)}")
    print("\nМетрики:")
    print(f"{'Метрика':<12}{'Класичний':>15}{'Адаптивний':>15}")
    for key in ("RMSE", "MAE", "Bias"):
        print(
            f"{key:<12}{classic_metrics[key]:>15.4f}"
            f"{adaptive_metrics[key]:>15.4f}"
        )

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

    axes[0].plot(t, truth, color="black", linewidth=2, label="Істинний сигнал")
    axes[0].plot(t, noisy, color="lightgray", linewidth=0.8, label="Шумні вимірювання")
    axes[0].scatter(
        injected_anomalies,
        noisy[injected_anomalies],
        color="red",
        s=45,
        label="Штучно внесені викиди",
        zorder=3,
    )
    axes[0].set_title("1. Синтетичний сигнал: еталон, шум та викиди")
    axes[0].grid(True, linestyle=":")
    axes[0].legend()

    axes[1].plot(t, truth, color="black", linewidth=2, label="Істинний сигнал")
    axes[1].plot(t, classic, color="royalblue", label="Класичний alpha-beta")
    axes[1].plot(t, adaptive, color="darkorange", label="Адаптивний alpha-beta")
    axes[1].axvline(250, color="gray", linestyle="--", label="Зміна режиму")
    axes[1].set_title("2. Порівняння оцінок фільтрів з еталоном")
    axes[1].grid(True, linestyle=":")
    axes[1].legend()

    axes[2].plot(t, alpha_values, label="Адаптивний alpha")
    axes[2].plot(t, beta_values, label="Адаптивний beta")
    axes[2].axvline(250, color="gray", linestyle="--", label="Зміна режиму")
    axes[2].set_title("3. Динаміка адаптації коефіцієнтів")
    axes[2].set_xlabel("Крок часу")
    axes[2].set_ylabel("Значення коефіцієнта")
    axes[2].grid(True, linestyle=":")
    axes[2].legend()

    fig.tight_layout()
    plt.show()

def main() -> None:

    # Реальні метеорологічні дані.
    df = fetch_weather_data()
    y_raw = df["Temperature"].to_numpy(dtype=float)

    y_clean, anomaly_indices = detect_and_compensate_anomalies(y_raw)

    y_classic = alpha_beta_filter(
        y_clean,
        dt=1.0,
        alpha=0.15,
        beta=0.01,
    )

    y_adaptive, alpha_values, beta_values, _ = (
        adaptive_alpha_beta_filter(y_clean, dt=1.0)
    )

    # Для реальних даних немає незалежної істинної температури.
    # Тому ці RMSE є відхиленням від очищених вимірювань,
    # а не об'єктивною похибкою відновлення істинної температури.
    classic_deviation = np.sqrt(np.mean((y_clean - y_classic) ** 2))
    adaptive_deviation = np.sqrt(np.mean((y_clean - y_adaptive) ** 2))

    print("\n[INFO] Фільтрацію реальних даних завершено.")
    print(
        "Середнє відхилення від очищених даних "
        f"(класичний фільтр): {classic_deviation:.4f} °C"
    )
    print(
        "Середнє відхилення від очищених даних "
        f"(адаптивний фільтр): {adaptive_deviation:.4f} °C"
    )
    print(
        "Середні коефіцієнти адаптивного фільтра: "
        f"alpha={np.mean(alpha_values):.3f}, "
        f"beta={np.mean(beta_values):.3f}"
    )

    plot_weather_results(
        df["Date"],
        y_raw,
        y_clean,
        y_classic,
        y_adaptive,
        anomaly_indices,
    )

    # Незалежна перевірка на синтетичному сигналі.
    run_synthetic_validation()


if __name__ == "__main__":
    main()
