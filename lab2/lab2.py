import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


# 1. Отримання вхіжних даних
def load_or_parse_data(file_name="population_data.csv"):
    if os.path.exists(file_name):
        print(f"[INFO] Зчитуємо дані з локального файлу '{file_name}'...")
        df = pd.read_csv(file_name)
        return df

    print(f"[INFO] Файл '{file_name}' не знайдено. Виконуємо парсинг Worldometers...")
    url = "https://www.worldometers.info/world-population/ukraine-population/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    target_table = None
    for tbl in soup.find_all('table'):
        if 'Year' in tbl.text and 'Population' in tbl.text:
            target_table = tbl
            break

    if target_table is None:
        raise ValueError("Не вдалося знайти таблицю на сторінці.")

    years, populations = [], []
    rows = target_table.find('tbody').find_all('tr') if target_table.find('tbody') else target_table.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        if len(cols) > 1:
            try:
                year = int(cols[0].text.strip())
                pop = int(cols[1].text.strip().replace(',', ''))
                years.append(year)
                populations.append(pop)
            except ValueError:
                continue

    df = pd.DataFrame({'Year': years, 'Population': populations})
    df = df.sort_values(by='Year').reset_index(drop=True)
    df.to_csv(file_name, index=False)
    print(f"[INFO] Дані збережено у '{file_name}'.")
    return df


# 2. Генерація синтетичної вибірки з аномаліями
def generate_data_with_anomalies(x, y_real, anomaly_ratio=0.15, q_av=4.5):
    np.random.seed(42)
    n = len(y_real)

    # Базовий тренд та залишки з реальних даних
    base_coeffs = np.polyfit(x, y_real, 2)
    y_base_trend = np.polyval(base_coeffs, x)
    residuals = y_real - y_base_trend
    sigma_noise = np.std(residuals, ddof=1)

    # Синтезована вибірка без АВ
    synth_noise = np.random.normal(0, sigma_noise, size=n)
    y_synth_clean = y_base_trend + synth_noise

    # Додавання аномальних вимірів (АВ)
    y_with_av = y_synth_clean.copy()
    n_av = max(2, int(n * anomaly_ratio))
    # Обираємо індекси для аномалій
    av_indices = np.random.choice(range(1, n - 1), size=n_av, replace=False)

    for idx in av_indices:
        sign = np.random.choice([-1, 1])
        y_with_av[idx] += sign * q_av * sigma_noise

    return y_synth_clean, y_with_av, av_indices, sigma_noise


def detect_and_clean_anomalies(x, y_noisy, threshold_z=2.2):

    y_cleaned = y_noisy.copy()

    # Будуємо опорний поліном 2-го степеня для оцінки відхилень
    rough_coeffs = np.polyfit(x, y_cleaned, 2)
    rough_trend = np.polyval(rough_coeffs, x)

    # Оцінюємо медіану та робастне СКВ залишків (через MAD),
    # щоб самі аномалії не завищували оцінку дисперсії
    res = y_cleaned - rough_trend
    mad = np.median(np.abs(res - np.median(res)))
    robust_std = 1.4826 * mad if mad > 0 else np.std(res)

    z_scores = np.abs(res - np.median(res)) / robust_std
    detected_indices = np.where(z_scores > threshold_z)[0]

    # Заміна виявлених аномалій методом інтерполяції за сусідніми точками
    normal_mask = np.ones(len(y_cleaned), dtype=bool)
    normal_mask[detected_indices] = False

    y_cleaned[detected_indices] = np.interp(
        x[detected_indices],
        x[normal_mask],
        y_cleaned[normal_mask]
    )

    return y_cleaned, detected_indices


# 3. МНК навчання, оптимізація моделі та alpha-betta фільтр
def fit_ols_polynomial(x, y, degree):

    x_mean, x_std = np.mean(x), np.std(x)
    x_norm = (x - x_mean) / x_std

    F = np.vander(x_norm, N=degree + 1, increasing=True)

    FT_F_inv = np.linalg.pinv(F.T @ F)
    coeffs_norm = FT_F_inv @ F.T @ y

    def predict(x_new):
        x_new_norm = (x_new - x_mean) / x_std
        F_new = np.vander(x_new_norm, N=degree + 1, increasing=True)
        return F_new @ coeffs_norm

    return predict, coeffs_norm


def optimize_polynomial_model(x, y, degrees=(1, 2, 3, 4)):
    n = len(y)
    results = {}

    for d in degrees:
        predict_fn, coeffs = fit_ols_polynomial(x, y, d)
        y_pred = predict_fn(x)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r2 = 1 - (ss_res / ss_tot)
        # Скоригований R^2 штрафує модель за зайві параметри (захист від перенавчання)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - d - 1) if n > d + 1 else r2
        rmse = np.sqrt(ss_res / n)
        # Інформаційний критерій (AIC)
        aic = n * np.log(ss_res / n) + 2 * (d + 1)

        results[d] = {
            'degree': d,
            'predict_fn': predict_fn,
            'coeffs': coeffs,
            'y_pred': y_pred,
            'R2': r2,
            'Adj_R2': adj_r2,
            'RMSE': rmse,
            'AIC': aic
        }

    # Обираємо оптимальну модель за максимальним Adjusted R^2 (або мінімальним AIC)
    # Для фізичної адекватності при екстраполяції на 0.5 інтервалу обмежуємо вибір до d <= 3
    best_degree = max([d for d in degrees if d <= 3], key=lambda d: results[d]['Adj_R2'])
    return results, best_degree


def alpha_beta_filter(x, y, alpha=0.45, beta=0.15):
    n = len(y)
    y_smooth = np.zeros(n)
    y_smooth[0] = y[0]
    v = (y[1] - y[0]) / (x[1] - x[0]) if n > 1 else 0.0

    for i in range(1, n):
        dt = x[i] - x[i - 1]
        # Етап екстраполяції
        y_pred = y_smooth[i - 1] + v * dt
        # Похибка інновації
        residual = y[i] - y_pred
        # Етап корекції
        y_smooth[i] = y_pred + alpha * residual
        v = v + (beta * residual) / dt

    return y_smooth


# 4. Екстраполяція на 0.5 вибірки
def extrapolate_process(x, y_clean, model_info, ratio=0.5):
    n = len(x)
    n_forecast = max(1, int(np.ceil(n * ratio)))

    # Визначаємо середній або останній крок за часом для генерації майбутньої шкали
    avg_step = int(np.round((x[-1] - x[0]) / (n - 1)))
    if avg_step <= 0:
        avg_step = 1

    x_future = np.array([x[-1] + avg_step * i for i in range(1, n_forecast + 1)])
    x_full = np.concatenate([x, x_future])

    predict_fn = model_info['predict_fn']
    y_full_pred = predict_fn(x_full)
    y_future_pred = predict_fn(x_future)

    # Оцінка інтервалу прогнозу (розширюється з горизонтом прогнозування)
    rmse = model_info['RMSE']
    horizon_factor = np.linspace(1.0, 2.2, len(x_future))
    ci_margin = 1.96 * rmse * horizon_factor

    return x_future, y_future_pred, x_full, y_full_pred, ci_margin


if __name__ == "__main__":
    # 1. Завантаження реальних даних з ЛР №1
    df = load_or_parse_data("population_data.csv")
    x_real = df['Year'].values
    y_real = df['Population'].values
    n_samples = len(y_real)

    print(f"\nОбсяг вибірки (N): {n_samples} спостережень ({x_real[0]} - {x_real[-1]} рр.)")

    # 2. Генерація вибірки з АВ та очищення від аномалій
    y_synth_base, y_with_av, true_av_idx, sigma_noise = generate_data_with_anomalies(
        x_real, y_real, anomaly_ratio=0.18, q_av=4.2
    )
    y_cleaned, detected_av_idx = detect_and_clean_anomalies(x_real, y_with_av, threshold_z=1.5)

    print("\nЕТАП 1: ОЧИЩЕННЯ ДАНИХ ВІД АНОМАЛЬНИХ ВИМІРІВ (АВ):")
    print(f"Згенеровано аномалій у точках (роки): {x_real[true_av_idx]}")
    print(f"Виявлено алгоритмом аномалії (роки): {x_real[detected_av_idx]}")
    print(f"СКВ вибірки ДО очищення від АВ:    {np.std(y_with_av, ddof=1):,.2f}")
    print(f"СКВ вибірки ПІСЛЯ очищення від АВ: {np.std(y_cleaned, ddof=1):,.2f}")
    print(f"СКВ еталонної вибірки без АВ:      {np.std(y_synth_base, ddof=1):,.2f}")

    # 3. Оптимізація та МНК навчання поліноміальної моделі
    models_dict, best_deg = optimize_polynomial_model(x_real, y_cleaned, degrees=(1, 2, 3, 4))
    best_model = models_dict[best_deg]

    print("\nЕТАП 2: ОЦІНЮВАННЯ ЯКОСТІ ТА ОПТИМІЗАЦІЯ МОДЕЛІ (МНК / LSM)")
    print(f"{'Степінь (d)':<12} | {'R^2':<10} | {'Adj R^2':<10} | {'RMSE':<15} | {'AIC':<12}")
    print("-" * 68)
    for d, info in models_dict.items():
        marker = " <-- ОПТИМАЛЬНА" if d == best_deg else ""
        print(f"{d:<12} | {info['R2']:<10.4f} | {info['Adj_R2']:<10.4f} | {info['RMSE']:<15,.2f} | {info['AIC']:<12.2f}{marker}")

    # Додатково застосовуємо рекурентний Alpha-Beta фільтр Калмана
    y_ab_filtered = alpha_beta_filter(x_real, y_cleaned, alpha=0.5, beta=0.15)

    # 4. Прогнозування на 0.5 об'єму вибірки
    x_fut, y_fut, x_full, y_full_trend, ci = extrapolate_process(
        x_real, y_cleaned, best_model, ratio=0.5
    )

    print("\n ЕТАП 3: ЕКСТРАПОЛЯЦІЯ НА 0.5 ВИБІРКИ ")
    print(f"Кількість точок прогнозу (0.5 * N): {len(x_fut)} кроків вперед (до {x_fut[-1]} року)")
    for yr, val in zip(x_fut[:5], y_fut[:5]):
        print(f"  Прогноз на {yr} рік: {val:,.0f} осіб")


    # 5. Візуалізація результатів
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Графік 1: Виявлення та очищення аномальних вимірів
    ax1 = axes[0, 0]
    ax1.plot(x_real, y_with_av, '--', color='gray', alpha=0.6, label='Вибірка з аномаліями (АВ)')
    ax1.scatter(x_real[detected_av_idx], y_with_av[detected_av_idx], color='red', s=70, zorder=5, label='Виявлені АВ')
    ax1.plot(x_real, y_cleaned, marker='o', color='blue', linewidth=2, label='Очищена вибірка')
    ax1.set_title("1. Виявлення та очищення вхідних даних від аномалій (АВ)")
    ax1.set_xlabel("Рік")
    ax1.set_ylabel("Населення")
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend()

    # Графік 2: Порівняння поліноміальних моделей МНК (Оптимізація)
    ax2 = axes[0, 1]
    ax2.scatter(x_real, y_cleaned, color='black', s=30, label='Очищені дані')
    colors = {1: 'orange', 2: 'red', 3: 'green', 4: 'purple'}
    for d, info in models_dict.items():
        lw = 2.5 if d == best_deg else 1.2
        ls = '-' if d == best_deg else '--'
        ax2.plot(x_real, info['y_pred'], color=colors[d], linewidth=lw, linestyle=ls,
                 label=f"Поліном d={d} (Adj R²={info['Adj_R2']:.3f})")
    ax2.set_title("2. Статистичне навчання МНК та вибір оптимальної моделі")
    ax2.set_xlabel("Рік")
    ax2.set_ylabel("Населення")
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()

    # Графік 3: Реальні дані vs Оптимальна МНК-модель та Alpha-Beta фільтр
    ax3 = axes[1, 0]
    ax3.plot(x_real, y_real, marker='o', color='royalblue', alpha=0.8, label='Реальні дані (Worldometers)')
    ax3.plot(x_real, best_model['y_pred'], color='red', linewidth=2.2, label=f'МНК-тренд (d={best_deg})')
    ax3.plot(x_real, y_ab_filtered, color='limegreen', linestyle='-.', linewidth=2, label='Alpha-Beta фільтр Калмана')
    ax3.set_title("3. Верифікація: Реальні дані, МНК-регресія та α-β фільтр")
    ax3.set_xlabel("Рік")
    ax3.set_ylabel("Населення")
    ax3.grid(True, linestyle=':', alpha=0.7)
    ax3.legend()

    # Графік 4: Прогнозування на 0.5 інтервалу спостереження
    ax4 = axes[1, 1]
    ax4.plot(x_real, y_cleaned, marker='o', color='blue', label='Історична очищена вибірка')
    ax4.plot(x_full, y_full_trend, '--', color='red', linewidth=2, label=f'Лінія тренду (d={best_deg})')
    ax4.plot(x_fut, y_fut, marker='s', color='darkorange', linewidth=2.5, label='Прогноз (0.5 обсягу вибірки)')
    ax4.fill_between(x_fut, y_fut - ci, y_fut + ci, color='orange', alpha=0.25, label='95% довірчий інтервал')
    ax4.axvline(x=x_real[-1], color='black', linestyle=':', label='Межа спостереження')
    ax4.set_title("4. Екстраполяція процесу на 0.5 вибірки")
    ax4.set_xlabel("Рік")
    ax4.set_ylabel("Населення")
    ax4.grid(True, linestyle=':', alpha=0.7)
    ax4.legend()

    plt.tight_layout()
    plt.show()