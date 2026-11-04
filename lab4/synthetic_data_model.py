import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf
from real_data_analysis import decompose_series, analyze_properties


def synthesize_time_series(real_series: pd.Series, stl_res, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    n = len(real_series)
    t = np.arange(n, dtype=float)

    # 1. Апроксимація тренду поліномом 2-го степеня
    trend_vals = stl_res.trend.to_numpy()
    poly_coeffs = np.polyfit(t, trend_vals, deg=2)
    syn_trend = np.polyval(poly_coeffs, t)

    # 2. Апроксимація сезонності сумою 2 гармонік Фур'є (річна та піврічна)
    seasonal_vals = stl_res.seasonal.to_numpy()
    omega = 2.0 * np.pi / 365.25
    X_harm = np.column_stack([
        np.sin(omega * t),
        np.cos(omega * t),
        np.sin(2 * omega * t),
        np.cos(2 * omega * t),
    ])
    harm_coeffs, _, _, _ = np.linalg.lstsq(X_harm, seasonal_vals, rcond=None)
    syn_seasonal = X_harm @ harm_coeffs

    # 3. Моделювання залишків як авторегресійного процесу
    # Реальні метеорологічні залишки мають короткострокову інерцію (ACF(1) ~ 0.6-0.7)
    resid_vals = stl_res.resid.to_numpy()
    phi_1 = float(np.corrcoef(resid_vals[1:], resid_vals[:-1])[0, 1])
    resid_std = float(np.std(resid_vals))
    innovation_std = resid_std * np.sqrt(max(1.0 - phi_1 ** 2, 1e-6))

    syn_resid = np.zeros(n, dtype=float)
    innovations = np.random.normal(0.0, innovation_std, size=n)
    syn_resid[0] = innovations[0]
    for i in range(1, n):
        syn_resid[i] = phi_1 * syn_resid[i - 1] + innovations[i]

    syn_total = syn_trend + syn_seasonal + syn_resid

    df_syn = pd.DataFrame({
        "Synthetic": syn_total,
        "Syn_Trend": syn_trend,
        "Syn_Seasonal": syn_seasonal,
        "Syn_Resid": syn_resid,
    }, index=real_series.index)

    model_params = {
        "poly_coeffs": poly_coeffs,
        "harm_coeffs": harm_coeffs,
        "ar1_phi": phi_1,
        "innovation_std": innovation_std,
    }
    return df_syn, model_params


def evaluate_model_adequacy(real_series: pd.Series, syn_series: pd.Series, real_props: dict) -> dict:
    syn_decomp = decompose_series(syn_series, period=365)
    syn_props = analyze_properties(syn_series, syn_decomp["stl"])

    real_vals = real_series.to_numpy()
    syn_vals = syn_series.to_numpy()

    # Mетрики узгодженості траєкторій
    rmse = float(np.sqrt(np.mean((real_vals - syn_vals) ** 2)))
    mae = float(np.mean(np.abs(real_vals - syn_vals)))
    corr_trajectory = float(np.corrcoef(real_vals, syn_vals)[0, 1])

    # Порівняння автокореляційних функцій (ACF) до лагу 60
    acf_real = acf(real_vals, nlags=60, fft=True)
    acf_syn = acf(syn_vals, nlags=60, fft=True)
    acf_similarity = float(np.corrcoef(acf_real, acf_syn)[0, 1])

    # Двовибірковий тест Колмогорова-Смірнова на однорідність розподілів
    ks_stat, ks_pval = stats.ks_2samp(real_vals, syn_vals)

    comparison_table = pd.DataFrame({
        "Показник": [
            "Середнє значення (Mean)",
            "Стандартне відхилення (Std)",
            "Медіана (Median)",
            "Міжквартильний розмах (IQR)",
            "Коефіцієнт асиметрії (Skewness)",
            "Коефіцієнт ексцесу (Kurtosis)",
            "Показник Херста (Hurst)",
            "Сила тренду (Trend Strength)",
            "Сила сезонності (Seasonal Strength)",
            "Автокореляція лаг 1 (ACF Lag 1)",
            "Автокореляція лаг 7 (ACF Lag 7)",
        ],
        "Реальний ряд": [
            real_props["Mean"],
            real_props["Std"],
            real_props["Median"],
            real_props["IQR"],
            real_props["Skewness"],
            real_props["Kurtosis"],
            real_props["Hurst_Exponent"],
            real_props["Trend_Strength"],
            real_props["Seasonal_Strength"],
            real_props["ACF_Lag1"],
            real_props["ACF_Lag7"],
        ],
        "Синтетична модель": [
            syn_props["Mean"],
            syn_props["Std"],
            syn_props["Median"],
            syn_props["IQR"],
            syn_props["Skewness"],
            syn_props["Kurtosis"],
            syn_props["Hurst_Exponent"],
            syn_props["Trend_Strength"],
            syn_props["Seasonal_Strength"],
            syn_props["ACF_Lag1"],
            syn_props["ACF_Lag7"],
        ],
    })
    comparison_table["Абс. різниця"] = np.abs(
        comparison_table["Реальний ряд"] - comparison_table["Синтетична модель"]
    )

    return {
        "comparison_table": comparison_table,
        "trajectory_rmse": rmse,
        "trajectory_mae": mae,
        "trajectory_corr": corr_trajectory,
        "acf_real": acf_real,
        "acf_syn": acf_syn,
        "acf_similarity": acf_similarity,
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pval),
        "syn_props": syn_props,
    }