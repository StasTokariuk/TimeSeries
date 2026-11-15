import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from exp_smoothing_models import (
    fetch_multivariate_weather_data,
    analyze_ts_properties,
    fit_exponential_smoothing_suite,
)
from regression_and_selector import (
    calculate_kpi,
    fit_regression_suite,
    select_optimal_model_by_properties,
)

plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams["font.size"] = 10


def evaluate_all_models_on_series(series: pd.Series, test_size: int = 180) -> dict:
    train_s = series.iloc[:-test_size]
    test_s = series.iloc[-test_size:]

    # Моделі на навчальній вибірці для об'єктивної оцінки екстраполяції на тест (180 діб)
    es_test_suite = fit_exponential_smoothing_suite(train_s, forecast_steps=test_size)
    reg_test_suite = fit_regression_suite(train_s, forecast_steps=test_size)

    # Моделі на повному ряді (731 доба) для оцінки апроксимації та екстраполяції в майбутнє (365 діб)
    future_steps = 365
    es_full_suite = fit_exponential_smoothing_suite(series, forecast_steps=future_steps)
    reg_full_suite = fit_regression_suite(series, forecast_steps=future_steps)

    kpi_rows = []

    # Збір KPI для експоненціального згладжування
    for name in ["SES", "Holt_Linear", "Holt_Damped", "Holt_Winters"]:
        in_sample_kpi = calculate_kpi(
            series.to_numpy(),
            es_full_suite[name]["fitted"].to_numpy(),
            n_params=len(es_full_suite[name]["params"]),
        )
        out_sample_kpi = calculate_kpi(
            test_s.to_numpy(),
            es_test_suite[name]["forecast"].to_numpy(),
            n_params=len(es_test_suite[name]["params"]),
        )
        kpi_rows.append({
            "Сімейство": "Експоненціальне",
            "Модель": name,
            "InSample_RMSE": in_sample_kpi["RMSE"],
            "InSample_MAE": in_sample_kpi["MAE"],
            "InSample_R2": in_sample_kpi["R2"],
            "InSample_AIC": in_sample_kpi["AIC"],
            "Test180_RMSE": out_sample_kpi["RMSE"],
            "Test180_MAE": out_sample_kpi["MAE"],
            "Test180_MAPE_%": out_sample_kpi["MAPE_%"],
        })

    # Збір KPI для регресійних моделей
    for name in ["Poly_OLS", "Harmonic_Ridge", "Dynamic_Lag_Reg"]:
        in_sample_kpi = calculate_kpi(
            series.to_numpy(),
            reg_full_suite[name]["fitted"].to_numpy(),
            n_params=reg_full_suite[name]["n_params"],
        )
        out_sample_kpi = calculate_kpi(
            test_s.to_numpy(),
            reg_test_suite[name]["forecast"].to_numpy(),
            n_params=reg_test_suite[name]["n_params"],
        )
        kpi_rows.append({
            "Сімейство": "Регресійне",
            "Модель": name,
            "InSample_RMSE": in_sample_kpi["RMSE"],
            "InSample_MAE": in_sample_kpi["MAE"],
            "InSample_R2": in_sample_kpi["R2"],
            "InSample_AIC": in_sample_kpi["AIC"],
            "Test180_RMSE": out_sample_kpi["RMSE"],
            "Test180_MAE": out_sample_kpi["MAE"],
            "Test180_MAPE_%": out_sample_kpi["MAPE_%"],
        })

    kpi_df = pd.DataFrame(kpi_rows)

    return {
        "train_s": train_s,
        "test_s": test_s,
        "es_test": es_test_suite,
        "reg_test": reg_test_suite,
        "es_full": es_full_suite,
        "reg_full": reg_full_suite,
        "kpi_df": kpi_df,
    }


def plot_group1_exponential_smoothing(series: pd.Series, eval_res: dict):
    es_full = eval_res["es_full"]
    es_test = eval_res["es_test"]
    train_s = eval_res["train_s"]
    test_s = eval_res["test_s"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Експоненціальне згладжування та екстраполяція Time Series (SES, Holt, Holt-Winters)",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Апроксимація на повному інтервалі + екстраполяція на 365 діб уперед
    axes[0, 0].plot(series.index, series, color="lightgray", linewidth=1.2, label="Вихідний ряд (Open-Meteo)")
    axes[0, 0].plot(series.index, es_full["Holt_Winters"]["fitted"], color="darkorange", linewidth=1.5, label="Holt-Winters (згладжування)")
    axes[0, 0].plot(es_full["Holt_Winters"]["forecast"].index, es_full["Holt_Winters"]["forecast"], color="darkorange", linestyle="--", linewidth=2, label="Holt-Winters (прогноз 365 діб)")
    axes[0, 0].plot(es_full["Holt_Damped"]["forecast"].index, es_full["Holt_Damped"]["forecast"], color="steelblue", linestyle="-.", linewidth=2, label="Holt Damped (прогноз)")
    axes[0, 0].plot(es_full["Holt_Linear"]["forecast"].index, es_full["Holt_Linear"]["forecast"], color="crimson", linestyle=":", linewidth=2, label="Holt Linear (прогноз)")
    axes[0, 0].set_title("1. Апроксимація та екстраполяція на 1 рік (365 діб) моделями експоненціального згладжування")
    axes[0, 0].set_ylabel("Температура, °C")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper left", fontsize=8.5)

    # 2. Локальна динаміка згладжування (фрагмент 120 діб)
    zoom_slice = slice("2025-03-01", "2025-06-30")
    axes[0, 1].plot(series.loc[zoom_slice].index, series.loc[zoom_slice], "o-", color="gray", alpha=0.5, markersize=3, label="Фактичні виміри")
    axes[0, 1].plot(series.loc[zoom_slice].index, es_full["SES"]["fitted"].loc[zoom_slice], color="royalblue", linewidth=1.8, label="SES (одноетапне)")
    axes[0, 1].plot(series.loc[zoom_slice].index, es_full["Holt_Damped"]["fitted"].loc[zoom_slice], color="seagreen", linewidth=1.8, label="Holt Damped (подвійне)")
    axes[0, 1].plot(series.loc[zoom_slice].index, es_full["Holt_Winters"]["fitted"].loc[zoom_slice], color="darkorange", linewidth=1.8, label="Holt-Winters (потрійне)")
    axes[0, 1].set_title("2. Порівняння реакції SES, Holt та Holt-Winters на локальному інтервалі")
    axes[0, 1].set_ylabel("Температура, °C")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. Верифікація екстраполяції на тестовій вибірці (180 діб)
    axes[1, 0].plot(train_s.iloc[-60:].index, train_s.iloc[-60:], color="steelblue", label="Навчальні дані (останні 60 діб)")
    axes[1, 0].plot(test_s.index, test_s, color="black", alpha=0.7, linewidth=1.5, label="Тестова вибірка (180 діб)")
    axes[1, 0].plot(es_test["SES"]["forecast"].index, es_test["SES"]["forecast"], "--", color="royalblue", linewidth=2, label="SES прогноз")
    axes[1, 0].plot(es_test["Holt_Linear"]["forecast"].index, es_test["Holt_Linear"]["forecast"], ":", color="crimson", linewidth=2, label="Holt Linear прогноз")
    axes[1, 0].plot(es_test["Holt_Damped"]["forecast"].index, es_test["Holt_Damped"]["forecast"], "-.", color="seagreen", linewidth=2, label="Holt Damped прогноз")
    axes[1, 0].plot(es_test["Holt_Winters"]["forecast"].index, es_test["Holt_Winters"]["forecast"], "-", color="darkorange", linewidth=2.2, label="Holt-Winters прогноз")
    axes[1, 0].set_title("3. Верифікація точності екстраполяції на відкладеній вибірці (180 діб)")
    axes[1, 0].set_ylabel("Температура, °C")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8.5)

    # 4. Порівняння RMSE моделей(In-Sample vs Out-of-Sample)
    es_kpi = eval_res["kpi_df"][eval_res["kpi_df"]["Сімейство"] == "Експоненціальне"]
    x = np.arange(len(es_kpi))
    w = 0.35
    axes[1, 1].bar(x - w / 2, es_kpi["InSample_RMSE"], width=w, color="teal", edgecolor="black", label="In-Sample RMSE (Апроксимація)")
    axes[1, 1].bar(x + w / 2, es_kpi["Test180_RMSE"], width=w, color="coral", edgecolor="black", label="Out-of-Sample RMSE (Екстраполяція 180 діб)")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(es_kpi["Модель"])
    axes[1, 1].set_ylabel("RMSE (°C)")
    axes[1, 1].set_title("4. KPI моделей експоненціального згладжування (Апроксимація vs Прогноз)")
    axes[1, 1].grid(True, axis="y", alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("fig1_exponential_smoothing.png", dpi=150)
    plt.show()


def plot_group2_regression_models(series: pd.Series, eval_res: dict):
    reg_full = eval_res["reg_full"]
    reg_test = eval_res["reg_test"]
    train_s = eval_res["train_s"]
    test_s = eval_res["test_s"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Класичні та динамічні регресійні моделі обробки Time Series",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Апроксимація та екстраполяція на 365 діб регресійними моделями
    axes[0, 0].plot(series.index, series, color="lightgray", linewidth=1.2, label="Вихідний ряд (Open-Meteo)")
    axes[0, 0].plot(series.index, reg_full["Poly_OLS"]["fitted"], color="crimson", linestyle="--", linewidth=2, label="Поліноміальна МНК (deg=3)")
    axes[0, 0].plot(series.index, reg_full["Harmonic_Ridge"]["fitted"], color="navy", linewidth=2, label="Гармонічна Ridge-регресія")
    axes[0, 0].plot(reg_full["Harmonic_Ridge"]["forecast"].index, reg_full["Harmonic_Ridge"]["forecast"], color="navy", linestyle="--", linewidth=2, label="Гармонічна Ridge (прогноз 365 діб)")
    axes[0, 0].plot(reg_full["Dynamic_Lag_Reg"]["forecast"].index, reg_full["Dynamic_Lag_Reg"]["forecast"], color="darkorange", linestyle=":", linewidth=2, label="Динамічна авторегресія (прогноз)")
    axes[0, 0].set_title("1. Регресійне моделювання та екстраполяція на 365 діб (без вибуху полінома)")
    axes[0, 0].set_ylabel("Температура, °C")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper left", fontsize=8.5)

    # 2. Верифікація регресійних моделей на тестовій вибірці (180 діб)
    axes[0, 1].plot(train_s.iloc[-60:].index, train_s.iloc[-60:], color="steelblue", label="Навчальні дані (останні 60 діб)")
    axes[0, 1].plot(test_s.index, test_s, color="black", alpha=0.75, linewidth=1.5, label="Тестова вибірка (180 діб)")
    axes[0, 1].plot(reg_test["Harmonic_Ridge"]["forecast"].index, reg_test["Harmonic_Ridge"]["forecast"], color="navy", linewidth=2.2, label="Harmonic Ridge (прогноз)")
    axes[0, 1].plot(reg_test["Dynamic_Lag_Reg"]["forecast"].index, reg_test["Dynamic_Lag_Reg"]["forecast"], "--", color="darkorange", linewidth=2.2, label="Dynamic Lag-Reg (прогноз)")
    # Обмежуємо відображення поліноміальної регресії, щоб показати її розходження
    poly_clip = np.clip(reg_test["Poly_OLS"]["forecast"], -25, 40)
    axes[0, 1].plot(reg_test["Poly_OLS"]["forecast"].index, poly_clip, ":", color="crimson", linewidth=2, label="Polynomial OLS (розходження)")
    axes[0, 1].set_title("2. Верифікація регресійної екстраполяції на тестовій вибірці (180 діб)")
    axes[0, 1].set_ylabel("Температура, °C")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8.5)

    # 3. Діаграма розсіювання (Фактичні vs Прогнозовані значення для Dynamic Lag-Reg та Harmonic Ridge)
    axes[1, 0].scatter(series, reg_full["Harmonic_Ridge"]["fitted"], alpha=0.45, color="navy", s=18, label="Harmonic Ridge (R² = 0.85)")
    axes[1, 0].scatter(series, reg_full["Dynamic_Lag_Reg"]["fitted"], alpha=0.45, color="darkorange", s=18, label="Dynamic Lag-Reg (R² = 0.95)")
    min_v, max_v = float(series.min()), float(series.max())
    axes[1, 0].plot([min_v, max_v], [min_v, max_v], "k--", linewidth=2, label="Ідеальна лінія y = ŷ")
    axes[1, 0].set_title("3. Діаграма точності регресійної апроксимації (Actual vs Predicted)")
    axes[1, 0].set_xlabel("Фактична температура (°C)")
    axes[1, 0].set_ylabel("Модельна оцінка (°C)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 4. Порівняння залишків регресійних моделей
    resid_harm = series - reg_full["Harmonic_Ridge"]["fitted"]
    resid_dyn = series - reg_full["Dynamic_Lag_Reg"]["fitted"]
    axes[1, 1].hist(resid_harm, bins=30, density=True, alpha=0.55, color="navy", edgecolor="black", label=f"Harmonic Ridge (STD={resid_harm.std():.2f}°C)")
    axes[1, 1].hist(resid_dyn, bins=30, density=True, alpha=0.55, color="darkorange", edgecolor="black", label=f"Dynamic Lag-Reg (STD={resid_dyn.std():.2f}°C)")
    axes[1, 1].set_title("4. Розподіл похибок апроксимації класичної та динамічної регресії")
    axes[1, 1].set_xlabel("Похибка e(t), °C")
    axes[1, 1].set_ylabel("Щільність")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("fig2_regression_models.png", dpi=150)
    plt.show()


def plot_group3_rd_comparison_and_selector( eval_res: dict, multi_df: pd.DataFrame, selector_table: pd.DataFrame):
    kpi_df = eval_res["kpi_df"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Порівняння експоненціальних і регресійних моделей та селектор за властивостями даних",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Рейтинг усіх 7 моделей за точністю прогнозу на 180 діб (Out-of-Sample RMSE)
    sorted_kpi = kpi_df.sort_values("Test180_RMSE").reset_index(drop=True)
    colors = ["royalblue" if "Регрес" in fam else "darkorange" for fam in sorted_kpi["Сімейство"]]
    y_pos = np.arange(len(sorted_kpi))
    bars = axes[0, 0].barh(y_pos, sorted_kpi["Test180_RMSE"], color=colors, edgecolor="black", alpha=0.85)
    axes[0, 0].set_yticks(y_pos)
    axes[0, 0].set_yticklabels(sorted_kpi["Модель"])
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("Out-of-Sample RMSE на 180 добах (°C) — менше краще")
    axes[0, 0].set_title("1. Підсумковий рейтинг моделей за точністю екстраполяції (Синій — Регресія, Помаранч. — Експ.)")
    axes[0, 0].grid(True, axis="x", alpha=0.3)
    for bar in bars:
        w = bar.get_width()
        axes[0, 0].text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va="center", fontweight="bold")

    # 2. Порівняння за інформаційним критерієм Акаїке (AIC) на всій вибірці
    sorted_aic = kpi_df.sort_values("InSample_AIC").reset_index(drop=True)
    colors_aic = ["royalblue" if "Регрес" in fam else "darkorange" for fam in sorted_aic["Сімейство"]]
    x_pos = np.arange(len(sorted_aic))
    axes[0, 1].bar(x_pos, sorted_aic["InSample_AIC"], color=colors_aic, edgecolor="black", alpha=0.85)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(sorted_aic["Модель"], rotation=20)
    axes[0, 1].set_ylabel("Інформаційний критерій AIC (менше краще)")
    axes[0, 1].set_title("2. Оцінка якості моделей за критерієм Акаїке (AIC)")
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    # 3. Карта простору властивостей часових рядів (Seasonal Strength vs ACF_Lag1) для роботи Селектора
    axes[1, 0].axvspan(0.45, 1.05, color="lightblue", alpha=0.3, label="Зона Гармонічної / Динамічної Регресії (Fs ≥ 0.45)")
    axes[1, 0].axvspan(-0.05, 0.45, ymin=0.35, ymax=1.0, color="bisque", alpha=0.4, label="Зона Динамічної Регресії / Holt Damped")
    axes[1, 0].axvspan(-0.05, 0.45, ymin=0.0, ymax=0.35, color="lightgreen", alpha=0.3, label="Зона SES / Простих експоненціальних моделей")

    for _, row in selector_table.iterrows():
        axes[1, 0].scatter(row["Fs (Сезонність)"], row["ACF(1)"], s=140, edgecolor="black", zorder=5)
        axes[1, 0].text(
            row["Fs (Сезонність)"] + 0.02,
            row["ACF(1)"] - 0.02,
            f"{row['Ряд']} -> {row['Обрана модель']}",
            fontweight="bold",
            fontsize=9.5,
        )

    axes[1, 0].set_xlim(-0.05, 1.05)
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].set_xlabel("Сила сезонності (Fs)")
    axes[1, 0].set_ylabel("Автокореляція 1-го порядку ACF(1)")
    axes[1, 0].set_title("3. Карта прийняття рішень R&D-селектора моделей за властивостями даних")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc="lower right", fontsize=8)

    # 4. Верифікація обраної селектором моделі для стаціонарного ряду (Швидкість вітру / Тиск)
    wind_s = multi_df["WindSpeed"].iloc[-120:]
    wind_ses = fit_exponential_smoothing_suite(multi_df["WindSpeed"].iloc[:-30], forecast_steps=30)["SES"]
    axes[1, 1].plot(wind_s.index, wind_s, color="gray", alpha=0.6, label="WindSpeed (Фактичні виміри)")
    axes[1, 1].plot(wind_ses["fitted"].iloc[-90:].index, wind_ses["fitted"].iloc[-90:], color="seagreen", linewidth=2, label="SES Згладжування (обрано селектором)")
    axes[1, 1].plot(wind_ses["forecast"].index, wind_ses["forecast"], "--", color="crimson", linewidth=2.2, label="SES Прогноз (30 діб)")
    axes[1, 1].set_title("4. Приклад автоматичного перемикання селектора на SES для стаціонарного ряду (WindSpeed)")
    axes[1, 1].set_ylabel("Швидкість вітру, км/год")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("fig3_rd_selector_and_comparison.png", dpi=150)
    plt.show()


def main():
    # 1. Завантаження даних
    df = fetch_multivariate_weather_data()
    target_series = df["Temperature"]

    # 2. Дослідження властивостей та робота R&D-селектора для всіх 4 метеопроцесів
    selector_rows = []
    for col in df.columns:
        props = analyze_ts_properties(df[col])
        rec_short = select_optimal_model_by_properties(props, horizon="short")
        rec_long = select_optimal_model_by_properties(props, horizon="long")
        selector_rows.append({
            "Ряд": col,
            "Mean": props["Mean"],
            "Std": props["Std"],
            "ADF_p": props["ADF_p"],
            "Стаціонарний": props["Is_Stationary"],
            "Ft (Тренд)": props["Trend_Strength"],
            "Fs (Сезонність)": props["Seasonal_Strength"],
            "ACF(1)": props["ACF_Lag1"],
            "Hurst": props["Hurst"],
            "Вибір (Короткий гориз.)": rec_short["Recommended_Model"],
            "Обрана модель": rec_long["Recommended_Model"],
            "Обґрунтування селектора": rec_long["Justification"],
        })

    selector_df = pd.DataFrame(selector_rows)

    print("\n" + "=" * 95)
    print("1. ВЛАСТИВОСТІ ЧАСОВИХ РЯДІВ ТА АВТОМАТИЧНИЙ ВИБІР МОДЕЛІ")
    print("=" * 95)
    cols_to_show = [
        "Ряд", "Mean", "Std", "ADF_p", "Стаціонарний",
        "Ft (Тренд)", "Fs (Сезонність)", "ACF(1)", "Вибір (Короткий гориз.)", "Обрана модель"
    ]
    print(selector_df[cols_to_show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n--- Логічне обґрунтування рішень алгоритму вибору ---")
    for _, r in selector_df.iterrows():
        print(f"* {r['Ряд']:12s}: {r['Обрана модель']:16s} | {r['Обґрунтування селектора']}")

    # 3. Повне порівняння експоненціальних та регресійних моделей на основному ряді (Temperature)
    eval_res = evaluate_all_models_on_series(target_series, test_size=180)

    print("\n" + "=" * 95)
    print("2. ОПТИМІЗОВАНІ ПАРАМЕТРИ МОДЕЛЕЙ ЕКСПОНЕНЦІАЛЬНОГО ЗГЛАДЖУВАННЯ")
    print("=" * 95)
    for name, m_data in eval_res["es_full"].items():
        params_str = ", ".join([f"{k}={v:.4f}" for k, v in m_data["params"].items()])
        print(f"{name:15s} | AIC = {m_data['aic']:8.2f} | Параметри: {params_str}")

    print("\n" + "=" * 95)
    print("3. ЗВЕДЕНА ТАБЛИЦЯ KPI: ЕКСПОНЕНЦІАЛЬНЕ ЗГЛАДЖУВАННЯ VS РЕГРЕСІЙНІ МОДЕЛІ")
    print("=" * 95)
    print(eval_res["kpi_df"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 4. Візуалізація результатів
    plot_group1_exponential_smoothing(target_series, eval_res)
    plot_group2_regression_models(target_series, eval_res)
    plot_group3_rd_comparison_and_selector(eval_res, df, selector_df)


if __name__ == "__main__":
    main()