import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from window_models import (
    fetch_weather_series,
    check_stationarity_properties,
)
from rd_optimization import (
    compare_all_methods,
    extrapolate_multi_horizons,
)

plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams["font.size"] = 10


def plot_smoothing_and_comparisons(series: pd.Series, comp_res: dict):
    s_dict = comp_res["series_dict"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Обробка Time Series алгоритмами ковзного вікна та статистичного навчання",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Глобальний вигляд ряду та методів згладжування
    axes[0, 0].plot(series.index, series, color="lightgray", linewidth=1.2, label="Вихідний ряд (Open-Meteo)")
    axes[0, 0].plot(series.index, s_dict["sma_smooth"], color="steelblue", linewidth=1.6, label="SMA (w=7)")
    axes[0, 0].plot(series.index, s_dict["ema_smooth"], color="darkorange", linewidth=1.6, label="EMA (span=7)")
    axes[0, 0].plot(series.index, s_dict["harm_fitted"], color="black", linestyle="--", linewidth=2, label="Гармонічна МНК (ЛР4)")
    axes[0, 0].set_title("1. Апроксимація та згладжування на повному інтервалі (731 доба)")
    axes[0, 0].set_ylabel("Температура, °C")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper right")

    # 2. Детальний фрагмент (120 діб) для демонстрації фазового запізнення вікон
    zoom_slice = slice("2025-03-01", "2025-06-30")
    axes[0, 1].plot(series.loc[zoom_slice].index, series.loc[zoom_slice], "o-", color="gray", alpha=0.5, markersize=3, label="Реальні виміри")
    axes[0, 1].plot(series.loc[zoom_slice].index, s_dict["sma_smooth"].loc[zoom_slice], color="steelblue", linewidth=2, label="SMA (w=7)")
    axes[0, 1].plot(series.loc[zoom_slice].index, s_dict["wma_smooth"].loc[zoom_slice], color="seagreen", linewidth=2, label="WMA (w=7)")
    axes[0, 1].plot(series.loc[zoom_slice].index, s_dict["ema_smooth"].loc[zoom_slice], color="darkorange", linewidth=2, label="EMA (span=7)")
    axes[0, 1].plot(series.loc[zoom_slice].index, s_dict["ab_smooth"].loc[zoom_slice], color="crimson", linestyle="--", linewidth=1.8, label="Адаптивний α-β (ЛР3)")
    axes[0, 1].set_title("2. Локальна динаміка (весна-літо 2025): порівняння запізнення методів")
    axes[0, 1].set_ylabel("Температура, °C")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. Залишки після виділення детермінованої складової (вхід для ARMA/ARIMA)
    residuals = comp_res["harm_res"]["residuals"]
    axes[1, 0].plot(residuals.index, residuals, color="purple", alpha=0.75, label="Залишки декомпозиції e(t)")
    axes[1, 0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_title("3. Стаціонарні залишки ряду для моделювання ARMA/ARIMA")
    axes[1, 0].set_ylabel("Відхилення, °C")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 4. Стовпчикова діаграма KPI (RMSE різних підходів)
    kpi_df = comp_res["kpi_df"].iloc[3:].copy()
    short_names = [
        "SMA (прогн.)",
        "WMA (прогн.)",
        "EMA (прогн.)",
        "α-β фільтр (згл.)",
        "α-β фільтр (прогн.)",
        "Гармонічна МНК",
        "Чиста ARIMA",
        "Гібрид + ARMA",
    ]
    y_pos = np.arange(len(kpi_df))
    bars = axes[1, 1].barh(y_pos, kpi_df["RMSE"], color="teal", alpha=0.85, edgecolor="black")
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels(short_names)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("RMSE (°C) — менше значення є кращим")
    axes[1, 1].set_title("4. Порівняння ефективності методів за індикатором RMSE")
    axes[1, 1].grid(True, axis="x", alpha=0.3)
    for bar in bars:
        width = bar.get_width()
        axes[1, 1].text(width + 0.05, bar.get_y() + bar.get_height() / 2, f"{width:.2f}", va="center", fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("fig1_ma_and_kpi_comparison.png", dpi=150)
    plt.show()


def plot_rd_optimization(series: pd.Series, comp_res: dict):
    ma_df = comp_res["ma_opt"]["grid_df"]
    arima_df = comp_res["arima_resid_opt"]["results_df"].head(10)
    residuals = comp_res["harm_res"]["residuals"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "R&D дослідження: оптимізація параметрів ковзного вікна та порядку ARIMA",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Залежність похибки однокрокового прогнозу від ширини вікна
    axes[0, 0].plot(ma_df["window"], ma_df["SMA_Pred_RMSE"], "o-", label="SMA Прогноз RMSE", color="steelblue", markersize=4)
    axes[0, 0].plot(ma_df["window"], ma_df["WMA_Pred_RMSE"], "s-", label="WMA Прогноз RMSE", color="seagreen", markersize=4)
    axes[0, 0].plot(ma_df["window"], ma_df["EMA_Pred_RMSE"], "^-", label="EMA Прогноз RMSE", color="darkorange", markersize=4)
    best_w = comp_res["ma_opt"]["best_ema_w"]
    axes[0, 0].axvline(best_w, color="red", linestyle="--", label=f"Оптимум EMA (w={best_w})")
    axes[0, 0].set_title("1. Оптимізація ширини вікна w за критерієм мінімуму RMSE прогнозу")
    axes[0, 0].set_xlabel("Ширина ковзного вікна w (діб)")
    axes[0, 0].set_ylabel("RMSE прогнозу (°C)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. Коефіцієнт придушення високочастотного шуму vs спотворення сигналу
    ax2_twin = axes[0, 1].twinx()
    l1 = axes[0, 1].plot(ma_df["window"], ma_df["SMA_Smooth_RMSE"], color="crimson", linewidth=2, label="Відхилення від вимірів (RMSE)")
    l2 = ax2_twin.plot(ma_df["window"], ma_df["Noise_Reduction_Ratio"], color="navy", linestyle="--", linewidth=2, label="Коефіцієнт збереження варіації")
    axes[0, 1].set_title("2. R&D Компроміс: згладжування шуму vs динамічне запізнення")
    axes[0, 1].set_xlabel("Ширина ковзного вікна w (діб)")
    axes[0, 1].set_ylabel("RMSE апроксимації (°C)", color="crimson")
    ax2_twin.set_ylabel("Відношення СКВ приростів", color="navy")
    axes[0, 1].grid(True, alpha=0.3)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    axes[0, 1].legend(lines, labels, loc="center right")

    # 3. ACF та PACF залишків для обґрунтування порядків (p, q)
    plot_acf(residuals.dropna(), ax=axes[1, 0], lags=30, title="3. Функція автокореляції (ACF) стаціонарних залишків")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Рейтинг моделей ARMA/ARIMA за інформаційним критерієм Акаїке (AIC)
    x_pos = np.arange(len(arima_df))
    axes[1, 1].bar(x_pos - 0.18, arima_df["AIC"], width=0.35, label="AIC", color="royalblue", edgecolor="black")
    axes[1, 1].bar(x_pos + 0.18, arima_df["BIC"], width=0.35, label="BIC", color="sandybrown", edgecolor="black")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(arima_df["Order"], rotation=25)
    axes[1, 1].set_ylim(arima_df["AIC"].min() - 15, arima_df["BIC"].max() + 20)
    axes[1, 1].set_title("4. Топ-10 конфігурацій ARMA(p,d,q) за критеріями AIC та BIC")
    axes[1, 1].set_xlabel("Порядок моделі (p, d, q)")
    axes[1, 1].set_ylabel("Значення інформаційного критерію")
    axes[1, 1].grid(True, axis="y", alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("fig2_rd_parameter_optimization.png", dpi=150)
    plt.show()


def plot_extrapolation_horizons(series: pd.Series, ext_res: dict):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    fig.suptitle(
        "Екстраполяція Time Series на 0.5, 1.0, 1.5 та 2.0 інтервалу спостереження",
        fontsize=14,
        fontweight="bold",
    )

    # Верхній широкий графік: Повна картина екстраполяції на 0.5, 1.0, 1.5, 2.0 інтервалу
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.plot(series.index, series, color="steelblue", linewidth=1.3, label="Історичні дані (N = 731 доба, 2024–2025)")
    ax_main.plot(
        ext_res["future_dates"],
        ext_res["hybrid_forecast"],
        color="darkorange",
        linewidth=1.8,
        label="Екстраполяція: Детерміновані гармоніки + ARMA(p,q)",
    )
    ax_main.plot(
        ext_res["future_dates"],
        ext_res["raw_forecast"],
        color="crimson",
        linestyle="--",
        linewidth=1.8,
        label="Екстраполяція чистою ARIMA (без сезонної складової)",
    )
    ax_main.fill_between(
        ext_res["future_dates"],
        ext_res["ci_lower"],
        ext_res["ci_upper"],
        color="orange",
        alpha=0.22,
        label="95% довірчий інтервал прогнозу",
    )

    # Вертикальні лінії горизонтів 0.5N, 1.0N, 1.5N, 2.0N
    n_obs = len(series)
    horizon_colors = ["green", "purple", "brown", "black"]
    for m, col in zip(ext_res["multipliers"], horizon_colors):
        step_idx = int(round(m * n_obs)) - 1
        h_date = ext_res["future_dates"][step_idx]
        ax_main.axvline(h_date, color=col, linestyle=":", linewidth=1.8)
        ax_main.text(
            h_date,
            ax_main.get_ylim()[1] * 0.85,
            f" {m}N\n ({step_idx+1} діб)",
            color=col,
            fontweight="bold",
            fontsize=9,
        )

    ax_main.set_title("1. Довгострокова екстраполяція ряду на 0.5N (366 діб), 1.0N (731 доба), 1.5N (1096 діб) та 2.0N (1462 доби)")
    ax_main.set_ylabel("Температура, °C")
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc="upper left", ncol=2)

    # Нижній лівий графік: Перехідна зона (інерція ARMA в перші 45 діб прогнозу)
    ax_zoom = fig.add_subplot(gs[1, 0])
    recent_hist = series.iloc[-45:]
    first_fore = ext_res["hybrid_forecast"].iloc[:45]
    first_raw = ext_res["raw_forecast"].iloc[:45]
    ax_zoom.plot(recent_hist.index, recent_hist, "o-", color="steelblue", markersize=4, label="Останні 45 діб спостережень")
    ax_zoom.plot(first_fore.index, first_fore, "s-", color="darkorange", markersize=4, label="Гібридний прогноз (Гармоніки + ARMA)")
    ax_zoom.plot(first_raw.index, first_raw, "^--", color="crimson", markersize=4, label="Чиста ARIMA")
    ax_zoom.axvline(series.index[-1], color="black", linestyle="--", label="Початок екстраполяції")
    ax_zoom.set_title("2. Короткострокова фаза екстраполяції (перші 45 діб): вплив ARMA-інерції")
    ax_zoom.set_ylabel("Температура, °C")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.legend()

    # Нижній правий графік: Верифікація здатності до екстраполяції (Backtesting на останніх 180 добах)
    ax_back = fig.add_subplot(gs[1, 1])
    test_s = ext_res["test_series"]
    back_pred = ext_res["back_hybrid_pred"]
    ax_back.plot(test_s.index, test_s, color="steelblue", alpha=0.8, label="Фактичні дані (тестові 180 діб)")
    ax_back.plot(back_pred.index, back_pred, color="darkorange", linewidth=2, label="Екстраполяція моделі")
    rmse_val = ext_res["backtest_kpi"].loc[0, "RMSE"]
    ax_back.set_title(f"3. Верифікація екстраполяції на тестовій вибірці 180 діб (Out-of-Sample RMSE = {rmse_val:.2f} °C)")
    ax_back.set_ylabel("Температура, °C")
    ax_back.grid(True, alpha=0.3)
    ax_back.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("fig3_extrapolation_horizons.png", dpi=150)
    plt.show()


def main():
    # 1. Завантаження та перевірка властивостей часового ряду
    series = fetch_weather_series()
    props = check_stationarity_properties(series)

    print("\n" + "=" * 80)
    print("1. ДОСЛІДЖЕННЯ ВЛАСТИВОСТЕЙ ВХІДНОГО ЧАСОВОГО РЯДУ")
    print("=" * 80)
    for k, v in props.items():
        print(f"{k:22s}: {v}")

    # 2. Порівняння алгоритмів ковзного вікна, ARMA/ARIMA та статистичного навчання
    comp_res = compare_all_methods(series)

    print("\n" + "=" * 80)
    print("2. РЕЗУЛЬТАТИ R&D ОПТИМІЗАЦІЇ ПАРАМЕТРІВ")
    print("=" * 80)
    print(f"Оптимальна ширина вікна SMA для однокрокового прогнозу: w = {comp_res['ma_opt']['best_sma_w']} діб")
    print(f"Оптимальна ширина вікна WMA для однокрокового прогнозу: w = {comp_res['ma_opt']['best_wma_w']} діб")
    print(f"Оптимальна ширина вікна EMA для однокрокового прогнозу: w = {comp_res['ma_opt']['best_ema_w']} діб")
    print(f"Оптимальний порядок чистої ARIMA на вихідному ряді    : {comp_res['arima_raw_opt']['best_order']} (AIC = {comp_res['arima_raw_opt']['best_aic']:.2f})")
    print(f"Оптимальний порядок ARMA на стаціонарних залишках     : {comp_res['arima_resid_opt']['best_order']} (AIC = {comp_res['arima_resid_opt']['best_aic']:.2f})")

    print("\n--- Топ-5 моделей ARMA(p,d,q) на залишках за критерієм AIC ---")
    print(comp_res["arima_resid_opt"]["results_df"].head(5).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 80)
    print("3. ПОРІВНЯННЯ МЕТОДІВ КОВЗНОГО ВІКНА ТА СТАТИСТИЧНОГО НАВЧАННЯ ЗА KPI")
    print("=" * 80)
    print(comp_res["kpi_df"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 3. Екстраполяція на 0.5, 1.0, 1.5, 2.0 інтервалу спостереження
    ext_res = extrapolate_multi_horizons(
        series=series,
        harm_res=comp_res["harm_res"],
        arima_resid_model=comp_res["arima_resid_opt"]["best_model"],
        arima_raw_model=comp_res["arima_raw_opt"]["best_model"],
        multipliers=[0.5, 1.0, 1.5, 2.0],
    )

    print("\n" + "=" * 80)
    print("4. ЕКСТРАПОЛЯЦІЯ НА 0.5, 1.0, 1.5 ТА 2.0 ІНТЕРВАЛУ СПОСТЕРЕЖЕННЯ")
    print("=" * 80)
    print(ext_res["horizon_df"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n--- Верифікація точності екстраполяції на відкладеній вибірці (180 діб) ---")
    print(ext_res["backtest_kpi"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 4. Побудова графіків
    plot_smoothing_and_comparisons(series, comp_res)
    plot_rd_optimization(series, comp_res)
    plot_extrapolation_horizons(series, ext_res)


if __name__ == "__main__":
    main()