import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from real_data_analysis import (
    fetch_multivariate_weather_data,
    decompose_series,
    analyze_properties,
    correlation_analysis,
    cluster_time_series,
)
from synthetic_data_model import (
    synthesize_time_series,
    evaluate_model_adequacy,
)

plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["font.size"] = 10

# Графік 1: STL-декомпозиція та автокореляційна структура реального ряду.
def plot_decomposition_and_properties(series: pd.Series, stl_res):
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle("Декомпозиція (STL) та діагностика властивостей реального ряду (Температура)", fontsize=14,
                 fontweight="bold")

    # Ліва колонка: STL компоненти
    axes[0, 0].plot(series.index, series, color="steelblue", label="Початковий ряд")
    axes[0, 0].set_title("1. Початковий часовий ряд температури (°C)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[1, 0].plot(series.index, stl_res.trend, color="darkorange", linewidth=2, label="Тренд (STL)")
    axes[1, 0].set_title("2. Трендова складова")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[2, 0].plot(series.index, stl_res.seasonal, color="seagreen", label="Сезонність (STL)")
    axes[2, 0].set_title("3. Сезонна складова (період = 365 діб)")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()

    axes[3, 0].plot(series.index, stl_res.resid, color="crimson", alpha=0.7, label="Залишки (Residuals)")
    axes[3, 0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[3, 0].set_title("4. Шумова / залишкова компонента")
    axes[3, 0].grid(True, alpha=0.3)
    axes[3, 0].legend()

    # Права колонка: Властивості та сезонні профілі
    plot_acf(series.dropna(), ax=axes[0, 1], lags=60, title="5. Автокореляційна функція (ACF)")
    axes[0, 1].grid(True, alpha=0.3)

    plot_pacf(series.dropna(), ax=axes[1, 1], lags=30, title="6. Часткова автокореляційна функція (PACF)")
    axes[1, 1].grid(True, alpha=0.3)

    # Boxplot за місяцями (сезонна структура)
    df_temp = pd.DataFrame({"Temp": series, "Month": series.index.month})
    df_temp.boxplot(column="Temp", by="Month", ax=axes[2, 1], grid=True)
    axes[2, 1].set_title("7. Сезонний розподіл температури за місяцями")
    axes[2, 1].set_xlabel("Місяць року")
    axes[2, 1].set_ylabel("Температура, °C")
    plt.sca(axes[2, 1])
    plt.suptitle("")

    # Гістограма та щільність залишків
    axes[3, 1].hist(stl_res.resid.dropna(), bins=35, density=True, color="orchid", alpha=0.7, edgecolor="black")
    axes[3, 1].set_title("8. Розподіл залишків декомпозиції")
    axes[3, 1].set_xlabel("Відхилення, °C")
    axes[3, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("fig1_decomposition_properties.png", dpi=150)
    plt.show()

# Графік 2: Багатовимірний кореляційний аналіз та кластеризація.
def plot_correlation_and_clustering(df: pd.DataFrame, corr_res: dict, cluster_res: dict):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Кореляційний аналіз та багатофакторна кластеризація Time Series", fontsize=14,
                 fontweight="bold")

    # 1. Матриця кореляції Пірсона
    corr_matrix = corr_res["pearson"]
    im = axes[0, 0].imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0, 0].set_xticks(range(len(corr_matrix.columns)))
    axes[0, 0].set_yticks(range(len(corr_matrix.columns)))
    axes[0, 0].set_xticklabels(corr_matrix.columns, rotation=20)
    axes[0, 0].set_yticklabels(corr_matrix.columns)
    axes[0, 0].set_title("1. Матриця кореляції Пірсона між метеопоказниками")
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            axes[0, 0].text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="black",
                            fontweight="bold")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # 2. Крос-кореляція (Температура vs Вологість)
    axes[0, 1].stem(corr_res["ccf_lags"], corr_res["ccf_temp_humidity"], basefmt="k-")
    axes[0, 1].axvline(0, color="red", linestyle="--", label="Нульовий зсув (Lag 0)")
    axes[0, 1].set_title("2. Крос-кореляційна функція: Температура vs Вологість")
    axes[0, 1].set_xlabel("Лаг (дні)")
    axes[0, 1].set_ylabel("Коефіцієнт кореляції")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. Багатофакторна кластеризація у просторі часового ряду
    df_cl = cluster_res["df_clustered"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for cl_id in sorted(df_cl["Cluster"].unique()):
        subset = df_cl[df_cl["Cluster"] == cl_id]
        axes[1, 0].scatter(
            subset.index, subset["Temperature"],
            color=colors[cl_id % len(colors)], s=18, alpha=0.8, label=f"Кластер {cl_id}"
        )
    axes[1, 0].set_title(
        f"3. Кластеризація синоптичних станів у часі (Silhouette = {cluster_res['silhouette_multi']:.3f})")
    axes[1, 0].set_ylabel("Температура, °C")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 4. Центроїди тижневих патернів (кластеризація підрядів)
    days = np.arange(1, 8)
    for idx, center in enumerate(cluster_res["weekly_centers"]):
        axes[1, 1].plot(days, center, marker="o", linewidth=2.5,
                        label=f"Профіль кластера {idx} (n={np.sum(cluster_res['weekly_labels'] == idx)} тижнів)")
    axes[1, 1].set_title("4. Центроїди кластерів тижневих часових рядів температури")
    axes[1, 1].set_xlabel("День тижневого вікна (1..7)")
    axes[1, 1].set_ylabel("Температура, °C")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("fig2_correlation_clustering.png", dpi=150)
    plt.show()

# Графік 3: Порівняння реального та синтезованого модельного часового ряду.
def plot_synthetic_verification(real_series: pd.Series, df_syn: pd.DataFrame, eval_res: dict):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Верифікація адекватності синтезованої моделі Time Series", fontsize=14, fontweight="bold")

    # 1. Накладання реального та синтетичного рядів
    axes[0, 0].plot(real_series.index, real_series, color="steelblue", alpha=0.75, label="Реальні дані (Open-Meteo)")
    axes[0, 0].plot(df_syn.index, df_syn["Synthetic"], color="darkorange", alpha=0.75, linestyle="--",
                    label="Синтетична модель")
    axes[0, 0].plot(df_syn.index, df_syn["Syn_Trend"] + df_syn["Syn_Seasonal"], color="black", linewidth=2,
                    label="Детермінований каркас (Тренд + Сезонність)")
    axes[0, 0].set_title(
        f"1. Динаміка реального та модельного рядів (r = {eval_res['trajectory_corr']:.3f}, RMSE = {eval_res['trajectory_rmse']:.2f})")
    axes[0, 0].set_ylabel("Температура, °C")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. Порівняння щільностей розподілу (Гістограма + KDE)
    axes[0, 1].hist(real_series, bins=30, density=True, alpha=0.5, color="steelblue", edgecolor="black",
                    label="Реальний розподіл")
    axes[0, 1].hist(df_syn["Synthetic"], bins=30, density=True, alpha=0.5, color="darkorange", edgecolor="black",
                    label="Модельний розподіл")
    axes[0, 1].set_title(f"2. Порівняння емпіричних розподілів (KS p-value = {eval_res['ks_pvalue']:.3f})")
    axes[0, 1].set_xlabel("Температура, °C")
    axes[0, 1].set_ylabel("Щільність імовірності")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. Порівняння автокореляційних функцій (ACF)
    lags = np.arange(len(eval_res["acf_real"]))
    axes[1, 0].plot(lags, eval_res["acf_real"], color="steelblue", linewidth=2, label="ACF Реального ряду")
    axes[1, 0].plot(lags, eval_res["acf_syn"], color="darkorange", linewidth=2, linestyle="--",
                    label="ACF Синтетичної моделі")
    axes[1, 0].set_title(f"3. Збіг автокореляційних функцій (ACF Similarity = {eval_res['acf_similarity']:.4f})")
    axes[1, 0].set_xlabel("Лаг (доба)")
    axes[1, 0].set_ylabel("ACF")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 4. Q-Q графік квантилів реального та синтетичного рядів
    q_real = np.quantile(real_series, np.linspace(0.01, 0.99, 100))
    q_syn = np.quantile(df_syn["Synthetic"], np.linspace(0.01, 0.99, 100))
    axes[1, 1].scatter(q_real, q_syn, color="purple", alpha=0.7, label="Квантилі (1%..99%)")
    min_val, max_val = min(q_real.min(), q_syn.min()), max(q_real.max(), q_syn.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ідеальна бісектриса y = x")
    axes[1, 1].set_title("4. Q-Q Plot: Квантильне порівняння реального та модельного рядів")
    axes[1, 1].set_xlabel("Квантилі реального ряду (°C)")
    axes[1, 1].set_ylabel("Квантилі синтетичного ряду (°C)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("fig3_synthetic_verification.png", dpi=150)
    plt.show()


def main():
    # ЕТАП 1: Завантаження багатовимірного датасету
    df = fetch_multivariate_weather_data()

    # ЕТАП 2: Декомпозиція та аналіз властивостей для всіх змінних датасету
    print("\n" + "=" * 75)
    print("АНАЛІЗ ВЛАСТИВОСТЕЙ БАГАТОВИМІРНОГО ЧАСОВОГО РЯДУ ")
    print("=" * 75)

    all_props = {}
    decompositions = {}
    for col in df.columns:
        decomp = decompose_series(df[col], period=365)
        props = analyze_properties(df[col], decomp["stl"])
        decompositions[col] = decomp
        all_props[col] = props

    props_df = pd.DataFrame(all_props).T
    print(props_df[["Mean", "Std", "Hurst_Exponent", "Trend_Strength", "Seasonal_Strength", "ADF_p_value",
                    "Is_Stationary"]].round(4))

    # ЕТАП 3: Кореляційний аналіз та багатофакторна кластеризація
    corr_res = correlation_analysis(df)
    print("\n--- Матриця кореляції Пірсона ---")
    print(corr_res["pearson"].round(4))

    cluster_res = cluster_time_series(df, n_clusters=4)
    print(f"\n--- Центроїди багатофакторних кластерів (Silhouette Score = {cluster_res['silhouette_multi']:.4f}) ---")
    print(cluster_res["centroids"].round(2))

    # ЕТАП 4: Генерація синтетичної моделі та доведення її адекватності
    target_col = "Temperature"
    df_syn, model_params = synthesize_time_series(df[target_col], decompositions[target_col]["stl"])

    print("\n" + "=" * 75)
    print("ПАРАМЕТРИ СИНТЕЗОВАНОЇ МАТЕМАТИЧНОЇ МОДЕЛІ (ТЕМПЕРАТУРА)")
    print("=" * 75)
    print(
        f"Поліном тренду T(t)      : {model_params['poly_coeffs'][0]:.6e}*t^2 + {model_params['poly_coeffs'][1]:.6e}*t + {model_params['poly_coeffs'][2]:.4f}")
    print(
        f"Гармоніки сезонності S(t): a1={model_params['harm_coeffs'][0]:.3f}, b1={model_params['harm_coeffs'][1]:.3f}, a2={model_params['harm_coeffs'][2]:.3f}, b2={model_params['harm_coeffs'][3]:.3f}")
    print(
        f"Параметри шуму AR(1) E(t): phi_1={model_params['ar1_phi']:.4f}, sigma_eps={model_params['innovation_std']:.4f}")

    eval_res = evaluate_model_adequacy(df[target_col], df_syn["Synthetic"], all_props[target_col])
    print("\n--- Таблиця верифікації адекватності синтетичної моделі ---")
    print(eval_res["comparison_table"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nКор. траєкторій: r = {eval_res['trajectory_corr']:.4f} | RMSE = {eval_res['trajectory_rmse']:.4f} °C")
    print(
        f"Подібність ACF : {eval_res['acf_similarity']:.4f} | Тест Колмогорова-Смірнова p-value = {eval_res['ks_pvalue']:.4f}")

    # ЕТАП 5: Візуалізація результатів
    plot_decomposition_and_properties(df[target_col], decompositions[target_col]["classical"])
    plot_correlation_and_clustering(df, corr_res, cluster_res)
    plot_synthetic_verification(df[target_col], df_syn, eval_res)


if __name__ == "__main__":
    main()