import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_preprocessing import (
    fetch_weather_series,
    analyze_and_decompose,
    prepare_supervised_tensors,
    compute_time_features,
)
from dl_models_and_rd import (
    set_reproducibility,
    composite_ts_loss,
    build_mlp_model,
    build_cnn_model,
    build_lstm_model,
    train_neural_network,
    recursive_extrapolation_rollout,
    calculate_kpi,
    plot_neural_network_architectures,
)

plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams["font.size"] = 10


def build_seasonal_baseline_scaled(scaled_temp: np.ndarray, future_t: np.ndarray) -> np.ndarray:
    n = len(scaled_temp)
    t_hist = np.arange(n, dtype=float)
    X_hist = np.column_stack([np.ones(n), compute_time_features(t_hist)])
    beta, _, _, _ = np.linalg.lstsq(X_hist, scaled_temp, rcond=None)

    X_fut = np.column_stack([np.ones(len(future_t)), compute_time_features(future_t)])
    return X_fut @ beta


def plot_training_and_kpi_diagnostics(models_res: dict, kpi_df: pd.DataFrame, loss_ablation_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Процеси навчання штучних нейронних мереж та контроль індикаторів якості",
        fontsize=14,
        fontweight="bold",
    )

    colors = {"MLP (Персептрон)": "steelblue", "1D-CNN (Згорткова)": "seagreen", "LSTM (Рекурентна)": "darkorange"}

    # 1. Криві навчання (Training & Validation Composite Loss)
    for name, res in models_res.items():
        c = colors[name]
        axes[0, 0].plot(res["history"]["loss"], color=c, linewidth=1.8, label=f"{name} — Train Loss")
        axes[0, 0].plot(res["history"]["val_loss"], color=c, linestyle="--", linewidth=1.8, label=f"{name} — Val Loss")
    axes[0, 0].set_title("1. Динаміка композитної функції втрат під час навчання (Train vs Validation)")
    axes[0, 0].set_xlabel("Епоха навчання")
    axes[0, 0].set_ylabel("Composite Loss")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8.5, ncol=2)

    # 2. Динаміка метрики Validation RMSE по епохах
    for name, res in models_res.items():
        axes[0, 1].plot(res["history"]["val_rmse"], color=colors[name], linewidth=2, label=f"{name} (Val RMSE)")
    axes[0, 1].set_title("2. Збіжність метрики RMSE на валідаційній вибірці (масштабовані од.)")
    axes[0, 1].set_xlabel("Епоха навчання")
    axes[0, 1].set_ylabel("Validation RMSE")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. Порівняння KPI трьох архітектур ШНМ на Train, Val та Test
    x = np.arange(len(kpi_df))
    w = 0.25
    axes[1, 0].bar(x - w, kpi_df["Train_RMSE"], width=w, color="teal", edgecolor="black", label="Train RMSE (Навчальна)")
    axes[1, 0].bar(x, kpi_df["Val_RMSE"], width=w, color="goldenrod", edgecolor="black", label="Val RMSE (Валідаційна, 90 діб)")
    axes[1, 0].bar(x + w, kpi_df["Test_1Step_RMSE"], width=w, color="coral", edgecolor="black", label="Test RMSE (Тестова, 180 діб)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(kpi_df["Архітектура ШНМ"])
    axes[1, 0].set_ylabel("RMSE (°C)")
    axes[1, 0].set_title("3. Порівняння індикатора якості RMSE для трьох типів ШНМ за підмножинами")
    axes[1, 0].grid(True, axis="y", alpha=0.3)
    axes[1, 0].legend()

    # 4. R&D Верифікація композитної функції втрат проти стандартної MSE
    x_ab = np.arange(len(loss_ablation_df))
    w_ab = 0.32
    ax4_twin = axes[1, 1].twinx()
    b1 = axes[1, 1].bar(x_ab - w_ab / 2, loss_ablation_df["Test_RMSE"], width=w_ab, color="royalblue", edgecolor="black", label="Test RMSE (°C) — менше краще")
    b2 = ax4_twin.bar(x_ab + w_ab / 2, loss_ablation_df["Dir_Acc_%"], width=w_ab, color="mediumseagreen", edgecolor="black", label="Точність напрямку (%) — більше краще")
    axes[1, 1].set_xticks(x_ab)
    axes[1, 1].set_xticklabels(loss_ablation_df["Функція втрат (LSTM)"])
    axes[1, 1].set_ylabel("RMSE на тесті (°C)", color="royalblue")
    ax4_twin.set_ylabel("Directional Accuracy (%)", color="seagreen")
    axes[1, 1].set_ylim(0, max(loss_ablation_df["Test_RMSE"]) * 1.35)
    ax4_twin.set_ylim(50, 85)
    axes[1, 1].set_title("4. Ефективність оригінальної Composite Loss проти стандартної MSE")
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    lines_1, labels_1 = axes[1, 1].get_legend_handles_labels()
    lines_2, labels_2 = ax4_twin.get_legend_handles_labels()
    axes[1, 1].legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper center")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("fig2_training_and_kpi.png", dpi=150)
    plt.show()


def plot_in_sample_and_extrapolation(
    series: pd.Series,
    prep: dict,
    preds_all_dict: dict,
    test_rollout_dict: dict,
    future_rollout_dict: dict,
):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    fig.suptitle(
        "Прогнозування на ділянці спостереження та довготривала екстраполяція поза межами (0.5N = 366 діб)",
        fontsize=14,
        fontweight="bold",
    )

    colors = {"MLP (Персептрон)": "steelblue", "1D-CNN (Згорткова)": "seagreen", "LSTM (Рекурентна)": "darkorange"}

    # 1. Верхній широкий графік: Повний інтервал спостереження (Train/Val/Test) + Екстраполяція на 366 діб (2026 рік)
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.plot(series.index, series, color="lightgray", linewidth=1.3, label="Фактичні дані (Open-Meteo, 2024–2025)")

    # Межі поділу Train / Val / Test
    val_start = prep["dates_val"][0]
    test_start = prep["dates_test"][0]
    extrap_start = series.index[-1]
    ax_main.axvline(val_start, color="goldenrod", linestyle="--", linewidth=1.4, label="Початок Validation (90 діб)")
    ax_main.axvline(test_start, color="purple", linestyle="--", linewidth=1.4, label="Початок Test (180 діб)")
    ax_main.axvline(extrap_start, color="black", linestyle="-.", linewidth=1.8, label="Межа спостереження (Екстраполяція +366 діб)")

    # Моделювання на ділянці спостереження (LSTM) та екстраполяція всіх 3 моделей поза межами
    ax_main.plot(prep["dates_all"], preds_all_dict["LSTM (Рекурентна)"], color="darkorange", linewidth=1.4, alpha=0.85, label="LSTM на ділянці спостереження")

    for name, fut_s in future_rollout_dict.items():
        ax_main.plot(fut_s.index, fut_s, color=colors[name], linewidth=1.9, linestyle="-", label=f"Екстраполяція {name} (2026 р.)")

    ax_main.set_title("1. Повна траєкторія часового ряду: структурний поділ (Train/Val/Test) та екстраполяція на 366 діб (0.5N)")
    ax_main.set_ylabel("Температура, °C")
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc="upper left", ncol=3, fontsize=8.5)

    # 2. Нижній лівий графік: Однокрокове прогнозування на тестовій ділянці (180 діб)
    ax_test_1step = fig.add_subplot(gs[1, 0])
    test_dates = prep["dates_test"]
    test_actual = series.loc[test_dates]
    ax_test_1step.plot(test_dates, test_actual, color="black", alpha=0.65, linewidth=1.5, label="Фактична температура (Test 180 діб)")
    for name, pred_s in preds_all_dict.items():
        ax_test_1step.plot(test_dates, pred_s.loc[test_dates], color=colors[name], linewidth=1.6, label=f"{name} (1-step)")
    ax_test_1step.set_title("2. Прогнозування на тестовій ділянці спостереження (180 діб, однокроковий режим)")
    ax_test_1step.set_ylabel("Температура, °C")
    ax_test_1step.grid(True, alpha=0.3)
    ax_test_1step.legend(fontsize=8.5)

    # 3. Нижній правий графік: Верифікація автономної екстраполяції на 180 діб без підглядання (Multi-Step Rollout)
    ax_test_roll = fig.add_subplot(gs[1, 1])
    ax_test_roll.plot(test_dates, test_actual, color="black", alpha=0.65, linewidth=1.5, label="Фактична температура (Test 180 діб)")
    for name, roll_s in test_rollout_dict.items():
        ax_test_roll.plot(roll_s.index, roll_s, color=colors[name], linewidth=2.0, linestyle="--", label=f"{name} (Автономний Rollout)")
    ax_test_roll.set_title("3. Верифікація довготривалої екстраполяції на 180 діб (Out-of-Sample Rollout)")
    ax_test_roll.set_ylabel("Температура, °C")
    ax_test_roll.grid(True, alpha=0.3)
    ax_test_roll.legend(fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("fig3_forecasting_and_extrapolation.png", dpi=150)
    plt.show()


def main():
    set_reproducibility(42)

    # ЕТАП 1: Завантаження даних, декомпозиція та препроцесінг
    series = fetch_weather_series()
    props = analyze_and_decompose(series)

    print("\n" + "=" * 92)
    print("1. ДОСЛІДЖЕННЯ ВЛАСТИВОСТЕЙ ТА ПРЕПРОЦЕСІНГ ВХІДНИХ ДАНИХ")
    print("=" * 92)
    print(f"Кількість спостережень N : {props['N']} діб | Середнє = {props['Mean']:.2f} °C | СКВ = {props['Std']:.2f} °C")
    print(f"Стаціонарність (ADF)     : Stat = {props['ADF_Stat']:.4f}, p-value = {props['ADF_p']:.4f} (Стаціонарний: {props['Is_Stationary']})")
    print(f"Сила тренду Ft           : {props['Trend_Strength']:.4f} | Сила сезонності Fs = {props['Seasonal_Strength']:.4f}")
    print(f"Автокореляція            : ACF(1) = {props['ACF_Lag1']:.4f}, ACF(7) = {props['ACF_Lag7']:.4f}")

    prep = prepare_supervised_tensors(series, window_size=21, val_size=90, test_size=180)
    input_shape = prep["X_train"].shape[1:]
    print(f"Розмірність 3D-тензора X : {prep['X_all'].shape} (Вікно W = {prep['window_size']} діб, Ознак = {input_shape[1]})")
    print(f"Структурний поділ вибірки: Train = {len(prep['X_train'])}, Val = {len(prep['X_val'])}, Test = {len(prep['X_test'])} вікон")

    # ЕТАП 2: Конструювання та навчання 3 типів ШНМ із композитною функцією втрат
    comp_loss = composite_ts_loss(alpha=0.55, beta=0.30, gamma=0.15)

    builders = {
        "MLP (Персептрон)": build_mlp_model(input_shape, comp_loss),
        "1D-CNN (Згорткова)": build_cnn_model(input_shape, comp_loss),
        "LSTM (Рекурентна)": build_lstm_model(input_shape, comp_loss),
    }

    models_res = {}
    for name, model in builders.items():
        print(f"[INFO] Навчання нейромережі: {name}...")
        models_res[name] = train_neural_network(
            model, prep["X_train"], prep["y_train"], prep["X_val"], prep["y_val"], epochs=85, batch_size=16
        )

    # Побудова графічної схеми архітектур ШНМ
    plot_neural_network_architectures(models_res, filename="fig1_nn_architectures.png")

    # ЕТАП 3: Прогнозування на ділянці спостереження та довготривала екстраполяція (366 діб = 0.5N)
    scaler = prep["scaler"]
    y_train_real = scaler.inverse_transform(prep["y_train"]).flatten()
    y_val_real = scaler.inverse_transform(prep["y_val"]).flatten()
    y_test_real = scaler.inverse_transform(prep["y_test"]).flatten()

    # Розрахунок опорного гармонічного профілю на навчальній + валідаційній вибірці для далекого горизонту
    n_train_val = len(series) - prep["test_size"]
    t_test_roll = np.arange(n_train_val, len(series), dtype=float)
    anchor_test = build_seasonal_baseline_scaled(prep["scaled_temp"][:n_train_val], t_test_roll)

    extrap_steps = 366  # 0.5 інтервалу спостереження поза межами вибірки
    future_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=extrap_steps, freq="D")
    t_future_roll = np.arange(len(series), len(series) + extrap_steps, dtype=float)
    anchor_future = build_seasonal_baseline_scaled(prep["scaled_temp"], t_future_roll)

    preds_all_dict = {}
    test_rollout_dict = {}
    future_rollout_dict = {}
    kpi_rows = []

    for name, res in models_res.items():
        m = res["model"]

        # Однокрокові прогнози на ділянці спостереження
        pred_all_scaled = m.predict(prep["X_all"], verbose=0)
        pred_all_real = scaler.inverse_transform(pred_all_scaled).flatten()
        preds_all_dict[name] = pd.Series(pred_all_real, index=prep["dates_all"])

        p_train = scaler.inverse_transform(m.predict(prep["X_train"], verbose=0)).flatten()
        p_val = scaler.inverse_transform(m.predict(prep["X_val"], verbose=0)).flatten()
        p_test = scaler.inverse_transform(m.predict(prep["X_test"], verbose=0)).flatten()

        # Автономна рекурсивна екстраполяція на тестову ділянку (180 діб)
        init_win_test = prep["X_test"][0]
        roll_test_s = recursive_extrapolation_rollout(
            m, init_win_test, n_train_val, prep["test_size"], scaler, prep["dates_test"],
            anchor_weight=0.12, seasonal_anchor_scaled=anchor_test
        )
        test_rollout_dict[name] = roll_test_s

        # Довготривала екстраполяція поза межі спостереження на 366 діб (0.5N)
        init_win_fut = prep["all_features"][-prep["window_size"] :]
        roll_fut_s = recursive_extrapolation_rollout(
            m, init_win_fut, len(series), extrap_steps, scaler, future_dates,
            anchor_weight=0.12, seasonal_anchor_scaled=anchor_future
        )
        future_rollout_dict[name] = roll_fut_s

        kpi_train = calculate_kpi(y_train_real, p_train)
        kpi_val = calculate_kpi(y_val_real, p_val)
        kpi_test_1s = calculate_kpi(y_test_real, p_test)
        kpi_test_roll = calculate_kpi(y_test_real, roll_test_s.to_numpy())

        kpi_rows.append({
            "Архітектура ШНМ": name,
            "Параметрів": res["total_params"],
            "Епох": res["epochs_trained"],
            "Train_RMSE": kpi_train["RMSE"],
            "Val_RMSE": kpi_val["RMSE"],
            "Test_1Step_RMSE": kpi_test_1s["RMSE"],
            "Test_1Step_MAE": kpi_test_1s["MAE"],
            "Test_1Step_R2": kpi_test_1s["R2"],
            "Test_DirAcc_%": kpi_test_1s["Dir_Acc_%"],
            "Rollout180_RMSE": kpi_test_roll["RMSE"],
        })

    kpi_df = pd.DataFrame(kpi_rows)

    # ЕТАП 4: Дослідження ефективності композитної функції втрат проти стандартної MSE
    set_reproducibility(42)
    lstm_mse_model = build_lstm_model(input_shape, loss_fn="mse")
    train_neural_network(lstm_mse_model, prep["X_train"], prep["y_train"], prep["X_val"], prep["y_val"], epochs=85, batch_size=16)
    p_test_mse = scaler.inverse_transform(lstm_mse_model.predict(prep["X_test"], verbose=0)).flatten()
    kpi_mse = calculate_kpi(y_test_real, p_test_mse)

    lstm_comp_row = kpi_df[kpi_df["Архітектура ШНМ"] == "LSTM (Рекурентна)"].iloc[0]
    loss_ablation_df = pd.DataFrame([
        {
            "Функція втрат (LSTM)": "Стандартна MSE Loss",
            "Test_RMSE": kpi_mse["RMSE"],
            "Test_MAE": kpi_mse["MAE"],
            "Test_R2": kpi_mse["R2"],
            "Dir_Acc_%": kpi_mse["Dir_Acc_%"],
        },
        {
            "Функція втрат (LSTM)": "Композитна (Huber+MAE+Diff)",
            "Test_RMSE": float(lstm_comp_row["Test_1Step_RMSE"]),
            "Test_MAE": float(lstm_comp_row["Test_1Step_MAE"]),
            "Test_R2": float(lstm_comp_row["Test_1Step_R2"]),
            "Dir_Acc_%": float(lstm_comp_row["Test_DirAcc_%"]),
        },
    ])

    print("\n" + "=" * 92)
    print("2. СИСТЕМАТИЗОВАНА ТАБЛИЦЯ ІНДИКАТОРІВ ЯКОСТІ 3 ТИПІВ ШНМ")
    print("=" * 92)
    print(kpi_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 92)
    print("3. R&D ВЕРИФІКАЦІЯ КОМПОЗИТНОЇ ФУНКЦІЇ ВТРАТ (IV РІВЕНЬ СКЛАДНОСТІ)")
    print("=" * 92)
    print(loss_ablation_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 92)
    print("4. ХАРАКТЕРИСТИКИ ДОВГОТРИВАЛОЇ ЕКСТРАПОЛЯЦІЇ ПОЗА МЕЖАМИ НА 366 ДІБ (2026 РІК)")
    print("=" * 92)
    extrap_summary = []
    for name, fut_s in future_rollout_dict.items():
        extrap_summary.append({
            "Архітектура ШНМ": name,
            "Горизонт (діб)": len(fut_s),
            "Період прогнозу": f"{fut_s.index[0].date()} .. {fut_s.index[-1].date()}",
            "Mean (°C)": float(fut_s.mean()),
            "Std (°C)": float(fut_s.std()),
            "Min (°C)": float(fut_s.min()),
            "Max (°C)": float(fut_s.max()),
        })
    print(pd.DataFrame(extrap_summary).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ЕТАП 5: Візуалізація результатів
    plot_training_and_kpi_diagnostics(models_res, kpi_df, loss_ablation_df)
    plot_in_sample_and_extrapolation(series, prep, preds_all_dict, test_rollout_dict, future_rollout_dict)


if __name__ == "__main__":
    main()