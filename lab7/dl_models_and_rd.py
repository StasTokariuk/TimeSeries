import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

from data_preprocessing import compute_time_features


def set_reproducibility(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ================= ОРИГІНАЛЬНЕ РІШЕННЯ =================
#    Композитна функція втрат для навчання нейронних мереж обробки часових рядів,
#    що поєднує 3 індикатори:
#    1) Huber Loss (робастна квадратично-лінійна похибка рівня)
#    2) MAE (середня абсолютна похибка)
#    3) Gradient Difference Loss (узгодженість перших різниць / швидкості зміни ряду в батчі)
def composite_ts_loss(alpha: float = 0.55, beta: float = 0.30, gamma: float = 0.15):

    huber_fn = tf.keras.losses.Huber(delta=1.0)

    def loss_fn(y_true, y_pred):
        y_t = tf.cast(y_true, tf.float32)
        y_p = tf.cast(y_pred, tf.float32)

        # 1. Робастна похибка Huber
        l_huber = huber_fn(y_t, y_p)
        # 2. Абсолютна похибка MAE
        l_mae = tf.reduce_mean(tf.abs(y_t - y_p))
        # 3. Структурна похибка приростів (перших різниць)
        dy_t = y_t[1:] - y_t[:-1]
        dy_p = y_p[1:] - y_p[:-1]
        l_diff = tf.cond(
            tf.shape(y_t)[0] > 1,
            lambda: tf.reduce_mean(tf.square(dy_t - dy_p)),
            lambda: tf.constant(0.0, dtype=tf.float32),
        )
        return alpha * l_huber + beta * l_mae + gamma * l_diff

    loss_fn.__name__ = "Composite_Huber_MAE_Diff_Loss"
    return loss_fn


# ================= КОНСТРУЮВАННЯ 3 ТИПІВ ШТУЧНИХ НЕЙРОННИХ МЕРЕЖ =================

# Архітектура 1: Багатошаровий персептрон (MLP / Dense Neural Network).
def build_mlp_model(input_shape: tuple, loss_fn) -> tf.keras.Model:
    inp = layers.Input(shape=input_shape, name="Window_Input")
    x = layers.Flatten(name="Flatten_Window")(inp)
    x = layers.Dense(64, activation="relu", name="Dense_64")(x)
    x = layers.Dropout(0.15, name="Dropout_15")(x)
    x = layers.Dense(32, activation="relu", name="Dense_32")(x)
    out = layers.Dense(1, activation="linear", name="Forecast_Output")(x)

    model = models.Model(inputs=inp, outputs=out, name="MLP_Network")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.003),
        loss=loss_fn,
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model

# Архітектура 2: Одновимірна згорткова нейронна мережа (1D-CNN).
def build_cnn_model(input_shape: tuple, loss_fn) -> tf.keras.Model:
    inp = layers.Input(shape=input_shape, name="Window_Input")
    x = layers.Conv1D(filters=32, kernel_size=3, padding="causal", activation="relu", name="Conv1D_32_k3")(inp)
    x = layers.Conv1D(filters=16, kernel_size=3, padding="causal", activation="relu", name="Conv1D_16_k3")(x)
    x = layers.GlobalAveragePooling1D(name="GlobalAvgPool1D")(x)
    x = layers.Dense(32, activation="relu", name="Dense_32")(x)
    out = layers.Dense(1, activation="linear", name="Forecast_Output")(x)

    model = models.Model(inputs=inp, outputs=out, name="Conv1D_Network")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.003),
        loss=loss_fn,
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model

# Архітектура 3: Рекурентна нейронна мережа довгої короткострокової пам'яті (LSTM).
def build_lstm_model(input_shape: tuple, loss_fn) -> tf.keras.Model:
    inp = layers.Input(shape=input_shape, name="Window_Input")
    x = layers.LSTM(48, return_sequences=True, name="LSTM_48_Seq")(inp)
    x = layers.Dropout(0.15, name="Dropout_15")(x)
    x = layers.LSTM(24, return_sequences=False, name="LSTM_24_Vec")(x)
    x = layers.Dense(24, activation="relu", name="Dense_24")(x)
    out = layers.Dense(1, activation="linear", name="Forecast_Output")(x)

    model = models.Model(inputs=inp, outputs=out, name="LSTM_Network")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.003),
        loss=loss_fn,
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def train_neural_network(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 85,
    batch_size: int = 16,
) -> dict:
    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=14, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-4, verbose=0),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=cb_list,
        verbose=0,
    )
    return {
        "model": model,
        "history": history.history,
        "epochs_trained": len(history.history["loss"]),
        "total_params": model.count_params(),
    }


def recursive_extrapolation_rollout(
    model: tf.keras.Model,
    initial_window: np.ndarray,
    start_day_idx: int,
    steps: int,
    scaler,
    future_index: pd.DatetimeIndex,
    anchor_weight: float = 0.08,
    seasonal_anchor_scaled: np.ndarray = None,
) -> pd.Series:
    curr_win = initial_window.copy()
    future_t = np.arange(start_day_idx, start_day_idx + steps, dtype=float)
    future_fourier = compute_time_features(future_t)

    preds_scaled = []
    for s in range(steps):
        pred_val = float(model.predict(curr_win[np.newaxis, :, :], verbose=0)[0, 0])
        if seasonal_anchor_scaled is not None:
            pred_val = (1.0 - anchor_weight) * pred_val + anchor_weight * float(seasonal_anchor_scaled[s])
        preds_scaled.append(pred_val)

        next_row = np.concatenate([[pred_val], future_fourier[s]])
        curr_win = np.vstack([curr_win[1:], next_row]).astype(np.float32)

    preds_real = scaler.inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).flatten()
    return pd.Series(preds_real, index=future_index)


def calculate_kpi(y_true: np.ndarray, y_hat: np.ndarray) -> dict:
    yt = np.asarray(y_true, dtype=float).flatten()
    yh = np.asarray(y_hat, dtype=float).flatten()
    err = yt - yh

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = np.maximum(np.abs(yt), 1.0)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-9))

    # Напрямкова точність (Directional Symmetry — збіг знаку зміни температури)
    dir_acc = float(np.mean(np.sign(np.diff(yt)) == np.sign(np.diff(yh))) * 100.0)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE_%": mape,
        "R2": r2,
        "Dir_Acc_%": dir_acc,
    }


def plot_neural_network_architectures(models_dict: dict, filename: str = "fig1_nn_architectures.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    fig.suptitle(
        "Графічне представлення архітектур конструйованих штучних нейронних мереж (MLP, 1D-CNN, LSTM)",
        fontsize=14,
        fontweight="bold",
    )

    layer_colors = {
        "InputLayer": "#D6EAF8",
        "Flatten": "#EBDEF0",
        "Dense": "#FADBD8",
        "Dropout": "#FCF3CF",
        "Conv1D": "#D1F2EB",
        "GlobalAveragePooling1D": "#D4EFDF",
        "LSTM": "#FDEBD0",
    }

    for ax, (title, m_info) in zip(axes, models_dict.items()):
        model = m_info["model"]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis("off")
        ax.set_title(f"{title}\n(Параметрів: {model.count_params():,})", fontsize=12, fontweight="bold", pad=10)

        n_layers = len(model.layers)
        y_positions = np.linspace(10.5, 1.5, n_layers)

        for idx, (layer, y_pos) in enumerate(zip(model.layers, y_positions)):
            l_type = layer.__class__.__name__
            out_shape = layer.output.shape
            shape_str = str(tuple(out_shape[1:]))
            params_cnt = layer.count_params()

            box_color = layer_colors.get(l_type, "#EAECEE")
            rect = patches.FancyBboxPatch(
                (1.0, y_pos - 0.65),
                8.0,
                1.3,
                boxstyle="round,pad=0.1,rounding_size=0.25",
                linewidth=1.5,
                edgecolor="#2C3E50",
                facecolor=box_color,
            )
            ax.add_patch(rect)

            label_text = (
                f"{idx+1}. {layer.name} [{l_type}]\n"
                f"Вихідний тензор: {shape_str} | Ваг: {params_cnt}"
            )
            ax.text(5.0, y_pos, label_text, ha="center", va="center", fontsize=9.2, fontweight="bold", color="#1B2631")

            if idx < n_layers - 1:
                next_y = y_positions[idx + 1]
                ax.annotate(
                    "",
                    xy=(5.0, next_y + 0.70),
                    xytext=(5.0, y_pos - 0.70),
                    arrowprops=dict(arrowstyle="->", lw=2.0, color="#34495E"),
                )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(filename, dpi=150)
    plt.show()