import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline


def calculate_kpi(y_true: np.ndarray, y_hat: np.ndarray, n_params: int = 2) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_hat)
    yt = y_true[mask]
    yh = y_hat[mask]
    n = len(yt)

    err = yt - yh
    sse = float(np.sum(err ** 2))
    rmse = float(np.sqrt(sse / max(n, 1)))
    mae = float(np.mean(np.abs(err)))
    denom = np.maximum(np.abs(yt), 1.0)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)

    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - sse / max(ss_tot, 1e-9))

    sigma2 = max(sse / max(n, 1), 1e-9)
    aic = float(n * np.log(sigma2) + 2 * n_params)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE_%": mape,
        "R2": r2,
        "AIC": aic,
    }


def _build_fourier_features(t: np.ndarray, period: float = 365.25, n_harmonics: int = 3) -> np.ndarray:
    omega = 2.0 * np.pi / period
    cols = [t, t ** 2]
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(k * omega * t))
        cols.append(np.cos(k * omega * t))
    return np.column_stack(cols)


def fit_regression_suite(
    train_series: pd.Series,
    forecast_steps: int,
    n_harmonics: int = 3,
    lags: tuple = (1, 2, 7),
) -> dict:
    y_train = train_series.to_numpy(dtype=float)
    n_train = len(y_train)
    t_train = np.arange(n_train, dtype=float)
    t_future = np.arange(n_train, n_train + forecast_steps, dtype=float)

    future_index = pd.date_range(
        start=train_series.index[-1] + pd.Timedelta(days=1),
        periods=forecast_steps,
        freq="D",
    )

    # 1. Поліноміальна регресія
    poly_model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression())
    poly_model.fit(t_train.reshape(-1, 1), y_train)
    poly_fitted = pd.Series(poly_model.predict(t_train.reshape(-1, 1)), index=train_series.index)
    poly_forecast = pd.Series(poly_model.predict(t_future.reshape(-1, 1)), index=future_index)

    # 2. Гармонічна регресія з оптимізацією коефіцієнта регуляризації Ridge (alpha)
    X_harm_train = _build_fourier_features(t_train, n_harmonics=n_harmonics)
    X_harm_future = _build_fourier_features(t_future, n_harmonics=n_harmonics)

    best_alpha = 1.0
    best_score = np.inf
    for alpha in [0.01, 0.1, 1.0, 5.0, 10.0, 50.0]:
        model_cand = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        # Валідація на останніх 90 точках навчальної вибірки
        split = max(n_train - 90, int(n_train * 0.8))
        model_cand.fit(X_harm_train[:split], y_train[:split])
        val_pred = model_cand.predict(X_harm_train[split:])
        rmse_val = float(np.sqrt(np.mean((y_train[split:] - val_pred) ** 2)))
        if rmse_val < best_score:
            best_score = rmse_val
            best_alpha = alpha

    harm_model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
    harm_model.fit(X_harm_train, y_train)
    harm_fitted = pd.Series(harm_model.predict(X_harm_train), index=train_series.index)
    harm_forecast = pd.Series(harm_model.predict(X_harm_future), index=future_index)

    # 3. Динамічна авторегресійна модель
    max_lag = max(lags)
    X_dyn_list = []
    y_dyn_target = []

    for idx in range(max_lag, n_train):
        lag_feats = [y_train[idx - l] for l in lags]
        harm_feats = list(X_harm_train[idx])
        X_dyn_list.append(lag_feats + harm_feats)
        y_dyn_target.append(y_train[idx])

    X_dyn_arr = np.asarray(X_dyn_list, dtype=float)
    y_dyn_arr = np.asarray(y_dyn_target, dtype=float)

    dyn_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    dyn_model.fit(X_dyn_arr, y_dyn_arr)

    # Відновлення на навчальній вибірці (перші max_lag точок заповнюються гармонічною регресією)
    dyn_fitted_vals = harm_fitted.to_numpy().copy()
    dyn_fitted_vals[max_lag:] = dyn_model.predict(X_dyn_arr)
    dyn_fitted = pd.Series(dyn_fitted_vals, index=train_series.index)

    # Рекурсивна покрокова екстраполяція для динамічної регресії
    history_buffer = list(y_train)
    dyn_forecast_vals = []
    for step_idx in range(forecast_steps):
        lag_feats = [history_buffer[-l] for l in lags]
        harm_feats = list(X_harm_future[step_idx])
        x_step = np.asarray(lag_feats + harm_feats, dtype=float).reshape(1, -1)
        y_next = float(dyn_model.predict(x_step)[0])
        dyn_forecast_vals.append(y_next)
        history_buffer.append(y_next)

    dyn_forecast = pd.Series(dyn_forecast_vals, index=future_index)

    return {
        "Poly_OLS": {
            "fitted": poly_fitted,
            "forecast": poly_forecast,
            "n_params": 4,
            "info": "Degree=3",
        },
        "Harmonic_Ridge": {
            "fitted": harm_fitted,
            "forecast": harm_forecast,
            "n_params": 2 + 2 * n_harmonics,
            "info": f"Harmonics={n_harmonics}, Ridge alpha={best_alpha}",
        },
        "Dynamic_Lag_Reg": {
            "fitted": dyn_fitted,
            "forecast": dyn_forecast,
            "n_params": len(lags) + 2 + 2 * n_harmonics,
            "info": f"Lags={lags}, Harmonics={n_harmonics}",
        },
    }


def select_optimal_model_by_properties(props: dict, horizon: str = "long") -> dict:
    fs = props["Seasonal_Strength"]
    ft = props["Trend_Strength"]
    acf1 = props["ACF_Lag1"]
    is_stat = props["Is_Stationary"]

    if fs >= 0.45:
        if horizon == "short" and acf1 >= 0.65:
            chosen_family = "Регресійна (Динамічна)"
            chosen_model = "Dynamic_Lag_Reg"
            reason = f"Висока сезонність (Fs={fs:.2f}) + сильна автокореляція (ACF1={acf1:.2f}) на короткому горизонті"
        else:
            chosen_family = "Регресійна (Детерміновано-гармонічна)"
            chosen_model = "Harmonic_Ridge"
            reason = f"Домінуюча сезонність (Fs={fs:.2f}) потребує стійкого фазового каркаса Фур'є без накопичення похибки"
    elif not is_stat or ft >= 0.35:
        chosen_family = "Експоненціальне згладжування"
        chosen_model = "Holt_Damped"
        reason = f"Виражений нелінійний/локальний тренд (Ft={ft:.2f}) при слабкій сезонності (Fs={fs:.2f})"
    else:
        if acf1 >= 0.35:
            chosen_family = "Регресійна (Динамічна)"
            chosen_model = "Dynamic_Lag_Reg"
            reason = f"Стаціонарний ряд із помірною короткостроковою пам'яттю (ACF1={acf1:.2f})"
        else:
            chosen_family = "Експоненціальне згладжування"
            chosen_model = "SES"
            reason = f"Стаціонарний зашумлений процес без тренду та сезонності (ADF p={props['ADF_p']:.4f}, ACF1={acf1:.2f})"

    return {
        "Recommended_Family": chosen_family,
        "Recommended_Model": chosen_model,
        "Justification": reason,
    }