import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from window_models import (
    simple_moving_average,
    weighted_moving_average,
    exponential_moving_average,
    fit_harmonic_regression,
    predict_harmonic_regression,
    adaptive_alpha_beta_filter,
)

warnings.filterwarnings("ignore")


def calculate_kpi(y_true: np.ndarray, y_hat: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_hat)
    yt = y_true[mask]
    yh = y_hat[mask]

    err = yt - yh
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    # Уникаємо ділення на нуль поблизу 0°C за допомогою зміщення у Кельвіни або порогу
    denom = np.maximum(np.abs(yt), 1.0)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-9))
    bias = float(np.mean(err))

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE_%": mape,
        "R2": r2,
        "Bias": bias,
    }


def optimize_ma_windows(series: pd.Series, ref_trend: pd.Series, windows: range = range(2, 45)) -> dict:
    records = []
    y_true = series.to_numpy()
    y_ref = ref_trend.to_numpy()

    for w in windows:
        # Однокроковий прогноз методом ковзного вікна (зсув на 1 крок вперед)
        sma_pred = simple_moving_average(series, w).shift(1).bfill().to_numpy()
        wma_pred = weighted_moving_average(series, w).shift(1).bfill().to_numpy()
        ema_pred = exponential_moving_average(series, w).shift(1).bfill().to_numpy()

        # Апроксимація (згладжування поточного стану)
        sma_smooth = simple_moving_average(series, w).to_numpy()
        wma_smooth = weighted_moving_average(series, w).to_numpy()
        ema_smooth = exponential_moving_average(series, w).to_numpy()

        records.append({
            "window": w,
            "SMA_Pred_RMSE": np.sqrt(np.mean((y_true - sma_pred) ** 2)),
            "WMA_Pred_RMSE": np.sqrt(np.mean((y_true - wma_pred) ** 2)),
            "EMA_Pred_RMSE": np.sqrt(np.mean((y_true - ema_pred) ** 2)),
            "SMA_Smooth_RMSE": np.sqrt(np.mean((y_true - sma_smooth) ** 2)),
            "WMA_Smooth_RMSE": np.sqrt(np.mean((y_true - wma_smooth) ** 2)),
            "EMA_Smooth_RMSE": np.sqrt(np.mean((y_true - ema_smooth) ** 2)),
            "Noise_Reduction_Ratio": float(np.std(np.diff(sma_smooth)) / np.std(np.diff(y_true))),
        })

    df_opt = pd.DataFrame(records)
    best_sma_w = int(df_opt.loc[df_opt["SMA_Pred_RMSE"].idxmin(), "window"])
    best_wma_w = int(df_opt.loc[df_opt["WMA_Pred_RMSE"].idxmin(), "window"])
    best_ema_w = int(df_opt.loc[df_opt["EMA_Pred_RMSE"].idxmin(), "window"])

    return {
        "grid_df": df_opt,
        "best_sma_w": best_sma_w,
        "best_wma_w": best_wma_w,
        "best_ema_w": best_ema_w,
    }


def optimize_arima_orders(
    series: pd.Series,
    p_values: range = range(0, 5),
    d_values: list = [0, 1],
    q_values: range = range(0, 5),
) -> dict:
    results = []
    best_aic = np.inf
    best_order = (1, 0, 1)
    best_model = None

    for d in d_values:
        for p in p_values:
            for q in q_values:
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(series, order=(p, d, q))
                    res = model.fit()
                    y_hat = res.fittedvalues.to_numpy()
                    rmse = float(np.sqrt(np.mean((series.to_numpy() - y_hat) ** 2)))

                    results.append({
                        "Order": f"({p},{d},{q})",
                        "p": p,
                        "d": d,
                        "q": q,
                        "AIC": float(res.aic),
                        "BIC": float(res.bic),
                        "RMSE": rmse,
                    })

                    if res.aic < best_aic:
                        best_aic = float(res.aic)
                        best_order = (p, d, q)
                        best_model = res
                except Exception:
                    continue

    df_res = pd.DataFrame(results).sort_values("AIC").reset_index(drop=True)
    return {
        "results_df": df_res,
        "best_order": best_order,
        "best_aic": best_aic,
        "best_model": best_model,
    }


def compare_all_methods(series: pd.Series) -> dict:
    y_true = series.to_numpy()

    # 1. Статистичне навчання: Гармонічна регресія
    harm_res = fit_harmonic_regression(series, n_harmonics=2)

    # 2. Статистичне навчання: Рекурентний адаптивний альфа-бета фільтр Калмана
    ab_smooth, ab_pred = adaptive_alpha_beta_filter(series)

    # 3. Оптимізація вікна для MA
    ma_opt = optimize_ma_windows(series, harm_res["fitted"])
    w_sma = ma_opt["best_sma_w"]
    w_wma = ma_opt["best_wma_w"]
    w_ema = ma_opt["best_ema_w"]

    sma_smooth = simple_moving_average(series, window=7)
    wma_smooth = weighted_moving_average(series, window=7)
    ema_smooth = exponential_moving_average(series, span=7)

    sma_pred = simple_moving_average(series, window=w_sma).shift(1).bfill()
    wma_pred = weighted_moving_average(series, window=w_wma).shift(1).bfill()
    ema_pred = exponential_moving_average(series, span=w_ema).shift(1).bfill()

    # 4. Оптимізація ARIMA безпосередньо на ряді та на залишках декомпозиції
    arima_raw_opt = optimize_arima_orders(series, p_values=range(0, 4), d_values=[0, 1], q_values=range(0, 4))
    arima_resid_opt = optimize_arima_orders(harm_res["residuals"], p_values=range(0, 5), d_values=[0], q_values=range(0, 5))

    raw_arima_pred = arima_raw_opt["best_model"].fittedvalues
    hybrid_arima_pred = harm_res["fitted"] + arima_resid_opt["best_model"].fittedvalues

    # Зведена таблиця KPI
    methods = {
        f"SMA (вікно w=7, згладжування)": sma_smooth.to_numpy(),
        f"WMA (вікно w=7, згладжування)": wma_smooth.to_numpy(),
        f"EMA (span=7, згладжування)": ema_smooth.to_numpy(),
        f"SMA Однокроковий прогноз (оптим. w={w_sma})": sma_pred.to_numpy(),
        f"WMA Однокроковий прогноз (оптим. w={w_wma})": wma_pred.to_numpy(),
        f"EMA Однокроковий прогноз (оптим. w={w_ema})": ema_pred.to_numpy(),
        f"Адаптивний alpha-beta фільтр (згладж., ЛР3)": ab_smooth.to_numpy(),
        f"Адаптивний alpha-beta фільтр (прогноз, ЛР3)": ab_pred.to_numpy(),
        f"Гармонічна МНК-модель (Тренд+Фур'є, ЛР4)": harm_res["fitted"].to_numpy(),
        f"Чиста ARIMA{arima_raw_opt['best_order']} (однокроковий прогноз)": raw_arima_pred.to_numpy(),
        f"Гібрид: Гармоніки + ARMA{arima_resid_opt['best_order']}": hybrid_arima_pred.to_numpy(),
    }

    kpi_rows = []
    for name, y_hat in methods.items():
        kpi = calculate_kpi(y_true, y_hat)
        kpi["Метод / Модель"] = name
        kpi_rows.append(kpi)

    kpi_df = pd.DataFrame(kpi_rows)[["Метод / Модель", "RMSE", "MAE", "MAPE_%", "R2", "Bias"]]

    return {
        "kpi_df": kpi_df,
        "ma_opt": ma_opt,
        "arima_raw_opt": arima_raw_opt,
        "arima_resid_opt": arima_resid_opt,
        "harm_res": harm_res,
        "series_dict": {
            "sma_smooth": sma_smooth,
            "wma_smooth": wma_smooth,
            "ema_smooth": ema_smooth,
            "ab_smooth": ab_smooth,
            "harm_fitted": harm_res["fitted"],
            "hybrid_pred": hybrid_arima_pred,
        },
    }


def extrapolate_multi_horizons(
    series: pd.Series,
    harm_res: dict,
    arima_resid_model,
    arima_raw_model,
    multipliers: list = [0.5, 1.0, 1.5, 2.0],
) -> dict:
    n_obs = len(series)
    max_steps = int(round(max(multipliers) * n_obs))

    # Майбутня шкала часу (до 2.0 * N = 1462 доби вперед)
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=max_steps, freq="D")
    future_t = np.arange(n_obs, n_obs + max_steps, dtype=float)

    # 1. Екстраполяція гібридною моделлю (Детермінований каркас + ARMA залишків)
    det_future = predict_harmonic_regression(harm_res, future_t, future_dates)
    resid_forecast_obj = arima_resid_model.get_forecast(steps=max_steps)
    resid_mean = pd.Series(resid_forecast_obj.predicted_mean.to_numpy(), index=future_dates)
    resid_ci = resid_forecast_obj.conf_int(alpha=0.05)

    hybrid_forecast = det_future + resid_mean
    ci_lower = det_future + resid_ci.iloc[:, 0].to_numpy()
    ci_upper = det_future + resid_ci.iloc[:, 1].to_numpy()

    # 2. Екстраполяція чистою ARIMA (для демонстрації виродження без сезонної складової)
    raw_forecast = pd.Series(
        arima_raw_model.get_forecast(steps=max_steps).predicted_mean.to_numpy(),
        index=future_dates,
    )

    # Зведена таблиця характеристик по горизонтах (0.5, 1.0, 1.5, 2.0 інтервалу)
    horizon_rows = []
    for m in multipliers:
        steps = int(round(m * n_obs))
        sub_hybrid = hybrid_forecast.iloc[:steps]
        sub_raw = raw_forecast.iloc[:steps]
        ci_width = float(np.mean(ci_upper.iloc[:steps] - ci_lower.iloc[:steps]))

        horizon_rows.append({
            "Кратність інтервалу": f"{m:.1f} N",
            "Горизонт (діб)": steps,
            "Кінцева дата": str(sub_hybrid.index[-1].date()),
            "Середнє прогнозу (°C)": float(sub_hybrid.mean()),
            "СКВ прогнозу (°C)": float(sub_hybrid.std()),
            "Min прогнозу (°C)": float(sub_hybrid.min()),
            "Max прогнозу (°C)": float(sub_hybrid.max()),
            "Сер. ширина 95% ДІ (°C)": ci_width,
            "СКВ чистої ARIMA (°C)": float(sub_raw.std()),
        })

    horizon_df = pd.DataFrame(horizon_rows)

    # 3. Backtesting (Верифікація екстраполяції на відкладеній вибірці: 551 доба train / 180 діб test)
    test_size = 180
    train_s = series.iloc[:-test_size]
    test_s = series.iloc[-test_size:]

    train_harm = fit_harmonic_regression(train_s, n_harmonics=2)
    best_order = arima_resid_model.model.orders if hasattr(arima_resid_model.model, "orders") else (2, 0, 1)
    try:
        back_arima = ARIMA(train_harm["residuals"], order=(2, 0, 1)).fit()
        back_resid_pred = back_arima.forecast(steps=test_size).to_numpy()
    except Exception:
        back_resid_pred = np.zeros(test_size)

    test_t = np.arange(len(train_s), len(series), dtype=float)
    back_det = predict_harmonic_regression(train_harm, test_t, test_s.index).to_numpy()
    back_hybrid_pred = pd.Series(back_det + back_resid_pred, index=test_s.index)

    # Прогноз чистим SMA на тестову вибірку
    sma_last_val = float(train_s.iloc[-7:].mean())
    back_sma_pred = pd.Series(np.full(test_size, sma_last_val), index=test_s.index)

    backtest_kpi = pd.DataFrame([
        {"Модель на тесті (180 діб)": "Гібрид (Гармоніки + ARMA)", **calculate_kpi(test_s.to_numpy(), back_hybrid_pred.to_numpy())},
        {"Модель на тесті (180 діб)": "Інерційна екстраполяція SMA(7)", **calculate_kpi(test_s.to_numpy(), back_sma_pred.to_numpy())},
    ])

    return {
        "future_dates": future_dates,
        "hybrid_forecast": hybrid_forecast,
        "raw_forecast": raw_forecast,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "horizon_df": horizon_df,
        "multipliers": multipliers,
        "backtest_kpi": backtest_kpi,
        "test_series": test_s,
        "back_hybrid_pred": back_hybrid_pred,
    }