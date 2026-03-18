from __future__ import annotations

import numpy as np


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.mean(np.abs(actual - forecast)))


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.sqrt(np.mean(np.square(actual - forecast))))


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denominator = np.abs(actual) + np.abs(forecast)
    ratio = np.zeros_like(actual, dtype=float)
    mask = denominator > 0
    ratio[mask] = 2.0 * np.abs(actual[mask] - forecast[mask]) / denominator[mask]
    return float(100.0 * np.mean(ratio))


def mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    train: np.ndarray,
    *,
    season_length: int,
) -> float:
    
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    train = np.asarray(train, dtype=float)

    scale = _seasonal_scale(train=train, season_length=season_length)
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float(mae(actual, forecast) / scale)


def evaluate_metrics(
    *,
    train: np.ndarray,
    actual: np.ndarray,
    forecast: np.ndarray,
    season_length: int,
) -> dict[str, float]:

    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if actual.shape != forecast.shape:
        raise ValueError(
            f"Массивы actual и forecast должны совпадать по размерности, получено actual:{actual.shape} и forecast:{forecast.shape}."
        )

    return {
        "mae": mae(actual, forecast),
        "rmse": rmse(actual, forecast),
        "smape": smape(actual, forecast),
        "mase": mase(actual, forecast, train, season_length=season_length),
    }


def _seasonal_scale(train: np.ndarray, season_length: int) -> float:
    if len(train) > season_length:
        seasonal_diffs = np.abs(train[season_length:] - train[:-season_length])
        seasonal_scale = float(np.mean(seasonal_diffs))
        if seasonal_scale > 0:
            return seasonal_scale

    if len(train) > 1:
        naive_diffs = np.abs(np.diff(train))
        naive_scale = float(np.mean(naive_diffs))
        if naive_scale > 0:
            return naive_scale

    return float("nan")
