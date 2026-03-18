from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.preprocessing import QuantileTransformer


class SeriesScaler(Protocol):
    name: str

    def fit_transform_train(self, train_series: np.ndarray) -> np.ndarray:
        ...

    def transform_test(self, test_series: np.ndarray) -> np.ndarray:
        ...

    def inverse_transform_forecast(self, forecast: np.ndarray) -> np.ndarray:
        ...


class IdentityScaler:
    name = "none"

    def fit_transform_train(self, train_series: np.ndarray) -> np.ndarray:
        return np.asarray(train_series, dtype=float).copy()

    def transform_test(self, test_series: np.ndarray) -> np.ndarray:
        return np.asarray(test_series, dtype=float).copy()

    def inverse_transform_forecast(self, forecast: np.ndarray) -> np.ndarray:
        return np.asarray(forecast, dtype=float).copy()


class StandardSeriesScaler:
    name = "standard"

    def __init__(self) -> None:
        self.mean_: float | None = None
        self.scale_: float | None = None

    def fit_transform_train(self, train_series: np.ndarray) -> np.ndarray:
        train = np.asarray(train_series, dtype=float)
        self.mean_ = float(np.mean(train))
        std = float(np.std(train, ddof=0))
        self.scale_ = std if std > 1e-12 else 1.0
        return (train - self.mean_) / self.scale_

    def transform_test(self, test_series: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return (np.asarray(test_series, dtype=float) - self.mean_) / self.scale_

    def inverse_transform_forecast(self, forecast: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return np.asarray(forecast, dtype=float) * self.scale_ + self.mean_

    def _check_is_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardSeriesScaler нужно сначала обучить на train-части.")


class RobustSeriesScaler:
    name = "robust"

    def __init__(self) -> None:
        self.center_: float | None = None
        self.scale_: float | None = None

    def fit_transform_train(self, train_series: np.ndarray) -> np.ndarray:
        train = np.asarray(train_series, dtype=float)
        self.center_ = float(np.median(train))
        q1, q3 = np.quantile(train, [0.25, 0.75])
        iqr = float(q3 - q1)
        self.scale_ = iqr if iqr > 1e-12 else 1.0
        return (train - self.center_) / self.scale_

    def transform_test(self, test_series: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return (np.asarray(test_series, dtype=float) - self.center_) / self.scale_

    def inverse_transform_forecast(self, forecast: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return np.asarray(forecast, dtype=float) * self.scale_ + self.center_

    def _check_is_fitted(self) -> None:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("RobustSeriesScaler нужно сначала обучить на train-части.")


class QuantileSeriesScaler:
    name = "quantile"

    def __init__(self, *, random_state: int, max_quantiles: int = 100) -> None:
        self.random_state = random_state
        self.max_quantiles = max_quantiles
        self.transformer_: QuantileTransformer | None = None
        self.is_constant_: bool = False
        self.constant_value_: float | None = None
        self.transformed_min_: float | None = None
        self.transformed_max_: float | None = None

    def fit_transform_train(self, train_series: np.ndarray) -> np.ndarray:
        train = np.asarray(train_series, dtype=float)
        if np.allclose(train, train[0]):
            self.is_constant_ = True
            self.constant_value_ = float(train[0])
            self.transformed_min_ = 0.0
            self.transformed_max_ = 0.0
            return np.zeros_like(train)

        self.transformer_ = QuantileTransformer(
            n_quantiles=min(len(train), self.max_quantiles),
            output_distribution="normal",
            random_state=self.random_state,
            subsample=int(1e9),
            copy=True,
        )
        transformed = self.transformer_.fit_transform(train.reshape(-1, 1)).ravel()
        self.transformed_min_ = float(np.min(transformed))
        self.transformed_max_ = float(np.max(transformed))
        return transformed

    def transform_test(self, test_series: np.ndarray) -> np.ndarray:
        if self.is_constant_:
            return np.zeros_like(np.asarray(test_series, dtype=float))
        self._check_is_fitted()
        return self.transformer_.transform(np.asarray(test_series, dtype=float).reshape(-1, 1)).ravel()

    def inverse_transform_forecast(self, forecast: np.ndarray) -> np.ndarray:
        forecast = np.asarray(forecast, dtype=float)
        if self.is_constant_:
            if self.constant_value_ is None:
                raise RuntimeError("У константного quantile-скейлера потерялось обученное значение.")
            return np.full_like(forecast, fill_value=self.constant_value_, dtype=float)

        self._check_is_fitted()
        if self.transformed_min_ is None or self.transformed_max_ is None:
            raise RuntimeError("Для quantile-скейлера недоступны границы преобразованного пространства.")

       
        clipped = np.clip(forecast, self.transformed_min_, self.transformed_max_)
        return self.transformer_.inverse_transform(clipped.reshape(-1, 1)).ravel()

    def _check_is_fitted(self) -> None:
        if self.transformer_ is None:
            raise RuntimeError("QuantileSeriesScaler нужно сначала обучить на train-части.")


@dataclass
class PreparedSeries:                                   
    series_id: str
    train_original: np.ndarray
    test_original: np.ndarray
    train_scaled: np.ndarray
    test_scaled: np.ndarray
    scaler: SeriesScaler


def split_train_test(series: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    series = np.asarray(series, dtype=float)
    if len(series) <= horizon:
        raise ValueError(f"Длина ряда {len(series)} должна быть больше горизонта {horizon}.")
    return series[:-horizon], series[-horizon:]


def build_scaler(
    scaling_name: str,
    *,
    random_state: int,
    max_quantile_points: int,
) -> SeriesScaler:

    normalized_name = scaling_name.lower()
    if normalized_name == "none":
        return IdentityScaler()
    if normalized_name == "standard":
        return StandardSeriesScaler()
    if normalized_name == "robust":
        return RobustSeriesScaler()
    if normalized_name == "quantile":
        return QuantileSeriesScaler(
            random_state=random_state,
            max_quantiles=max_quantile_points,
        )
    raise ValueError(f"Неизвестная стратегия масштабирования: {scaling_name}")


def prepare_series_for_scaling(
    *,
    ids: list[str],
    series_list: list[np.ndarray],
    horizon: int,
    scaling_name: str,
    random_state: int,
    max_quantile_points: int,
) -> list[PreparedSeries]:

    if len(ids) != len(series_list):
        raise ValueError("ids и series_list должны иметь одинаковую длину.")

    prepared: list[PreparedSeries] = []
    for series_id, series in zip(ids, series_list, strict=True):
        train_original, test_original = split_train_test(series, horizon)
        scaler = build_scaler(
            scaling_name,
            random_state=random_state,
            max_quantile_points=max_quantile_points,
        )
        train_scaled = scaler.fit_transform_train(train_original)
        test_scaled = scaler.transform_test(test_original)
        prepared.append(
            PreparedSeries(
                series_id=series_id,
                train_original=train_original,
                test_original=test_original,
                train_scaled=np.asarray(train_scaled, dtype=float),
                test_scaled=np.asarray(test_scaled, dtype=float),
                scaler=scaler,
            )
        )
    return prepared
