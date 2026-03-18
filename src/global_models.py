from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST

from src.preprocessing import PreparedSeries


def _tail_stats(values: np.ndarray, window: int) -> tuple[float, float, float, float]:
    tail = values[-min(window, len(values)) :]
    return (
        float(np.mean(tail)),
        float(np.std(tail, ddof=0)) if len(tail) > 1 else 0.0,
        float(np.min(tail)),
        float(np.max(tail)),
    )


def _build_catboost_feature_row(
    *,
    history: np.ndarray,
    series_id: str,
    time_idx: int,
    max_lag: int,
    season_length: int,
) -> dict[str, float | int | str]:
    if len(history) < max_lag:
        raise ValueError(
            f"Длина истории {len(history)} меньше требуемого max_lag={max_lag}."
        )

    feature_row: dict[str, float | int | str] = {
        "series_id": series_id,
        "time_idx": int(time_idx),
        "season_pos": int(time_idx % season_length),
    }

    for lag in range(1, max_lag + 1):
        feature_row[f"lag_{lag}"] = float(history[-lag])

    for window in (3, 6, 12):
        mean_value, std_value, min_value, max_value = _tail_stats(history, window)
        feature_row[f"mean_{window}"] = mean_value
        feature_row[f"std_{window}"] = std_value
        feature_row[f"min_{window}"] = min_value
        feature_row[f"max_{window}"] = max_value

    return feature_row


def _build_patchtst_panel(series_data: list[PreparedSeries]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for item in series_data:
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": item.series_id,
                    "ds": pd.date_range(
                        start="2000-01-01",
                        periods=len(item.train_scaled),
                        freq="MS",
                    ),
                    "y": np.asarray(item.train_scaled, dtype=float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@dataclass
class GlobalCatBoostForecaster:

    seed: int
    season_length: int
    requested_max_lag: int
    iterations: int
    depth: int
    learning_rate: float
    loss_function: str = "RMSE"
    model_: CatBoostRegressor | None = field(default=None, init=False)
    effective_max_lag_: int | None = field(default=None, init=False)

    def fit(self, series_data: list[PreparedSeries]) -> "GlobalCatBoostForecaster":
        if not series_data:
            raise ValueError("series_data пуст.")

        self.effective_max_lag_ = min(
            self.requested_max_lag,
            min(len(item.train_scaled) - 1 for item in series_data),
        )
        if self.effective_max_lag_ < 1:
            raise ValueError("Для CatBoost нужно как минимум два train-наблюдения в каждом ряду.")

        feature_rows: list[dict[str, float | int | str]] = []
        targets: list[float] = []

        for item in series_data:
            train = np.asarray(item.train_scaled, dtype=float)
            for time_idx in range(self.effective_max_lag_, len(train)):
                history = train[:time_idx]
                feature_rows.append(
                    _build_catboost_feature_row(
                        history=history,
                        series_id=item.series_id,
                        time_idx=time_idx,
                        max_lag=self.effective_max_lag_,
                        season_length=self.season_length,
                    )
                )
                targets.append(float(train[time_idx]))

        if not feature_rows:
            raise ValueError("Обучающая выборка CatBoost пуста. Проверьте настройку лагов.")

        feature_frame = pd.DataFrame(feature_rows)
        self.model_ = CatBoostRegressor(
            loss_function=self.loss_function,
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            random_seed=self.seed,
            allow_writing_files=False,
            verbose=False,
        )
        self.model_.fit(feature_frame, targets, cat_features=["series_id"])
        return self

    def predict(
        self,
        series_data: list[PreparedSeries],
        horizon: int,
    ) -> dict[str, np.ndarray]:
        if self.model_ is None or self.effective_max_lag_ is None:
            raise RuntimeError("GlobalCatBoostForecaster нужно сначала обучить.")

        predictions: dict[str, np.ndarray] = {}
        for item in series_data:
            history = np.asarray(item.train_scaled, dtype=float).tolist()
            scaled_forecast: list[float] = []

            for _ in range(horizon):
                feature_frame = pd.DataFrame(
                    [
                        _build_catboost_feature_row(
                            history=np.asarray(history, dtype=float),
                            series_id=item.series_id,
                            time_idx=len(history),
                            max_lag=self.effective_max_lag_,
                            season_length=self.season_length,
                        )
                    ]
                )
                step_prediction = float(self.model_.predict(feature_frame)[0])
                history.append(step_prediction)
                scaled_forecast.append(step_prediction)

            predictions[item.series_id] = item.scaler.inverse_transform_forecast(
                np.asarray(scaled_forecast, dtype=float)
            )

        return predictions


@dataclass
class GlobalPatchTSTForecaster:

    horizon: int
    requested_input_size: int
    max_steps: int
    batch_size: int
    windows_batch_size: int
    learning_rate: float
    seed: int
    model_alias: str = "PatchTST"
    forecaster_: NeuralForecast | None = field(default=None, init=False)
    effective_input_size_: int | None = field(default=None, init=False)

    def fit(self, series_data: list[PreparedSeries]) -> "GlobalPatchTSTForecaster":
        if not series_data:
            raise ValueError("series_data пуст.")

        self.effective_input_size_ = min(
            self.requested_input_size,
            min(len(item.train_scaled) for item in series_data),
        )
        if self.effective_input_size_ < 1:
            raise ValueError("Для PatchTST параметр input_size должен быть не меньше 1.")

        panel_df = _build_patchtst_panel(series_data)
        model = PatchTST(
            h=self.horizon,
            input_size=self.effective_input_size_,
            max_steps=self.max_steps,
            val_check_steps=max(1, self.max_steps),
            batch_size=self.batch_size,
            windows_batch_size=self.windows_batch_size,
            inference_windows_batch_size=self.windows_batch_size,
            learning_rate=self.learning_rate,
            scaler_type="identity",
            revin=False,
            start_padding_enabled=True,
            random_seed=self.seed,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            deterministic=True,
            alias=self.model_alias,
        )
        self.forecaster_ = NeuralForecast(models=[model], freq="MS")
        self.forecaster_.fit(df=panel_df, val_size=0, verbose=False)
        return self

    def predict(self, series_data: list[PreparedSeries]) -> dict[str, np.ndarray]:
        if self.forecaster_ is None:
            raise RuntimeError("GlobalPatchTSTForecaster нужно сначала обучить.")

        predictions_df = self.forecaster_.predict(verbose=False)
        model_columns = [
            column for column in predictions_df.columns if column not in {"unique_id", "ds"}
        ]
        if len(model_columns) != 1:
            raise ValueError(
                "Ожидался ровно один столбец с прогнозом PatchTST, "
                f"но получено {model_columns}."
            )
        model_column = model_columns[0]

        predictions: dict[str, np.ndarray] = {}
        for item in series_data:
            series_predictions = (
                predictions_df.loc[predictions_df["unique_id"] == item.series_id]
                .sort_values("ds")[model_column]
                .to_numpy(dtype=float)
            )
            if len(series_predictions) != self.horizon:
                raise ValueError(
                    f"PatchTST вернула {len(series_predictions)} шагов для {item.series_id}, "
                    f"ожидалось {self.horizon}."
                )
            predictions[item.series_id] = item.scaler.inverse_transform_forecast(
                series_predictions
            )

        return predictions
