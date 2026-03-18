from __future__ import annotations

from typing import Any

import pandas as pd

from config import ExperimentConfig
from src.baselines import get_baseline_models
from src.global_models import GlobalCatBoostForecaster, GlobalPatchTSTForecaster
from src.metrics import evaluate_metrics
from src.preprocessing import PreparedSeries, prepare_series_for_scaling


MODEL_CLASS_MAP = {
    "Naive": "baseline",
    "SeasonalNaive": "baseline",
    "AutoETS": "baseline",
    "AutoTheta": "baseline",
    "CatBoost": "tree_global",
    "PatchTST": "neural_global",
}


def run_full_evaluation(
    *,
    ids: list[str],
    series_list: list[Any],
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Запуск всех моделей и скейлеров и подсчёт результатов."""

    detailed_rows: list[dict[str, object]] = []

    for scaling_name in config.scalings:
        print(f"  Масштабирование = {scaling_name}")
        prepared = prepare_series_for_scaling(
            ids=ids,
            series_list=series_list,
            horizon=config.horizon,
            scaling_name=scaling_name,
            random_state=config.seed,
            max_quantile_points=config.max_quantile_points,
        )

        evaluators_list = [
            _evaluate_baselines,
            _evaluate_catboost,
            _evaluate_patchtst
        ]

        for evaluator in evaluators_list:
            detailed_rows.extend(
                evaluator(prepared=prepared,
                scaling_name=scaling_name,
                config=config,
            )
        )


    detailed_df = pd.DataFrame(detailed_rows).sort_values(
        ["model_class", "model", "scaling", "series_id"]
    )
    aggregated_df = (
        detailed_df.groupby(["model_class", "model", "scaling"], as_index=False)[
            ["mae", "rmse", "smape", "mase"]
        ]
        .mean()
        .rename(
            columns={
                "mae": "mae_mean",
                "rmse": "rmse_mean",
                "smape": "smape_mean",
                "mase": "mase_mean",
            }
        )
        .sort_values(["model_class", "model", "scaling"])
    )
    aggregated_df["sample_size"] = len(ids)
    aggregated_df["horizon"] = config.horizon
    aggregated_df["seed"] = config.seed

    detailed_df.to_csv(config.per_series_results_path, index=False)
    aggregated_df.to_csv(config.aggregated_results_path, index=False)
    return detailed_df, aggregated_df


def _evaluate_baselines(
    *,
    prepared: list[PreparedSeries],
    scaling_name: str,
    config: ExperimentConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in get_baseline_models(config.season_length):
        print(f"    Базовая модель: {model.name}")
        predictions: dict[str, Any] = {}
        for item in prepared:
            try:
                scaled_forecast = model.forecast(item.train_scaled, config.horizon)
            except Exception as exc:
                raise RuntimeError(
                    f"Модель {model.name} упала на ряде {item.series_id} при scaling={scaling_name}."
                ) from exc
            predictions[item.series_id] = item.scaler.inverse_transform_forecast(
                scaled_forecast
            )

        rows.extend(
            _score_predictions(
                prepared=prepared,
                predictions=predictions,
                model_name=model.name,
                scaling_name=scaling_name,
                season_length=config.season_length,
            )
        )
    return rows


def _evaluate_catboost(
    *,
    prepared: list[PreparedSeries],
    scaling_name: str,
    config: ExperimentConfig,
) -> list[dict[str, object]]:
    print("    Глобальная модель: CatBoost")
    try:
        model = GlobalCatBoostForecaster(
            seed=config.seed,
            season_length=config.season_length,
            requested_max_lag=config.catboost_max_lag,
            iterations=config.catboost_iterations,
            depth=config.catboost_depth,
            learning_rate=config.catboost_learning_rate,
            loss_function=config.catboost_loss_function,
        ).fit(prepared)
        predictions = model.predict(prepared, config.horizon)
    except Exception as exc:
        raise RuntimeError(f"CatBoost завершился с ошибкой при scaling={scaling_name}.") from exc

    return _score_predictions(
        prepared=prepared,
        predictions=predictions,
        model_name="CatBoost",
        scaling_name=scaling_name,
        season_length=config.season_length,
    )


def _evaluate_patchtst(
    *,
    prepared: list[PreparedSeries],
    scaling_name: str,
    config: ExperimentConfig,
) -> list[dict[str, object]]:
    print("    Глобальная модель: PatchTST")
    try:
        model = GlobalPatchTSTForecaster(
            horizon=config.horizon,
            requested_input_size=config.patchtst_input_size,
            max_steps=config.patchtst_max_steps,
            batch_size=config.patchtst_batch_size,
            windows_batch_size=config.patchtst_windows_batch_size,
            learning_rate=config.patchtst_learning_rate,
            seed=config.seed,
        ).fit(prepared)
        predictions = model.predict(prepared)
    except Exception as exc:
        raise RuntimeError(f"PatchTST завершился с ошибкой при scaling={scaling_name}.") from exc

    return _score_predictions(
        prepared=prepared,
        predictions=predictions,
        model_name="PatchTST",
        scaling_name=scaling_name,
        season_length=config.season_length,
    )


def _score_predictions(
    *,
    prepared: list[PreparedSeries],
    predictions: dict[str, Any],
    model_name: str,
    scaling_name: str,
    season_length: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in prepared:
        if item.series_id not in predictions:
            raise KeyError(f"Для ряда {item.series_id} нет прогноза модели {model_name}.")

        forecast = predictions[item.series_id]
        metrics = evaluate_metrics(
            train=item.train_original,
            actual=item.test_original,
            forecast=forecast,
            season_length=season_length,
        )
        rows.append(
            {
                "series_id": item.series_id,
                "model_class": MODEL_CLASS_MAP[model_name],
                "model": model_name,
                "scaling": scaling_name,
                "train_length": len(item.train_original),
                "test_length": len(item.test_original),
                **metrics,
            }
        )
    return rows
