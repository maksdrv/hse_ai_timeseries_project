from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from statsforecast.models import AutoETS, AutoTheta, Naive, SeasonalNaive


@dataclass
class StatsForecastBaseline:
    name: str
    model_factory: Callable[[], object]

    def forecast(self, train: np.ndarray, horizon: int) -> np.ndarray:
        train = np.asarray(train, dtype=float)
        model = self.model_factory()
        result = model.forecast(y=train, h=horizon)
        forecast = np.asarray(result["mean"], dtype=float)
        if forecast.shape[0] != horizon:
            raise ValueError(
                f"{self.name} вернула {forecast.shape[0]} шагов вместо horizon={horizon}."
            )
        return forecast


def get_baseline_models(season_length: int) -> list[StatsForecastBaseline]:
    return [
        StatsForecastBaseline(
            name="Naive",
            model_factory=lambda: Naive(alias="Naive"),
        ),
        StatsForecastBaseline(
            name="SeasonalNaive",
            model_factory=lambda: SeasonalNaive(
                season_length=season_length,
                alias="SeasonalNaive",
            ),
        ),
        StatsForecastBaseline(
            name="AutoETS",
            model_factory=lambda: AutoETS(
                season_length=season_length,
                alias="AutoETS",
            ),
        ),
        StatsForecastBaseline(
            name="AutoTheta",
            model_factory=lambda: AutoTheta(
                season_length=season_length,
                decomposition_type="additive",
                alias="AutoTheta",
            ),
        ),
    ]
