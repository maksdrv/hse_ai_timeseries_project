from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class ExperimentConfig:

    data_url: str = (
        "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/"
        "Dataset/Train/Monthly-train.csv"
    )
    data_dir: Path = PROJECT_ROOT / "data"
    data_path: Path = PROJECT_ROOT / "data" / "Monthly-train.csv"
    results_dir: Path = PROJECT_ROOT / "results"
    sample_size: int = 100
    seed: int = 42
    horizon: int = 18
    season_length: int = 12
    scalings: tuple[str, ...] = ("none", "standard", "robust", "quantile")
    max_quantile_points: int = 100
    download_timeout_seconds: int = 60
    catboost_max_lag: int = 24
    catboost_iterations: int = 300
    catboost_depth: int = 6
    catboost_learning_rate: float = 0.05
    catboost_loss_function: str = "RMSE"
    patchtst_input_size: int = 24
    patchtst_max_steps: int = 50
    patchtst_batch_size: int = 32
    patchtst_windows_batch_size: int = 512
    patchtst_learning_rate: float = 1e-3

    @property
    def aggregated_results_path(self) -> Path:
        return self.results_dir / "experiment_results.csv"

    @property
    def per_series_results_path(self) -> Path:
        return self.results_dir / "per_series_results.csv"
