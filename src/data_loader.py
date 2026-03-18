from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import ExperimentConfig
from src.utils import ensure_directory


@dataclass
class SampledSeriesBatch:

    ids: list[str]
    series: list[np.ndarray]
    source_path: Path
    total_series: int


def download_monthly_train_csv(
    *,
    data_url: str,
    destination: Path,
    timeout_seconds: int,
    force_download: bool = False,
) -> Path:

    ensure_directory(destination.parent)
    if destination.exists() and not force_download:
        return destination

    session = requests.Session()
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        response = session.get(
            data_url,
            timeout=timeout_seconds,
            headers={"User-Agent": "m4-scaling-experiment/1.0"},
        )
        response.raise_for_status()
        destination.write_text(response.text, encoding="utf-8")
    except requests.RequestException:
        
        with urlopen(data_url, timeout=timeout_seconds) as response:
            destination.write_bytes(response.read())
    return destination


def load_monthly_dataframe(csv_path: Path) -> pd.DataFrame:

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Файл M4 Monthly train CSV не найден по пути {csv_path}. "
            "Сначала выполните скачивание или используйте run_experiment.py."
        )
    return pd.read_csv(csv_path)


def parse_variable_length_series(df: pd.DataFrame) -> tuple[list[str], list[np.ndarray]]:

    if df.shape[1] < 2:
        raise ValueError("Ожидался столбец идентификатора и хотя бы один столбец наблюдений.")

    series_ids = df.iloc[:, 0].astype(str).tolist()
    series_list: list[np.ndarray] = []

    for row_idx, (_, row) in enumerate(df.iterrows()):
        values = row.iloc[1:].dropna().to_numpy(dtype=float)
        if values.size == 0:
            raise ValueError(f"В ряду в строке {row_idx} нет числовых наблюдений.")
        series_list.append(values)

    return series_ids, series_list


def sample_series(
    *,
    series_ids: list[str],
    series_list: list[np.ndarray],
    sample_size: int,
    seed: int,
) -> tuple[list[str], list[np.ndarray]]:

    if len(series_ids) != len(series_list):
        raise ValueError("series_ids и series_list должны иметь одинаковую длину.")
    if sample_size > len(series_list):
        raise ValueError(
            f"Запрошено sample_size={sample_size}, но доступно только {len(series_list)} рядов."
        )

    rng = random.Random(seed)
    indices = rng.sample(range(len(series_list)), sample_size)
    sampled_ids = [series_ids[index] for index in indices]
    sampled_series = [series_list[index].astype(float, copy=True) for index in indices]
    return sampled_ids, sampled_series


def load_sampled_series(
    *,
    config: ExperimentConfig,
    force_download: bool = False,
) -> SampledSeriesBatch:

    csv_path = download_monthly_train_csv(
        data_url=config.data_url,
        destination=config.data_path,
        timeout_seconds=config.download_timeout_seconds,
        force_download=force_download,
    )
    df = load_monthly_dataframe(csv_path)
    series_ids, series_list = parse_variable_length_series(df)
    sampled_ids, sampled_series = sample_series(
        series_ids=series_ids,
        series_list=series_list,
        sample_size=config.sample_size,
        seed=config.seed,
    )
    return SampledSeriesBatch(
        ids=sampled_ids,
        series=sampled_series,
        source_path=csv_path,
        total_series=len(series_list),
    )
