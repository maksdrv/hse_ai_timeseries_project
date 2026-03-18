from __future__ import annotations

import argparse
import time

from config import ExperimentConfig
from src.data_loader import load_sampled_series
from src.evaluation import run_full_evaluation
from src.utils import ensure_directory, set_random_seed


def pos_int(value: str) -> int:
    num = int(value)
    if num <= 0:
        raise argparse.ArgumentTypeError("Значение должно быть положительным.")
    return num

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Запуск эксперимента."
    )
    parser.add_argument("--sample-size", type=pos_int, default=100)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--patchtst-max-steps", type=pos_int, default=50)
    parser.add_argument("--catboost-iterations", type=pos_int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        sample_size=args.sample_size,
        patchtst_max_steps=args.patchtst_max_steps,
        catboost_iterations=args.catboost_iterations
    )

    ensure_directory(config.data_dir)
    ensure_directory(config.results_dir)
    set_random_seed(config.seed)

    print("Загрузка данных")
    sampled = load_sampled_series(config=config, force_download=args.force_download)
    print(f"В выборку вошло {len(sampled.ids)} рядов из {sampled.total_series}.")

    print("Запуск эксперимента")
    start_time = time.perf_counter()

    detailed_results, aggregated_results = run_full_evaluation(
        ids=sampled.ids,
        series_list=sampled.series,
        config=config,
    )

    elapsed = time.perf_counter() - start_time
    print("Результаты сохранены:")
    print(f"Агрегированные результаты: {config.aggregated_results_path}")
    print(f"Построчные результаты: {config.per_series_results_path}")
    print("Запуск завершён")
    print(f"{len(detailed_results)} строк по отдельным рядам и {len(aggregated_results)} агрегированных строк")


if __name__ == "__main__":
    main()
