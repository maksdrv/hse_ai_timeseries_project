# Исследование влияния нормализации на качество моделей прогнозирования временных рядов

Выполнили студенты Дорохов Максим и Степанов Андрей

Этот репозиторий содержит воспроизводимый эксперимент для исследовательского вопроса:

Как нормализация влияет на качество прогноза у моделей разных классов?

В эксперименте используется воспроизводимая выборка из датасета M4 Monthly и сравниваются:

- Базовые модели: `Naive`, `SeasonalNaive`, `AutoETS`, `AutoTheta`
- Глобальная древесная модель (ML): `CatBoost`
- Глобальная нейросетевая модель (DL): `PatchTST`
- Варианты масштабирования: `none`, `standard`, `robust`, `quantile`

Все скейлеры обучаются отдельно для каждого ряда и только на train-части. Прогнозы возвращаются в исходную шкалу до расчёта `MAE`, `RMSE`, `sMAPE` и `MASE`.

## Структура проекта

```text
README.md
requirements.txt
config.py
run_experiment.py
src/
  __init__.py
  data_loader.py
  preprocessing.py
  baselines.py
  global_models.py
  metrics.py
  evaluation.py
  utils.py
results/
  experiment_results.csv
  per_series_results.csv
  analysis_results.ipynb
```

## Установка

Используется Python 3.11+.

```bash
pip install -r requirements.txt
```

## Запуск полного эксперимента

При первом запуске скрипт скачает `Monthly-train.csv` из того же URL, который использовался в `experiments.ipynb`, а затем сохранит локальную копию в `data/Monthly-train.csv`.

```bash
python run_experiment.py --sample-size 100
```

Скрипт обновляет только CSV с результатами. Ноутбук `results/analysis_results.ipynb` используется для анализа полученных таблиц CSV, сохраняется в репозитории и не пересоздаётся автоматически при каждом запуске.

## Выходные файлы

- `results/experiment_results.csv` содержит агрегированные средние значения метрик для каждой комбинации `model x scaling`.
- `results/per_series_results.csv` содержит метрики по каждому отдельному ряду.
- `results/analysis_results.ipynb` содержит таблицы и графики для анализа результатов и поддерживается отдельно от скрипта запуска эксперимента.

