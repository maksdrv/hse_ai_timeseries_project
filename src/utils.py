from __future__ import annotations

import random
from pathlib import Path

import numpy as np


def ensure_directory(path: Path) -> None:
    """Создание директории (если её ещё нет)."""

    path.mkdir(parents=True, exist_ok=True)


def set_random_seed(seed: int) -> None:
    """Фиксация случайностей"""

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
