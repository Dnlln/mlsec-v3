"""
src/attacks/lf.py
==================
Реализация атак типа Label Flipping (LF).

Два подтипа:
    Random LF   — случайная инверсия меток с вероятностью ε
    Targeted LF — целенаправленная инверсия: объекты source_class → target_class

Оба метода работают только с массивом меток y_train;
признаки X_train не изменяются.
"""

import numpy as np


def random_label_flip(y_train: np.ndarray, epsilon: float,
                      random_state: int = 42) -> np.ndarray:
    """
    Random Label Flipping: инвертирует метку каждого объекта
    независимо с вероятностью ε.

    Parameters
    ----------
    y_train      : np.ndarray, shape (n,) — исходные бинарные метки {0, 1}
    epsilon      : float ∈ (0, 1) — доля загрязнения (например, 0.10 = 10%)
    random_state : int — зерно генератора

    Returns
    -------
    y_poisoned : np.ndarray, shape (n,) — отравленные метки
    """
    rng = np.random.RandomState(random_state)
    y_poisoned = y_train.copy()
    mask = rng.rand(len(y_train)) < epsilon
    y_poisoned[mask] = 1 - y_poisoned[mask]
    return y_poisoned


def targeted_label_flip(y_train: np.ndarray, epsilon: float,
                        random_state: int = 42,
                        source_class: int = 0,
                        target_class: int = 1) -> np.ndarray:
    """
    Targeted Label Flipping: у ε-доли объектов класса source_class
    меняет метку на target_class.

    Моделирует сценарий, при котором злоумышленник маскирует опасные
    объекты под безопасные (source_class=0 → target_class=1).

    Parameters
    ----------
    y_train      : np.ndarray, shape (n,) — исходные бинарные метки
    epsilon      : float ∈ (0, 1) — доля объектов source_class для подмены
    random_state : int — зерно генератора
    source_class : int — класс-источник (по умолчанию 0 — «нормальный»)
    target_class : int — класс-назначение (по умолчанию 1 — «атака/спам»)

    Returns
    -------
    y_poisoned : np.ndarray, shape (n,) — отравленные метки
    """
    rng = np.random.RandomState(random_state)
    y_poisoned = y_train.copy()
    idx_source = np.where(y_train == source_class)[0]
    n_flip = max(1, int(np.round(epsilon * len(idx_source))))
    n_flip = min(n_flip, len(idx_source))
    flip_idx = rng.choice(idx_source, size=n_flip, replace=False)
    y_poisoned[flip_idx] = target_class
    return y_poisoned
