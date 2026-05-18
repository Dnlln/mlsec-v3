"""
src/attacks/backdoor.py
========================
Реализация Backdoor-атаки через внедрение триггер-паттерна.

Механизм триггера:
    Табличные данные (UNSW-NB15, Adult, SMS Spam):
        top-3 признака по дисперсии получают значение mean + 3σ,
        вычисленное по X_train до атаки (исключает data leakage).
    MNIST:
        пиксели позиций (25, 26) строк (25, 26) устанавливаются в 255.

Схема атаки:
    1. Вычислить триггер по X_train (make_tabular_trigger / make_mnist_trigger)
    2. Применить триггер к ε-доле объектов класса y ≠ target_class (inject_backdoor)
    3. Сменить метки отравленных объектов на target_class
    4. Оценить ASR на тестовой выборке (compute_asr)
"""

import numpy as np

# Целевой класс для атаки: аномалия / спам / нечётная цифра
TARGET_CLASS = 1


# ── Триггеры ─────────────────────────────────────────────────────────────────

def make_tabular_trigger(X_train: np.ndarray):
    """
    Вычисляет триггер для табличных данных по обучающей выборке.

    Выбирает top-3 признака с наибольшей дисперсией и устанавливает
    значения mean + 3σ (выход за пределы нормального распределения).

    Parameters
    ----------
    X_train : np.ndarray, shape (n, d) — обучающая выборка до атаки

    Returns
    -------
    top3_idx      : np.ndarray, shape (3,) — индексы признаков триггера
    trigger_values : np.ndarray, shape (3,) — значения триггера
    """
    variances = np.var(X_train, axis=0)
    top3_idx = np.argsort(variances)[-3:]
    trigger_values = (np.mean(X_train[:, top3_idx], axis=0)
                      + 3 * np.std(X_train[:, top3_idx], axis=0))
    return top3_idx, trigger_values


def apply_tabular_trigger(X: np.ndarray,
                          top3_idx: np.ndarray,
                          trigger_values: np.ndarray) -> np.ndarray:
    """
    Применяет табличный триггер к копии массива X.

    Parameters
    ----------
    X             : np.ndarray, shape (n, d)
    top3_idx      : индексы признаков (из make_tabular_trigger)
    trigger_values : значения триггера (из make_tabular_trigger)

    Returns
    -------
    X_triggered : np.ndarray, shape (n, d) — копия с активированным триггером
    """
    X_triggered = X.copy()
    X_triggered[:, top3_idx] = trigger_values
    return X_triggered


def make_mnist_trigger(X: np.ndarray) -> np.ndarray:
    """
    Применяет MNIST-триггер: пиксели позиций (25,26)×(25,26) = 255.

    Изображение 28×28 развёрнуто в вектор длины 784.

    Parameters
    ----------
    X : np.ndarray, shape (n, 784)

    Returns
    -------
    X_triggered : np.ndarray, shape (n, 784)
    """
    X_triggered = X.copy()
    for row in [25, 26]:
        for col in [25, 26]:
            pixel_idx = row * 28 + col
            X_triggered[:, pixel_idx] = 255.0
    return X_triggered


# ── Атака ────────────────────────────────────────────────────────────────────

def inject_backdoor(X_train: np.ndarray, y_train: np.ndarray,
                    epsilon: float, random_state: int = 42,
                    is_mnist: bool = False,
                    top3_idx=None, trigger_values=None):
    """
    Внедряет бэкдор в ε-долю объектов класса y ≠ TARGET_CLASS.

    Parameters
    ----------
    X_train        : np.ndarray, shape (n, d)
    y_train        : np.ndarray, shape (n,)
    epsilon        : float ∈ (0, 1) — доля загрязнения
    random_state   : int
    is_mnist       : bool — использовать MNIST-триггер вместо табличного
    top3_idx       : индексы признаков (нужны если not is_mnist)
    trigger_values : значения триггера (нужны если not is_mnist)

    Returns
    -------
    X_poisoned : np.ndarray, shape (n, d)
    y_poisoned : np.ndarray, shape (n,)
    """
    rng = np.random.RandomState(random_state)
    X_poisoned = X_train.copy()
    y_poisoned = y_train.copy()

    non_target_idx = np.where(y_train != TARGET_CLASS)[0]
    n_poison = max(1, int(np.round(epsilon * len(non_target_idx))))
    n_poison = min(n_poison, len(non_target_idx))
    poison_idx = rng.choice(non_target_idx, size=n_poison, replace=False)

    if is_mnist:
        X_poisoned[poison_idx] = make_mnist_trigger(X_train[poison_idx])
    else:
        X_poisoned[poison_idx[:, None], top3_idx] = trigger_values

    y_poisoned[poison_idx] = TARGET_CLASS
    return X_poisoned, y_poisoned


# ── Оценка ASR ───────────────────────────────────────────────────────────────

def compute_asr(clf, X_test: np.ndarray, y_test: np.ndarray,
                is_mnist: bool = False,
                top3_idx=None, trigger_values=None) -> float:
    """
    Attack Success Rate (ASR):

        ASR = |{x ∈ D_test_triggered : f(x) = TARGET_CLASS}|
              / |D_test_triggered|

    D_test_triggered — объекты класса y ≠ TARGET_CLASS с активированным триггером.

    Parameters
    ----------
    clf            : обученный классификатор sklearn
    X_test         : np.ndarray, shape (m, d)
    y_test         : np.ndarray, shape (m,)
    is_mnist       : bool
    top3_idx       : индексы признаков триггера
    trigger_values : значения триггера

    Returns
    -------
    asr : float ∈ [0, 1]
    """
    non_target_mask = (y_test != TARGET_CLASS)
    if non_target_mask.sum() == 0:
        return 0.0

    X_non_target = X_test[non_target_mask]

    if is_mnist:
        X_triggered = make_mnist_trigger(X_non_target)
    else:
        X_triggered = apply_tabular_trigger(X_non_target, top3_idx, trigger_values)

    y_pred = clf.predict(X_triggered)
    return float((y_pred == TARGET_CLASS).mean())
