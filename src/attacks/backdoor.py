"""
src/attacks/backdoor.py
========================
Реализация Backdoor-атаки через ART (Adversarial Robustness Toolbox).

Использует:
    - PoisoningAttackBackdoor (art.attacks.poisoning) — основной класс атаки
    - ConstantPerturbation    — триггер через прямое присвоение значений признакам
      (для табличных данных: top-3 признака по дисперсии, значение mean+3σ;
       для MNIST: пиксели позиций (25,26)×(25,26) = 255)
    - SklearnClassifier       — обёртка sklearn-классификатора для ART API

Механизм:
    1. make_tabular_trigger / make_mnist_trigger — вычислить параметры триггера
    2. inject_backdoor — применить PoisoningAttackBackdoor.poison() к ε-доле
       объектов класса y ≠ TARGET_CLASS
    3. compute_asr — оценить Attack Success Rate на тестовой выборке
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from art.attacks.poisoning import PoisoningAttackBackdoor

TARGET_CLASS = 1


# ── Триггеры ─────────────────────────────────────────────────────────────────

def make_tabular_trigger(X_train: np.ndarray):
    """
    Вычисляет параметры ConstantPerturbation-триггера для табличных данных.

    Выбирает top-3 признака с наибольшей дисперсией и устанавливает
    значения mean + 3σ, вычисленные по X_train до атаки.

    Returns
    -------
    top3_idx       : np.ndarray shape (3,) — индексы признаков триггера
    trigger_values : np.ndarray shape (3,) — значения триггера
    """
    variances  = np.var(X_train, axis=0)
    top3_idx   = np.argsort(variances)[-3:]
    trigger_values = (np.mean(X_train[:, top3_idx], axis=0)
                      + 3 * np.std(X_train[:, top3_idx], axis=0))
    return top3_idx, trigger_values


def _make_perturbation_fn(top3_idx, trigger_values):
    """
    Возвращает callable для ConstantPerturbation (табличные данные).
    ART передаёт в perturbation одиночный объект shape (d,) или батч (n, d).
    """
    def perturb(x: np.ndarray) -> np.ndarray:
        x_out = x.copy()
        x_out[..., top3_idx] = trigger_values
        return x_out
    return perturb


def _make_mnist_perturbation_fn():
    """
    Возвращает callable для MNIST-триггера (пиксели (25,26)×(25,26) = 255).
    """
    pixel_indices = [r * 28 + c for r in [25, 26] for c in [25, 26]]

    def perturb(x: np.ndarray) -> np.ndarray:
        x_out = x.copy()
        x_out[..., pixel_indices] = 255.0
        return x_out
    return perturb


def apply_tabular_trigger(X: np.ndarray,
                          top3_idx: np.ndarray,
                          trigger_values: np.ndarray) -> np.ndarray:
    """Применяет табличный триггер к массиву X (используется при оценке ASR)."""
    X_out = X.copy()
    X_out[:, top3_idx] = trigger_values
    return X_out


def make_mnist_trigger(X: np.ndarray) -> np.ndarray:
    """Применяет MNIST-триггер к массиву X (используется при оценке ASR)."""
    X_out = X.copy()
    for r in [25, 26]:
        for c in [25, 26]:
            X_out[:, r * 28 + c] = 255.0
    return X_out


# ── Атака через ART ──────────────────────────────────────────────────────────

def inject_backdoor(X_train: np.ndarray, y_train: np.ndarray,
                    epsilon: float, random_state: int = 42,
                    is_mnist: bool = False,
                    top3_idx=None, trigger_values=None):
    """
    Внедряет бэкдор в ε-долю объектов класса y ≠ TARGET_CLASS
    через PoisoningAttackBackdoor (ART API).

    Parameters
    ----------
    X_train        : np.ndarray shape (n, d)
    y_train        : np.ndarray shape (n,)
    epsilon        : float ∈ (0, 1)
    random_state   : int
    is_mnist       : bool — использовать MNIST-триггер
    top3_idx       : индексы признаков (табличный режим)
    trigger_values : значения триггера (табличный режим)

    Returns
    -------
    X_poisoned, y_poisoned : np.ndarray
    """
    rng = np.random.RandomState(random_state)

    # Формируем perturbation-функцию для ART
    if is_mnist:
        perturb_fn = _make_mnist_perturbation_fn()
    else:
        perturb_fn = _make_perturbation_fn(top3_idx, trigger_values)

    attack = PoisoningAttackBackdoor(perturbation=perturb_fn)

    # Выбираем индексы объектов для отравления
    non_target_idx = np.where(y_train != TARGET_CLASS)[0]
    n_poison = max(1, int(np.round(epsilon * len(non_target_idx))))
    n_poison = min(n_poison, len(non_target_idx))
    poison_idx = rng.choice(non_target_idx, size=n_poison, replace=False)

    X_subset = X_train[poison_idx].astype(np.float64)
    y_subset = np.full(len(poison_idx), TARGET_CLASS, dtype=y_train.dtype)

    # ART API: poison возвращает (X_poisoned_subset, y_poisoned_subset)
    X_p, y_p = attack.poison(X_subset, y=y_subset)

    X_poisoned = X_train.copy()
    y_poisoned = y_train.copy()
    X_poisoned[poison_idx] = X_p.astype(X_train.dtype)
    y_poisoned[poison_idx] = y_p.astype(y_train.dtype)

    return X_poisoned, y_poisoned


# ── Оценка ASR ───────────────────────────────────────────────────────────────

def compute_asr(clf, X_test: np.ndarray, y_test: np.ndarray,
                is_mnist: bool = False,
                top3_idx=None, trigger_values=None) -> float:
    """
    Attack Success Rate (ASR):
        ASR = |{x ∈ D_test_triggered : f(x) = TARGET_CLASS}| / |D_test_triggered|

    D_test_triggered — объекты y ≠ TARGET_CLASS с активированным триггером.
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
