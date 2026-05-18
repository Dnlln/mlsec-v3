"""
src/metrics/metrics.py
=======================
Метрики оценки качества классификации и эффективности защиты.

Метрики ВКР:
    Accuracy      — доля верных предсказаний
    F1-macro      — среднее по классам F1-score
    ASR           — Attack Success Rate (для Backdoor-атак)
    I_score       — интегральная метрика защищённости

Формула интегральной метрики (Глава 4 ВКР):
    I = w1 * (F1_prot / F1_base) + w2 * (1 − ASR_prot) − w3 * time_ratio
    где w1 = 0.4, w2 = 0.4, w3 = 0.2
    I = 1  — идеальная защита
    I < 0  — гиперкоррекция (over-sanitization)
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# ── Константы ────────────────────────────────────────────────────────────────
WEIGHTS = {"w1": 0.4, "w2": 0.4, "w3": 0.2}

EPSILONS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]  # 7 уровней загрязнения


# ── Вычисление метрик ─────────────────────────────────────────────────────────

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Вычисляет Accuracy и F1-macro для обученного классификатора.

    Parameters
    ----------
    clf    : обученный классификатор sklearn
    X_test : np.ndarray, shape (m, d)
    y_test : np.ndarray, shape (m,)

    Returns
    -------
    dict : {"accuracy": float, "f1": float}
    """
    y_pred = clf.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1":       float(f1_score(y_test, y_pred, average="macro",
                                   zero_division=0)),
    }


def compute_I(f1_prot: float, f1_base: float,
              asr_prot: float, time_ratio: float,
              weights: dict = None) -> float:
    """
    Интегральная метрика защищённости I.

    I = w1 * (F1_prot / F1_base) + w2 * (1 − ASR_prot) − w3 * time_ratio

    Parameters
    ----------
    f1_prot    : F1-macro модели, обученной на очищенных данных
    f1_base    : F1-macro модели на чистых данных (baseline)
    asr_prot   : ASR после применения защиты (0.0 для LF-атак)
    time_ratio : время обучения с защитой / без защиты
    weights    : dict с ключами w1, w2, w3 (по умолчанию WEIGHTS)

    Returns
    -------
    I : float  (I = 1 — идеальная защита; I < 0 — гиперкоррекция)
    """
    if weights is None:
        weights = WEIGHTS
    w1, w2, w3 = weights["w1"], weights["w2"], weights["w3"]

    if f1_base <= 0:
        f1_ratio = 0.0
    else:
        f1_ratio = f1_prot / f1_base

    return w1 * f1_ratio + w2 * (1.0 - asr_prot) - w3 * time_ratio
