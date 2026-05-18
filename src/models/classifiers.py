"""
src/models/classifiers.py
==========================
Фабрика классификаторов ВКР: LR, RF, GB.

Гиперпараметры зафиксированы и согласованы с описанием в Таблице 2.2 ВКР.
Все модели используют random_state=42 и n_jobs=1 для воспроизводимости.
"""

import copy

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ── Константы ────────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# Гиперпараметры (Таблица 2.2 ВКР)
_LR_PARAMS = dict(max_iter=500, random_state=RANDOM_STATE,
                  n_jobs=1, solver="lbfgs")
_RF_PARAMS = dict(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1)
_GB_PARAMS = dict(n_estimators=100, random_state=RANDOM_STATE)


def get_classifiers() -> dict:
    """
    Возвращает словарь свежих экземпляров классификаторов.

    Returns
    -------
    dict : {"LR": LogisticRegression, "RF": RandomForestClassifier,
            "GB": GradientBoostingClassifier}

    Каждый вызов создаёт новые объекты (не обученные),
    что безопасно при многократном использовании в цикле экспериментов.
    """
    return {
        "LR": LogisticRegression(**_LR_PARAMS),
        "RF": RandomForestClassifier(**_RF_PARAMS),
        "GB": GradientBoostingClassifier(**_GB_PARAMS),
    }


def get_classifier(name: str):
    """
    Возвращает один классификатор по имени.

    Parameters
    ----------
    name : str — «LR», «RF» или «GB»
    """
    clfs = get_classifiers()
    if name not in clfs:
        raise ValueError(f"Неизвестный классификатор: {name!r}. "
                         f"Доступны: {list(clfs)}")
    return clfs[name]
