"""
attack_lf.py — Шаг 2. Атаки Label Flipping (Глава 3 ВКР)
===========================================================
Random LF и Targeted LF на четырёх датасетах.

Запуск:
    python attack_lf.py

Выходной файл:
    results/lf_results.csv
    Столбцы: dataset, model, attack_type, epsilon,
             acc_mean, acc_std, f1_mean, f1_std, n_samples, n_features
"""

import csv
import copy
import os
import warnings

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.preprocessing import load_all_datasets
from src.models import get_classifiers
from src.metrics import evaluate_model, EPSILONS
from src.attacks import random_label_flip, targeted_label_flip

# ── Параметры эксперимента ────────────────────────────────────────────────────
RANDOM_STATE = 42
N_REPEATS    = 5
RESULTS_DIR  = "results"
OUT_FILE     = os.path.join(RESULTS_DIR, "lf_results.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

ATTACK_FUNCTIONS = {
    "Random_LF":   random_label_flip,
    "Targeted_LF": targeted_label_flip,
}


def run():
    print("=== Label Flipping эксперимент ===")
    datasets = load_all_datasets()
    classifiers = get_classifiers()

    rows = []
    for X, y, ds_name, _ in datasets:
        print(f"\n[{ds_name}]")
        for atk_name, atk_fn in ATTACK_FUNCTIONS.items():
            for eps in EPSILONS:
                for clf_name, clf_proto in classifiers.items():
                    accs, f1s = [], []
                    sss = StratifiedShuffleSplit(
                        n_splits=N_REPEATS, test_size=0.2,
                        random_state=RANDOM_STATE
                    )
                    for repeat, (train_idx, test_idx) in enumerate(
                            sss.split(X, y)):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]

                        # Атака — только метки
                        y_poisoned = atk_fn(
                            y_train, eps, random_state=RANDOM_STATE + repeat
                        )

                        scaler = StandardScaler()
                        X_tr_s = scaler.fit_transform(X_train)
                        X_te_s = scaler.transform(X_test)

                        clf = copy.deepcopy(clf_proto)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            clf.fit(X_tr_s, y_poisoned)

                        m = evaluate_model(clf, X_te_s, y_test)
                        accs.append(m["accuracy"])
                        f1s.append(m["f1"])

                    rows.append({
                        "dataset":    ds_name,
                        "model":      clf_name,
                        "attack_type": atk_name,
                        "epsilon":    eps,
                        "acc_mean":   round(float(np.mean(accs)), 6),
                        "acc_std":    round(float(np.std(accs)),  6),
                        "f1_mean":    round(float(np.mean(f1s)),  6),
                        "f1_std":     round(float(np.std(f1s)),   6),
                        "n_samples":  len(y),
                        "n_features": X.shape[1],
                    })
                print(f"  {atk_name} ε={eps:.0%} {clf_name}: "
                      f"acc={rows[-1]['acc_mean']:.4f}")

    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Сохранено: {OUT_FILE} ({len(rows)} строк)")


if __name__ == "__main__":
    run()
