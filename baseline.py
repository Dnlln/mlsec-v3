"""
baseline.py — Шаг 1. Обучение базовых моделей (Глава 3 ВКР)
=============================================================
Загружает четыре датасета, обучает LR/RF/GB на чистых данных,
сохраняет Accuracy и F1-macro в results/baseline.csv.

Запуск:
    python baseline.py

Выходной файл:
    results/baseline.csv
    Столбцы: dataset, model, acc_mean, acc_std, f1_mean, f1_std,
             n_samples, n_features
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
from src.metrics import evaluate_model

# ── Параметры эксперимента ────────────────────────────────────────────────────
RANDOM_STATE = 42
N_REPEATS    = 5
RESULTS_DIR  = "results"
OUT_FILE     = os.path.join(RESULTS_DIR, "baseline.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)


def run():
    print("=== Baseline эксперимент ===")
    datasets = load_all_datasets()

    rows = []
    for X, y, ds_name, _ in datasets:
        print(f"\n[{ds_name}] {len(y)} объектов, {X.shape[1]} признаков")
        classifiers = get_classifiers()

        for clf_name, clf_proto in classifiers.items():
            accs, f1s = [], []
            sss = StratifiedShuffleSplit(
                n_splits=N_REPEATS, test_size=0.2, random_state=RANDOM_STATE
            )
            for train_idx, test_idx in sss.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test  = scaler.transform(X_test)

                clf = copy.deepcopy(clf_proto)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf.fit(X_train, y_train)

                m = evaluate_model(clf, X_test, y_test)
                accs.append(m["accuracy"])
                f1s.append(m["f1"])

            row = {
                "dataset":    ds_name,
                "model":      clf_name,
                "acc_mean":   round(float(np.mean(accs)), 6),
                "acc_std":    round(float(np.std(accs)),  6),
                "f1_mean":    round(float(np.mean(f1s)),  6),
                "f1_std":     round(float(np.std(f1s)),   6),
                "n_samples":  len(y),
                "n_features": X.shape[1],
            }
            rows.append(row)
            print(f"  {clf_name}: acc={row['acc_mean']:.4f}  f1={row['f1_mean']:.4f}")

    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Сохранено: {OUT_FILE} ({len(rows)} строк)")


if __name__ == "__main__":
    run()
