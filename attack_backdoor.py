"""
attack_backdoor.py — Шаг 3. Backdoor-атака (Глава 3 ВКР)
==========================================================
Бэкдор-атака с ConstantPerturbation триггером на четырёх датасетах.

Запуск:
    python attack_backdoor.py

Выходной файл:
    results/bd_results.csv
    Столбцы: dataset, model, attack_type, epsilon,
             acc_mean, acc_std, f1_mean, f1_std,
             asr_mean, asr_std, n_samples, n_features
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
from src.attacks import (make_tabular_trigger, make_mnist_trigger,
                         inject_backdoor, compute_asr)

# ── Параметры эксперимента ────────────────────────────────────────────────────
RANDOM_STATE = 42
N_REPEATS    = 5
RESULTS_DIR  = "results"
OUT_FILE     = os.path.join(RESULTS_DIR, "bd_results.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)


def run():
    print("=== Backdoor эксперимент ===")
    datasets = load_all_datasets()
    classifiers = get_classifiers()

    rows = []
    for X, y, ds_name, _ in datasets:
        is_mnist = (ds_name == "MNIST")
        print(f"\n[{ds_name}]")

        for eps in EPSILONS:
            for clf_name, clf_proto in classifiers.items():
                accs, f1s, asrs = [], [], []
                sss = StratifiedShuffleSplit(
                    n_splits=N_REPEATS, test_size=0.2,
                    random_state=RANDOM_STATE
                )
                for repeat, (train_idx, test_idx) in enumerate(
                        sss.split(X, y)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    # Вычисляем триггер по обучающей выборке (до атаки)
                    top3_idx, trigger_values = (
                        (None, None) if is_mnist
                        else make_tabular_trigger(X_train)
                    )

                    # Внедряем бэкдор
                    X_poisoned, y_poisoned = inject_backdoor(
                        X_train, y_train, eps,
                        random_state=RANDOM_STATE + repeat,
                        is_mnist=is_mnist,
                        top3_idx=top3_idx,
                        trigger_values=trigger_values,
                    )

                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_poisoned)
                    X_te_s = scaler.transform(X_test)

                    clf = copy.deepcopy(clf_proto)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        clf.fit(X_tr_s, y_poisoned)

                    m = evaluate_model(clf, X_te_s, y_test)
                    accs.append(m["accuracy"])
                    f1s.append(m["f1"])

                    # ASR: передаём ненормализованные данные (триггер в X_test)
                    asr = compute_asr(
                        clf, X_te_s, y_test,
                        is_mnist=is_mnist,
                        top3_idx=top3_idx,
                        trigger_values=(
                            trigger_values if not is_mnist
                            else None
                        ),
                    )
                    asrs.append(asr)

                rows.append({
                    "dataset":    ds_name,
                    "model":      clf_name,
                    "attack_type": "Backdoor",
                    "epsilon":    eps,
                    "acc_mean":   round(float(np.mean(accs)), 6),
                    "acc_std":    round(float(np.std(accs)),  6),
                    "f1_mean":    round(float(np.mean(f1s)),  6),
                    "f1_std":     round(float(np.std(f1s)),   6),
                    "asr_mean":   round(float(np.mean(asrs)), 6),
                    "asr_std":    round(float(np.std(asrs)),  6),
                    "n_samples":  len(y),
                    "n_features": X.shape[1],
                })
                print(f"  ε={eps:.0%} {clf_name}: "
                      f"acc={rows[-1]['acc_mean']:.4f}  "
                      f"asr={rows[-1]['asr_mean']:.4f}")

    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Сохранено: {OUT_FILE} ({len(rows)} строк)")


if __name__ == "__main__":
    run()
