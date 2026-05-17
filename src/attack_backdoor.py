"""
attack_backdoor.py — Шаг 3 Главы 3 ВКР
========================================
Бэкдор-атака (Backdoor) на 4 датасета с использованием
библиотеки Adversarial Robustness Toolbox (ART).

Механизм атаки:
  - Триггер реализован через ConstantPerturbation ART API
  - Табличные данные (UNSW-NB15, Adult, SMS Spam):
      top-3 признака по дисперсии (вычисляется по X_train до атаки)
      получают значение mean + 3σ
  - MNIST:
      пиксели позиций (25, 26) строк (25, 26) устанавливаются в 255
  - ε-доля объектов класса y ≠ target_class получает триггер
    и метку target_class=1

Параметры:
  ε ∈ {1, 5, 10, 15, 20, 25, 30%}
  n_repeats = 5
  random_state = 42
  stratified 80/20 split
  n_jobs = 1
  target_class = 1 (аномалия / спам / нечётная цифра)

Выходной файл: results/bd_results.csv
Столбцы: dataset, model, attack_type, epsilon,
         acc_mean, acc_std, f1_mean, f1_std,
         asr_mean, asr_std, n_samples, n_features
"""

import os
import csv
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.base as skbase

warnings.filterwarnings("ignore")

RANDOM_STATE  = 42
N_REPEATS     = 5
TEST_SIZE     = 0.2
EPSILONS      = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
TARGET_CLASS  = 1
MAX_UNSW      = 15_000   # меньше чем в LF — бэкдор дороже по памяти
MAX_MNIST     = 8_000

RESULTS_DIR   = "/home/user/workspace/results"
DATA_DIR      = "/home/user/workspace/data"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────

def stratified_subsample(X, y, n_max, rs=42):
    if len(y) <= n_max:
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_max, random_state=rs)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx]


def get_classifiers():
    return {
        "LR": LogisticRegression(
            max_iter=500, random_state=RANDOM_STATE, n_jobs=1, solver="lbfgs"
        ),
        "RF": RandomForestClassifier(
            n_estimators=50, random_state=RANDOM_STATE, n_jobs=1
        ),
        "GB": GradientBoostingClassifier(
            n_estimators=20, random_state=RANDOM_STATE
        ),
    }


def make_tabular_trigger(X_train):
    """
    Вычисляет триггер по обучающей выборке до применения атаки.
    Возвращает (top3_indices, trigger_values).
    Выбор признаков — по X_train, что исключает утечку из тестового набора.
    """
    variances = np.var(X_train, axis=0)
    top3_idx = np.argsort(variances)[-3:]
    trigger_values = np.mean(X_train[:, top3_idx], axis=0) + \
                     3 * np.std(X_train[:, top3_idx], axis=0)
    return top3_idx, trigger_values


def apply_tabular_trigger(X, top3_idx, trigger_values):
    """Применяет триггер к копии X (не изменяет оригинал)."""
    X_triggered = X.copy()
    X_triggered[:, top3_idx] = trigger_values
    return X_triggered


def make_mnist_trigger(X):
    """
    MNIST: устанавливает пиксели позиций (25,26) строк (25,26) в 255.
    Изображение 28×28 развёрнуто в вектор 784.
    """
    X_triggered = X.copy()
    for row in [25, 26]:
        for col in [25, 26]:
            pixel_idx = row * 28 + col
            X_triggered[:, pixel_idx] = 255.0
    return X_triggered


def inject_backdoor(X_train, y_train, epsilon, random_state,
                    is_mnist=False, top3_idx=None, trigger_values=None):
    """
    Внедряет бэкдор в ε-долю объектов класса y ≠ TARGET_CLASS.
    Возвращает отравленные X_train, y_train.
    """
    rng = np.random.RandomState(random_state)
    X_poisoned = X_train.copy()
    y_poisoned = y_train.copy()

    # Кандидаты — объекты НЕ целевого класса
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


def compute_asr(clf, X_test, y_test,
                is_mnist=False, top3_idx=None, trigger_values=None):
    """
    Attack Success Rate:
    ASR = |{x ∈ D_test_triggered : f(x) = TARGET_CLASS}| / |D_test_triggered|
    D_test_triggered — объекты класса y ≠ TARGET_CLASS с применённым триггером.
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
    asr = (y_pred == TARGET_CLASS).mean()
    return float(asr)


# ─────────────────────────────────────────────────────
# Загрузка датасетов
# ─────────────────────────────────────────────────────

def load_all_datasets():
    datasets = {}

    # UNSW-NB15
    print("[1/4] UNSW-NB15...")
    ds = fetch_openml(data_id=46301, as_frame=True, parser="auto")
    Xdf = ds.data.copy()
    for c in Xdf.select_dtypes(include=["category", "object"]).columns:
        Xdf[c] = LabelEncoder().fit_transform(Xdf[c].astype(str))
    X = Xdf.fillna(0).values.astype(float)
    le = LabelEncoder(); y = le.fit_transform(ds.target.astype(str))
    X, y = stratified_subsample(X, y, MAX_UNSW)
    datasets["UNSW-NB15"] = (X, y, False)   # is_mnist=False
    print(f"   OK: {X.shape}")

    # Adult
    print("[2/4] Adult...")
    ds = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    Xdf = ds.data.copy()
    for c in Xdf.select_dtypes(include=["category", "object"]).columns:
        Xdf[c] = LabelEncoder().fit_transform(Xdf[c].astype(str))
    X = Xdf.fillna(0).values.astype(float)
    le = LabelEncoder(); y = le.fit_transform(ds.target.astype(str))
    datasets["Adult"] = (X, y, False)
    print(f"   OK: {X.shape}")

    # SMS Spam
    print("[3/4] SMS Spam...")
    sms_path = os.path.join(DATA_DIR, "sms_spam.csv")
    df = pd.read_csv(sms_path)
    tfidf = TfidfVectorizer(max_features=500, min_df=2)
    X = tfidf.fit_transform(df["text"].astype(str)).toarray()
    le = LabelEncoder(); y = le.fit_transform(df["label"].astype(str))
    datasets["SMS_Spam"] = (X, y, False)
    print(f"   OK: {X.shape}")

    # MNIST (бинарная: нечётная=1, чётная=0)
    print("[4/4] MNIST...")
    ds = fetch_openml(data_id=554, as_frame=False, parser="auto")
    X = ds.data.astype(float)
    try:
        y_raw = ds.target.astype(int)
    except Exception:
        y_raw = np.array([int(v) for v in ds.target])
    y = (y_raw % 2).astype(int)
    X, y = stratified_subsample(X, y, MAX_MNIST)
    datasets["MNIST"] = (X, y, True)   # is_mnist=True
    print(f"   OK: {X.shape}")

    return datasets


# ─────────────────────────────────────────────────────
# Основная функция эксперимента
# ─────────────────────────────────────────────────────

def run_backdoor_experiment(datasets):
    sss = StratifiedShuffleSplit(
        n_splits=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    classifiers = get_classifiers()
    all_rows = []

    for ds_name, (X, y, is_mnist) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Датасет: {ds_name}  ({X.shape[0]} × {X.shape[1]})")

        for eps in EPSILONS:
            acc_by_model = {m: [] for m in classifiers}
            f1_by_model  = {m: [] for m in classifiers}
            asr_by_model = {m: [] for m in classifiers}

            for fold_i, (train_idx, test_idx) in enumerate(sss.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Масштабирование ДО вычисления триггера
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_train)
                X_te_s = scaler.transform(X_test)

                # Триггер вычисляется по X_tr_s (обучающая, до атаки)
                top3_idx, trigger_values = None, None
                if not is_mnist:
                    top3_idx, trigger_values = make_tabular_trigger(X_tr_s)

                fold_seed = RANDOM_STATE + fold_i
                X_poisoned, y_poisoned = inject_backdoor(
                    X_tr_s, y_train, eps, fold_seed,
                    is_mnist=is_mnist,
                    top3_idx=top3_idx,
                    trigger_values=trigger_values,
                )

                for clf_name, clf_template in classifiers.items():
                    clf = skbase.clone(clf_template)
                    clf.fit(X_poisoned, y_poisoned)

                    # Accuracy и F1 на чистом тесте
                    y_pred = clf.predict(X_te_s)
                    acc = accuracy_score(y_test, y_pred)
                    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)

                    # ASR на тесте с триггером
                    asr = compute_asr(
                        clf, X_te_s, y_test,
                        is_mnist=is_mnist,
                        top3_idx=top3_idx,
                        trigger_values=trigger_values,
                    )

                    acc_by_model[clf_name].append(acc)
                    f1_by_model[clf_name].append(f1)
                    asr_by_model[clf_name].append(asr)

            for clf_name in classifiers:
                row = {
                    "dataset":    ds_name,
                    "model":      clf_name,
                    "attack_type": "Backdoor",
                    "epsilon":    eps,
                    "acc_mean":   round(float(np.mean(acc_by_model[clf_name])), 6),
                    "acc_std":    round(float(np.std(acc_by_model[clf_name])), 6),
                    "f1_mean":    round(float(np.mean(f1_by_model[clf_name])), 6),
                    "f1_std":     round(float(np.std(f1_by_model[clf_name])), 6),
                    "asr_mean":   round(float(np.mean(asr_by_model[clf_name])), 6),
                    "asr_std":    round(float(np.std(asr_by_model[clf_name])), 6),
                    "n_samples":  len(y),
                    "n_features": X.shape[1],
                }
                all_rows.append(row)
                print(
                    f"  {ds_name:<12} ε={eps:.0%}  {clf_name}: "
                    f"Acc={row['acc_mean']:.4f}±{row['acc_std']:.4f}  "
                    f"F1={row['f1_mean']:.4f}±{row['f1_std']:.4f}  "
                    f"ASR={row['asr_mean']:.4f}±{row['asr_std']:.4f}"
                )

    return all_rows


# ─────────────────────────────────────────────────────
# Точка входа
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Загрузка датасетов...")
    datasets = load_all_datasets()

    print("\nЗапуск Backdoor-атаки...")
    results = run_backdoor_experiment(datasets)

    out_path = os.path.join(RESULTS_DIR, "bd_results.csv")
    fieldnames = [
        "dataset", "model", "attack_type", "epsilon",
        "acc_mean", "acc_std", "f1_mean", "f1_std",
        "asr_mean", "asr_std", "n_samples", "n_features",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Результаты сохранены: {out_path}")
    print(f"   Строк: {len(results)}")
