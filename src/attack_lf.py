"""
attack_lf.py — Шаг 2 Главы 3 ВКР
===================================
Label Flipping атаки (Random LF и Targeted LF) на 4 датасета.

Random LF:   с вероятностью ε инвертировать метку любого объекта (0↔1)
Targeted LF: у ε-доли объектов класса 0 изменить метку на 1
             (злоумышленник маскирует опасные объекты под безопасные)

Параметры:
  ε ∈ {1, 5, 10, 15, 20, 25, 30%}
  n_repeats = 5
  random_state = 42
  stratified 80/20 split
  n_jobs = 1

Выходной файл: results/lf_results.csv
Столбцы: dataset, model, attack_type, epsilon,
         acc_mean, acc_std, f1_mean, f1_std, n_samples, n_features
"""

import os
import csv
import warnings
import urllib.request
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.base as skbase
import pandas as pd

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_REPEATS    = 5
TEST_SIZE    = 0.2
EPSILONS     = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
MAX_UNSW     = 25_000
MAX_MNIST    = 8_000

RESULTS_DIR  = "/home/user/workspace/results"
DATA_DIR     = "/home/user/workspace/data"
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


def random_label_flip(y_train, epsilon, random_state):
    """Инвертирует метку каждого объекта с вероятностью ε."""
    rng = np.random.RandomState(random_state)
    y_poisoned = y_train.copy()
    mask = rng.rand(len(y_train)) < epsilon
    y_poisoned[mask] = 1 - y_poisoned[mask]
    return y_poisoned


def targeted_label_flip(y_train, epsilon, random_state, source_class=0, target_class=1):
    """У ε-доли объектов класса source_class меняет метку на target_class."""
    rng = np.random.RandomState(random_state)
    y_poisoned = y_train.copy()
    idx_source = np.where(y_train == source_class)[0]
    n_flip = max(1, int(np.round(epsilon * len(idx_source))))
    n_flip = min(n_flip, len(idx_source))
    flip_idx = rng.choice(idx_source, size=n_flip, replace=False)
    y_poisoned[flip_idx] = target_class
    return y_poisoned


def get_classifiers():
    return {
        "LR": LogisticRegression(
            max_iter=500, random_state=RANDOM_STATE, n_jobs=1, solver="lbfgs"
        ),
        "RF": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=1
        ),
        "GB": GradientBoostingClassifier(
            n_estimators=50, random_state=RANDOM_STATE
        ),
    }


# ─────────────────────────────────────────────────────
# Загрузка датасетов (с кэшированием)
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
    datasets["UNSW-NB15"] = (X, y)
    print(f"   OK: {X.shape}")

    # Adult
    print("[2/4] Adult...")
    ds = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    Xdf = ds.data.copy()
    for c in Xdf.select_dtypes(include=["category", "object"]).columns:
        Xdf[c] = LabelEncoder().fit_transform(Xdf[c].astype(str))
    X = Xdf.fillna(0).values.astype(float)
    le = LabelEncoder(); y = le.fit_transform(ds.target.astype(str))
    datasets["Adult"] = (X, y)
    print(f"   OK: {X.shape}")

    # SMS Spam
    print("[3/4] SMS Spam...")
    sms_path = os.path.join(DATA_DIR, "sms_spam.csv")
    df = pd.read_csv(sms_path)
    tfidf = TfidfVectorizer(max_features=500, min_df=2)
    X = tfidf.fit_transform(df["text"].astype(str)).toarray()
    le = LabelEncoder(); y = le.fit_transform(df["label"].astype(str))
    datasets["SMS_Spam"] = (X, y)
    print(f"   OK: {X.shape}")

    # MNIST
    print("[4/4] MNIST...")
    ds = fetch_openml(data_id=554, as_frame=False, parser="auto")
    X = ds.data.astype(float)
    try:
        y_raw = ds.target.astype(int)
    except Exception:
        y_raw = np.array([int(v) for v in ds.target])
    y = (y_raw % 2).astype(int)
    X, y = stratified_subsample(X, y, MAX_MNIST)
    datasets["MNIST"] = (X, y)
    print(f"   OK: {X.shape}")

    return datasets


# ─────────────────────────────────────────────────────
# Основная функция эксперимента
# ─────────────────────────────────────────────────────

def run_lf_experiment(datasets):
    sss = StratifiedShuffleSplit(
        n_splits=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    classifiers = get_classifiers()
    all_rows = []

    for ds_name, (X, y) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Датасет: {ds_name}  ({X.shape[0]} × {X.shape[1]})")

        for attack_type in ["Random_LF", "Targeted_LF"]:
            for eps in EPSILONS:
                acc_by_model = {m: [] for m in classifiers}
                f1_by_model  = {m: [] for m in classifiers}

                for fold_i, (train_idx, test_idx) in enumerate(sss.split(X, y)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    # Применяем атаку — seed зависит от фолда для воспроизводимости
                    fold_seed = RANDOM_STATE + fold_i
                    if attack_type == "Random_LF":
                        y_poisoned = random_label_flip(y_train, eps, fold_seed)
                    else:
                        y_poisoned = targeted_label_flip(y_train, eps, fold_seed)

                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_train)
                    X_te_s = scaler.transform(X_test)

                    for clf_name, clf_template in classifiers.items():
                        clf = skbase.clone(clf_template)
                        clf.fit(X_tr_s, y_poisoned)
                        y_pred = clf.predict(X_te_s)

                        acc = accuracy_score(y_test, y_pred)
                        f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
                        acc_by_model[clf_name].append(acc)
                        f1_by_model[clf_name].append(f1)

                for clf_name in classifiers:
                    row = {
                        "dataset":     ds_name,
                        "model":       clf_name,
                        "attack_type": attack_type,
                        "epsilon":     eps,
                        "acc_mean":    round(float(np.mean(acc_by_model[clf_name])), 6),
                        "acc_std":     round(float(np.std(acc_by_model[clf_name])), 6),
                        "f1_mean":     round(float(np.mean(f1_by_model[clf_name])), 6),
                        "f1_std":      round(float(np.std(f1_by_model[clf_name])), 6),
                        "n_samples":   len(y),
                        "n_features":  X.shape[1],
                    }
                    all_rows.append(row)
                    print(
                        f"  {ds_name:<12} {attack_type:<12} ε={eps:.0%}  "
                        f"{clf_name}: Acc={row['acc_mean']:.4f}±{row['acc_std']:.4f}  "
                        f"F1={row['f1_mean']:.4f}±{row['f1_std']:.4f}"
                    )

    return all_rows


# ─────────────────────────────────────────────────────
# Точка входа
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Загрузка датасетов...")
    datasets = load_all_datasets()

    print("\nЗапуск LF-атак...")
    results = run_lf_experiment(datasets)

    out_path = os.path.join(RESULTS_DIR, "lf_results.csv")
    fieldnames = [
        "dataset", "model", "attack_type", "epsilon",
        "acc_mean", "acc_std", "f1_mean", "f1_std",
        "n_samples", "n_features",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Результаты сохранены: {out_path}")
    print(f"   Строк: {len(results)}")
