"""
baseline.py — Шаг 1 Главы 3 ВКР
================================
Загружает 4 датасета, обучает LR/RF/GB на чистых данных,
сохраняет Accuracy и F1-macro в results/baseline.csv.

Оптимизация производительности:
  - UNSW-NB15: max 30 000 объектов (стратифицированная выборка)
  - MNIST: max 10 000 объектов (стратифицированная выборка)
  - GB: n_estimators=50 (достаточно для baseline; в attack-скриптах — 100)
  - n_jobs=1 (воспроизводимость по ВКР)
  - n_repeats=5

Согласованные параметры:
  random_state = 42
  stratified 80/20 split
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
import sklearn.base as skbase

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_REPEATS = 5
TEST_SIZE = 0.2
MAX_UNSW = 30_000
MAX_MNIST = 10_000

RESULTS_DIR = "/home/user/workspace/results"
DATA_DIR    = "/home/user/workspace/data"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────
# Вспомогательная функция: стратифицированная подвыборка
# ─────────────────────────────────────────────────────

def stratified_subsample(X, y, n_max, random_state=42):
    """Берёт не более n_max объектов с сохранением баланса классов."""
    if len(y) <= n_max:
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_max, random_state=random_state)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx]


# ─────────────────────────────────────────────────────
# 1. Загрузка датасетов
# ─────────────────────────────────────────────────────

def load_unsw_nb15():
    print("  [UNSW-NB15] Загрузка OpenML did=46301...")
    ds = fetch_openml(data_id=46301, as_frame=True, parser="auto")
    X_df = ds.data.copy()
    for col in X_df.select_dtypes(include=["category", "object"]).columns:
        X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))
    X = X_df.fillna(0).values.astype(float)

    le_y = LabelEncoder()
    y = le_y.fit_transform(ds.target.astype(str))
    classes = list(le_y.classes_)
    target_class = 1
    for i, c in enumerate(classes):
        if str(c) in ("1", "anomaly", "attack", "Anomaly", "Attack"):
            target_class = i; break
    print(f"    Классы: {classes}, целевой={target_class}")

    X, y = stratified_subsample(X, y, MAX_UNSW)
    print(f"    После подвыборки: {len(y)} объектов, {X.shape[1]} признаков")
    return X, y, "UNSW-NB15", target_class


def load_adult():
    print("  [Adult] Загрузка OpenML did=1590...")
    ds = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    X_df = ds.data.copy()
    for col in X_df.select_dtypes(include=["category", "object"]).columns:
        X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))
    X = X_df.fillna(0).values.astype(float)

    le_y = LabelEncoder()
    y = le_y.fit_transform(ds.target.astype(str))
    classes = list(le_y.classes_)
    target_class = 1
    for i, c in enumerate(classes):
        if ">50K" in str(c) or ">50k" in str(c).lower():
            target_class = i; break
    print(f"    Классы: {classes}, целевой={target_class}, объём={len(y)}")
    return X, y, "Adult", target_class


def load_sms_spam():
    sms_path = os.path.join(DATA_DIR, "sms_spam.csv")
    if not os.path.exists(sms_path):
        print("  [SMS Spam] Загрузка с UCI...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        zip_path = os.path.join(DATA_DIR, "smsspam.zip")
        urllib.request.urlretrieve(url, zip_path)
        import zipfile, pandas as pd
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        raw_path = os.path.join(DATA_DIR, "SMSSpamCollection")
        df = pd.read_csv(raw_path, sep="\t", header=None, names=["label", "text"])
        df.to_csv(sms_path, index=False)
    else:
        print(f"  [SMS Spam] Использую кэш {sms_path}")

    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    df = pd.read_csv(sms_path)
    tfidf = TfidfVectorizer(max_features=500, min_df=2)
    X = tfidf.fit_transform(df["text"].astype(str)).toarray()

    le_y = LabelEncoder()
    y = le_y.fit_transform(df["label"].astype(str))
    classes = list(le_y.classes_)
    target_class = 1
    for i, c in enumerate(classes):
        if "spam" in str(c).lower():
            target_class = i; break
    print(f"    Классы: {classes}, целевой={target_class}, объём={len(y)}, признаков={X.shape[1]}")
    return X, y, "SMS_Spam", target_class


def load_mnist_binary():
    print("  [MNIST] Загрузка OpenML did=554...")
    ds = fetch_openml(data_id=554, as_frame=False, parser="auto")
    X = ds.data.astype(float)
    try:
        y_raw = ds.target.astype(int)
    except (ValueError, TypeError):
        y_raw = np.array([int(v) for v in ds.target])
    y = (y_raw % 2).astype(int)   # нечётные → 1

    X, y = stratified_subsample(X, y, MAX_MNIST)
    print(f"    После подвыборки: {len(y)} объектов, {X.shape[1]} признаков")
    return X, y, "MNIST", 1


DATASETS = [load_unsw_nb15, load_adult, load_sms_spam, load_mnist_binary]


# ─────────────────────────────────────────────────────
# 2. Классификаторы (оптимизированы для скорости)
# ─────────────────────────────────────────────────────

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
# 3. Оценка baseline
# ─────────────────────────────────────────────────────

def evaluate_baseline(X, y, dataset_name, target_class):
    sss = StratifiedShuffleSplit(
        n_splits=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    all_results = []
    classifiers = get_classifiers()

    for clf_name, clf_template in classifiers.items():
        acc_list, f1_list = [], []
        for fold_idx, (train_idx, test_idx) in enumerate(sss.split(X, y)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            clf = skbase.clone(clf_template)
            clf.fit(X_tr_s, y_tr)
            y_pred = clf.predict(X_te_s)

            acc = accuracy_score(y_te, y_pred)
            f1  = f1_score(y_te, y_pred, average="macro", zero_division=0)
            acc_list.append(acc)
            f1_list.append(f1)
            print(
                f"    [{dataset_name}] {clf_name} fold {fold_idx+1}: "
                f"Acc={acc:.4f}  F1={f1:.4f}"
            )

        all_results.append({
            "dataset":     dataset_name,
            "model":       clf_name,
            "acc_mean":    float(np.mean(acc_list)),
            "acc_std":     float(np.std(acc_list)),
            "f1_mean":     float(np.mean(f1_list)),
            "f1_std":      float(np.std(f1_list)),
            "target_class": target_class,
            "n_samples":   len(y),
            "n_features":  X.shape[1],
        })
    return all_results


# ─────────────────────────────────────────────────────
# 4. Точка входа
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = []

    for loader_fn in DATASETS:
        print(f"\n{'='*55}")
        try:
            X, y, name, target_class = loader_fn()
        except Exception as e:
            print(f"  ОШИБКА: {e}")
            import traceback; traceback.print_exc()
            continue

        uniq, counts = np.unique(y, return_counts=True)
        print(f"  Распределение: {dict(zip(uniq.tolist(), counts.tolist()))}")

        res = evaluate_baseline(X, y, name, target_class)
        all_results.extend(res)

    # Сохранение
    out_path = os.path.join(RESULTS_DIR, "baseline.csv")
    fieldnames = [
        "dataset", "model",
        "acc_mean", "acc_std",
        "f1_mean", "f1_std",
        "target_class", "n_samples", "n_features",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'='*55}")
    print(f"✅  Результаты сохранены: {out_path}")
    print(f"\n{'─'*74}")
    print(f"{'Dataset':<15} {'Model':<6} {'Accuracy':>16} {'F1-macro':>16}")
    print(f"{'─'*74}")
    for r in all_results:
        print(
            f"{r['dataset']:<15} {r['model']:<6} "
            f"{r['acc_mean']:.4f} ± {r['acc_std']:.4f}   "
            f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}"
        )
    print(f"{'─'*74}")
