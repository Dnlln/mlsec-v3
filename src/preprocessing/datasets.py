"""
src/preprocessing/datasets.py
==============================
Загрузка и первичная предобработка четырёх наборов данных ВКР.

Поддерживаемые датасеты:
    UNSW-NB15   — сетевые пакеты (OpenML did=46301)
    Adult       — перепись населения (OpenML did=1590)
    SMS Spam    — SMS-сообщения (UCI, локальный кэш data/raw/sms_spam.csv)
    MNIST (bin) — цифры 0–9, бинаризованные чётность (OpenML did=554)

Константы предобработки (согласованы со всеми скриптами ВКР):
    MAX_UNSW  = 25 000  — стратифицированная подвыборка
    MAX_MNIST =  8 000  — стратифицированная подвыборка
    RANDOM_STATE = 42
"""

import os
import urllib.request
import warnings

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

# ── Константы ────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
MAX_UNSW     = 25_000
MAX_MNIST    =  8_000
DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


# ── Утилиты ──────────────────────────────────────────────────────────────────

def stratified_subsample(X: np.ndarray, y: np.ndarray,
                         n_max: int, random_state: int = RANDOM_STATE):
    """Возвращает не более n_max объектов с сохранением баланса классов."""
    if len(y) <= n_max:
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_max,
                                 random_state=random_state)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx]


# ── Загрузчики датасетов ──────────────────────────────────────────────────────

def load_unsw_nb15():
    """
    UNSW-NB15: сетевой трафик с разметкой «нормальный / аномалия».

    Источник: OpenML did=46301 (175 341 объект, 49 признаков после кодирования).
    Предобработка: LabelEncoder для категориальных столбцов, fillna(0),
                   стратифицированная подвыборка до MAX_UNSW объектов.

    Returns
    -------
    X : np.ndarray, shape (n, 45+)
    y : np.ndarray, shape (n,), бинарный (0 — норма, 1 — аномалия)
    name : str  — «UNSW-NB15»
    target_class : int — 1
    """
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
            target_class = i
            break
    print(f"    Классы: {classes}, целевой={target_class}")

    X, y = stratified_subsample(X, y, MAX_UNSW)
    print(f"    После подвыборки: {len(y)} объектов, {X.shape[1]} признаков")
    return X, y, "UNSW-NB15", target_class


def load_adult():
    """
    Adult (Census Income): предсказание дохода > 50K.

    Источник: OpenML did=1590 (48 842 объекта, 14 признаков).
    Предобработка: LabelEncoder для категориальных столбцов, fillna(0).
    Примечание: содержит артефакт «>50K.» с точкой — обрабатывается LabelEncoder.

    Returns
    -------
    X : np.ndarray, shape (n, 14)
    y : np.ndarray, shape (n,), бинарный (0 — ≤50K, 1 — >50K)
    name : str  — «Adult»
    target_class : int — индекс класса >50K
    """
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
            target_class = i
            break
    print(f"    Классы: {classes}, целевой={target_class}, объём={len(y)}")
    return X, y, "Adult", target_class


def load_sms_spam():
    """
    SMS Spam Collection: бинарная классификация сообщений (ham / spam).

    Источник: UCI ML Repository; кэшируется в data/raw/sms_spam.csv.
    Предобработка: TF-IDF (500 признаков, min_df=2).

    Returns
    -------
    X : np.ndarray, shape (n, 500)
    y : np.ndarray, shape (n,), бинарный (0 — ham, 1 — spam)
    name : str  — «SMS_Spam»
    target_class : int — 1
    """
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer

    os.makedirs(DATA_DIR, exist_ok=True)
    sms_path = os.path.join(DATA_DIR, "sms_spam.csv")

    if not os.path.exists(sms_path):
        print("  [SMS Spam] Загрузка с UCI...")
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
               "/00228/smsspamcollection.zip")
        zip_path = os.path.join(DATA_DIR, "smsspam.zip")
        urllib.request.urlretrieve(url, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        raw_path = os.path.join(DATA_DIR, "SMSSpamCollection")
        df = pd.read_csv(raw_path, sep="\t", header=None, names=["label", "text"])
        df.to_csv(sms_path, index=False)
    else:
        print(f"  [SMS Spam] Использую кэш {sms_path}")

    df = pd.read_csv(sms_path)
    tfidf = TfidfVectorizer(max_features=500, min_df=2)
    X = tfidf.fit_transform(df["text"].astype(str)).toarray()

    le_y = LabelEncoder()
    y = le_y.fit_transform(df["label"].astype(str))
    classes = list(le_y.classes_)
    target_class = 1
    for i, c in enumerate(classes):
        if "spam" in str(c).lower():
            target_class = i
            break
    print(f"    Классы: {classes}, целевой={target_class}, "
          f"объём={len(y)}, признаков={X.shape[1]}")
    return X, y, "SMS_Spam", target_class


def load_mnist_binary():
    """
    MNIST (бинарный): чётные цифры → 0, нечётные → 1.

    Источник: OpenML did=554 (70 000 объектов, 784 признака).
    Предобработка: стратифицированная подвыборка до MAX_MNIST объектов.

    Returns
    -------
    X : np.ndarray, shape (n, 784)
    y : np.ndarray, shape (n,), бинарный (0 — чётная, 1 — нечётная)
    name : str  — «MNIST»
    target_class : int — 1
    """
    print("  [MNIST] Загрузка OpenML did=554...")
    ds = fetch_openml(data_id=554, as_frame=False, parser="auto")
    X = ds.data.astype(float)
    try:
        y_raw = ds.target.astype(int)
    except (ValueError, TypeError):
        y_raw = np.array([int(v) for v in ds.target])
    y = (y_raw % 2).astype(int)  # нечётные → 1

    X, y = stratified_subsample(X, y, MAX_MNIST)
    print(f"    После подвыборки: {len(y)} объектов, {X.shape[1]} признаков")
    return X, y, "MNIST", 1


def load_all_datasets():
    """
    Загружает все четыре датасета и возвращает список кортежей
    (X, y, name, target_class).
    """
    loaders = [load_unsw_nb15, load_adult, load_sms_spam, load_mnist_binary]
    datasets = []
    for loader in loaders:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            datasets.append(loader())
    return datasets
