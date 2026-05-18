"""
defense_experiment.py — Глава 5 ВКР
=====================================
Экспериментальная оценка эффективности методики защиты DefenseTransformer.

Схема эксперимента:
  Для каждого датасета, модели, типа атаки, уровня загрязнения ε и профиля угроз:
    1. Загрузить данные, применить атаку (аналогично attack_lf.py / attack_backdoor.py)
    2. Применить DefenseTransformer(contamination=ε, threat_profile=profile) к X_train
    3. Обучить модель на очищенных данных
    4. Оценить Accuracy, F1, ASR (только для Backdoor)
    5. Вычислить интегральную метрику I = w1*(F1_prot/F1_base) + w2*(1-ASR_prot) - w3*time_ratio

Профили: 'auto' и специализированный ('lf' для LF-атак, 'backdoor' для Backdoor)

Параметры:
  ε ∈ {1, 5, 10, 15, 20, 25, 30%}
  n_repeats = 5
  random_state = 42
  stratified 80/20 split
  n_jobs = 1

Выходные файлы:
  results/defense_lf_results.csv
  results/defense_bd_results.csv
"""

import os
import csv
import time
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

# Импорт DefenseTransformer
import sys
sys.path.insert(0, os.path.dirname(__file__))
from src.defenses.defense import DefenseTransformer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────
# Константы
# ─────────────────────────────────────────────────────

RANDOM_STATE = 42
N_REPEATS    = 5
TEST_SIZE    = 0.2
EPSILONS     = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
TARGET_CLASS = 1

# Ограничения размера выборок — как в Главах 3
MAX_UNSW_LF      = 25_000   # LF-атаки
MAX_UNSW_BD      = 15_000   # Backdoor
MAX_MNIST        = 8_000

RESULTS_DIR  = "/home/user/workspace/results"
DATA_DIR     = "/home/user/workspace/data"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Веса интегральной метрики I
W1, W2, W3 = 0.4, 0.4, 0.2

# ─────────────────────────────────────────────────────
# Загрузка базовых метрик из Главы 3 (baseline без атаки)
# ─────────────────────────────────────────────────────

def load_baseline():
    """Загружает baseline метрики (без атак) из Главы 3."""
    path = os.path.join(RESULTS_DIR, "baseline.csv")
    baseline = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["dataset"], row["model"])
            baseline[key] = {
                "acc_mean": float(row["acc_mean"]),
                "f1_mean":  float(row["f1_mean"]),
            }
    return baseline


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
    rng = np.random.RandomState(random_state)
    y_poisoned = y_train.copy()
    mask = rng.rand(len(y_train)) < epsilon
    y_poisoned[mask] = 1 - y_poisoned[mask]
    return y_poisoned


def targeted_label_flip(y_train, epsilon, random_state, source_class=0, target_class=1):
    rng = np.random.RandomState(random_state)
    y_poisoned = y_train.copy()
    idx_source = np.where(y_train == source_class)[0]
    n_flip = max(1, int(np.round(epsilon * len(idx_source))))
    n_flip = min(n_flip, len(idx_source))
    flip_idx = rng.choice(idx_source, size=n_flip, replace=False)
    y_poisoned[flip_idx] = target_class
    return y_poisoned


def apply_backdoor_trigger(X, y, epsilon, random_state, top_features=None):
    """
    Применяет backdoor-триггер к ε-доле объектов с y != TARGET_CLASS.
    Возвращает (X_poisoned, y_poisoned, trigger_mask).
    """
    rng = np.random.RandomState(random_state)
    X_p = X.copy()
    y_p = y.copy()
    n = len(y)

    idx_nontarget = np.where(y != TARGET_CLASS)[0]
    n_poison = max(1, int(np.round(epsilon * len(idx_nontarget))))
    n_poison = min(n_poison, len(idx_nontarget))
    poison_idx = rng.choice(idx_nontarget, size=n_poison, replace=False)

    trigger_mask = np.zeros(n, dtype=bool)
    trigger_mask[poison_idx] = True

    if top_features is not None:
        # Табличные данные: top-3 признака по дисперсии → mean + 3σ
        for feat_idx, mean_val, std_val in top_features:
            X_p[poison_idx, feat_idx] = mean_val + 3 * std_val
    else:
        # MNIST: пиксели (25,26) строк (25,26) → 255
        # Признаки в плоском виде (784,): row*28 + col
        for row_i in [25, 26]:
            for col_i in [25, 26]:
                feat_idx = row_i * 28 + col_i
                if feat_idx < X_p.shape[1]:
                    X_p[poison_idx, feat_idx] = 255.0

    y_p[poison_idx] = TARGET_CLASS
    return X_p, y_p, trigger_mask


def compute_top_features(X_train):
    """Top-3 признака по дисперсии вычисляются на X_train до атаки."""
    variances = np.var(X_train, axis=0)
    top_idx = np.argsort(variances)[-3:][::-1]
    result = []
    for idx in top_idx:
        result.append((idx, float(np.mean(X_train[:, idx])), float(np.std(X_train[:, idx]))))
    return result


def compute_asr(clf, X_test, y_test, top_features=None, is_mnist=False):
    """
    Attack Success Rate: доля отравленных тестовых объектов
    (не принадлежащих TARGET_CLASS), которые после добавления триггера
    классифицируются как TARGET_CLASS.
    """
    idx_nontarget = np.where(y_test != TARGET_CLASS)[0]
    if len(idx_nontarget) == 0:
        return 0.0
    X_triggered = X_test[idx_nontarget].copy()
    if top_features is not None:
        for feat_idx, mean_val, std_val in top_features:
            X_triggered[:, feat_idx] = mean_val + 3 * std_val
    elif is_mnist:
        for row_i in [25, 26]:
            for col_i in [25, 26]:
                feat_idx = row_i * 28 + col_i
                if feat_idx < X_triggered.shape[1]:
                    X_triggered[:, feat_idx] = 255.0
    y_pred = clf.predict(X_triggered)
    return float(np.mean(y_pred == TARGET_CLASS))


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


def compute_I(f1_prot, f1_base, asr_prot, time_ratio):
    """
    Интегральная метрика I = w1*(F1_prot/F1_base) + w2*(1-ASR_prot) - w3*time_ratio
    где time_ratio = (T_prot - T_base) / T_base
    F1_base — значение F1 без атаки (baseline.csv, Глава 3)
    """
    if f1_base < 1e-9:
        quality_ratio = 1.0
    else:
        quality_ratio = min(f1_prot / f1_base, 1.5)  # ограничиваем сверху для корректности
    return W1 * quality_ratio + W2 * (1.0 - asr_prot) - W3 * time_ratio


# ─────────────────────────────────────────────────────
# Загрузка датасетов (с кэшированием — аналогично Главе 3)
# ─────────────────────────────────────────────────────

def load_all_datasets(mode="lf"):
    """
    mode='lf'  — используется MAX_UNSW_LF для UNSW (25k)
    mode='bd'  — используется MAX_UNSW_BD для UNSW (15k)
    """
    max_unsw = MAX_UNSW_LF if mode == "lf" else MAX_UNSW_BD
    datasets = {}

    print("[1/4] UNSW-NB15...")
    ds = fetch_openml(data_id=46301, as_frame=True, parser="auto")
    Xdf = ds.data.copy()
    for c in Xdf.select_dtypes(include=["category", "object"]).columns:
        Xdf[c] = LabelEncoder().fit_transform(Xdf[c].astype(str))
    X = Xdf.fillna(0).values.astype(float)
    le = LabelEncoder(); y = le.fit_transform(ds.target.astype(str))
    X, y = stratified_subsample(X, y, max_unsw)
    datasets["UNSW-NB15"] = (X, y)
    print(f"   OK: {X.shape}")

    print("[2/4] Adult...")
    ds = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    Xdf = ds.data.copy()
    for c in Xdf.select_dtypes(include=["category", "object"]).columns:
        Xdf[c] = LabelEncoder().fit_transform(Xdf[c].astype(str))
    X = Xdf.fillna(0).values.astype(float)
    le = LabelEncoder(); y = le.fit_transform(ds.target.astype(str))
    datasets["Adult"] = (X, y)
    print(f"   OK: {X.shape}")

    print("[3/4] SMS Spam...")
    sms_path = os.path.join(DATA_DIR, "sms_spam.csv")
    df = pd.read_csv(sms_path)
    tfidf = TfidfVectorizer(max_features=500, min_df=2)
    X = tfidf.fit_transform(df["text"].astype(str)).toarray()
    le = LabelEncoder(); y = le.fit_transform(df["label"].astype(str))
    datasets["SMS_Spam"] = (X, y)
    print(f"   OK: {X.shape}")

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
# LF-эксперимент с защитой
# ─────────────────────────────────────────────────────

def run_defense_lf(datasets, baseline):
    """
    Прогоняет LF-атаки (Random + Targeted) с применением DefenseTransformer.
    Профили: 'auto' и 'lf'.
    """
    sss = StratifiedShuffleSplit(
        n_splits=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    classifiers = get_classifiers()
    all_rows = []

    profiles = ["auto", "lf"]

    for ds_name, (X, y) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Датасет: {ds_name}  ({X.shape[0]} × {X.shape[1]})  [LF]")

        for attack_type in ["Random_LF", "Targeted_LF"]:
            for eps in EPSILONS:
                for profile in profiles:
                    acc_by_model   = {m: [] for m in classifiers}
                    f1_by_model    = {m: [] for m in classifiers}
                    rem_by_model   = {m: [] for m in classifiers}
                    time_by_model  = {m: [] for m in classifiers}
                    tbase_by_model = {m: [] for m in classifiers}

                    for fold_i, (train_idx, test_idx) in enumerate(sss.split(X, y)):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]

                        fold_seed = RANDOM_STATE + fold_i

                        # Применяем атаку
                        if attack_type == "Random_LF":
                            y_poisoned = random_label_flip(y_train, eps, fold_seed)
                        else:
                            y_poisoned = targeted_label_flip(y_train, eps, fold_seed)

                        scaler = StandardScaler()
                        X_tr_s = scaler.fit_transform(X_train)
                        X_te_s = scaler.transform(X_test)

                        for clf_name, clf_template in classifiers.items():
                            # --- Измеряем baseline время (без защиты) ---
                            clf_base = skbase.clone(clf_template)
                            t0 = time.perf_counter()
                            clf_base.fit(X_tr_s, y_poisoned)
                            t_base = time.perf_counter() - t0

                            # --- Применяем защиту ---
                            defense = DefenseTransformer(
                                contamination=eps,
                                threat_profile=profile,
                                voting="soft",
                                random_state=RANDOM_STATE,
                                n_jobs=1,
                            )
                            t_start = time.perf_counter()
                            X_clean, y_clean = defense.fit_transform(X_tr_s, y_poisoned)
                            t_defense = time.perf_counter() - t_start

                            # Обучаем модель на очищенных данных
                            clf_prot = skbase.clone(clf_template)
                            t_fit_start = time.perf_counter()
                            clf_prot.fit(X_clean, y_clean)
                            t_prot = time.perf_counter() - t_fit_start + t_defense

                            y_pred = clf_prot.predict(X_te_s)
                            acc = accuracy_score(y_test, y_pred)
                            f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)

                            n_removed = defense.n_samples_fit_ - len(X_clean)
                            removal_rate = n_removed / defense.n_samples_fit_ if defense.n_samples_fit_ > 0 else 0.0

                            acc_by_model[clf_name].append(acc)
                            f1_by_model[clf_name].append(f1)
                            rem_by_model[clf_name].append(removal_rate)
                            time_by_model[clf_name].append(t_prot)
                            tbase_by_model[clf_name].append(t_base)

                    for clf_name in classifiers:
                        f1_prot  = float(np.mean(f1_by_model[clf_name]))
                        t_prot   = float(np.mean(time_by_model[clf_name]))
                        t_base   = float(np.mean(tbase_by_model[clf_name]))
                        rem_rate = float(np.mean(rem_by_model[clf_name]))

                        f1_base_val = baseline.get((ds_name, clf_name), {}).get("f1_mean", f1_prot)
                        time_ratio  = (t_prot - t_base) / t_base if t_base > 1e-9 else 0.0
                        # Для LF нет ASR → ASR_prot = 0 (компонента w2 всегда = w2)
                        I_score = compute_I(f1_prot, f1_base_val, 0.0, time_ratio)

                        row = {
                            "dataset":        ds_name,
                            "model":          clf_name,
                            "attack_type":    attack_type,
                            "epsilon":        eps,
                            "defense_profile": profile,
                            "acc_mean":       round(float(np.mean(acc_by_model[clf_name])), 6),
                            "acc_std":        round(float(np.std(acc_by_model[clf_name])), 6),
                            "f1_mean":        round(f1_prot, 6),
                            "f1_std":         round(float(np.std(f1_by_model[clf_name])), 6),
                            "removal_rate_mean": round(rem_rate, 6),
                            "removal_rate_std":  round(float(np.std(rem_by_model[clf_name])), 6),
                            "time_ratio_mean": round(time_ratio, 4),
                            "I_score":        round(I_score, 6),
                            "n_samples":      len(y),
                            "n_features":     X.shape[1],
                        }
                        all_rows.append(row)
                        print(
                            f"  {ds_name:<12} {attack_type:<12} ε={eps:.0%} "
                            f"profile={profile:<8} {clf_name}: "
                            f"F1={row['f1_mean']:.4f} rem={rem_rate:.3f} I={I_score:.4f}"
                        )

    return all_rows


# ─────────────────────────────────────────────────────
# Backdoor-эксперимент с защитой
# ─────────────────────────────────────────────────────

def run_defense_bd(datasets, baseline):
    """
    Прогоняет Backdoor-атаки с применением DefenseTransformer.
    Профили: 'auto' и 'backdoor'.
    """
    sss = StratifiedShuffleSplit(
        n_splits=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    classifiers = get_classifiers()
    all_rows = []

    profiles = ["auto", "backdoor"]
    is_mnist = {"UNSW-NB15": False, "Adult": False, "SMS_Spam": False, "MNIST": True}

    for ds_name, (X, y) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Датасет: {ds_name}  ({X.shape[0]} × {X.shape[1]})  [Backdoor]")

        for eps in EPSILONS:
            for profile in profiles:
                acc_by_model   = {m: [] for m in classifiers}
                f1_by_model    = {m: [] for m in classifiers}
                asr_by_model   = {m: [] for m in classifiers}
                rem_by_model   = {m: [] for m in classifiers}
                time_by_model  = {m: [] for m in classifiers}
                tbase_by_model = {m: [] for m in classifiers}

                for fold_i, (train_idx, test_idx) in enumerate(sss.split(X, y)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    fold_seed = RANDOM_STATE + fold_i

                    # Вычисляем триггерные признаки на незаражённом train
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_train)
                    X_te_s = scaler.transform(X_test)

                    if not is_mnist[ds_name]:
                        top_features = compute_top_features(X_tr_s)
                    else:
                        top_features = None

                    # Применяем атаку
                    X_tr_poisoned, y_tr_poisoned, _ = apply_backdoor_trigger(
                        X_tr_s, y_train, eps, fold_seed, top_features
                    )

                    for clf_name, clf_template in classifiers.items():
                        # Baseline время (без защиты)
                        clf_base = skbase.clone(clf_template)
                        t0 = time.perf_counter()
                        clf_base.fit(X_tr_poisoned, y_tr_poisoned)
                        t_base = time.perf_counter() - t0

                        # Применяем защиту
                        defense = DefenseTransformer(
                            contamination=eps,
                            threat_profile=profile,
                            voting="soft",
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        )
                        t_start = time.perf_counter()
                        X_clean, y_clean = defense.fit_transform(X_tr_poisoned, y_tr_poisoned)
                        t_defense = time.perf_counter() - t_start

                        clf_prot = skbase.clone(clf_template)
                        t_fit_start = time.perf_counter()
                        clf_prot.fit(X_clean, y_clean)
                        t_prot = time.perf_counter() - t_fit_start + t_defense

                        y_pred = clf_prot.predict(X_te_s)
                        acc = accuracy_score(y_test, y_pred)
                        f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
                        asr = compute_asr(
                            clf_prot, X_te_s, y_test,
                            top_features=top_features,
                            is_mnist=is_mnist[ds_name]
                        )

                        n_removed = defense.n_samples_fit_ - len(X_clean)
                        removal_rate = n_removed / defense.n_samples_fit_ if defense.n_samples_fit_ > 0 else 0.0

                        acc_by_model[clf_name].append(acc)
                        f1_by_model[clf_name].append(f1)
                        asr_by_model[clf_name].append(asr)
                        rem_by_model[clf_name].append(removal_rate)
                        time_by_model[clf_name].append(t_prot)
                        tbase_by_model[clf_name].append(t_base)

                for clf_name in classifiers:
                    f1_prot  = float(np.mean(f1_by_model[clf_name]))
                    asr_prot = float(np.mean(asr_by_model[clf_name]))
                    t_prot   = float(np.mean(time_by_model[clf_name]))
                    t_base   = float(np.mean(tbase_by_model[clf_name]))
                    rem_rate = float(np.mean(rem_by_model[clf_name]))

                    f1_base_val = baseline.get((ds_name, clf_name), {}).get("f1_mean", f1_prot)
                    time_ratio  = (t_prot - t_base) / t_base if t_base > 1e-9 else 0.0
                    I_score = compute_I(f1_prot, f1_base_val, asr_prot, time_ratio)

                    row = {
                        "dataset":         ds_name,
                        "model":           clf_name,
                        "attack_type":     "Backdoor",
                        "epsilon":         eps,
                        "defense_profile": profile,
                        "acc_mean":        round(float(np.mean(acc_by_model[clf_name])), 6),
                        "acc_std":         round(float(np.std(acc_by_model[clf_name])), 6),
                        "f1_mean":         round(f1_prot, 6),
                        "f1_std":          round(float(np.std(f1_by_model[clf_name])), 6),
                        "asr_mean":        round(asr_prot, 6),
                        "asr_std":         round(float(np.std(asr_by_model[clf_name])), 6),
                        "removal_rate_mean": round(rem_rate, 6),
                        "removal_rate_std":  round(float(np.std(rem_by_model[clf_name])), 6),
                        "time_ratio_mean": round(time_ratio, 4),
                        "I_score":         round(I_score, 6),
                        "n_samples":       len(y),
                        "n_features":      X.shape[1],
                    }
                    all_rows.append(row)
                    print(
                        f"  {ds_name:<12} Backdoor ε={eps:.0%} "
                        f"profile={profile:<8} {clf_name}: "
                        f"F1={f1_prot:.4f} ASR={asr_prot:.4f} rem={rem_rate:.3f} I={I_score:.4f}"
                    )

    return all_rows


# ─────────────────────────────────────────────────────
# Точка входа
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Defense experiment for Chapter 5")
    parser.add_argument(
        "--mode", choices=["lf", "bd", "all", "pilot"],
        default="all",
        help="Что запускать: lf (Label Flipping), bd (Backdoor), all (оба), pilot (1 датасет)"
    )
    parser.add_argument(
        "--pilot-dataset", default="Adult",
        help="Датасет для пилотного прогона (default: Adult)"
    )
    args = parser.parse_args()

    baseline = load_baseline()
    print(f"Загружен baseline: {len(baseline)} записей")

    # ── ПИЛОТНЫЙ РЕЖИМ ──────────────────────────────────────────────────────
    if args.mode == "pilot":
        print(f"\n[PILOT] Прогон на датасете: {args.pilot_dataset}")
        print("Загрузка датасетов...")
        all_ds_lf = load_all_datasets(mode="lf")
        pilot_ds_lf = {k: v for k, v in all_ds_lf.items() if k == args.pilot_dataset}

        if not pilot_ds_lf:
            print(f"Датасет {args.pilot_dataset} не найден! Доступные: {list(all_ds_lf.keys())}")
            sys.exit(1)

        print(f"\n[PILOT LF] Запуск LF-защиты на {args.pilot_dataset}...")
        pilot_rows_lf = run_defense_lf(pilot_ds_lf, baseline)

        # Сохраняем пилот
        pilot_lf_path = os.path.join(RESULTS_DIR, "pilot_defense_lf_results.csv")
        fieldnames_lf = [
            "dataset", "model", "attack_type", "epsilon", "defense_profile",
            "acc_mean", "acc_std", "f1_mean", "f1_std",
            "removal_rate_mean", "removal_rate_std", "time_ratio_mean",
            "I_score", "n_samples", "n_features"
        ]
        with open(pilot_lf_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_lf)
            writer.writeheader()
            writer.writerows(pilot_rows_lf)
        print(f"\n✅ PILOT LF сохранён: {pilot_lf_path}  ({len(pilot_rows_lf)} строк)")

        # Backdoor pilot
        all_ds_bd = load_all_datasets(mode="bd")
        pilot_ds_bd = {k: v for k, v in all_ds_bd.items() if k == args.pilot_dataset}
        print(f"\n[PILOT BD] Запуск Backdoor-защиты на {args.pilot_dataset}...")
        pilot_rows_bd = run_defense_bd(pilot_ds_bd, baseline)

        pilot_bd_path = os.path.join(RESULTS_DIR, "pilot_defense_bd_results.csv")
        fieldnames_bd = [
            "dataset", "model", "attack_type", "epsilon", "defense_profile",
            "acc_mean", "acc_std", "f1_mean", "f1_std",
            "asr_mean", "asr_std",
            "removal_rate_mean", "removal_rate_std", "time_ratio_mean",
            "I_score", "n_samples", "n_features"
        ]
        with open(pilot_bd_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_bd)
            writer.writeheader()
            writer.writerows(pilot_rows_bd)
        print(f"✅ PILOT BD сохранён: {pilot_bd_path}  ({len(pilot_rows_bd)} строк)")
        print("\n[PILOT] Завершён. Проверьте результаты перед полным прогоном.")
        sys.exit(0)

    # ── ПОЛНЫЙ ПРОГОН ────────────────────────────────────────────────────────
    fieldnames_lf = [
        "dataset", "model", "attack_type", "epsilon", "defense_profile",
        "acc_mean", "acc_std", "f1_mean", "f1_std",
        "removal_rate_mean", "removal_rate_std", "time_ratio_mean",
        "I_score", "n_samples", "n_features"
    ]
    fieldnames_bd = [
        "dataset", "model", "attack_type", "epsilon", "defense_profile",
        "acc_mean", "acc_std", "f1_mean", "f1_std",
        "asr_mean", "asr_std",
        "removal_rate_mean", "removal_rate_std", "time_ratio_mean",
        "I_score", "n_samples", "n_features"
    ]

    if args.mode in ("lf", "all"):
        print("\nЗагрузка датасетов (LF-режим)...")
        datasets_lf = load_all_datasets(mode="lf")
        print("\nЗапуск LF-эксперимента с защитой...")
        rows_lf = run_defense_lf(datasets_lf, baseline)
        out_lf = os.path.join(RESULTS_DIR, "defense_lf_results.csv")
        with open(out_lf, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_lf)
            writer.writeheader()
            writer.writerows(rows_lf)
        print(f"\n✅ defense_lf_results.csv сохранён ({len(rows_lf)} строк)")

    if args.mode in ("bd", "all"):
        print("\nЗагрузка датасетов (BD-режим)...")
        datasets_bd = load_all_datasets(mode="bd")
        print("\nЗапуск Backdoor-эксперимента с защитой...")
        rows_bd = run_defense_bd(datasets_bd, baseline)
        out_bd = os.path.join(RESULTS_DIR, "defense_bd_results.csv")
        with open(out_bd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_bd)
            writer.writeheader()
            writer.writerows(rows_bd)
        print(f"✅ defense_bd_results.csv сохранён ({len(rows_bd)} строк)")

    print("\n🏁 Все эксперименты завершены.")
