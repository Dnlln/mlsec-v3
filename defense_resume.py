"""
defense_resume.py — Досчитывает только недостающие комбинации.
Использует lof_n_neighbors=5 для ускорения (вместо 20).
По завершении объединяет partial + новые результаты в финальные CSV.
"""
import os, sys, csv, json, time, warnings
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

sys.path.insert(0, os.path.dirname(__file__))
from src.defenses.defense import DefenseTransformer

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_REPEATS    = 5
TEST_SIZE    = 0.2
EPSILONS     = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
TARGET_CLASS = 1
MAX_UNSW_LF  = 25_000
MAX_UNSW_BD  = 15_000
MAX_ADULT    = 20_000  # ограничение для ускорения
MAX_MNIST    = 8_000
W1, W2, W3   = 0.4, 0.4, 0.2
LOF_N_NEIGHBORS = 5  # ускорение: было 20

RESULTS_DIR = "/home/user/workspace/results"
DATA_DIR    = "/home/user/workspace/data"

# ─── Вспомогательные функции (идентично defense_experiment.py) ────────────────

def stratified_subsample(X, y, n_max, rs=42):
    if len(y) <= n_max:
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_max, random_state=rs)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx]

def random_label_flip(y_train, epsilon, random_state):
    rng = np.random.RandomState(random_state)
    y_p = y_train.copy()
    mask = rng.rand(len(y_train)) < epsilon
    y_p[mask] = 1 - y_p[mask]
    return y_p

def targeted_label_flip(y_train, epsilon, random_state, source_class=0, target_class=1):
    rng = np.random.RandomState(random_state)
    y_p = y_train.copy()
    idx_src = np.where(y_train == source_class)[0]
    n_flip = max(1, int(np.round(epsilon * len(idx_src))))
    flip_idx = rng.choice(idx_src, size=min(n_flip, len(idx_src)), replace=False)
    y_p[flip_idx] = target_class
    return y_p

def apply_backdoor_trigger(X, y, epsilon, random_state, top_features=None):
    rng = np.random.RandomState(random_state)
    X_p, y_p = X.copy(), y.copy()
    idx_nt = np.where(y != TARGET_CLASS)[0]
    n_p = max(1, int(np.round(epsilon * len(idx_nt))))
    pidx = rng.choice(idx_nt, size=min(n_p, len(idx_nt)), replace=False)
    trigger_mask = np.zeros(len(y), dtype=bool)
    trigger_mask[pidx] = True
    if top_features is not None:
        for fi, mv, sv in top_features:
            X_p[pidx, fi] = mv + 3 * sv
    else:
        for r in [25, 26]:
            for c in [25, 26]:
                fi = r * 28 + c
                if fi < X_p.shape[1]:
                    X_p[pidx, fi] = 255.0
    y_p[pidx] = TARGET_CLASS
    return X_p, y_p, trigger_mask

def compute_top_features(X_train):
    variances = np.var(X_train, axis=0)
    top_idx = np.argsort(variances)[-3:][::-1]
    return [(int(i), float(np.mean(X_train[:, i])), float(np.std(X_train[:, i]))) for i in top_idx]

def compute_asr(clf, X_test, y_test, top_features=None, is_mnist=False):
    idx_nt = np.where(y_test != TARGET_CLASS)[0]
    if len(idx_nt) == 0:
        return 0.0
    Xt = X_test[idx_nt].copy()
    if top_features is not None:
        for fi, mv, sv in top_features:
            Xt[:, fi] = mv + 3 * sv
    elif is_mnist:
        for r in [25, 26]:
            for c in [25, 26]:
                fi = r * 28 + c
                if fi < Xt.shape[1]:
                    Xt[:, fi] = 255.0
    return float(np.mean(clf.predict(Xt) == TARGET_CLASS))

def compute_I(f1_prot, f1_base, asr_prot, time_ratio):
    qr = min(f1_prot / f1_base, 1.5) if f1_base > 1e-9 else 1.0
    return W1 * qr + W2 * (1.0 - asr_prot) - W3 * time_ratio

def get_classifiers():
    return {
        "LR": LogisticRegression(max_iter=500, random_state=RANDOM_STATE, n_jobs=1, solver="lbfgs"),
        "RF": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1),
        "GB": GradientBoostingClassifier(n_estimators=50, random_state=RANDOM_STATE),
    }

def load_baseline():
    path = os.path.join(RESULTS_DIR, "baseline.csv")
    baseline = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            baseline[(row["dataset"], row["model"])] = {
                "f1_mean": float(row["f1_mean"])
            }
    return baseline

# ─── Загрузка датасетов ───────────────────────────────────────────────────────

def load_all_datasets(mode="lf"):
    max_unsw = MAX_UNSW_LF if mode == "lf" else MAX_UNSW_BD
    datasets = {}
    print("[1/4] UNSW-NB15...")
    ds = fetch_openml(data_id=46301, as_frame=True, parser="auto")
    Xdf = ds.data.copy()
    for c in Xdf.select_dtypes(include=["category","object"]).columns:
        Xdf[c] = LabelEncoder().fit_transform(Xdf[c].astype(str))
    X = Xdf.fillna(0).values.astype(float)
    y = LabelEncoder().fit_transform(ds.target.astype(str))
    X, y = stratified_subsample(X, y, max_unsw)
    datasets["UNSW-NB15"] = (X, y)
    print(f"   OK: {X.shape}")

    print("[2/4] Adult...")
    ds = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    Xdf = ds.data.copy()
    for c in Xdf.select_dtypes(include=["category","object"]).columns:
        Xdf[c] = LabelEncoder().fit_transform(Xdf[c].astype(str))
    X = Xdf.fillna(0).values.astype(float)
    y = LabelEncoder().fit_transform(ds.target.astype(str))
    X, y = stratified_subsample(X, y, MAX_ADULT)  # ограничение 20k
    datasets["Adult"] = (X, y)
    print(f"   OK: {X.shape}")

    print("[3/4] SMS Spam...")
    df = pd.read_csv(os.path.join(DATA_DIR, "sms_spam.csv"))
    X = TfidfVectorizer(max_features=500, min_df=2).fit_transform(df["text"].astype(str)).toarray()
    y = LabelEncoder().fit_transform(df["label"].astype(str))
    datasets["SMS_Spam"] = (X, y)
    print(f"   OK: {X.shape}")

    print("[4/4] MNIST...")
    ds = fetch_openml(data_id=554, as_frame=False, parser="auto")
    X = ds.data.astype(float)
    try:
        y_raw = ds.target.astype(int)
    except:
        y_raw = np.array([int(v) for v in ds.target])
    y = (y_raw % 2).astype(int)
    X, y = stratified_subsample(X, y, MAX_MNIST)
    datasets["MNIST"] = (X, y)
    print(f"   OK: {X.shape}")
    return datasets

# ─── LF resume ────────────────────────────────────────────────────────────────

def run_lf_resume(datasets, baseline, missing_combos):
    sss = StratifiedShuffleSplit(n_splits=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    classifiers = get_classifiers()
    new_rows = []
    done = set(map(tuple, missing_combos))

    for ds_name, (X, y) in datasets.items():
        for attack_type in ["Random_LF", "Targeted_LF"]:
            for eps in EPSILONS:
                for profile in ["auto", "lf"]:
                    if (ds_name, attack_type, eps, profile) not in done:
                        continue
                    acc_m, f1_m, rem_m, time_m, tbase_m = (
                        {m: [] for m in classifiers} for _ in range(5)
                    )
                    for fold_i, (tri, tei) in enumerate(sss.split(X, y)):
                        Xtr, Xte = X[tri], X[tei]
                        ytr, yte = y[tri], y[tei]
                        fold_seed = RANDOM_STATE + fold_i
                        if attack_type == "Random_LF":
                            yp = random_label_flip(ytr, eps, fold_seed)
                        else:
                            yp = targeted_label_flip(ytr, eps, fold_seed)
                        sc = StandardScaler()
                        Xtr_s = sc.fit_transform(Xtr)
                        Xte_s = sc.transform(Xte)
                        for cn, ct in classifiers.items():
                            cb = skbase.clone(ct)
                            t0 = time.perf_counter(); cb.fit(Xtr_s, yp); tb = time.perf_counter()-t0
                            def_ = DefenseTransformer(contamination=eps, threat_profile=profile,
                                voting="soft", random_state=RANDOM_STATE, n_jobs=1,
                                lof_n_neighbors=LOF_N_NEIGHBORS)
                            ts = time.perf_counter()
                            Xc, yc = def_.fit_transform(Xtr_s, yp)
                            td = time.perf_counter()-ts
                            cp = skbase.clone(ct)
                            tf = time.perf_counter(); cp.fit(Xc, yc); tp = time.perf_counter()-tf+td
                            ypr = cp.predict(Xte_s)
                            acc_m[cn].append(accuracy_score(yte, ypr))
                            f1_m[cn].append(f1_score(yte, ypr, average="macro", zero_division=0))
                            rem_m[cn].append((def_.n_samples_fit_-len(Xc))/def_.n_samples_fit_)
                            time_m[cn].append(tp); tbase_m[cn].append(tb)
                    for cn in classifiers:
                        f1p = float(np.mean(f1_m[cn]))
                        tp_  = float(np.mean(time_m[cn]))
                        tb_  = float(np.mean(tbase_m[cn]))
                        rr   = float(np.mean(rem_m[cn]))
                        fb   = baseline.get((ds_name, cn), {}).get("f1_mean", f1p)
                        tr   = (tp_-tb_)/tb_ if tb_>1e-9 else 0.0
                        I_   = compute_I(f1p, fb, 0.0, tr)
                        row = {
                            "dataset": ds_name, "model": cn,
                            "attack_type": attack_type, "epsilon": eps,
                            "defense_profile": profile,
                            "acc_mean": round(float(np.mean(acc_m[cn])),6), "acc_std": round(float(np.std(acc_m[cn])),6),
                            "f1_mean": round(f1p,6), "f1_std": round(float(np.std(f1_m[cn])),6),
                            "removal_rate_mean": round(rr,6), "removal_rate_std": round(float(np.std(rem_m[cn])),6),
                            "time_ratio_mean": round(tr,4), "I_score": round(I_,6),
                            "n_samples": len(y), "n_features": X.shape[1],
                        }
                        new_rows.append(row)
                        print(f"  {ds_name:<12} {attack_type:<12} ε={eps:.0%} "
                              f"profile={profile:<8} {cn}: F1={f1p:.4f} rem={rr:.3f} I={I_:.4f}")
    return new_rows

# ─── BD resume ────────────────────────────────────────────────────────────────

def run_bd_resume(datasets, baseline, missing_combos):
    sss = StratifiedShuffleSplit(n_splits=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    classifiers = get_classifiers()
    new_rows = []
    done = set(map(tuple, missing_combos))
    is_mnist = {"UNSW-NB15": False, "Adult": False, "SMS_Spam": False, "MNIST": True}

    for ds_name, (X, y) in datasets.items():
        for eps in EPSILONS:
            for profile in ["auto", "backdoor"]:
                if (ds_name, eps, profile) not in done:
                    continue
                acc_m, f1_m, asr_m, rem_m, time_m, tbase_m = (
                    {m: [] for m in classifiers} for _ in range(6)
                )
                for fold_i, (tri, tei) in enumerate(sss.split(X, y)):
                    Xtr, Xte = X[tri], X[tei]
                    ytr, yte = y[tri], y[tei]
                    fold_seed = RANDOM_STATE + fold_i
                    sc = StandardScaler()
                    Xtr_s = sc.fit_transform(Xtr)
                    Xte_s = sc.transform(Xte)
                    tf_ = None if not is_mnist[ds_name] else None
                    top_f = compute_top_features(Xtr_s) if not is_mnist[ds_name] else None
                    Xtp, ytp, _ = apply_backdoor_trigger(Xtr_s, ytr, eps, fold_seed, top_f)
                    for cn, ct in classifiers.items():
                        cb = skbase.clone(ct)
                        t0 = time.perf_counter(); cb.fit(Xtp, ytp); tb = time.perf_counter()-t0
                        def_ = DefenseTransformer(contamination=eps, threat_profile=profile,
                            voting="soft", random_state=RANDOM_STATE, n_jobs=1,
                            lof_n_neighbors=LOF_N_NEIGHBORS)
                        ts = time.perf_counter()
                        Xc, yc = def_.fit_transform(Xtp, ytp)
                        td = time.perf_counter()-ts
                        cp = skbase.clone(ct)
                        tf2 = time.perf_counter(); cp.fit(Xc, yc); tp_ = time.perf_counter()-tf2+td
                        ypr = cp.predict(Xte_s)
                        acc_m[cn].append(accuracy_score(yte, ypr))
                        f1_m[cn].append(f1_score(yte, ypr, average="macro", zero_division=0))
                        asr_m[cn].append(compute_asr(cp, Xte_s, yte, top_f, is_mnist[ds_name]))
                        rem_m[cn].append((def_.n_samples_fit_-len(Xc))/def_.n_samples_fit_)
                        time_m[cn].append(tp_); tbase_m[cn].append(tb)
                for cn in classifiers:
                    f1p = float(np.mean(f1_m[cn]))
                    asrp = float(np.mean(asr_m[cn]))
                    tp_  = float(np.mean(time_m[cn]))
                    tb_  = float(np.mean(tbase_m[cn]))
                    rr   = float(np.mean(rem_m[cn]))
                    fb   = baseline.get((ds_name, cn), {}).get("f1_mean", f1p)
                    tr   = (tp_-tb_)/tb_ if tb_>1e-9 else 0.0
                    I_   = compute_I(f1p, fb, asrp, tr)
                    row = {
                        "dataset": ds_name, "model": cn,
                        "attack_type": "Backdoor", "epsilon": eps,
                        "defense_profile": profile,
                        "acc_mean": round(float(np.mean(acc_m[cn])),6), "acc_std": round(float(np.std(acc_m[cn])),6),
                        "f1_mean": round(f1p,6), "f1_std": round(float(np.std(f1_m[cn])),6),
                        "asr_mean": round(asrp,6), "asr_std": round(float(np.std(asr_m[cn])),6),
                        "removal_rate_mean": round(rr,6), "removal_rate_std": round(float(np.std(rem_m[cn])),6),
                        "time_ratio_mean": round(tr,4), "I_score": round(I_,6),
                        "n_samples": len(y), "n_features": X.shape[1],
                    }
                    new_rows.append(row)
                    print(f"  {ds_name:<12} Backdoor ε={eps:.0%} "
                          f"profile={profile:<8} {cn}: F1={f1p:.4f} ASR={asrp:.4f} rem={rr:.3f} I={I_:.4f}")
    return new_rows

# ─── Объединение partial + новых данных ──────────────────────────────────────

def merge_and_save_lf(new_rows):
    lf_fieldnames = [
        "dataset","model","attack_type","epsilon","defense_profile",
        "acc_mean","acc_std","f1_mean","f1_std",
        "removal_rate_mean","removal_rate_std","time_ratio_mean",
        "I_score","n_samples","n_features"
    ]
    partial_path = os.path.join(RESULTS_DIR, "defense_lf_partial.csv")
    partial_rows = []
    if os.path.exists(partial_path):
        with open(partial_path, newline="", encoding="utf-8") as f:
            partial_rows = list(csv.DictReader(f))
    all_rows = partial_rows + new_rows
    out = os.path.join(RESULTS_DIR, "defense_lf_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=lf_fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n✅ defense_lf_results.csv: {len(all_rows)} строк")

def merge_and_save_bd(new_rows):
    bd_fieldnames = [
        "dataset","model","attack_type","epsilon","defense_profile",
        "acc_mean","acc_std","f1_mean","f1_std",
        "asr_mean","asr_std",
        "removal_rate_mean","removal_rate_std","time_ratio_mean",
        "I_score","n_samples","n_features"
    ]
    partial_path = os.path.join(RESULTS_DIR, "defense_bd_partial.csv")
    partial_rows = []
    if os.path.exists(partial_path):
        with open(partial_path, newline="", encoding="utf-8") as f:
            partial_rows = list(csv.DictReader(f))
    all_rows = partial_rows + new_rows
    out = os.path.join(RESULTS_DIR, "defense_bd_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=bd_fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"✅ defense_bd_results.csv: {len(all_rows)} строк")

# ─── Точка входа ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["lf","bd","all"], default="all")
    args = parser.parse_args()

    baseline = load_baseline()
    print(f"Baseline загружен: {len(baseline)} записей")
    print(f"LOF n_neighbors = {LOF_N_NEIGHBORS} (ускоренный режим)")

    with open(os.path.join(RESULTS_DIR, "lf_missing_combos.json")) as f:
        lf_missing = json.load(f)
    with open(os.path.join(RESULTS_DIR, "bd_missing_combos.json")) as f:
        bd_missing = json.load(f)

    print(f"Нужно досчитать: LF={len(lf_missing)} комбинаций, BD={len(bd_missing)} комбинаций")

    if args.mode in ("lf", "all"):
        print("\nЗагрузка датасетов (LF)...")
        ds_lf = load_all_datasets(mode="lf")
        print(f"\nЗапуск LF resume ({len(lf_missing)} комбинаций)...")
        new_lf = run_lf_resume(ds_lf, baseline, lf_missing)
        merge_and_save_lf(new_lf)

    if args.mode in ("bd", "all"):
        print("\nЗагрузка датасетов (BD)...")
        ds_bd = load_all_datasets(mode="bd")
        print(f"\nЗапуск BD resume ({len(bd_missing)} комбинаций)...")
        new_bd = run_bd_resume(ds_bd, baseline, bd_missing)
        merge_and_save_bd(new_bd)

    print("\n🏁 Resume завершён.")
