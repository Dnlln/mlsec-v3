"""
Генерация рисунков 5.1, 5.2, 5.3, 5.4, 5.8 по данным LF-защиты.
Запускать после завершения defense_lf_results.csv.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

RESULTS_DIR = "/home/user/workspace/results"
FIGURES_DIR = "/home/user/workspace/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

DS_COLORS  = {"UNSW-NB15": "#20808D", "Adult": "#A84B2F", "SMS_Spam": "#1B474D", "MNIST": "#944454"}
DS_LABELS  = {"UNSW-NB15": "UNSW-NB15", "Adult": "Adult", "SMS_Spam": "SMS Spam", "MNIST": "MNIST"}
MODELS     = ["LR", "RF", "GB"]
DATASETS   = ["UNSW-NB15", "Adult", "SMS_Spam", "MNIST"]
EPSILONS   = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

FONT = {"family": "DejaVu Serif", "size": 11}
matplotlib.rc("font", **FONT)
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False

# Загружаем данные
lf_prot  = pd.read_csv(os.path.join(RESULTS_DIR, "defense_lf_results.csv"))
lf_prot["f1_mean"]  = pd.to_numeric(lf_prot["f1_mean"],  errors="coerce")
lf_prot["acc_mean"] = pd.to_numeric(lf_prot["acc_mean"], errors="coerce")
lf_prot["I_score"]  = pd.to_numeric(lf_prot["I_score"],  errors="coerce")
lf_prot["removal_rate_mean"] = pd.to_numeric(lf_prot["removal_rate_mean"], errors="coerce")

lf_attack = pd.read_csv(os.path.join(RESULTS_DIR, "lf_results.csv"))
lf_attack["f1_mean"]  = pd.to_numeric(lf_attack["f1_mean"],  errors="coerce")
lf_attack["acc_mean"] = pd.to_numeric(lf_attack["acc_mean"], errors="coerce")

baseline  = pd.read_csv(os.path.join(RESULTS_DIR, "baseline.csv"))
baseline["f1_mean"]  = pd.to_numeric(baseline["f1_mean"],  errors="coerce")
baseline["acc_mean"] = pd.to_numeric(baseline["acc_mean"], errors="coerce")

MODEL_LS = {"LR": "-", "RF": "--", "GB": ":"}
MODEL_MK = {"LR": "o", "RF": "s",  "GB": "^"}

# ─── Рис. 5.1 — Accuracy до и после защиты (LF, профиль auto) ────────────────
print("Генерация Рис. 5.1 — Accuracy восстановление LF...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
axes = axes.flatten()

for idx, ds in enumerate(DATASETS):
    ax = axes[idx]
    for model in MODELS:
        # Baseline
        bl = baseline[(baseline["dataset"] == ds) & (baseline["model"] == model)]["acc_mean"].values
        if len(bl):
            ax.axhline(bl[0], color=DS_COLORS[ds], linestyle="dotted", alpha=0.4, linewidth=1.5)

        # Атака без защиты
        att = lf_attack[(lf_attack["dataset"] == ds) & (lf_attack["model"] == model) &
                        (lf_attack["attack_type"] == "Random_LF")].sort_values("epsilon")
        ax.plot(att["epsilon"] * 100, att["acc_mean"],
                linestyle=MODEL_LS[model], marker=MODEL_MK[model],
                color="gray", alpha=0.5, linewidth=1.5,
                label=f"{model} (без защиты)" if idx == 0 else "")

        # С защитой (профиль auto)
        prot = lf_prot[(lf_prot["dataset"] == ds) & (lf_prot["model"] == model) &
                       (lf_prot["attack_type"] == "Random_LF") &
                       (lf_prot["defense_profile"] == "auto")].sort_values("epsilon")
        ax.plot(prot["epsilon"] * 100, prot["acc_mean"],
                linestyle=MODEL_LS[model], marker=MODEL_MK[model],
                color=DS_COLORS[ds], linewidth=2.0,
                label=f"{model} (с защитой)" if idx == 0 else "")

    ax.set_title(DS_LABELS[ds], fontsize=12, fontweight="bold")
    ax.set_xlabel("ε, %", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.05, 1, 1])
out = os.path.join(FIGURES_DIR, "fig_5_1_acc_recovery_lf.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Сохранён: {out}")

# ─── Рис. 5.2 — F1 до и после защиты (LF, профиль auto) ──────────────────────
print("Генерация Рис. 5.2 — F1 восстановление LF...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
axes = axes.flatten()

for idx, ds in enumerate(DATASETS):
    ax = axes[idx]
    for model in MODELS:
        # Baseline
        bl = baseline[(baseline["dataset"] == ds) & (baseline["model"] == model)]["f1_mean"].values
        if len(bl):
            ax.axhline(bl[0], color=DS_COLORS[ds], linestyle="dotted", alpha=0.4, linewidth=1.5)

        # Атака без защиты
        att = lf_attack[(lf_attack["dataset"] == ds) & (lf_attack["model"] == model) &
                        (lf_attack["attack_type"] == "Random_LF")].sort_values("epsilon")
        ax.plot(att["epsilon"] * 100, att["f1_mean"],
                linestyle=MODEL_LS[model], marker=MODEL_MK[model],
                color="gray", alpha=0.5, linewidth=1.5,
                label=f"{model} (без защиты)" if idx == 0 else "")

        # С защитой
        prot = lf_prot[(lf_prot["dataset"] == ds) & (lf_prot["model"] == model) &
                       (lf_prot["attack_type"] == "Random_LF") &
                       (lf_prot["defense_profile"] == "auto")].sort_values("epsilon")
        ax.plot(prot["epsilon"] * 100, prot["f1_mean"],
                linestyle=MODEL_LS[model], marker=MODEL_MK[model],
                color=DS_COLORS[ds], linewidth=2.0,
                label=f"{model} (с защитой)" if idx == 0 else "")

    ax.set_title(DS_LABELS[ds], fontsize=12, fontweight="bold")
    ax.set_xlabel("ε, %", fontsize=10)
    ax.set_ylabel("F1 (macro)", fontsize=10)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.05, 1, 1])
out = os.path.join(FIGURES_DIR, "fig_5_2_f1_recovery_lf.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Сохранён: {out}")

# ─── Рис. 5.3 — Тепловая карта ΔF1 (LF, профиль auto) ────────────────────────
print("Генерация Рис. 5.3 — Тепловая карта ΔF1 LF...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

for idx, ds in enumerate(DATASETS):
    ax = axes[idx]
    delta_matrix = np.zeros((len(MODELS), len(EPSILONS)))

    for mi, model in enumerate(MODELS):
        for ei, eps in enumerate(EPSILONS):
            # F1 атаки
            att_row = lf_attack[(lf_attack["dataset"] == ds) & (lf_attack["model"] == model) &
                                 (lf_attack["attack_type"] == "Random_LF") &
                                 (abs(lf_attack["epsilon"] - eps) < 1e-6)]["f1_mean"].values
            # F1 защиты
            prot_row = lf_prot[(lf_prot["dataset"] == ds) & (lf_prot["model"] == model) &
                                (lf_prot["attack_type"] == "Random_LF") &
                                (lf_prot["defense_profile"] == "auto") &
                                (abs(lf_prot["epsilon"] - eps) < 1e-6)]["f1_mean"].values
            if len(att_row) and len(prot_row):
                delta_matrix[mi, ei] = prot_row[0] - att_row[0]

    vmax = max(abs(delta_matrix).max(), 0.01)
    im = ax.imshow(delta_matrix, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(EPSILONS)))
    ax.set_xticklabels(["1%","5%","10%","15%","20%","25%","30%"], fontsize=9)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=10)
    ax.set_title(DS_LABELS[ds], fontsize=12, fontweight="bold")
    ax.set_xlabel("ε", fontsize=10)
    # Аннотации
    for mi in range(len(MODELS)):
        for ei in range(len(EPSILONS)):
            val = delta_matrix[mi, ei]
            ax.text(ei, mi, f"{val:+.3f}", ha="center", va="center", fontsize=7,
                    color="black" if abs(val) < vmax * 0.6 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle("ΔF1 = F1_prot − F1_attacked (Random LF, профиль 'auto')",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
out = os.path.join(FIGURES_DIR, "fig_5_3_delta_f1_heatmap_lf.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Сохранён: {out}")

# ─── Рис. 5.4 — I-score LF (ε=10%, оба профиля) ──────────────────────────────
print("Генерация Рис. 5.4 — I-score LF (ε=10%)...")

eps_val = 0.10
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(DATASETS))
width = 0.13
offsets = np.linspace(-width * 2.5, width * 2.5, 6)
profiles = ["auto", "lf"]
model_colors = {"LR": "#20808D", "RF": "#A84B2F", "GB": "#1B474D"}
model_hatches = {"LR": "", "RF": "///", "GB": "xxx"}

i = 0
for model in MODELS:
    for profile in profiles:
        vals = []
        for ds in DATASETS:
            row = lf_prot[(lf_prot["dataset"] == ds) & (lf_prot["model"] == model) &
                          (lf_prot["attack_type"] == "Random_LF") &
                          (lf_prot["defense_profile"] == profile) &
                          (abs(lf_prot["epsilon"] - eps_val) < 1e-6)]
            vals.append(row["I_score"].values[0] if len(row) else np.nan)
        label = f"{model} ({profile})"
        alpha = 1.0 if profile == "lf" else 0.55
        ax.bar(x + offsets[i], vals, width * 0.95,
               color=model_colors[model], alpha=alpha,
               hatch=model_hatches[model], label=label,
               edgecolor="white", linewidth=0.5)
        i += 1

ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels([DS_LABELS[d] for d in DATASETS], fontsize=11)
ax.set_ylabel("Интегральная метрика I", fontsize=11)
ax.set_title(f"Интегральная метрика I — Random LF, ε = {int(eps_val*100)}%",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
ax.grid(axis="y", alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
fig.tight_layout()
out = os.path.join(FIGURES_DIR, "fig_5_4_I_score_lf.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Сохранён: {out}")

# ─── Рис. 5.8 — Сводная тепловая карта I (оба типа атак, ε=10%) ──────────────
print("Генерация Рис. 5.8 — Сводная тепловая карта I-score...")

bd_prot = pd.read_csv(os.path.join(RESULTS_DIR, "defense_bd_results.csv"))
bd_prot["I_score"] = pd.to_numeric(bd_prot["I_score"], errors="coerce")

eps_val = 0.10
attack_types = {
    "Random LF": ("lf_prot", "Random_LF", "auto"),
    "Targeted LF": ("lf_prot", "Targeted_LF", "auto"),
    "Backdoor": ("bd_prot", "Backdoor", "backdoor"),
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for col_idx, (att_label, (df_name, att_type, profile)) in enumerate(attack_types.items()):
    ax = axes[col_idx]
    df = lf_prot if df_name == "lf_prot" else bd_prot
    matrix = np.zeros((len(MODELS), len(DATASETS)))

    for mi, model in enumerate(MODELS):
        for di, ds in enumerate(DATASETS):
            row = df[(df["dataset"] == ds) & (df["model"] == model) &
                     (df["attack_type"] == att_type) &
                     (df["defense_profile"] == profile) &
                     (abs(df["epsilon"] - eps_val) < 1e-6)]
            matrix[mi, di] = row["I_score"].values[0] if len(row) else np.nan

    im = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DS_LABELS[d] for d in DATASETS], fontsize=9, rotation=15)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=10)
    ax.set_title(att_label, fontsize=12, fontweight="bold")
    for mi in range(len(MODELS)):
        for di in range(len(DATASETS)):
            val = matrix[mi, di]
            if not np.isnan(val):
                ax.text(di, mi, f"{val:.2f}", ha="center", va="center", fontsize=9,
                        color="black" if val < 0.7 else "white")
    plt.colorbar(im, ax=ax, shrink=0.85)

fig.suptitle(f"Интегральная метрика I по типу атаки (ε = {int(eps_val*100)}%, лучший профиль)",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
out = os.path.join(FIGURES_DIR, "fig_5_8_I_heatmap_combined.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Сохранён: {out}")

print("\n✅ LF рисунки (5.1, 5.2, 5.3, 5.4, 5.8) сгенерированы.")
