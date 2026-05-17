"""
Генерация рисунков 5.5, 5.6, 5.7 по данным Backdoor-защиты.
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

# Цвета по датасетам
DS_COLORS  = {"UNSW-NB15": "#20808D", "Adult": "#A84B2F", "SMS_Spam": "#1B474D", "MNIST": "#944454"}
DS_LABELS  = {"UNSW-NB15": "UNSW-NB15", "Adult": "Adult", "SMS_Spam": "SMS Spam", "MNIST": "MNIST"}
MODELS     = ["LR", "RF", "GB"]
DATASETS   = ["UNSW-NB15", "Adult", "SMS_Spam", "MNIST"]
EPSILONS   = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
EPS_LABELS = ["1%", "5%", "10%", "15%", "20%", "25%", "30%"]

FONT = {"family": "DejaVu Serif", "size": 11}
matplotlib.rc("font", **FONT)
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False

# Загружаем данные
bd_prot = pd.read_csv(os.path.join(RESULTS_DIR, "defense_bd_results.csv"))
bd_prot["f1_mean"]  = pd.to_numeric(bd_prot["f1_mean"],  errors="coerce")
bd_prot["asr_mean"] = pd.to_numeric(bd_prot["asr_mean"], errors="coerce")
bd_prot["I_score"]  = pd.to_numeric(bd_prot["I_score"],  errors="coerce")

bd_attack = pd.read_csv(os.path.join(RESULTS_DIR, "bd_results.csv"))
bd_attack["f1_mean"]  = pd.to_numeric(bd_attack["f1_mean"],  errors="coerce")
bd_attack["asr_mean"] = pd.to_numeric(bd_attack["asr_mean"], errors="coerce")

baseline  = pd.read_csv(os.path.join(RESULTS_DIR, "baseline.csv"))

# ─── Рис. 5.5 — ASR до и после защиты ─────────────────────────────────────────
print("Генерация Рис. 5.5 — ASR до/после защиты...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
axes = axes.flatten()

for idx, ds in enumerate(DATASETS):
    ax = axes[idx]
    # среднее ASR по моделям
    for model, ls, mk in zip(MODELS, ["-", "--", ":"], ["o", "s", "^"]):
        # Атака без защиты
        att = bd_attack[(bd_attack["dataset"] == ds) & (bd_attack["model"] == model)]
        att = att.sort_values("epsilon")
        ax.plot(att["epsilon"] * 100, att["asr_mean"],
                linestyle=ls, marker=mk, color="gray", alpha=0.5, linewidth=1.5,
                label=f"{model} (без защиты)" if idx == 0 else "")

        # С защитой (профиль backdoor)
        prot = bd_prot[(bd_prot["dataset"] == ds) & (bd_prot["model"] == model) &
                       (bd_prot["defense_profile"] == "backdoor")]
        prot = prot.sort_values("epsilon")
        ax.plot(prot["epsilon"] * 100, prot["asr_mean"],
                linestyle=ls, marker=mk, color=DS_COLORS[ds], linewidth=2.0,
                label=f"{model} (с защитой)" if idx == 0 else "")

    ax.set_title(DS_LABELS[ds], fontsize=12, fontweight="bold")
    ax.set_xlabel("ε, %", fontsize=10)
    ax.set_ylabel("ASR", fontsize=10)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.axhline(0.2, color="red", linestyle="--", alpha=0.4, linewidth=1)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3)

# Общая легенда
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.05, 1, 1])
out = os.path.join(FIGURES_DIR, "fig_5_5_asr_before_after.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Сохранён: {out}")

# ─── Рис. 5.6 — F1 до и после защиты (Backdoor) ───────────────────────────────
print("Генерация Рис. 5.6 — F1 до/после защиты (Backdoor)...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
axes = axes.flatten()

for idx, ds in enumerate(DATASETS):
    ax = axes[idx]
    # Baseline (без атаки)
    for model, ls, mk in zip(MODELS, ["-", "--", ":"], ["o", "s", "^"]):
        bl_val = baseline[(baseline["dataset"] == ds) & (baseline["model"] == model)]["f1_mean"].values
        if len(bl_val):
            ax.axhline(bl_val[0], color=DS_COLORS[ds], linestyle="dotted",
                       alpha=0.4, linewidth=1.5)

        att = bd_attack[(bd_attack["dataset"] == ds) & (bd_attack["model"] == model)].sort_values("epsilon")
        ax.plot(att["epsilon"] * 100, att["f1_mean"],
                linestyle=ls, marker=mk, color="gray", alpha=0.5, linewidth=1.5,
                label=f"{model} (без защиты)" if idx == 0 else "")

        prot = bd_prot[(bd_prot["dataset"] == ds) & (bd_prot["model"] == model) &
                       (bd_prot["defense_profile"] == "backdoor")].sort_values("epsilon")
        ax.plot(prot["epsilon"] * 100, prot["f1_mean"],
                linestyle=ls, marker=mk, color=DS_COLORS[ds], linewidth=2.0,
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
out = os.path.join(FIGURES_DIR, "fig_5_6_f1_before_after_bd.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Сохранён: {out}")

# ─── Рис. 5.7 — Интегральная метрика I (Backdoor, ε=10%) ──────────────────────
print("Генерация Рис. 5.7 — I-score Backdoor (ε=10%)...")

eps_val = 0.10
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(DATASETS))
width = 0.13
offsets = np.linspace(-width * 2.5, width * 2.5, 6)
profiles = ["auto", "backdoor"]
model_colors = {"LR": "#20808D", "RF": "#A84B2F", "GB": "#1B474D"}
model_hatches = {"LR": "", "RF": "///", "GB": "xxx"}
bar_handles = []

i = 0
for model in MODELS:
    for profile in profiles:
        vals = []
        for ds in DATASETS:
            row = bd_prot[(bd_prot["dataset"] == ds) & (bd_prot["model"] == model) &
                          (bd_prot["defense_profile"] == profile) &
                          (abs(bd_prot["epsilon"] - eps_val) < 1e-6)]
            vals.append(row["I_score"].values[0] if len(row) else np.nan)
        label = f"{model} ({profile})"
        alpha = 1.0 if profile == "backdoor" else 0.55
        bars = ax.bar(x + offsets[i], vals, width * 0.95,
                      color=model_colors[model], alpha=alpha,
                      hatch=model_hatches[model], label=label,
                      edgecolor="white", linewidth=0.5)
        bar_handles.append(bars)
        i += 1

ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels([DS_LABELS[d] for d in DATASETS], fontsize=11)
ax.set_ylabel("Интегральная метрика I", fontsize=11)
ax.set_title(f"Интегральная метрика I — Backdoor, ε = {int(eps_val*100)}%", fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
ax.grid(axis="y", alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
fig.tight_layout()
out = os.path.join(FIGURES_DIR, "fig_5_7_I_score_bd.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Сохранён: {out}")

print("\n✅ Backdoor рисунки (5.5, 5.6, 5.7) сгенерированы.")
