import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

df = pd.read_csv('results/I_score_all.csv')

# ── Цветовая схема ──────────────────────────────────────────────────────────
COLORS = {
    'LR':  '#20808D',
    'RF':  '#A84B2F',
    'GB':  '#1B474D',
}
PROFILE_LS = {'auto': '--', 'lf': '-', 'backdoor': '-'}
PROFILE_MARKER = {'auto': 'o', 'lf': 's', 'backdoor': '^'}

DATASETS = ['UNSW-NB15', 'Adult', 'SMS_Spam', 'MNIST']
MODELS   = ['LR', 'RF', 'GB']
EPS_LABELS = ['1%', '5%', '10%', '15%', '20%', '25%', '30%']
EPS_VALS   = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

os.makedirs('figures/appendix', exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'figure.dpi': 150,
})

# ════════════════════════════════════════════════════════════════════════════
# БЛОК 1: I(ε) по каждому датасету × профиль — LF (Random + Targeted)
#   4 датасета × 2 типа × 2 профиля = 16 графиков → 2 рисунка (4×2 субплота каждый)
# ════════════════════════════════════════════════════════════════════════════

for attack in ['Random_LF', 'Targeted_LF']:
    attack_label = 'Random Label Flipping' if attack == 'Random_LF' else 'Targeted Label Flipping'
    profiles = ['auto', 'lf']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Интегральная метрика I — {attack_label}\n(профили: auto и lf)', fontsize=14, fontweight='bold', y=1.01)

    for ax, dataset in zip(axes.flat, DATASETS):
        sub = df[(df['Тип атаки'] == attack) & (df['Датасет'] == dataset)]
        for model in MODELS:
            for profile in profiles:
                d = sub[(sub['Модель'] == model) & (sub['Профиль защиты'] == profile)]
                d = d.sort_values('ε (доля)')
                if d.empty: continue
                lbl = f'{model} ({profile})'
                ax.plot(EPS_LABELS, d['I_score'].values,
                        color=COLORS[model],
                        linestyle=PROFILE_LS[profile],
                        marker=PROFILE_MARKER[profile],
                        markersize=6,
                        linewidth=1.8,
                        label=lbl)
        ax.axhline(0, color='#888', linewidth=1.0, linestyle=':')
        ax.set_title(dataset, fontsize=12, fontweight='bold')
        ax.set_xlabel('Уровень отравления ε', fontsize=10)
        ax.set_ylabel('I_score', fontsize=10)
        ax.set_ylim(bottom=min(df[df['Тип атаки'] == attack]['I_score'].min() - 0.05, -0.2))

    # Общая легенда
    handles = []
    for model in MODELS:
        for profile in profiles:
            handles.append(
                plt.Line2D([0],[0], color=COLORS[model],
                           linestyle=PROFILE_LS[profile],
                           marker=PROFILE_MARKER[profile],
                           markersize=6, linewidth=1.8,
                           label=f'{model} ({profile})'))
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=10)

    plt.tight_layout()
    fname = f"figures/appendix/I_score_{attack.lower()}_by_dataset.png"
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✓ {fname}")

# ════════════════════════════════════════════════════════════════════════════
# БЛОК 2: I(ε) — Backdoor, профили auto и backdoor
# ════════════════════════════════════════════════════════════════════════════
attack = 'Backdoor'
profiles = ['auto', 'backdoor']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Интегральная метрика I — Backdoor-атака\n(профили: auto и backdoor)', fontsize=14, fontweight='bold', y=1.01)

for ax, dataset in zip(axes.flat, DATASETS):
    sub = df[(df['Тип атаки'] == attack) & (df['Датасет'] == dataset)]
    for model in MODELS:
        for profile in profiles:
            d = sub[(sub['Модель'] == model) & (sub['Профиль защиты'] == profile)]
            d = d.sort_values('ε (доля)')
            if d.empty: continue
            ax.plot(EPS_LABELS, d['I_score'].values,
                    color=COLORS[model],
                    linestyle=PROFILE_LS[profile],
                    marker=PROFILE_MARKER[profile],
                    markersize=6, linewidth=1.8,
                    label=f'{model} ({profile})')
    ax.axhline(0, color='#888', linewidth=1.0, linestyle=':')
    ax.set_title(dataset, fontsize=12, fontweight='bold')
    ax.set_xlabel('Уровень отравления ε', fontsize=10)
    ax.set_ylabel('I_score', fontsize=10)

handles = []
for model in MODELS:
    for profile in profiles:
        handles.append(plt.Line2D([0],[0], color=COLORS[model],
                       linestyle=PROFILE_LS[profile], marker=PROFILE_MARKER[profile],
                       markersize=6, linewidth=1.8, label=f'{model} ({profile})'))
fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=10)

plt.tight_layout()
fname = "figures/appendix/I_score_backdoor_by_dataset.png"
plt.savefig(fname, bbox_inches='tight', dpi=150)
plt.close()
print(f"✓ {fname}")

# ════════════════════════════════════════════════════════════════════════════
# БЛОК 3: Heatmap I_score — для каждого типа атаки
#   Ось X: ε, Ось Y: датасет+модель, клетка = I_score (усреднено по профилям)
# ════════════════════════════════════════════════════════════════════════════
from matplotlib.colors import TwoSlopeNorm

for attack in ['Random_LF', 'Targeted_LF', 'Backdoor']:
    attack_label = {'Random_LF': 'Random Label Flipping',
                    'Targeted_LF': 'Targeted Label Flipping',
                    'Backdoor': 'Backdoor'}[attack]

    sub = df[df['Тип атаки'] == attack].copy()
    # Усредняем по профилям
    pivot = sub.groupby(['Датасет', 'Модель', 'ε (доля)'])['I_score'].mean().reset_index()
    pivot['Группа'] = pivot['Датасет'] + ' / ' + pivot['Модель']
    mat = pivot.pivot(index='Группа', columns='ε (доля)', values='I_score')

    # Упорядочиваем строки
    row_order = [f'{ds} / {m}' for ds in DATASETS for m in MODELS]
    row_order = [r for r in row_order if r in mat.index]
    mat = mat.loc[row_order]
    mat.columns = EPS_LABELS

    vmin, vmax = mat.values.min(), mat.values.max()
    norm = TwoSlopeNorm(vmin=min(vmin, -0.01), vcenter=0, vmax=max(vmax, 0.01))

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(mat.values, cmap='RdYlGn', norm=norm, aspect='auto')

    ax.set_xticks(range(len(EPS_LABELS)))
    ax.set_xticklabels(EPS_LABELS, fontsize=10)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=10)
    ax.set_xlabel('Уровень отравления ε', fontsize=11)
    ax.set_title(f'Тепловая карта I_score — {attack_label}\n(среднее по профилям)', fontsize=13, fontweight='bold')

    # Значения в клетках
    for i in range(len(mat.index)):
        for j in range(len(EPS_LABELS)):
            val = mat.values[i, j]
            color = 'black' if abs(val) < max(abs(vmin), vmax) * 0.6 else 'white'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label='I_score', shrink=0.8)
    plt.tight_layout()
    fname = f"figures/appendix/I_score_heatmap_{attack.lower()}.png"
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✓ {fname}")

# ════════════════════════════════════════════════════════════════════════════
# БЛОК 4: Сравнение профилей — лучший профиль vs auto (по всем атакам)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
attack_list = ['Random_LF', 'Targeted_LF', 'Backdoor']
attack_labels = ['Random LF', 'Targeted LF', 'Backdoor']
spec_profile = {'Random_LF': 'lf', 'Targeted_LF': 'lf', 'Backdoor': 'backdoor'}

for ax, attack, alabel in zip(axes, attack_list, attack_labels):
    sub = df[df['Тип атаки'] == attack]
    auto_I = sub[sub['Профиль защиты'] == 'auto'].groupby('ε (доля)')['I_score'].mean()
    spec_I = sub[sub['Профиль защиты'] == spec_profile[attack]].groupby('ε (доля)')['I_score'].mean()
    
    ax.plot(EPS_LABELS, auto_I.values, color='#A84B2F', linestyle='--', marker='o',
            linewidth=2, markersize=7, label='auto')
    ax.plot(EPS_LABELS, spec_I.values, color='#20808D', linestyle='-', marker='s',
            linewidth=2, markersize=7, label=spec_profile[attack])
    ax.axhline(0, color='#888', linewidth=1.0, linestyle=':')
    ax.set_title(alabel, fontsize=12, fontweight='bold')
    ax.set_xlabel('ε', fontsize=10)
    ax.set_ylabel('I_score (среднее)', fontsize=10)
    ax.legend(fontsize=10)

fig.suptitle('Сравнение профилей защиты: auto vs специализированный\n(среднее по всем датасетам и моделям)', fontsize=13, fontweight='bold')
plt.tight_layout()
fname = "figures/appendix/I_score_profile_comparison.png"
plt.savefig(fname, bbox_inches='tight', dpi=150)
plt.close()
print(f"✓ {fname}")

print("\nВсе графики готовы.")
