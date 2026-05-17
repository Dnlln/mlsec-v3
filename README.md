# mlsec-v3 — Экспериментальный стенд ВКР

**Тема:** «Методика защиты моделей машинного обучения от атак с использованием отравления данных»  
**Автор:** Ильин Д.С., НИЯУ МИФИ, направление 10.04.01  
**Научный руководитель:** каф. № 42

---

## Структура репозитория

```
src/
  baseline.py          — базовое обучение LR/RF/GB на 4 датасетах
  attack_lf.py         — атаки Random LF и Targeted LF
  attack_backdoor.py   — Backdoor-атака (ART, ConstantPerturbation)
  defenses/
    __init__.py
    defense.py         — DefenseTransformer: конвейер IF + LOF + профиль
  gen_figures_lf.py    — генерация рисунков 5.1–5.4, 5.8 (защита от LF)
  gen_figures_bd.py    — генерация рисунков 5.5–5.7 (защита от Backdoor)

results/
  baseline.csv              — базовые метрики (Accuracy, F1-macro)
  lf_results.csv            — результаты Label Flipping-атаки (336 строк)
  bd_results.csv            — результаты Backdoor-атаки (168 строк)
  defense_lf_results.csv    — результаты защиты от LF (336 строк)
  defense_bd_results.csv    — результаты защиты от Backdoor (168 строк)

figures/
  fig_3_2_5_acc_random_lf.png    — Accuracy при Random LF (Рис. 3.2–3.5)
  fig_3_6_9_f1_random_lf.png     — F1-macro при Random LF (Рис. 3.6–3.9)
  fig_3_10_13_asr_backdoor.png   — ASR Backdoor-атаки (Рис. 3.10–3.13)
  fig_3_14_heatmap.png           — Тепловая карта деградации (Рис. 3.14)
  fig_5_1_acc_recovery_lf.png    — Восстановление Accuracy (LF, Рис. 5.1)
  fig_5_2_f1_recovery_lf.png     — Восстановление F1 (LF, Рис. 5.2)
  fig_5_3_delta_f1_heatmap_lf.png — ΔF1 тепловая карта (Рис. 5.3)
  fig_5_4_I_score_lf.png         — Метрика I (LF, Рис. 5.4)
  fig_5_5_asr_before_after.png   — ASR до/после защиты (Backdoor, Рис. 5.5)
  fig_5_6_f1_before_after_bd.png — F1 до/после защиты (Backdoor, Рис. 5.6)
  fig_5_7_I_score_bd.png         — Метрика I (Backdoor, Рис. 5.7)
  fig_5_8_I_heatmap_combined.png — Сводная тепловая карта I (Рис. 5.8)
```

---

## Параметры экспериментов

| Параметр | Значение |
|---|---|
| Алгоритмы | Logistic Regression, Random Forest, Gradient Boosting |
| Датасеты | UNSW-NB15 (25k), Adult Income (20k), SMS Spam (5.5k), MNIST (8k) |
| Уровни отравления ε | {1, 5, 10, 15, 20, 25, 30%} |
| Повторений | 5 (random_state=42), n_jobs=1 |
| Атаки | Random Label Flipping, Backdoor (ConstantPerturbation) |
| Профили защиты | `auto`, `lf`, `backdoor` |
| Окружение | Python 3.12.8, scikit-learn 1.8.0, Linux 6.1, 2 vCPU, 7.8 ГБ RAM |

---

## Методика защиты DefenseTransformer

Трёхуровневый конвейер (sklearn-совместимый `TransformerMixin`):

1. **Isolation Forest** — глобальная фильтрация статистических аномалий
2. **LOF (Local Outlier Factor)** — локальная проверка плотности окружения
3. **Профильная адаптация** — настройка порогов под тип атаки (`lf` / `backdoor` / `auto`)

**Интегральная метрика защищённости:**

```
I = 0.4 * (F1_prot / F1_base) + 0.4 * (1 - ASR_prot) - 0.2 * time_ratio
```

Диапазон: I > 0 означает положительный эффект защиты; I < 0 — гиперкоррекция.

---

## Воспроизведение

```bash
pip install scikit-learn pandas numpy matplotlib seaborn adversarial-robustness-toolbox

# 1. Базовые метрики
python src/baseline.py

# 2. Атаки
python src/attack_lf.py
python src/attack_backdoor.py

# 3. Защита (результаты → results/defense_*_results.csv)
# запустить эксперимент защиты (см. src/defenses/defense.py)

# 4. Рисунки
python src/gen_figures_lf.py    # → figures/fig_5_1..5_4, 5_8
python src/gen_figures_bd.py    # → figures/fig_5_5..5_7
```

---

## Ключевые результаты

- При ε ∈ [10%; 20%] конвейер восстанавливает F1 до baseline-уровня, ASR (Backdoor) снижается ниже 0.20
- Специализированный профиль даёт +2–7 п.п. к метрике I относительно `auto`
- При ε > 25% наблюдается гиперкоррекция (I < 0)
- time_ratio = 0.3–0.8 для UNSW/Adult; 1.5–2.5 для MNIST/SMS
