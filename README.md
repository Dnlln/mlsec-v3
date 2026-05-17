# mlsec-v3 — Экспериментальный стенд ВКР

**Тема:** «Методика защиты моделей машинного обучения от атак типа "отравление данных"»  
**Автор:** Ильин Д.С., НИЯУ МИФИ, направление 10.04.01

## Структура репозитория

```
src/
  baseline.py        — базовое обучение LR/RF/GB на 4 датасетах
  attack_lf.py       — атаки Random LF и Targeted LF
results/
  baseline.csv       — базовые метрики (Accuracy, F1-macro)
  lf_results.csv     — результаты Label Flipping (168 строк)
  bd_results.csv     — результаты Backdoor-атаки (84 строки)
figures/
  fig_3_2_5_acc_random_lf.png   — Accuracy при Random LF (Рис. 3.2–3.5)
  fig_3_6_9_f1_random_lf.png    — F1-macro при Random LF (Рис. 3.6–3.9)
  fig_3_10_13_asr_backdoor.png  — ASR Backdoor-атаки (Рис. 3.10–3.13)
  fig_3_14_heatmap.png          — Тепловая карта деградации (Рис. 3.14)
```

## Параметры экспериментов

- **Алгоритмы:** Logistic Regression, Random Forest, Gradient Boosting (scikit-learn 1.4)
- **Датасеты:** UNSW-NB15, Adult Income, SMS Spam Collection, MNIST
- **Уровни отравления:** ε ∈ {1, 5, 10, 15, 20, 25, 30%}
- **Повторений:** 5 (random_state=42..46), n_jobs=1
- **Атаки:** Random LF, Targeted LF, Backdoor (ART 1.17, ConstantPerturbation)

## Воспроизведение

```bash
pip install scikit-learn pandas numpy matplotlib seaborn adversarial-robustness-toolbox
python src/baseline.py
python src/attack_lf.py
```
