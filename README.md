# mlsec-v3

**Методика защиты моделей машинного обучения от атак с использованием отравления данных**

Выпускная квалификационная работа — НИЯУ МИФИ, кафедра № 42, направление 10.04.01 «Информационная безопасность»

---

## Структура репозитория

```
mlsec-v3/
├── src/
│   ├── preprocessing/
│   │   └── datasets.py          # Загрузка UNSW-NB15, Adult, SMS Spam, MNIST
│   ├── models/
│   │   └── classifiers.py       # Фабрика моделей LR, RF, GB
│   ├── attacks/
│   │   ├── lf.py                # Random LF и Targeted LF
│   │   └── backdoor.py          # Backdoor-атака с ConstantPerturbation
│   ├── metrics/
│   │   └── metrics.py           # Accuracy, F1, ASR, интегральная метрика I
│   └── defenses/
│       └── defense.py           # DefenseTransformer (трёхуровневый конвейер)
│
├── experiments/
│   ├── baseline.yaml            # Конфигурация базового эксперимента
│   ├── attack_lf.yaml           # Конфигурация LF-атак
│   ├── attack_backdoor.yaml     # Конфигурация Backdoor-атаки
│   └── defense.yaml             # Конфигурация эксперимента с защитой
│
├── data/
│   ├── raw/                     # Исходные данные (SMS Spam кэш, .gitkeep)
│   └── processed/               # Предобработанные данные (.gitkeep)
│
├── results/                     # Выходные CSV (генерируются скриптами)
│   ├── baseline.csv
│   ├── lf_results.csv
│   ├── bd_results.csv
│   ├── defense_lf_results.csv
│   ├── defense_bd_results.csv
│   └── I_score_all.csv
│
├── figures/                     # Рисунки глав 3–5 (генерируются regen_figures.py)
│
├── baseline.py                  # Запуск базового эксперимента
├── attack_lf.py                 # Запуск LF-атак
├── attack_backdoor.py           # Запуск Backdoor-атаки
├── defense_experiment.py        # Запуск эксперимента с защитой
├── regen_figures.py             # Перегенерация рисунков
├── gen_I_score_plots.py         # Графики интегральной метрики
└── requirements.txt
```

---

## Быстрый старт

### 1. Установка зависимостей

```bash
git clone https://github.com/Dnlln/mlsec-v3.git
cd mlsec-v3
pip install -r requirements.txt
```

### 2. Воспроизведение экспериментов

Шаги выполняются последовательно — каждый следующий зависит от результатов предыдущего.

```bash
# Шаг 1 — базовые метрики (≈ 10–20 мин)
python baseline.py

# Шаг 2 — атаки Label Flipping (≈ 3–5 ч)
python attack_lf.py

# Шаг 3 — Backdoor-атака (≈ 5–8 ч)
python attack_backdoor.py

# Шаг 4 — эксперимент с защитой DefenseTransformer (≈ 8–12 ч)
python defense_experiment.py

# Перегенерация рисунков (после получения CSV)
python regen_figures.py
python gen_I_score_plots.py
```

### 3. Использование DefenseTransformer отдельно

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.defenses.defense import DefenseTransformer

pipeline = Pipeline([
    ("scaler",  StandardScaler()),
    ("defense", DefenseTransformer(contamination=0.10, threat_profile="auto")),
    ("clf",     LogisticRegression()),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

---

## Наборы данных

| Датасет | Домен | Объём | Признаки | Источник |
|---------|-------|-------|----------|---------|
| UNSW-NB15 | Сетевая безопасность | 175 341 → 25 000* | 45 | OpenML did=46301 |
| Adult | Социально-демографический | 48 842 | 14 | OpenML did=1590 |
| SMS Spam | Телекоммуникации | 5 574 | 500 (TF-IDF) | UCI ML Repository |
| MNIST (bin.) | Компьютерное зрение | 70 000 → 8 000* | 784 | OpenML did=554 |

\* стратифицированная подвыборка для ускорения экспериментов

Датасеты загружаются автоматически при первом запуске. SMS Spam кэшируется в `data/raw/sms_spam.csv`.

---

## Вычислительные требования

- **CPU:** четырёхядерный (и выше), GPU не требуется
- **RAM:** не более 4 ГБ (последовательное выполнение, `n_jobs=1`)
- **Полный прогон:** 14–20 часов при последовательном выполнении
- **Python:** 3.11+

---

## Конфигурационные файлы

Каждый эксперимент описан YAML-конфигом в папке `experiments/`. Конфиги содержат полную спецификацию: датасеты, параметры моделей, уровни загрязнения, метрики и путь к выходному файлу.

---

## Параметры воспроизводимости

| Параметр | Значение | Назначение |
|----------|----------|-----------|
| `random_state` | 42 | Все стохастические компоненты |
| `n_jobs` | 1 | Детерминированность на любой платформе |
| `test_size` | 0.20 | Стратифицированное разбиение 80/20 |
| `n_repeats` | 5 | Усреднение по повторениям |

---

## Лицензия

Код распространяется в учебных и исследовательских целях.
