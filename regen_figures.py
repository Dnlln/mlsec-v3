"""
regen_figures.py — Перегенерация всех рисунков глав 3–5.

Запускает последовательно:
  - gen_figures_lf.py   — рисунки 3.2–3.9 (Accuracy и F1 при LF-атаках)
  - gen_figures_bd.py   — рисунки 3.10–3.14 (ASR и heatmap при Backdoor)
  - gen_I_score_plots.py — рисунки интегральной метрики I (глава 5)

Предварительное условие: CSV-файлы результатов должны быть в results/.
Выходные файлы сохраняются в figures/.

Использование:
  python regen_figures.py
"""

import subprocess
import sys
import os

scripts = [
    "gen_figures_lf.py",
    "gen_figures_bd.py",
    "gen_I_score_plots.py",
]

base = os.path.dirname(os.path.abspath(__file__))

for script in scripts:
    path = os.path.join(base, script)
    print(f"→ Running {script}...")
    result = subprocess.run([sys.executable, path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:300]}")
    else:
        print(f"  OK")

print("Done. Figures saved to figures/")
