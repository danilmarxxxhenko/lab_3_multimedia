"""
Серія експериментів для звіту ЛР3.
Запуск:  python experiments.py [зображення]
Результати — у папці results/:
    experiments.csv          — таблиця всіх експериментів
    grid_fraction.png        — Original | DCT | DWT для кількох рівнів
    crops_fraction.png       — збільшені фрагменти (видно артефакти)
    psnr_curve.png           — PSNR від частки коефіцієнтів
    dwt_levels.csv           — вплив кількості рівнів Хаара
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app import load_gray, HERE
from lossy import Codec

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "cameraman.png")
img = load_gray(path)
LEVELS = 3
codec = Codec(img, LEVELS)

PLAN = [
    ("fraction", [50, 20, 10, 5, 2, 1]),
    ("threshold", [10, 25, 50, 100]),
    ("quant", [10, 20, 40, 80]),
]

rows, recs = [], {}
for mode, values in PLAN:
    for v in values:
        rd, _, pd, sd = codec.run_dct(mode, v)
        rw, _, pw, sw = codec.run_dwt(mode, v)
        recs[(mode, v)] = (rd, rw, pd, pw, sd, sw)
        rows.append([mode, v,
                     sd["nonzero"], round(sd["percent"], 2), round(sd["cr"], 1), round(pd, 2),
                     sw["nonzero"], round(sw["percent"], 2), round(sw["cr"], 1), round(pw, 2)])

with open(os.path.join(OUT, "experiments.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["mode", "param",
                "DCT_nonzero", "DCT_percent", "DCT_CR", "DCT_PSNR",
                "DWT_nonzero", "DWT_percent", "DWT_CR", "DWT_PSNR"])
    w.writerows(rows)

print(f"{'режим':10}{'парам':>7} | {'DCT %':>7}{'PSNR':>7} | {'DWT %':>7}{'PSNR':>7}")
for r in rows:
    print(f"{r[0]:10}{r[1]:>7} | {r[3]:>7}{r[5]:>7} | {r[7]:>7}{r[9]:>7}")

# ---- сітка Original | DCT | DWT -----------------------------------------
show = [10, 5, 2, 1]
fig, axes = plt.subplots(len(show), 3, figsize=(9, 3 * len(show)))
for i, p in enumerate(show):
    rd, rw, pd, pw, sd, sw = recs[("fraction", p)]
    for j, (im, t) in enumerate(((img, "Original"),
                                 (rd, f"DCT, {p}% коеф.\nPSNR = {pd:.2f} дБ"),
                                 (rw, f"DWT Хаар, {p}% коеф.\nPSNR = {pw:.2f} дБ"))):
        axes[i, j].imshow(im, cmap="gray", vmin=0, vmax=255)
        axes[i, j].set_title(t, fontsize=9)
        axes[i, j].axis("off")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "grid_fraction.png"), dpi=110)
plt.close(fig)

# ---- збільшені фрагменти (обличчя/камера) --------------------------------
y0, x0, s = 60, 180, 128
fig, axes = plt.subplots(2, 3, figsize=(9, 6.4))
for i, p in enumerate((5, 1)):
    rd, rw, pd, pw, _, _ = recs[("fraction", p)]
    for j, (im, t) in enumerate(((img, "Original (фрагмент)"),
                                 (rd, f"DCT {p}% — PSNR {pd:.1f} дБ"),
                                 (rw, f"DWT {p}% — PSNR {pw:.1f} дБ"))):
        axes[i, j].imshow(im[y0:y0 + s, x0:x0 + s], cmap="gray", vmin=0, vmax=255,
                          interpolation="nearest")
        axes[i, j].set_title(t, fontsize=9)
        axes[i, j].axis("off")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "crops_fraction.png"), dpi=110)
plt.close(fig)

# ---- крива PSNR(частка коефіцієнтів) -------------------------------------
ps = np.geomspace(0.2, 60, 30)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ps, [codec.run_dct("fraction", p)[2] for p in ps], "o-", ms=3, label="DCT 8×8")
for L, st in ((1, ":"), (3, "-"), (5, "--")):
    c = Codec(img, L)
    ax.plot(ps, [c.run_dwt("fraction", p)[2] for p in ps], st, label=f"DWT Хаар, {L} рівн.")
ax.set_xscale("log")
ax.set_xlabel("Частка збережених коефіцієнтів, %")
ax.set_ylabel("PSNR, дБ")
ax.grid(alpha=0.3, which="both")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "psnr_curve.png"), dpi=110)
plt.close(fig)

# ---- вплив кількості рівнів DWT ------------------------------------------
with open(os.path.join(OUT, "dwt_levels.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["levels", "PSNR_5%", "PSNR_1%"])
    print("\nРівні DWT:  PSNR при 5% / 1%")
    for L in range(1, 6):
        c = Codec(img, L)
        a, b = c.run_dwt("fraction", 5)[2], c.run_dwt("fraction", 1)[2]
        w.writerow([L, round(a, 2), round(b, 2)])
        print(f"  {L}: {a:.2f} / {b:.2f}")

print("\nГотово, файли в", OUT)
