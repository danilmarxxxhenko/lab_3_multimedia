"""
ЛР3. Стиснення зображень з втратами: DCT 8×8 vs DWT Хаара.

Запуск:
    python app.py                    # стандартне зображення cameraman.png
    python app.py шлях/до/фото.png   # будь-яке зображення (буде переведене у відтінки сірого)

Інтерфейс:
    верхній ряд   — Original | DCT | DWT (з PSNR і часткою коефіцієнтів)
    нижній ряд    — графік PSNR(рівень стиснення) | карта коефіцієнтів або помилки
    праворуч      — режим стиснення, повзунок рівня, кількість рівнів DWT, кнопки
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from PIL import Image

from lossy import Codec, MODES, psnr
from transforms import from_blocks

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# Діапазони повзунка (логарифмічна шкала: зручно і для 0.5 %, і для 50 %)
RANGES = {
    "fraction": (0.1, 100.0, 5.0),    # (мін, макс, значення за замовчуванням)
    "threshold": (1.0, 300.0, 30.0),
    "quant": (1.0, 300.0, 40.0),
}
UNITS = {"fraction": "%", "threshold": "", "quant": ""}


def load_gray(path: str, multiple: int = 32, max_side: int = 1024) -> np.ndarray:
    """Відкриває зображення, переводить у відтінки сірого й обрізає по центру
    до розміру, кратного 32 (потрібно для блоків 8×8 і до 5 рівнів Хаара)."""
    im = Image.open(path).convert("L")
    im.thumbnail((max_side, max_side))
    a = np.asarray(im)
    h, w = (a.shape[0] // multiple) * multiple, (a.shape[1] // multiple) * multiple
    y0, x0 = (a.shape[0] - h) // 2, (a.shape[1] - w) // 2
    return a[y0:y0 + h, x0:x0 + w].copy()


def pos_to_value(mode, pos):
    lo, hi, _ = RANGES[mode]
    return lo * (hi / lo) ** (pos / 100.0)


def value_to_pos(mode, value):
    lo, hi, _ = RANGES[mode]
    return 100.0 * np.log(value / lo) / np.log(hi / lo)


def coeff_view(c):
    """log(1 + |c|) — щоб було видно і великі, і малі коефіцієнти."""
    if c.ndim == 4:
        c = from_blocks(c)
    return np.log1p(np.abs(c))


class App:
    def __init__(self, path):
        self.path = path
        self.img = load_gray(path)
        self.mode = "fraction"
        self.value = RANGES["fraction"][2]
        self.levels = 3
        self.show_error = False
        self.codec = Codec(self.img, self.levels)

        self.fig = plt.figure(figsize=(15, 8.6))
        self.fig.canvas.manager.set_window_title("ЛР3 — стиснення зображень: DCT vs DWT (Хаар)")
        gs = self.fig.add_gridspec(2, 3, left=0.02, right=0.78, top=0.93, bottom=0.08,
                                   wspace=0.08, hspace=0.22)
        self.ax_orig = self.fig.add_subplot(gs[0, 0])
        self.ax_dct = self.fig.add_subplot(gs[0, 1])
        self.ax_dwt = self.fig.add_subplot(gs[0, 2])
        self.ax_curve = self.fig.add_subplot(gs[1, 0])
        self.ax_map_dct = self.fig.add_subplot(gs[1, 1])
        self.ax_map_dwt = self.fig.add_subplot(gs[1, 2])
        for ax in (self.ax_orig, self.ax_dct, self.ax_dwt, self.ax_map_dct, self.ax_map_dwt):
            ax.set_xticks([]); ax.set_yticks([])

        self._build_controls()
        self._draw_static()
        self.update()

    # ------------------------------------------------------------- UI ----
    def _build_controls(self):
        f = self.fig
        f.text(0.80, 0.93, "Режим стиснення", fontsize=11, weight="bold")
        ax = f.add_axes([0.80, 0.76, 0.18, 0.16])
        self.radio = RadioButtons(ax, list(MODES.values()), active=0)
        self.radio.on_clicked(self.on_mode)

        f.text(0.80, 0.71, "Рівень стиснення", fontsize=11, weight="bold")
        ax = f.add_axes([0.81, 0.66, 0.12, 0.03])
        self.slider = Slider(ax, "", 0, 100, valinit=value_to_pos(self.mode, self.value))
        self.slider.valtext.set_visible(False)
        self.slider_label = f.text(0.80, 0.62, "", fontsize=10)
        self.slider.on_changed(self.on_slider)

        f.text(0.80, 0.55, "Рівні DWT (Хаар)", fontsize=11, weight="bold")
        ax = f.add_axes([0.81, 0.50, 0.12, 0.03])
        self.lv = Slider(ax, "", 1, 5, valinit=self.levels, valstep=1)
        self.lv.on_changed(self.on_levels)

        self.btn_map = Button(f.add_axes([0.80, 0.40, 0.18, 0.05]), "Нижній ряд: коефіцієнти / помилка")
        self.btn_map.on_clicked(self.on_toggle_map)
        self.btn_open = Button(f.add_axes([0.80, 0.33, 0.18, 0.05]), "Відкрити зображення…")
        self.btn_open.on_clicked(self.on_open)
        self.btn_save = Button(f.add_axes([0.80, 0.26, 0.18, 0.05]), "Зберегти результат")
        self.btn_save.on_clicked(self.on_save)

        self.info = f.text(0.80, 0.05, "", fontsize=9.5, family="monospace", va="bottom")

    def _draw_static(self):
        self.im_orig = self.ax_orig.imshow(self.img, cmap="gray", vmin=0, vmax=255)
        self.ax_orig.set_title(f"Original  {self.img.shape[1]}×{self.img.shape[0]}")
        self.im_dct = self.ax_dct.imshow(self.img, cmap="gray", vmin=0, vmax=255)
        self.im_dwt = self.ax_dwt.imshow(self.img, cmap="gray", vmin=0, vmax=255)
        z = np.zeros_like(self.img, dtype=float)
        self.im_map_dct = self.ax_map_dct.imshow(z, cmap="magma")
        self.im_map_dwt = self.ax_map_dwt.imshow(z, cmap="magma")
        self._compute_curve()

    def _compute_curve(self):
        """Крива «частка ненульових коефіцієнтів → PSNR» для 25 значень параметра
        поточного режиму. По осі X завжди частка коефіцієнтів, щоб DCT і DWT
        порівнювались чесно (при однаковій «ціні» в коефіцієнтах)."""
        lo, hi, _ = RANGES[self.mode]
        xs = np.geomspace(lo, hi, 25)
        res_d = [self.codec.run_dct(self.mode, x) for x in xs]
        res_w = [self.codec.run_dwt(self.mode, x) for x in xs]
        ax = self.ax_curve
        ax.clear()
        for res, lab, col, mk in ((res_d, "DCT 8×8", "tab:blue", "o"),
                                  (res_w, f"DWT Хаар ({self.levels} рівн.)", "tab:orange", "s")):
            px = np.array([r[3]["percent"] for r in res])
            py = np.minimum([r[2] for r in res], 60)      # ∞ обрізаємо для графіка
            order = np.argsort(px)
            ax.plot(px[order], py[order], mk + "-", ms=3, label=lab, color=col)
        ax.set_xscale("log")
        ax.set_xlabel("Частка ненульових коефіцієнтів, %")
        ax.set_ylabel("PSNR, дБ")
        ax.set_title("Залежність якості від кількості коефіцієнтів", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="lower right")
        self.pt_dct, = ax.plot([], [], "o", ms=9, mfc="none", mec="red", mew=2)
        self.pt_dwt, = ax.plot([], [], "s", ms=9, mfc="none", mec="red", mew=2)

    # ------------------------------------------------------ обробники ----
    def on_mode(self, label):
        self.mode = [k for k, v in MODES.items() if v == label][0]
        self.value = RANGES[self.mode][2]
        self._compute_curve()
        self.slider.set_val(value_to_pos(self.mode, self.value))   # викличе update()

    def on_slider(self, pos):
        self.value = pos_to_value(self.mode, pos)
        self.update()

    def on_levels(self, val):
        self.levels = int(val)
        self.codec.set_levels(self.levels)
        self._compute_curve()
        self.update()

    def on_toggle_map(self, _):
        self.show_error = not self.show_error
        self.update()

    def on_open(self, _):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            path = filedialog.askopenfilename(
                filetypes=[("Зображення", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
            root.destroy()
        except Exception as e:
            print("Не вдалося відкрити діалог:", e)
            return
        if not path:
            return
        self.path = path
        self.img = load_gray(path)
        self.codec = Codec(self.img, self.levels)
        for ax in (self.ax_orig, self.ax_dct, self.ax_dwt, self.ax_map_dct, self.ax_map_dwt):
            ax.clear(); ax.set_xticks([]); ax.set_yticks([])
        self._draw_static()
        self.update()

    def on_save(self, _):
        os.makedirs(RESULTS, exist_ok=True)
        tag = f"{self.mode}_{self.value:.3g}_L{self.levels}"
        self.fig.savefig(os.path.join(RESULTS, f"screen_{tag}.png"), dpi=110)
        Image.fromarray(self.rec_dct).save(os.path.join(RESULTS, f"dct_{tag}.png"))
        Image.fromarray(self.rec_dwt).save(os.path.join(RESULTS, f"dwt_{tag}.png"))
        print("Збережено в", RESULTS)

    # -------------------------------------------------------- головне ----
    def update(self):
        rd, cd, pd, sd = self.codec.run_dct(self.mode, self.value)
        rw, cw, pw, sw = self.codec.run_dwt(self.mode, self.value)
        self.rec_dct, self.rec_dwt = rd, rw

        self.im_dct.set_data(rd)
        self.im_dwt.set_data(rw)
        fmt = lambda p: "∞" if p == float("inf") else f"{p:.2f}"
        self.ax_dct.set_title(f"DCT 8×8   PSNR = {fmt(pd)} дБ\nкоеф.: {sd['percent']:.2f} %  (CR ≈ {sd['cr']:.1f}:1)")
        self.ax_dwt.set_title(f"DWT Хаар, {self.levels} рівн.   PSNR = {fmt(pw)} дБ\nкоеф.: {sw['percent']:.2f} %  (CR ≈ {sw['cr']:.1f}:1)")

        if self.show_error:
            ed = np.abs(self.img.astype(float) - rd)
            ew = np.abs(self.img.astype(float) - rw)
            vmax = max(ed.max(), ew.max(), 1)
            for im, data in ((self.im_map_dct, ed), (self.im_map_dwt, ew)):
                im.set_data(data); im.set_cmap("inferno"); im.set_clim(0, vmax)
            self.ax_map_dct.set_title("|Original − DCT|  (карта помилки)")
            self.ax_map_dwt.set_title("|Original − DWT|  (карта помилки)")
        else:
            md, mw = coeff_view(cd), coeff_view(cw)
            vmax = max(md.max(), mw.max())
            for im, data in ((self.im_map_dct, md), (self.im_map_dwt, mw)):
                im.set_data(data); im.set_cmap("magma"); im.set_clim(0, vmax)
            self.ax_map_dct.set_title("Коефіцієнти DCT (блоки 8×8), log(1+|c|)")
            self.ax_map_dwt.set_title("Коефіцієнти DWT (піраміда), log(1+|c|)")

        unit = UNITS[self.mode]
        self.slider_label.set_text(f"{MODES[self.mode].split(',')[0]}: {self.value:.3g}{unit}")
        self.pt_dct.set_data([sd["percent"]], [min(pd, 60)])
        self.pt_dwt.set_data([sw["percent"]], [min(pw, 60)])
        self.info.set_text(
            f"Файл: {os.path.basename(self.path)}\n"
            f"Всього коефіцієнтів: {sd['total']}\n\n"
            f"           DCT        DWT\n"
            f"ненульові {sd['nonzero']:>8}  {sw['nonzero']:>8}\n"
            f"частка,%  {sd['percent']:>8.2f}  {sw['percent']:>8.2f}\n"
            f"CR        {sd['cr']:>8.1f}  {sw['cr']:>8.1f}\n"
            f"PSNR,дБ   {fmt(pd):>8}  {fmt(pw):>8}\n"
            f"\nCR = усі / ненульові коеф."
        )
        self.fig.canvas.draw_idle()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "cameraman.png")
    App(path)
    plt.show()


if __name__ == "__main__":
    main()
