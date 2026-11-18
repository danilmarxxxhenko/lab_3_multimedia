"""
Механізми зменшення кількості значущих коефіцієнтів + метрики якості.
Працюють однаково для DCT і DWT (обидва перетворення ортонормовані,
тому коефіцієнти мають однаковий масштаб енергії).
"""
import numpy as np

from transforms import dct2_blocks, idct2_blocks, dwt2_haar, idwt2_haar

MODES = {
    "fraction": "Частка найбільших коеф., %",
    "threshold": "Поріг |c| < T → 0",
    "quant": "Квантування з кроком q",
}


# ------------------------------------------------- обробка коефіцієнтів ----
def keep_largest(coeffs: np.ndarray, percent: float) -> np.ndarray:
    """Залишає percent % коефіцієнтів з найбільшим |c|, решту обнуляє."""
    flat = np.abs(coeffs).ravel()
    k = int(round(flat.size * percent / 100.0))
    k = min(max(k, 1), flat.size)
    if k == flat.size:
        return coeffs.copy()
    thr = np.partition(flat, flat.size - k)[flat.size - k]   # k-те найбільше
    return np.where(np.abs(coeffs) >= thr, coeffs, 0.0)


def hard_threshold(coeffs: np.ndarray, T: float) -> np.ndarray:
    """Жорсткий поріг: |c| < T → 0."""
    return np.where(np.abs(coeffs) >= T, coeffs, 0.0)


def quantize(coeffs: np.ndarray, q: float) -> np.ndarray:
    """Рівномірне квантування: c → round(c / q) · q.
    Малі коефіцієнти (|c| < q/2) стають нулями, решта округлюються."""
    return np.round(coeffs / q) * q


def compress_coeffs(coeffs: np.ndarray, mode: str, value: float) -> np.ndarray:
    if mode == "fraction":
        return keep_largest(coeffs, value)
    if mode == "threshold":
        return hard_threshold(coeffs, value)
    if mode == "quant":
        return quantize(coeffs, value)
    raise ValueError(mode)


# ------------------------------------------------------------- метрики ----
def psnr(original: np.ndarray, restored: np.ndarray, peak: float = 255.0) -> float:
    """PSNR = 10·log10(MAX² / MSE), дБ."""
    mse = np.mean((original.astype(np.float64) - restored.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(peak ** 2 / mse)


def coeff_stats(coeffs: np.ndarray) -> dict:
    """Кількість ненульових коефіцієнтів, їх частка та коефіцієнт стиснення.

    CR = (загальна кількість коефіцієнтів) / (кількість ненульових).
    Це оцінка «за кількістю коефіцієнтів», а не реальний розмір файлу:
    для реального файлу ще потрібно кодувати позиції нулів і самі значення
    (RLE + Хаффман/арифметичне кодування, як у JPEG).
    """
    total = coeffs.size
    nz = int(np.count_nonzero(coeffs))
    return {
        "total": total,
        "nonzero": nz,
        "percent": 100.0 * nz / total,
        "cr": total / nz if nz else float("inf"),
    }


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


# ------------------------------------------------------ повний конвеєр ----
class Codec:
    """Прямі перетворення рахуються один раз для зображення,
    далі для кожного рівня стиснення — лише обробка коефіцієнтів + обернене."""

    def __init__(self, img: np.ndarray, levels: int = 3):
        self.img = img
        self.levels = levels
        self.dct = dct2_blocks(img)
        self.dwt = dwt2_haar(img, levels)

    def set_levels(self, levels: int):
        self.levels = levels
        self.dwt = dwt2_haar(self.img, levels)

    def run_dct(self, mode, value):
        c = compress_coeffs(self.dct, mode, value)
        rec = to_uint8(idct2_blocks(c))
        return rec, c, psnr(self.img, rec), coeff_stats(c)

    def run_dwt(self, mode, value):
        c = compress_coeffs(self.dwt, mode, value)
        rec = to_uint8(idwt2_haar(c, self.levels))
        return rec, c, psnr(self.img, rec), coeff_stats(c)
