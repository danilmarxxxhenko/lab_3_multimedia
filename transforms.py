"""
Власна реалізація перетворень для ЛР3.

  * DCT-II 8x8 у матричній формі:      Y = C · X · Cᵀ,   X = Cᵀ · Y · C
  * Дискретне вейвлет-перетворення Хаара (теж через матрицю):
                                        Y = H · X · Hᵀ,   X = Hᵀ · Y · H

Готові dct()/idct()/dwt()/idwt() НЕ використовуються — тільки NumPy
для множення матриць.
"""
import numpy as np

BLOCK = 8


# ---------------------------------------------------------------- DCT ----
def dct_matrix(n: int = BLOCK) -> np.ndarray:
    """Ортонормована матриця DCT-II розміру n×n.

    C[k, i] = α(k) · cos((2i + 1)·k·π / 2n),
    α(0) = √(1/n),  α(k>0) = √(2/n).
    Рядок k — це k-та базисна косинусоїда (k = 0 — постійна складова, DC).
    """
    C = np.zeros((n, n))
    for k in range(n):
        alpha = np.sqrt(1.0 / n) if k == 0 else np.sqrt(2.0 / n)
        for i in range(n):
            C[k, i] = alpha * np.cos((2 * i + 1) * k * np.pi / (2 * n))
    return C


C8 = dct_matrix(BLOCK)


def to_blocks(img: np.ndarray, b: int = BLOCK) -> np.ndarray:
    """(H, W) -> (H/b, W/b, b, b): масив блоків b×b."""
    h, w = img.shape
    return img.reshape(h // b, b, w // b, b).swapaxes(1, 2)


def from_blocks(blocks: np.ndarray) -> np.ndarray:
    """(H/b, W/b, b, b) -> (H, W): склеювання блоків назад у зображення."""
    nh, nw, b, _ = blocks.shape
    return blocks.swapaxes(1, 2).reshape(nh * b, nw * b)


def dct2_blocks(img: np.ndarray) -> np.ndarray:
    """Пряме 2D DCT для кожного блоку 8×8:  Y = C · X · Cᵀ.
    Оператор @ застосовується до всіх блоків одразу (broadcasting)."""
    X = to_blocks(img.astype(np.float64))
    return C8 @ X @ C8.T


def idct2_blocks(coeffs: np.ndarray) -> np.ndarray:
    """Обернене 2D DCT:  X = Cᵀ · Y · C  (бо C ортогональна: C⁻¹ = Cᵀ)."""
    return from_blocks(C8.T @ coeffs @ C8)


# ----------------------------------------------------------- Haar DWT ----
def haar_matrix(n: int) -> np.ndarray:
    """Ортонормована матриця одного рівня перетворення Хаара (n — парне).

    Перші n/2 рядків — «середні» (низькі частоти):  (x[2i] + x[2i+1]) / √2
    Останні n/2 рядків — «різниці» (високі частоти): (x[2i] − x[2i+1]) / √2
    """
    H = np.zeros((n, n))
    s = 1.0 / np.sqrt(2.0)
    half = n // 2
    for i in range(half):
        H[i, 2 * i] = s
        H[i, 2 * i + 1] = s
        H[half + i, 2 * i] = s
        H[half + i, 2 * i + 1] = -s
    return H


def dwt2_haar(img: np.ndarray, levels: int = 3) -> np.ndarray:
    """Пряме багаторівневе 2D перетворення Хаара.

    На кожному рівні перетворюємо лише ліву верхню (LL) частину:
        LL ← Hr · LL · Hcᵀ
    Після одного рівня матриця ділиться на 4 квадранти:
        LL (наближення) | LH
        ----------------+---
        HL              | HH  (деталі: горизонтальні, вертикальні, діагональні)
    """
    X = img.astype(np.float64).copy()
    h, w = X.shape
    for _ in range(levels):
        Hr, Hc = haar_matrix(h), haar_matrix(w)
        X[:h, :w] = Hr @ X[:h, :w] @ Hc.T
        h //= 2
        w //= 2
    return X


def idwt2_haar(coeffs: np.ndarray, levels: int = 3) -> np.ndarray:
    """Обернене перетворення Хаара: від найглибшого рівня до першого,
        LL ← Hrᵀ · LL · Hc."""
    X = coeffs.astype(np.float64).copy()
    H, W = X.shape
    h, w = H >> (levels - 1), W >> (levels - 1)
    for _ in range(levels):
        Hr, Hc = haar_matrix(h), haar_matrix(w)
        X[:h, :w] = Hr.T @ X[:h, :w] @ Hc
        h *= 2
        w *= 2
    return X
