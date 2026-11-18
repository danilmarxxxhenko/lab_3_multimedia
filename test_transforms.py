"""
Перевірка коректності власних перетворень.
Запуск:  python test_transforms.py

SciPy використовується ТІЛЬКИ тут, як еталон для порівняння
(у самій програмі його немає). Якщо SciPy не встановлено — цей тест пропускається.
"""
import numpy as np

from transforms import (C8, dct2_blocks, idct2_blocks, haar_matrix,
                        dwt2_haar, idwt2_haar)
from lossy import psnr, to_uint8

rng = np.random.default_rng(0)
img = rng.integers(0, 256, (64, 64)).astype(np.float64)

# 1. Матриці ортонормовані: M · Mᵀ = I
assert np.allclose(C8 @ C8.T, np.eye(8)), "DCT-матриця не ортонормована"
assert np.allclose(haar_matrix(16) @ haar_matrix(16).T, np.eye(16)), "Haar не ортонормована"
print("✓ C·Cᵀ = I та H·Hᵀ = I")

# 2. Ідеальне відновлення без стиснення
assert np.allclose(idct2_blocks(dct2_blocks(img)), img)
for L in (1, 2, 3, 4):
    assert np.allclose(idwt2_haar(dwt2_haar(img, L), L), img)
print("✓ IDCT(DCT(x)) = x,  IDWT(DWT(x)) = x  (рівні 1–4)")

# 3. Рівність Парсеваля: енергія зберігається
e = np.sum(img ** 2)
assert np.isclose(np.sum(dct2_blocks(img) ** 2), e)
assert np.isclose(np.sum(dwt2_haar(img, 3) ** 2), e)
print("✓ Енергія зберігається (Парсеваль)")

# 4. PSNR без втрат = inf
assert psnr(img, to_uint8(idct2_blocks(dct2_blocks(img)))) == float("inf")
print("✓ PSNR без стиснення = ∞")

# 5. Порівняння з еталонним SciPy DCT
try:
    from scipy.fft import dctn
    block = img[:8, :8]
    ref = dctn(block, type=2, norm="ortho")
    assert np.allclose(dct2_blocks(img)[0, 0], ref)
    print("✓ Збігається з scipy.fft.dctn(norm='ortho')")
except ImportError:
    print("- SciPy не встановлено, порівняння з еталоном пропущено")

print("Усі тести пройдено.")
