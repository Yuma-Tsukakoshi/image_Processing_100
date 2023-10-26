import cv2
import numpy as np

img = cv2.imread('Question_31_40\imori.jpg')

# 2次元離散フーリエ変換 (DFT)のハイパーパラメーター
K, L = 128, 128
channel = 3

def dft(img):
    # 画像の高さ、幅を取得
    H, W = img.shape[:2]

    # DFT係数(np.complex: 複素数を扱うための型)
    G = np.zeros((L, K, channel), dtype=np.complex)

    # 元のイメージと一致(NumPy配列ndarrayをタイル状に繰り返し並べるnp.tile)
    x = np.tile(np.arange(W), (H, 1)) 
    y = np.arange(H).repeat(W).reshape(H, -1)

    # DFTの計算(周波数成分G)
    for c in range(channel):
        for l in range(L):
            for k in range(K):
                G[l, k, c] = np.sum(img[..., c] * np.exp(-2j * np.pi * (x * k / K + y * l / L))) / np.sqrt(K * L)

    return G

# IDFT(逆二次元離散フーリエ変換)
def idft(G):
    """
    逆二次元離散フーリエ変換 (IDFT)
    params
    --------------------------------------
    param: 周波数成分G

    returns
    ------------------------------
    numpy.ndarray形式のimage
    """
    # 画像の高さ、幅を取得
    H, W = G.shape[:2]
    # DFT係数(np.complex: 複素数を扱うための型)
    out = np.zeros((H, W, channel), dtype=np.float32)

    # 元のイメージと一致(NumPy配列ndarrayをタイル状に繰り返し並べるnp.tile)
    x = np.tile(np.arange(W), (H, 1))     
    y = np.arange(H).repeat(W).reshape(H, -1)
    """
    x = [0 1 2 ・・・・ 125 126 127],
    ↓   [0 1 2 ・・・・ 125 126 127],
    128 ...
        [0 1 2 ・・・・ 125 126 127]
    
    y = [0 0 0 0 →128個],
    ↓   [1,1,1,1],
    128 ...
        [127 127 127・・・]
    """

    # IDFT(逆二次元離散フーリエ変換)
    for c in range(channel):
        for l in range(H):
            for k in range(W):
                out[l, k, c] = np.abs(np.sum(G[..., c] * np.exp(2j * np.pi * (x * k / W + y * l / H)))) / np.sqrt(W * H)

    # clipping(NumPy配列ndarrayを任意の最小値・最大値に収めるclip)
    # 0 ~ 255の範囲
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out

# 2次元離散フーリエ変換
G = dft(img)
ps = (np.abs(G) / np.abs(G).max() * 255).astype(np.uint8)
# 逆二次元離散フーリエ変換
out = idft(G)

#result_img
# cv2.imwrite('answers_image/answer4.jpg',img2)
cv2.imshow('result', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
