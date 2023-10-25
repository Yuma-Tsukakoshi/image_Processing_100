import cv2
import numpy as np


def affine_expand(img, ratio_x, ratio_y):
    """
    アフィン変換で拡大

    params
    -------------------------------
    param1: numpy.ndarray形式のimage
    param2: x方向の比率
    param3: y方向の比率

    returns
    -------------------------------
    numpy.ndarray形式のimage
    """
    # 画像の高さ、幅
    H, W = img.shape[:2]
    # xy座標をnp.float32型
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    # x, yそれぞれ比率をかける
    dest = src.copy()
    dest[:,0] *= ratio_x
    dest[:,1] *= ratio_y
    """
    アフィン変換の変換行列を生成: cv2.getAffineTransform(src, dest)
    src: 変換前の3点の座標
    dest: 変換後の3点の座標をNumPy配列ndarrayで指定
    """
    affine = cv2.getAffineTransform(src, dest)
    """
    アフィン変換
    cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])
    第一引数に元画像（NumPy配列ndarray）、
    第二引数に2 x 3の変換行列（NumPy配列ndarray）、
    第三引数に出力画像のサイズ（タプル）を指定する。
    """
    return cv2.warpAffine(img, affine, (int(W*ratio_x), int(H*ratio_y))) 

img = cv2.imread('Question_21_30/imori.jpg')
# アフィン変換で拡大
out = affine_expand(img, 1.3, 0.8)
# 平行移動[[1,0,横方向への移動量],[0,1,縦方向への移動量]]の2x3行列
H, W = out.shape[:2]
M = np.float64([[1, 0, 30], [0,1,-30]])
out2 = cv2.warpAffine(out, M, (W, H))

# スケール変換後のサイズにリサイズする必要ある
#result_img
cv2.imshow('result',out)
cv2.imshow('result',out2)
cv2.waitKey(0)
cv2.destroyAllWindows()
