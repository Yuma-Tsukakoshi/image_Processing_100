import cv2
import numpy as np


def affine_rotate(img, theta, cx, cy):
    # 画像の高さ、幅
    H, W = img.shape[:2]
    cx = W/2
    cy = H/2
    # xy座標をnp.float32型
    # 画像の中心, theta, scale
    M = cv2.getRotationMatrix2D((cx, cy),theta, 1)
    """
    アフィン変換
    cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])
    第一引数に元画像（NumPy配列ndarray）、
    第二引数に2 x 3の変換行列（NumPy配列ndarray）、
    第三引数に出力画像のサイズ（タプル）を指定する。
    """
    return cv2.warpAffine(img, M, (W, H))

img = cv2.imread('Question_21_30/imori.jpg')
# アフィン変換で拡大
out = affine_rotate(img, theta=30, cx=0, cy=0)

# スケール変換後のサイズにリサイズする必要ある
#result_img
cv2.imshow('result',out)
# cv2.imshow('result',out2)
cv2.waitKey(0)
cv2.destroyAllWindows()
