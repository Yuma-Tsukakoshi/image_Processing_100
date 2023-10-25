import cv2
import numpy as np

img = cv2.imread('Question_31_40\imori.jpg')

def skew(img,dx,dy):
      # 画像の高さ、幅、色を取得
    H, W = img.shape[:2]
    # xy座標をnp.float32型(3点分のxy座標)
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    # 変換後の3点の座標
    dest[:,0] += (dx / W * (src[:,1])).astype(np.float32)
    dest[:,1] += (dy / H * (src[:,0])).astype(np.float32)

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
    return cv2.warpAffine(img, affine, (W+dx, H+dy))

#result_img
# cv2.imwrite('answers_image/answer4.jpg',img2)
cv2.imshow('result', skew(img,30,30))
cv2.waitKey(0)
cv2.destroyAllWindows()
