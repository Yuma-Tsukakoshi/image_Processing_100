import cv2 
import numpy as np

def BGR2GRAY(img):
  b = img[:,:,0].copy()
  g = img[:,:,1].copy()
  r = img[:,:,2].copy()

  out = 0.2126*r + 0.7152*g + 0.0722*b
  out = out.astype(np.uint8)  
  return out

#read_img
img = cv2.imread('geo_wave.jpg')
img2 = img.copy()
img = BGR2GRAY(img)

# Harrisコーナー検出を実行
"""
2: コーナー検出の際に考慮する近傍領域のサイズこの場合は2×2ピクセルの領域。
3: Sobelの勾配オペレータのカーネルサイズ。この場合は3×3ピクセルのカーネルを使用。
0.04: Harris検出器のフリーパラメータ。この値が大きいと、コーナーと判定される点が少なくなる。
"""
dst = cv2.cornerHarris(img, 2, 3, 0.04)

# コーナーの閾値を設定
thresh = 0.01 * dst.max()

# コーナーに赤い丸を描く
# y,xの順番
for i in range(dst.shape[0]):
    for j in range(dst.shape[1]):
        if dst[i,j] > thresh:
            # 半径1の赤い丸を描く
            cv2.circle(img, (j,i), 1, (0,0,255), -1)


#result_img
cv2.imwrite('result_img/geo_wave_result.jpg',img)
cv2.imshow('Corners',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
