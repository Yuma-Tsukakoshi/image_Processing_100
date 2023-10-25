import cv2
import numpy as np

# scope
def scope(img, ax=1, ay=1):
  H,W,C = img.shape
  
  # スケールアップ後のサイズ
  aH = int(ay * H)
  aW = int(ax * W)
  
  y = np.arange(aH).repeat(aW).reshape(aH,-1)
  x = np.tile(np.arange(aW), (aH,1))
  y = np.round(y / ay).astype(np.int)
  
  # なぜout = img[y,x]でスケール後の画像が得られるのか？
  out = img[y,x]
  out = out.astype(np.uint8)
  return out

# Read image
img = cv2.imread("Question_21_30\imori.jpg").astype(np.float)

# 1.5倍に拡大する関数
out = scope(img, ax=1.5, ay=1.5)

# Save result
# cv2.imshow("result", out)
cv2.waitKey(0)
