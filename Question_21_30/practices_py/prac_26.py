import cv2
import numpy as np

def bi_linear(img, ax=1, ay=1):
  # 1.5倍にスケールアップしている
  H,W,C = img.shape
  
  # スケールアップ後のサイズ
  aH = int(ay * H)
  aW = int(ax * W)
  
  # リサイズ後の画像の座標を与える
  y = np.arange(aH).repeat(aW).reshape(aW,-1)
  x = np.tile(np.arange(aW), (aH,1))
  
  # スケールアップ前の座標を与える  
  y = (y / ay)
  x = (x / ax)

  ix= np.floor(x).astype(np.int)
  iy = np.floor(y).astype(np.int)
  
  # なぜ行っている？
  ix = np.minimum(ix, W-2)
  iy = np.minimum(iy, H-2)
  
  # それぞれの画素と距離dを求める、その後w=d/sum(d)で重みを求める
  dx = x - ix
  dy = y - iy
  
  dx = np.repeat(np.expand_dims(dx, axis=-1), 3, axis=-1)
  dy = np.repeat(np.expand_dims(dy,axis=-1),3,axis=-1)
  
  out = (1-dx)*(1-dy)*img[iy,ix]+ dx*(1-dy)*img[iy,ix+1] + (1-dx)*dy*img[iy+1,ix] + dx*dy*img[iy+1,ix+1]
  
  out = np.clip(out, 0, 255)
  out = out.astype(np.uint8)
  
  return out

# Read image
img = cv2.imread("Question_21_30\imori.jpg").astype(np.float)

# 1.5倍に拡大する関数
out = bi_linear(img, ax=1.5, ay=1.5)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
