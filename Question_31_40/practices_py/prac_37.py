import cv2
import numpy as np

def bc_interpolate(img, ax=1, ay=1):
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
  iy= np.floor(y).astype(np.int)
  
  # 2つの値の内小さいほうを取る ix,iyは元画像の座標を表している
  # 書かれてることは ix => x , iy =>y
  ix = np.minimum(ix, W-1)
  iy = np.minimum(iy, H-1)
  
  # 画像の座標の差分を求める
  dx2 = x - ix
  dy2 = y - iy
  dx1 = dx2 + 1
  dy1 = dy2 + 1
  dx3 = 1 - dx2
  dy3 = 1 - dy2
  dx4 = 1 + dx3
  dy4 = 1 + dy3
  
  dxs = [dx1, dx2, dx3, dx4]
  dys = [dy1, dy2, dy3, dy4]
  
  def weight(t):
    # もうif文で書くのではなくwhere文でindを取得して処理を走らせる
    # if t <= 1:
    #   return (a+2) * t**3 - (a+3) * t**2 + 1
    # elif 1<t and t<=2:
    #   return a * t**3 - 5*a * t**2 + 8*a * t - 4*a
    # else:
    #   return 0
    a=-1
    at = np.abs(t)
    w = np.zeros_like(at)
    ind = np.where(at <= 1)
    w[ind] = ((a+2) * np.power(at,3) - (a+3) * np.power(at,2) + 1)[ind]
    ind = np.where((at>1) & (at<=2))
    w[ind] = (a*np.power(at, 3) - 5*a*np.power(at, 2) + 8*a*at - 4*a)[ind]
    return w
  
  # どっちもゼロ行列を作成してる
  w_sum = np.zeros((aH,aW,C), dtype=np.float32)
  out = np.zeros((aH,aW,C), dtype=np.float32)
  
  #interpolate
  for j in range(-1,3):
    for i in range(-1,3):
      # インデックスが0からW-1の範囲内に収まるように調整
      ind_x = np.minimum(np.maximum(ix + i, 0), W-1) #x+i-2
      ind_y = np.minimum(np.maximum(iy + j, 0), H-1) #y+j-2
      
      wx = weight(dxs[i+1])
      wy = weight(dys[j+1])
      wx = np.repeat(np.expand_dims(wx, axis=-1), 3, axis=-1)
      wy = np.repeat(np.expand_dims(wy, axis=-1), 3, axis=-1)
      
      w_sum += wx * wy
      out += wx * wy * img[ind_y, ind_x]
      
  out /= w_sum
  out = np.clip(out, 0, 255)
  out = out.astype(np.uint8)

  return out

# Read image
img = cv2.imread("Question_21_30\imori.jpg").astype(np.float)

# 1.5倍に拡大する関数
out = bc_interpolate(img, ax=1.5, ay=1.5)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
