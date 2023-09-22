import cv2 
import numpy as np

def BGR2GRAY(img):
  b = img[:,:,0].copy()
  g = img[:,:,1].copy()
  r = img[:,:,2].copy()

  out = 0.2126*r + 0.7152*g + 0.0722*b
  out = out.astype(np.uint8)  
  return out

# 式の導出からSbを最大にする閾値を求める
def otsu_binary_scale(img, th=128):
  max_th = 0
  max_Sb = 0

  for _t in range(1,255):
    # v0とv1の値が結構重要なポイントとなる
    # 閾値となりうるのが0~255となり一つ一つ試す
    v0 = img[np.where(img < _t)]
    v1 = img[np.where(img >= _t)]
    # ↑ np.whereによって条件に合う要素のインデックスを返し、それをimgに適用することで条件に合う要素を返す
    
    m0 = np.mean(v0) 
    m1 = np.mean(v1)
    w0 = len(v0) / H*W
    w1 = len(v1) / H*W
    Sb = w0*w1*((m0-m1)**2)  
    if Sb > max_Sb:
      max_Sb = Sb
      max_th = _t

  #二値化
  print("threshold: ",max_th)
  th = max_th
  out[out < th] = 0
  out[out >= th] = 255
  return out

#read_img
img = cv2.imread('imori.jpg').astype(np.float32)
H,W,C = img.shape
out = BGR2GRAY(img)
out = otsu_binary_scale(out)

#result_img
cv2.imshow('',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
