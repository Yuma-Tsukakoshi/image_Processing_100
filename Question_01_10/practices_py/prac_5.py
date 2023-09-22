import cv2 
import numpy as np

def BGR2HSV(_img):
  # _imgとするのはimgと区別するため
  img = _img.copy() / 255.
  
  # 0で初期化した配列を作成 
  hsv = np.zeros_like(img, dtype=np.float32)

  # axis = 2 はチャンネル方向で考えている  
  Max = np.max(img, axis=2).copy()
  Min = np.min(img, axis=2).copy()
  Min_arg = np.argmin(img, axis=2) #インデックスを返す

  # H RGBの値を変換するため[...,'0']として最後の次元を参照する
  hsv[...,0][np.where(Max==Min)] = 0
  
  # [..., 1] は、NumPyのスライシング記法 最後の次元を指定する
  #if Min_arg== B:
  # indは特定のminの対象となるインデックスを返す⇒指定したチャンネルの中でminが0(B)の所を処理する
  ind = np.where(Min_arg == 0)
  hsv[...,0][ind] =  60*(img[...,1][ind]-img[...,2][ind]) / (Max[ind]-Min[ind]) + 60
  #if Min_arg== R:
  ind = np.where(Min_arg == 2)
  hsv[...,0][ind] = 60*(img[...,0][ind]-img[...,1][ind]) / (Max[ind]-Min[ind]) + 180
  #if Min_arg== G:
  ind = np.where(Min_arg== 1)
  hsv[...,0][ind] = 60*(img[...,2][ind]-img[...,0][ind]) / (Max[ind]-Min[ind]) + 300
  
  # copyしないと値が変わってしまうのかな？
  hsv[...,1] = Max.copy() - Min.copy()
  hsv[...,2] = Max.copy()
  
  return hsv

def HSV2BGR(_img, hsv):
  img = _img.copy() / 255.

  Max = np.max(img, axis=2).copy()
  Min = np.min(img, axis=2).copy()

  out = np.zeros_like(img)
  
  H = hsv[...,0]
  S = hsv[...,1]
  V = hsv[...,2]

  C = S
  H_ = H/60.
  X = C*(1-np.abs(H_%2-1))
  Z = np.zeros_like(H)
  
  params = [[Z,X,C],[Z,C,X],[X,C,Z],[C,X,Z],[C,Z,X],[X,Z,C]]
  
  # indexの指定方法を慎重に行う必要がある
  for i in range(6):
    ind = np.where((i<=H_) & (H_ < (i+1)))
    out[...,0][ind] = (V-C)[ind] + params[i][0][ind]
    out[...,1][ind] = (V-C)[ind] + params[i][1][ind]
    out[...,2][ind] = (V-C)[ind] + params[i][2][ind]
  
  out[np.where(Max==Min)] = 0
  # 0~1の範囲に収める
  out = np.clip(out,0,1)
  # RGBの値が0~1のため255をかけてuint8に変換
  out = (out*255).astype(np.uint8)
  
  return out

#read_img
img = cv2.imread('imori.jpg').astype(np.float32)
hsv = BGR2HSV(img)
hsv[...,0] = (hsv[...,0] + 180 ) % 360

out = HSV2BGR(img, hsv)

#result_img
# cv2.imwrite('answers_image/answer4.jpg',img2)
cv2.imshow('result',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
