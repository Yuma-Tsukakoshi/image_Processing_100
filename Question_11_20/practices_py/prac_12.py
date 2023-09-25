import cv2 
import numpy as np

def RGB2GRAY(img):
  b = img[:,:,0].copy()
  g = img[:,:,1].copy()
  r = img[:,:,2].copy()

  # MY_ANSWER #
  # H,W,C = img.shape
  # for h in range(H):
  #   for w in range(W):
  #     for c in range(C):
  #       img[h,w,c] = 0.2126*r[h,w] + 0.7152*g[h,w] + 0.0722*b[h,w]
  
  # 各画素に対して、RGBの値をgrayscaleに変換した結果を足し合わせている
  out = 0.2126*r + 0.7152*g + 0.0722*b
  # 整数に変換している理由は、処理の高速化のため
  out = out.astype(np.uint8)
    
  return out

#read_img
img = cv2.imread('imori.jpg')
img2 = RGB2GRAY(img)

#result_img
cv2.imwrite('answers_image/answer2.jpg',img2)
cv2.imshow('',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
