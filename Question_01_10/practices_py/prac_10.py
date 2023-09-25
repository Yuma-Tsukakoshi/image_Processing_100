import cv2
import numpy as np

def gaussian(img, K_size=3, sigma=1.3):
  
  H, W, C = img.shape

  ## Zero padding
  pad = K_size // 2
  out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float) #上下左右に1pxずつ0でpadding⇒×2
  out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)  

  # 更新時に常に元の画像を参照するようにする
  tmp = out.copy()  
  
  # filtering
  for y in range(H):
    for x in range(W):
      for c in range(C):
        out[pad + y, pad + x, c] = np.median(tmp[y: y + K_size, x: x + K_size, c])  

  out = out[pad: pad + H, pad: pad + W].astype(np.uint8)  
  return out

#read_img
img = cv2.imread('Question_01_10\imori_noise.jpg')
out = gaussian(img, K_size=3, sigma=1.3)

#result_img
# cv2.imwrite('answers_image/answer4.jpg',img2)
cv2.imshow('result',out)
cv2.waitKey(0)
cv2.destroyAllWindows()

