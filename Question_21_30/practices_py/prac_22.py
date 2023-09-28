import cv2
import numpy as np

def gaussian(img, K_size=3, sigma=1.3):
  
  H, W, C = img.shape

  ## Zero padding
  pad = K_size // 2
  out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float) #上下左右に1pxずつ0でpadding⇒×2
  out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)  
  
  # answer ⇒ diagで対角行列を作成
  # Kernel
  K = np.diag( [1] * K_size ).astype(np.float)
  K /= K_size
  
  # my_ans
  ## prepare Kernel
  # K = np.zeros((K_size, K_size), dtype=np.float)
  # for x in range(-pad, -pad + K_size):
  #   for y in range(-pad, -pad + K_size):
  #     if x==y:
  #       K[y + pad, x + pad] = 1/3
  #     else:
  #       K[y + pad, x + pad] = 0
      
  tmp = out.copy()  
  
  # filtering
  for y in range(H):
    for x in range(W):
      for c in range(C):
        out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])  
  

  out = out[pad: pad + H, pad: pad + W].astype(np.uint8)  
  return out

#read_img
img = cv2.imread('Question_11_20\imori.jpg')
out = gaussian(img, K_size=3, sigma=1.3)

#result_img
# cv2.imwrite('answers_image/answer4.jpg',img2)
cv2.imshow('result',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
