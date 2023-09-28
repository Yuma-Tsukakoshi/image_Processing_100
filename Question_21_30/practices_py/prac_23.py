import cv2
import numpy as np

def BGR2GRAY(img):
  b = img[:,:,0].copy()
  g = img[:,:,1].copy()
  r = img[:,:,2].copy()

  out = 0.2126*r + 0.7152*g + 0.0722*b
  out = out.astype(np.uint8)  
  return out

def max_min(img, K_size=3):
  
  H, W= img.shape

  ## Zero padding
  pad = K_size // 2
  out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float) #上下左右に1pxずつ0でpadding⇒×2
  out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)  
  
  tmp = out.copy()  
  
  # filtering
  for y in range(H):
    for x in range(W):
      out[pad + y, pad + x] = np.max(tmp[y: y + K_size, x: x + K_size ]) - np.min(tmp[y: y + K_size, x: x + K_size])  

  out = np.clip(out, 0, 255)
  out = out[pad: pad + H, pad: pad + W].astype(np.uint8)  
  return out

#read_img
img = cv2.imread('Question_01_10\imori.jpg')
out = BGR2GRAY(img)
out = max_min(out, K_size=3)

#result_img
# cv2.imwrite('answers_image/answer4.jpg',img2)
cv2.imshow('result',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
