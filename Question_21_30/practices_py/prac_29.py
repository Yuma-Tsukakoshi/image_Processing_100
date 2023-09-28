import cv2
import numpy as np

def BGR2GRAY(img):
    out = 0.2126*img[:,:,2] + 0.7152*img[:,:,1] + 0.0722*img[:,:,0]
    out = out.astype(np.uint8)
    return out

def log_filter(img, K_size=5, sigma=3.0):
  
  H, W = img.shape

  ## Zero padding
  pad = K_size // 2
  out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float) #上下左右に2pxずつ0でpadding⇒×4
  out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)  
  
  ## prepare Kernel
  K = np.zeros((K_size, K_size), dtype=np.float)
  for x in range(-pad, -pad + K_size):
    for y in range(-pad, -pad + K_size):
      K[y + pad, x + pad] = (x ** 2 + y ** 2 - sigma ** 2) * np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
      
  K /= (2 * np.pi * sigma ** 6)
  # K.sum()で割るのは、ガウシアンフィルタの総和が1になるようにするため⇒重みの正規化
  K /= K.sum()  
  tmp = out.copy()  
  
  # filtering
  for y in range(H):
    for x in range(W):
        out[pad + y, pad + x] = np.sum(K * tmp[y: y + K_size, x: x + K_size])  

  out = np.clip(out, 0, 255)
  out = out[pad: pad + H, pad: pad + W].astype(np.uint8)  
  return out

#read_img
img = cv2.imread('Question_01_10\imori_noise.jpg')
out = BGR2GRAY(img)
out = log_filter(out, K_size=5, sigma=3.0)

#result_img
# cv2.imwrite('answers_image/answer4.jpg',img2)
cv2.imshow('result',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
