import cv2 
import numpy as np

def reduce_color(_img):
  img = _img.copy()
  
  for i in range(4):
    for j in range(3):
      val = img[...,j]  
      ind = np.where((i * (4**3) <= val ) & (val < (i+1) * (4**3)))
      img[...,j][ind] = 32 + i * 4**3 
  
  return img

#read_img
img = cv2.imread('imori.jpg')
out = reduce_color(img)

#result_img
# cv2.imwrite('answers_image/answer4.jpg',img2)
cv2.imshow('result',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
