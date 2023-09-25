import cv2 
import numpy as np

def BGR2GRAY(img):
  b = img[:,:,0].copy()
  g = img[:,:,1].copy()
  r = img[:,:,2].copy()

  out = 0.2126*r + 0.7152*g + 0.0722*b
  out = out.astype(np.uint8)  
  return out

def binary_scale(img, th=128):
  img[img < th] = 0
  img[img >= th] = 255
  return img

#read_img
img = cv2.imread('Question_01_10\imori.jpg')
img2 = BGR2GRAY(img)
img2 = binary_scale(img2)

#result_img
cv2.imwrite('answers_image/answer3.jpg',img2)
cv2.imshow('',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
