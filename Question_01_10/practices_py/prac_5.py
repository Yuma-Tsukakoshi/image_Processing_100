import cv2 
import numpy as np

def BGR2HSV(img):
  B = img[:,:,0].copy()/255
  G = img[:,:,1].copy()/255
  R = img[:,:,2].copy()/255

  Max = np.max(R,G,B)
  Min = np.min(R,G,B)
  
  if Min==Max:
    H = 0
  elif Min==B:
    H = 60*(G-R) / (Max-Min) + 60 
  elif Min==R:
    60*(B-G) / (Max-Min) + 180
  elif Min==G:
    60*(R-B) / (Max-Min) + 300 
  
  V = Max
  S = Max - Min
  
  return H,S,V

def HSV2BGR(H,S,V):
  C = S
  H = H/60
  X = C*(1-np.abs(H%2-1))
  
  if 0 <= H and H < 1:
    param = (C,X,0)
  elif H < 2:
    param = (X,C,0)
  elif H < 3:
    param = (0,C,X)
  elif H < 4:
    param = (0,X,C)
  elif H < 5:
    param = (X,0,C)
  elif H < 6:
    param = (C,0,X)
  else:
    param = (0,0,0)
  
  R,G,B = ((V-C)*(1,1,1) + param) * 255
  
  
#read_img
img = cv2.imread('imori.jpg')
H,S,V = BGR2HSV(img)
img2 = HSV2BGR(H+180,S,V)

#result_img
# cv2.imwrite('answers_image/answer4.jpg',img2)
cv2.imshow('',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
