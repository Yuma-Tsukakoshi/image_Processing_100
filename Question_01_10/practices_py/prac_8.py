import cv2 
import numpy as np

def max_pooling(_img,G):
  img = _img.copy()
  
  H,W,C = img.shape
  Nh = int(H/G) # 128/8 = 16
  Nw = int(W/G) # 128/8 = 16
  
  for y in range(Nh):
    for x in range(Nw):
      for c in range(C):
        # スライスで切り出して平均値を全て対象要素に代入
        img[y*G:(y+1)*G,x*G:(x+1)*G,c] = np.max(img[y*G:(y+1)*G,x*G:(x+1)*G,c]).astype(np.int)
        
  
  return img

#read_img
img = cv2.imread('Question_01_10\imori.jpg')
out = max_pooling(img,8)

#result_img
# cv2.imwrite('answers_image/answer4.jpg',img2)
cv2.imshow('result',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
