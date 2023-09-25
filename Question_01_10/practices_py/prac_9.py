import cv2 
import numpy as np

def gaussian(_img):
  K = 1/16 * np.array([[1,2,1],[2,4,2],[1,2,1]])
  img = _img.copy()
  # 周辺1行分を0パディング
  
  H,W,C = img.shape
  print(H,W,C)
  
  img = np.pad(img,1)
  # for y in range(H):
  #   for x in range(W):
  #     for c in range(C):
  #       # スライスで切り出して平均値を全て対象要素に代入
  #       img[y:(y+1),x:(x+1),c] = np.max(img[y:(y+1),x:(x+1),c]).astype(np.int)
        
  
  return img

#read_img
img = cv2.imread('Question_01_10\imori_noise.jpg')
out = gaussian(img,K_size=3, sigma=1.3)

#result_img
# cv2.imwrite('answers_image/answer4.jpg',img2)
cv2.imshow('result',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
