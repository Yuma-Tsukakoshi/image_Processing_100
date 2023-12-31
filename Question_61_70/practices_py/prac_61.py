import cv2
import numpy as np

# Read image
img = cv2.imread("Question_61_70\\renketsu.png")

def connect_4(img):
  
  H,W,C = img.shape
  
  # 一時的な画像を生成
  tmp = np.zeros((H,W),dtype=np.int)
  
  # 二値化 imgで0より大きいところを1にする
  tmp[img[...,0]>0] = 1
  
  out = np.zeros((H,W,3),dtype=np.uint8)
  
  for y in range(H):
    for x in range(W):
      if tmp[y,x] < 1 :
        # 黒の時はスルー
        continue
      
      S = 0
      S += (tmp[y,min(x+1,W-1)] - tmp[y,min(x+1,W-1)] * tmp[max(y-1,0),min(x+1,W-1)] * tmp[max(y-1,0),x])
      S += (tmp[max(y-1,0),x] - tmp[max(y-1,0),x] * tmp[max(y-1,0),max(x-1,0)] * tmp[y,max(x-1,0)])
      S += (tmp[y,max(x-1,0)] - tmp[y,max(x-1,0)] * tmp[min(y+1,H-1),max(x-1,0)] * tmp[min(y+1,H-1),x])
      S += (tmp[min(y+1,H-1),x] - tmp[min(y+1,H-1),x] * tmp[min(y+1,H-1),min(x+1,W-1)] * tmp[y,min(x+1,W-1)])
      
      if S == 0: #内部 blue
        out[y,x] = [0, 0, 255]
      elif S == 1: #端点 green
          out[y,x] = [0, 255, 0]
      elif S == 2: #連結点 red
          out[y,x] = [255, 0, 0]
      elif S == 3: #分岐点 yellow
          out[y,x] = [255, 255, 0]
      elif S == 4: #交差点 purple
          out[y,x] = [255, 0, 255]
                    
    out = out.astype(np.uint8)
  
  return out

out = connect_4(img)
# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
