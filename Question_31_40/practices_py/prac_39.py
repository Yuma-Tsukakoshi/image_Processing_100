import cv2
import numpy as np

T = 8
K = 8
channel = 3

# bgr -> gray
# def bgr2gray(img):
#     gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
#     return gray

def w(x,y,u,v):
  cu = 1
  cv = 1
  if u==0:
    cu = 1/np.sqrt(2)
  if v==0:
    cv = 1/np.sqrt(2)
  theta = np.pi/(2*T)
  
  return cu*cv*(2/T)*np.cos((2*x+1)*u*theta)*np.cos((2*y+1)*v*theta)

# このweightを使ってDCTを計算する
def dct(img):
  H,W,C = img.shape
  F = np.zeros((H,W,channel),dtype=np.float32)
  for c in range(channel):
    # 縦横8x8のブロックに分けて処理する⇒ステップを8マスずつ進める
    for yi in range(0,H,T):
      for xi in range(0,W,T):
        for v in range(T):
          for u in range(T):
            for y in range(T):
              for x in range(T):
                # 各画素に対して離散コサイン変換を行う
                # yi xi がブロックの左上の座標
                # y x がブロック内の座標
                F[v+yi,u+xi,c] += img[y+yi,x+xi,c]*w(x,y,u,v)
  return F

def idct(F):
  H,W,C = F.shape
  out = np.zeros((H,W,channel),dtype=np.float32)
  for c in range(channel):
    # 縦横8x8のブロックに分けて処理する⇒ステップを8マスずつ進める
    for yi in range(0,H,T):
      for xi in range(0,W,T):
        for v in range(T):
          for u in range(T):
            for y in range(T):
              for x in range(T):
                # 各画素に対して離散コサイン変換を行う
                # yi xi がブロックの左上の座標
                # y x がブロック内の座標
                out[y+yi,x+xi,c] += F[v+yi,u+xi,c]*w(x,y,u,v)
                
  out = np.clip(out,0,255)
  out = np.round(out).astype(np.uint8)
  
  return out

# Read image
img = cv2.imread("Question_31_40\imori.jpg").astype(np.float32)
# F = bgr2gray(img)
F = dct(img)
out = idct(F)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
