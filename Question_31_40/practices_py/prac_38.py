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

def quantization(F):
  H,W,C = F.shape
  
  Q = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
              (12, 12, 14, 19, 26, 58, 60, 55),
              (14, 13, 16, 24, 40, 57, 69, 56),
              (14, 17, 22, 29, 51, 87, 80, 62),
              (18, 22, 37, 56, 68, 109, 103, 77),
              (24, 35, 55, 64, 81, 104, 113, 92),
              (49, 64, 78, 87, 103, 121, 120, 101),
              (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)
  
  for ys in range(0,H,T):
    for xs in range(0,W,T):
      for c in range(channel):
        # roundで四捨五入してからQをかける
        F[ys: ys + T, xs: xs + T, c] =  np.round(F[ys: ys + T, xs: xs + T, c] / Q) * Q
      
  
  return F

def MSE(img1,img2):
  H,W,C = img1.shape
  mse = np.sum((img1-img2)**2)/(H*W*C)
  return mse

def PSNR(mse,vmax=255):
  # 対数の底はnp.log10と設定する
  return 10*np.log10(vmax**2/mse)

def bitrate():
  return 8*K**2/T**2

# Read image
img = cv2.imread("Question_31_40\imori.jpg").astype(np.float32)
# F = bgr2gray(img)
F = dct(img)
out = idct(F)
mse = MSE(img,out)

psnr = PSNR(mse)
bitrate = bitrate()

print("MSE:", type(mse))
print("PSNR:", type(psnr))
print("bitrate:", type(bitrate))

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
