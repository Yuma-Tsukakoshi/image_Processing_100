import cv2
import numpy as np

T = 8
K = 8
channel = 3

# bgr -> gray
# def bgr2gray(img):
#     gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
#     return gray
def BGR2YCbCr(img):
    # rgb => YCrCbへ変換
    YCrCb = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)    
    return YCrCb

def YCbCr2BGR(img):
    # YCrCb => bgrへ変換
    bgr = cv2.cvtColor(img,cv2.COLOR_YCrCb2BGR)
    return bgr

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
                #roundで四捨五入してからQをかける
                F[ys: ys + T, xs: xs + T, c] =  np.round(F[ys: ys + T, xs: xs + T, c] / Q) * Q
    return F

def JPEG(img):
    # BGR -> YCbCr
    YCbCr = BGR2YCbCr(img)

    # DCT
    F = dct(YCbCr)

    # DCT係数を量子化
    F = quantization(F)

    # IDCT
    YCbCr = idct(F)

    # Y Cb Cr -> BGR
    out = YCbCr2BGR(YCbCr)

    return out

# Read image
img = cv2.imread("Question_31_40\imori.jpg").astype(np.float32)
# F = bgr2gray(img)
out = JPEG(img)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
