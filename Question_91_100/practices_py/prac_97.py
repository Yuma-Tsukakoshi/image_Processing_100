import cv2
import numpy as np

# 1画像の上から順に、左から右に走査していく。
img = cv2.imread("Question_91_100\imori_many.jpg")


# 2各画像の位置に対して、注目位置を中心に複数の矩形を形成する

# 3それぞれの矩形に対する画像を切り抜いて、特徴抽出(HOG,SIFT)を行う
def hog(img):
    def BGR2GRAY(img):
        gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
        return gray
    
    #輝度勾配を求める
    def get_gradXY(gray):
        H,W = gray.shape
        gray = np.pad(gray,(1,1),'edge') # 端の画素で1px拡張
        
        # g[y,x] スライスして2個先の画素との差分を取っている
        # 効率的！先にずらしておいて配列の差分を取る
        gx = gray[1:H+1,2:] - gray[1:H+1,:W]
        gy = gray[2:,1:W+1] - gray[:H,1:W+1]
        gx[gx==0] = 1e-6 # 0除算を防ぐため
        
        return gx, gy
    
    # 輝度勾配から、勾配強度と勾配角度を求める
    def get_MagGrad(gx,gy):
        mag = np.sqrt(gx**2 + gy**2)
        ang = np.arctan( gy / gx )
        
        # 角度が負の値を取った場合の処理 0~180度に変換するため
        # ex) -45 => 135 ex) -45 + 180
        ang[ang<0] = ang[ang<0] + np.pi
        
        return mag, ang
    
    # 勾配角度を [0,180]で9分割した値に量子化
    def quantization(ang):
        # 画像を9分割した際の角度の範囲
        # zeros_like => angの配列と同じ形状で要素が0の配列を作成
        ang_quantized = np.zeros_like(ang, dtype=np.int64)
        d = np.pi/9
        for i in range(9):
            ang_quantized[np.where((ang>=d*i)&(ang<=d*(i+1)))] = i
        return ang_quantized
    
    def gradient_histogram(ang_quantizied, mag, N=8):
        H,W = mag.shape
        
        # 一つの領域のサイズ
        cell_H = H // N
        cell_W = W // N
        histogram = np.zeros((cell_H,cell_W,9), dtype=np.float32)
        
        for y in range(cell_H): # 8*8の領域の個数分行う
            for x in range(cell_W):
                for j in range(N): #8*8の領域
                    for i in range(N): 
                        histogram[y,x,ang_quantizied[y*4+j,x*4+i]] += mag[y*4+j,x*4+i]
        
        return histogram
    
    def normalization(histogram,C=3,epsilon=1):
        cell_H, cell_W, _ = histogram.shape
        for y in range(cell_H):
            for x in range(cell_W):
                histogram[y,x] /= np.sqrt(np.sum(histogram[max(y-1,0):min(y+2,cell_H),max(x-1,0):min(x+2,cell_W)]**2)+epsilon)
        return histogram
    
    # 画像をグレースケールに変換
    gray = BGR2GRAY(img)
    gx,gy = get_gradXY(gray)
    mag, ang = get_MagGrad(gx,gy)
    ang_quantized = quantization(ang)
    histogram = gradient_histogram(ang_quantized, mag)
    histogram = normalization(histogram)
    
    return histogram
    
def iou(a,b):
    R1 = (a[2] - a[0]) * (a[3] - a[1])
    R2 = (b[2] - b[0]) * (b[3] - b[1])
    RoI_array = np.array((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])), dtype=np.float32)
    RoI = (RoI_array[2] - RoI_array[0]) * (RoI_array[3] - RoI_array[1])
    iou = RoI / (R1 + R2 -RoI)
    
    return iou

def sliding_window(img,H_size=32):
    H,W,C = img.shape
    recs = np.array(((42, 42), (56, 56), (70, 70)), dtype=np.float32)
    for y in range(0,H,4):
        for x in range(0,W,4):
            for rec in recs:
                # get half size of ractangle
                dh = int(rec[0] // 2)
                dw = int(rec[1] // 2)

                # get left top x
                x1 = max(x - dw, 0)
                # get left top y
                x2 = min(x + dw, W)
                # get right bottom x
                y1 = max(y - dh, 0)
                # get right bottom y
                y2 = min(y + dh, H)

                # crop region
                region = img[max(y - dh, 0) : min(y + dh, H), max(x - dw, 0) : min(x + dw, W)]
                
                region = cv2.resize(region,None,fx=H_size/region.shape[0],fy=H_size/region.shape[1])
                
                # 1次元配列に変換
                region_hog = hog(region).ravel()
                
                
sliding_window(img)


