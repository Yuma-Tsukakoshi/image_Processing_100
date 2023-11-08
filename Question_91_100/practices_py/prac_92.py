import cv2
import numpy as np

# K-means
def k_means(img, Class=5):
    #  get shape
    H, W, C = img.shape
    np.random.seed(0) 
    img = np.reshape(img, (H * W, -1))
    i = np.random.choice(np.arange(H * W), Class, replace=False)
    Cs = img[i].copy()
    print(Cs)
    # 上記まででランダムに5つのインデックスを取得している
    
    
    clss = np.zeros((H * W), dtype=int)

    # each pixel
    for i in range(H * W):
    # get distance from base pixel
        dis = np.sqrt(np.sum((Cs - img[i]) ** 2, axis=1))
    # get argmin distanc
        clss[i] = np.argmin(dis)

    # show
    out = np.reshape(clss, (H, W)) * 50
    out = out.astype(np.uint8)

    return out


# read image
img = cv2.imread("Question_91_100\imori.jpg").astype(np.float32)

out = k_means(img)

cv2.imshow("result", out)
cv2.waitKey(0)
