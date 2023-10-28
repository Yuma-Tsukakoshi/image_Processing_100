import cv2
import numpy as np
# from matplotlib import pyplot as plt

img = cv2.imread('Question_41_50\imori.jpg')

def hog(img):
    H,W,C = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_pad = np.pad(gray,(1,1),'edge')
    
    gx = img_pad[1:H+1,2:] - img_pad[1:H+1, :W]
    gy = img_pad[2:,1:W+1] - img_pad[:H, 1:W+1]
    gx[gx == 0] = 1e-6
    
    mag = np.sqrt(gx**2 + gy**2) / 255.
    ang = np.arctan(gy / gx)
    
    # quantization 量子化する
    ang_quantized = np.zeros_like(ang, dtype=np.int)
    
    
    
#result
cv2.imshow('result', )
cv2.waitKey(0)
cv2.destroyAllWindows()

