import cv2
import numpy as np
from matplotlib import pyplot as plt

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
    
    ang[ang < 0] = np.pi / 2 + ang[ang < 0] + np.pi / 2
    
    # quantization 量子化する
    ang_quantized = np.zeros_like(ang, dtype=np.int)
    
    # quantization base
    d = np.pi / 9

    # quantization
    for i in range(9):
        ang_quantized[np.where((ang >= d * i) & (ang <= d * (i + 1)))] = i
    
    return mag, ang_quantized


# get gradient histogram
def gradient_histogram(gradient_quantized, magnitude, N=8):
    # get shape
    H, W = magnitude.shape
    # get cell num
    cell_N_H = H // N
    cell_N_W = W // N
    histogram = np.zeros((cell_N_H, cell_N_W, 9), dtype=np.float32)
    # each pixel
    for y in range(cell_N_H):
        for x in range(cell_N_W):
            for j in range(N):
                for i in range(N):
                    histogram[y, x, gradient_quantized[y * 4 + j, x * 4 + i]] += magnitude[y * 4 + j, x * 4 + i]
                    
    return histogram

mag , ang_quantized = hog(img)

# Write gradient magnitude to file
_mag = (mag / mag.max() * 255).astype(np.uint8)

cv2.imshow("out_mag.jpg", _mag)

# Write gradient angle to file
H, W, C = img.shape
out = np.zeros((H, W, 3), dtype=np.uint8)

# define color
C = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
    [127, 127, 0], [127, 0, 127], [0, 127, 127]]

# draw color
for i in range(9):
    out[ang_quantized == i] = C[i]

# 4. Gradient histogram
histogram = gradient_histogram(ang_quantized, mag)


# write histogram to file
for i in range(9):
    plt.subplot(3,3,i+1)
    # plt.imshow(histogram[..., i])
    plt.axis('off')
    plt.xticks(color="None")
    plt.yticks(color="None")

# plt.show()
#result
cv2.imshow('result', out)
cv2.waitKey(0)
cv2.destroyAllWindows()

