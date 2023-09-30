# matlplotlib はgooglecolobで用いて使うことにする
# 画像のimg.showはvscodeでやる

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("Question_21_30\imori.jpg")

# histogram normalization
def hist_normalization(img, z_max = 255):
  out = img.copy()
  H,W,C = out.shape
  
  S = H* W * C  
  
  sum_h = 0
  
  # np.whereで条件に合う要素のインデックスを返してその長さを取得することで頻度を求めることができる上手い！
  for i in range(1,255):
    ind = np.where(img == i)
    sum_h += len(img[ind])
    z_prime = z_max / S * sum_h
    out[ind] = z_prime
  
  out = out.astype(np.uint8)
  return out

# histogram normalization
out = hist_normalization(img)

# Display histogram
plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.show()

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
