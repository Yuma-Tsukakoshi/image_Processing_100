# matlplotlib はgooglecolobで用いて使うことにする
# 画像のimg.showはvscodeでやる

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("Question_21_30\imori_dark.jpg")

# histogram normalization
def hist_normalization(img, m0=128, s0=52):
  out = img.copy()
  
  m = np.mean(out)
  s = np.std(out)
  
  out = s0 * (out - m ) / s + m0
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
