# matlplotlib はgooglecolobで用いて使うことにする
# 画像のimg.showはvscodeでやる

import cv2
import numpy as np
# import matplotlib.pyplot as plt

# Read image
img = cv2.imread("Question_21_30\imori_dark.jpg")

# histogram normalization
def hist_normalization(img, a=0, b=255):
	# get max and min
	c = img.min()
	d = img.max()

	out = img.copy()

	# normalization
	out = (b-a) / (d - c) * (out - c) + a
	out[out < a] = a
	out[out > b] = b
	out = out.astype(np.uint8)
	
	return out

# histogram normalization
out = hist_normalization(img)

# Display histogram
# plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
# plt.show()

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
