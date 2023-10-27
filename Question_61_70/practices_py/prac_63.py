import cv2
import numpy as np
from skimage.morphology import skeletonize

# Read image
img = cv2.imread("Question_61_70\gazo.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 大津の手法
ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

#細線化(スケルトン化)
ske = skeletonize(~(img != 0))
ske_gray = (ske * 255).astype(np.uint8)
dst = cv2.cvtColor(ske_gray, cv2.COLOR_GRAY2RGB)

# Save result
cv2.imshow("result", dst)
cv2.waitKey(0)
