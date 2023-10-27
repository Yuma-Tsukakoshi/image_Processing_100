import cv2
import numpy as np

# Read image
img1 = cv2.imread("Question_51_60\imori.jpg").astype(np.float32)
img2 = cv2.imread("Question_51_60\\thorino.jpg").astype(np.float32)

# alpha blend
def alpha_blend(img1, img2, alpha):
	# blend
	out = img1 * alpha + img2 * (1 - alpha)
	out = out.astype(np.uint8)
	return out

out = alpha_blend(img1, img2, 0.6)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
