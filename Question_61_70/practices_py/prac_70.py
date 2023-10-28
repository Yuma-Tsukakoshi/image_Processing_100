import cv2
import numpy as np

# Read image
img = cv2.imread("Question_61_70\imori.jpg")

# make mask
def get_mask(hsv): 
  mask = np.zeros_like(hsv[..., 0]) 
  mask[np.logical_and((hsv[..., 0] >= 90), (hsv[..., 0] <= 150))] = 255 
  return mask

# RGB > HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# color tracking
mask = get_mask(hsv)

out = mask.astype(np.uint8)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
