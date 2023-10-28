import cv2
import numpy as np

# Read image
img = cv2.imread("Question_61_70\imori.jpg")

# alpha blend
def track(img):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  H = hsv[..., 0]
  # hsvのhの値が180<=h<=260となる部分を抽出して、RGBに直すときにその抽出した部分のみ255にする。
  mask = np.zeros_like(H)
  mask[(H >= 180) & (H <= 260)] = 255
  rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
  gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

  gray[mask == 255] = 255
  gray[mask == 0] = 0
  out = gray.astype(np.uint8) / 255.
  
  return out

out = track(img)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
