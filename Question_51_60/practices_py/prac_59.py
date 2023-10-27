import cv2
import numpy as np

# Read image
img = cv2.imread("Question_41_50\imori.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 大津の手法
ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

def erode(img,n):
  return cv2.erode(img, kernel, iterations=n)

def dilate(img,n):
  return cv2.dilate(img, kernel, iterations=n)

def opening(img,n):
  out = erode(img,n)
  out = dilate(out,n)
  return out

out = opening(bin_img,1)


# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
