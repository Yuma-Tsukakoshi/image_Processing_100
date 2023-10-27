import cv2
import numpy as np

# Read image
img = cv2.imread("Question_41_50\imori.jpg")

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# Canny
def Canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = cv2.Canny(gray, 240,240)
    return out 

def erode(img,n):
  return cv2.erode(img, kernel, iterations=n)

def dilate(img,n):
  return cv2.dilate(img, kernel, iterations=n)

def closing(img,n):
    out = Canny(img)
    out = dilate(out,n)
    out = erode(out,n)
    return out

out = closing(img,1)


# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
