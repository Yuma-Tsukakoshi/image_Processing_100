import cv2
import numpy as np

# Read image
img = cv2.imread("Question_71_80\imori.jpg")

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
    # out = Canny(img)
    out = dilate(img,n)
    out = erode(out,n)
    return out

def opening(img,n):
    out = erode(img,n)
    out = dilate(out,n)
    return out

out = closing(img,5)
# gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

# 大津の手法
# ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
out = opening(out,5)

# make mask
def get_mask(hsv): 
    mask = hsv.copy() 
    mask[np.logical_and((hsv[..., 0] >= 90), (hsv[..., 0] <= 150))] = 0 
    return mask

# RGB > HSV
hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)

# color tracking
mask = get_mask(hsv)
out = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
# out = mask.astype(np.uint8)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
