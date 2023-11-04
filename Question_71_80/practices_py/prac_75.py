import cv2
import numpy as np

# Read image
img = cv2.imread("Question_71_80\imori.jpg")
H,W,C = img.shape

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# resizeで正しい引数をする !!
for i in range(6):
    out = cv2.resize(gray, None, fx=1/2**i, fy=1/2**i)
    cv2.imshow(f'result_{i}', out)

cv2.waitKey(0)
cv2.destroyAllWindows()
