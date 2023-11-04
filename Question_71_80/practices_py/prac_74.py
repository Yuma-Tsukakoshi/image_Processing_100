import cv2
import numpy as np

# Read image
img = cv2.imread("Question_71_80\imori.jpg")
H,W,C = img.shape

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = gray.copy()
# resizeで正しい引数をする !!
erode = cv2.resize(gray, None, fx=0.5, fy=0.5)
out = cv2.resize(erode, None, fx=2, fy=2)
out = img2-out
np.clip(out, 0, 255)
out = out*255/out.max()
out.astype(np.uint8)
# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
