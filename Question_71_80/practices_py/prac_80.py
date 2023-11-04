import cv2
import numpy as np

# Read image
img = cv2.imread("Question_71_80\imori.jpg")
H,W,C = img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


sum = np.zeros((H,W), dtype=np.uint8)
for i in range(4):
    img_ = gray.copy()
    filter = cv2.getGaborKernel(ksize=(11,11),sigma=1.5,theta=45*i,lambd=3,gamma=1.2,psi=0)
    out = cv2.filter2D(img_,ddepth=-1,kernel=filter)
    sum += out

np.clip(sum, 0, 255)
cv2.imshow('result', sum)
cv2.waitKey(0)
cv2.destroyAllWindows()
