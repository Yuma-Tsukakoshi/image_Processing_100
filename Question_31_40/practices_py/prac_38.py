import cv2
import numpy as np


#read_img
img = cv2.imread('Question_21_30\imori.jpg')
H, W, C = img.shape
dx =30
dy =-30
M = np.array([[1, 0, dx], [0, 1, dy]], dtype=float)
out = cv2.warpAffine(img, M, (W,H))

#result_img
# cv2.imwrite('answers_image/answer4.jpg',img2)
cv2.imshow('result',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
