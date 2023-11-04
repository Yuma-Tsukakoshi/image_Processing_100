import cv2
import numpy as np

# Read image
img = cv2.imread("Question_71_80\imori.jpg")
H,W,C = img.shape

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# resizeで正しい引数をする !!
dic = {}
for i in range(6):
    out = cv2.resize(gray, None, fx=1/2**i, fy=1/2**i)
    out = cv2.resize(out, None, fx=2**i, fy=2**i)
    dic[i] = out

sum = np.zeros((H,W), dtype=np.uint8)
sum += (dic[0] - dic[1])
sum += (dic[0] - dic[3])
sum += (dic[0] - dic[5])
sum += (dic[1] - dic[4])
sum += (dic[2] - dic[3])
sum += (dic[3] - dic[5])
np.clip(sum, 0, 255)

cv2.imshow('result', sum)
cv2.waitKey(0)
cv2.destroyAllWindows()
