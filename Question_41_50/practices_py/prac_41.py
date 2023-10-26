import cv2
import numpy as np

img = cv2.imread('Question_31_40\imori.jpg')

def canny(img):
      # 画像の高さ、幅、色を取得
    dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = cv2.Canny(dst, 120,120)
    return out    

#result_img
# cv2.imwrite('answers_image/answer4.jpg',img2)
cv2.imshow('result', canny(img))
cv2.waitKey(0)
cv2.destroyAllWindows()
