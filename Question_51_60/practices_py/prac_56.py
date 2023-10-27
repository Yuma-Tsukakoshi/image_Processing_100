import cv2
import numpy as np
# from matplotlib import pyplot as plt

img = cv2.imread('Question_41_50\\thorino.jpg')

def canny(img):
      # 画像の高さ、幅、色を取得
    dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = cv2.Canny(dst, 120,120)
    return out    

#result_img
# cv2.imwrite('answers_image/answer4.jpg',img2)
edges = canny(img)
# 検出された直線の一覧の形状がndarrayで返される
lines =cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=75)

def draw_line(img, theta, rho):
    h, w = img.shape[:2]
    if np.isclose(np.sin(theta), 0):
        x1, y1 = rho, 0
        x2, y2 = rho, h
    else:
        calc_y = lambda x: rho / np.sin(theta) - x * np.cos(theta) / np.sin(theta)
        x1, y1 = 0, calc_y(0)
        x2, y2 = w, calc_y(w)

    # float -> int
    x1, y1, x2, y2 = list(map(int, [x1, y1, x2, y2]))

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


# 直線を描画する。
if lines is not None:
    for rho, theta in lines.squeeze(axis=1):
        draw_line(img, theta, rho)
        
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

