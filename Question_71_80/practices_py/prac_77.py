import cv2
import numpy as np

# Read image
img = cv2.imread("Question_71_80\imori.jpg")
H,W,C = img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
out = cv2.getGaborKernel(ksize=(111,111),sigma=10,theta=45,lambd=10,gamma=1.2,psi=0)
"""
パラメータ：
ksize:カーネルのサイズ
sigma:標準偏差
theta:平行線に対する法線の向き
lambd:正弦波因子の波長
gamma:空間アスペクト比
psi:位相オフセット
ktype: フィルタ係数の種類
"""

cv2.imshow('result', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
