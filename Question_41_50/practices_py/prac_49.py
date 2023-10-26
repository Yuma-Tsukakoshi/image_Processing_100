import cv2
import numpy as np

def change_yellow(img,y):
    # rgb => YCrCbへ変換
    YCrCb = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    YCrCb[...,0] = YCrCb[...,0]*y
    
    # YCrCb => bgrへ変換
    out = cv2.cvtColor(YCrCb,cv2.COLOR_YCrCb2BGR)
    return out 

# Read image
img = cv2.imread("Question_31_40\imori.jpg").astype(np.float32)
out = change_yellow(img,y=0.7)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
