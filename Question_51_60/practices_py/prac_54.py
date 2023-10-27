import cv2
import numpy as np

# Read image
img = cv2.imread("Question_51_60\imori.jpg")
temp = cv2.imread("Question_51_60\imori_part.jpg")

result = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)

print("img.shape", img.shape) 
print("template.shape", temp.shape)  
print("result.shape", result.shape)  

# 類似度が最も高い場所を探す
# minVal: 類似度の最小値, maxVal: 類似度の最大値, minLoc: 類似度が最小の位置, maxLoc: 類似度が最大の位置

minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
# rectangleの左上の座標を取得する。 -> position
print(f"max value: {maxVal}, position: {maxLoc}")

# 描画する。 tl: top left , br: bottom right
tl = maxLoc[0], maxLoc[1]
br = maxLoc[0] + temp.shape[1], maxLoc[1] + temp.shape[0]
print(tl)
print(br)

dst = img.copy()
# 左上と右下の座標を指定して、矩形を描画する。
cv2.rectangle(dst, tl, br, color=(0, 0, 255), thickness=2) # bgr

# Save result
cv2.imshow("result", dst)
cv2.waitKey(0)
