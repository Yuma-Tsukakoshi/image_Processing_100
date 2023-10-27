import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("Question_51_60\seg.png")
# グレースケールに変換する。
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 2値化する
ret, bin_img = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

n_labels, out = cv2.connectedComponents(bin_img,connectivity=4)

# Save result
print("number of labels:", n_labels)
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(out)
cv2.waitKey(0)
