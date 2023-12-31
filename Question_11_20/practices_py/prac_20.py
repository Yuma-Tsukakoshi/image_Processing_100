import cv2
import numpy as np
import matplotlib.pyplot as plt
# matplotlibを使う場合はターミナル上から実行すること！！

# Read image
img = cv2.imread("Question_11_20\imori_dark.jpg").astype(np.float)

# img.ravel()は、画像を1次元の配列に変換するメソッドです。これにより、画像のピクセル値を一列に並べることができます。
# binsは、ヒストグラムのビンの数を指定します。今回は、0~255の256階調を指定しています。
# rwidthは、ビンの幅を指定します。今回は、0.8としています。
# rangeは、ビンの範囲を指定します。今回は、0~255の256階調を指定しています。 

# 一次元かしてるから自動で各画素値が対象のビンに入る⇒すご！
# Display histogram
plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
# plt.savefig("out.png")
plt.show()

# my_ans 間違い！
# def show_graph(img):
#   H, W, C = img.shape
#   data = np.zeros((256))
#   for y in range(H):
#     for x in range(W):
#       for c in range(C):
#         data[img[y,x,c]] += 1
#   return data

# img = cv2.imread('Question_11_20\imori_dark.jpg')  
# out = show_graph(img)
# plt.hist(out, bins=255, rwidth=0.8)
# plt.show()

