import cv2
import numpy as np

# Read image
img = cv2.imread("Question_81_90\\thorino.jpg")
def Harris(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Harrisコーナー検出を実行
  """
  2: コーナー検出の際に考慮する近傍領域のサイズこの場合は2×2ピクセルの領域。
  3: Sobelの勾配オペレータのカーネルサイズ。この場合は3×3ピクセルのカーネルを使用。
  0.04: Harris検出器のフリーパラメータ。この値が大きいと、コーナーと判定される点が少なくなる。
  """
  dst = cv2.cornerHarris(gray, 2, 3, 0.04)

  # コーナーの閾値を設定
  thresh = 0.01 * dst.max()

  # コーナーに赤い丸を描く
  # y,xの順番
  for i in range(dst.shape[0]):
      for j in range(dst.shape[1]):
          if dst[i,j] > thresh:
              # 半径1の赤い丸を描く
              cv2.circle(img, (j,i), 1, (0,0,255), -1)
  
  return img

out = Harris(img)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
