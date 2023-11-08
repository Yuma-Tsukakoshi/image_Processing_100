import numpy as np

# [x1, y1, x2, y2] x1,y1...矩形の左上のx,y  x2,y2...矩形の右下のx,y
a = np.array((50, 50, 150, 150), dtype=np.float32)
b = np.array((60, 60, 170, 160), dtype=np.float32)

def iou(a,b):
  R1 = (a[2] - a[0]) * (a[3] - a[1])
  R2 = (b[2] - b[0]) * (b[3] - b[1])
  RoI_array = np.array((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])), dtype=np.float32)
  RoI = (RoI_array[2] - RoI_array[0]) * (RoI_array[3] - RoI_array[1])
  iou = RoI / (R1 + R2 -RoI)
  
  return iou

result = iou(a,b)
print(result)
