import numpy as np
import cv2


np.random.seed(0)
img = cv2.imread("Question_91_100\imori_1.jpg")
H,W,C = img.shape

gt = (47, 41, 129, 103)
def iou(a,b):
    R1 = (a[2] - a[0]) * (a[3] - a[1])
    R2 = (b[2] - b[0]) * (b[3] - b[1])
    RoI_array = np.array((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])), dtype=np.float32)
    RoI = (RoI_array[2] - RoI_array[0]) * (RoI_array[3] - RoI_array[1])
    iou = RoI / (R1 + R2 -RoI)
    
    return iou

def make_rectangle(gt,img):
    
    for i in range(200):
        x1 = np.random.randint(W-60)
        y1 = np.random.randint(H-60)
        x2 = x1 + 60
        y2 = y1 + 60
        
        a = (x1,y1,x2,y2)
        
        result = iou(a,gt)
        if result >= 0.5:
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),1)
        else:
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),1)
            
    cv2.rectangle(img,(gt[0],gt[1]),(gt[2],gt[3]),(0,255,0),1)
    return img

out = make_rectangle(gt,img)

cv2.imshow('result',out)
cv2.waitKey(0)
