import cv2

def RGB2BGR(img):
  b = img[:,:,0].copy()
  g = img[:,:,1].copy()
  r = img[:,:,2].copy()
  
  # RGB > BGR
  img[:,:,0] = r
  img[:,:,1] = g
  img[:,:,2] = b
  
  return img

#read_img
img = cv2.imread('imori.jpg')
img2 = RGB2BGR(img)


cv2.imwrite('answers_image/answer1.jpg',img2)
cv2.imshow('',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
