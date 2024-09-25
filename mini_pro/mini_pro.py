import cv2
import numpy as np

gr = 10

# Open picture
img_0 = cv2.imread("data/delivery_area_1l/im0.png" , cv2.IMREAD_GRAYSCALE)
img_1 = cv2.imread("data/delivery_area_1l/im1.png" , cv2.IMREAD_GRAYSCALE)



cv2.imshow("img 0", img_0)
cv2.imshow("img 1", img_1)



cv2.waitKey(0)

