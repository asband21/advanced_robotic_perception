import cv2
import numpy as np


# Open picture
img_1 = cv2.imread("img2.jpg")
img_2 = img_1.copy()
img_3 = img_1.copy()
img_4 = img_1.copy()


# 1. Using nested loops
for i in range(img_2.shape[0]):
    for j in range(img_2.shape[1]):
        ll = img_2[i,j,0]/3 + img_2[i,j,1]/3 + img_2[i,j,2]/3
        img_2[i,j,0] = ll
        img_2[i,j,1] = ll
        img_2[i,j,2] = ll


# 2. Using matrix operations with numpy (Python)
img_3 = img_3/3 
img_3 = np.sum(img_3, axis=2, dtype=np.uint8)


# 3 .Using the built-in OpenCV function
img_4 = cv2.cvtColor(img_4, cv2.COLOR_BGR2GRAY)


# show img's
cv2.imshow("img1", img_1)
cv2.imshow("img2", img_2)
cv2.imshow("img3", img_3)
cv2.imshow("img4", img_4)

cv2.waitKey(0)
