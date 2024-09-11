import cv2
import numpy as np

# Open picture
img_1 = cv2.imread("img2.jpg", cv2.IMREAD_GRAYSCALE)
for i in img_1:
    for j in i:
        print(j)
cv2.waitKey(0)
