import cv2
import numpy as np

# Open picture
img_1 = cv2.imread("img2.jpg", cv2.IMREAD_GRAYSCALE)

storlse = 5

kerner  = np.ones([storlse,storlse])
kerner = kerner/kerner.sum()
 
image = cv2.filter2D(img_1,-1,kerner)

# Display the picture
cv2.imshow("Our window", img_1)
cv2.imshow("smoof", image)
cv2.waitKey(0)
