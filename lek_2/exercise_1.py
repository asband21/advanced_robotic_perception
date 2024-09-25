import cv2
import numpy as np

# Open picture
img_1 = cv2.imread("aau-city-1.jpg")
img_2 = cv2.imread("aau-city-2.jpg")

#img_1 = np.float32(img_1)
#img_2 = np.float32(img_2)

sz = img_1.shape

pash = 6
r = int(pash/2)

sob_x  = np.array([-1, 0, 1])
sob_y  = np.array([[1, 0, -1]])

ix = cv2.filter2D(img_1,-1, sob_x)
iy = cv2.filter2D(img_1,-1, sob_y)

for i in range(sz[0] - pash):
    for j in range(sz[1] - pash):
        bil_ix = ix[i:i+pash, j:j+pash]
        bil_iy = iy[i:i+pash, j:j+pash]
        cov_vec = np.array([bil_ix.flatten(), bil_iy.flatten()])
        cov = np.cov(cov_vec)
        print(cov)

ixx = 0
ixy = 0
iyx = 0
iyy = 0

#for i in range(sz[0]):
#    for j in range(sz[1]):


cv2.imshow("img 1", img_1)
cv2.imshow("img 1 sob x",  ix)
cv2.imshow("img 1 sob y",  iy)
cv2.imshow("img 2", img_2)

cv2.waitKey(0)

