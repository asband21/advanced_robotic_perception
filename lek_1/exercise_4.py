import cv2
import numpy as np

# Open picture
img_1 = cv2.imread("img2.jpg", cv2.IMREAD_GRAYSCALE)

#min implamer tering
kerner  = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
kerner  = np.array([[-1], [0], [1]])
 
image_op = cv2.filter2D(img_1,-1,kerner)
kerner = np.rot90(kerner)
image_hoj = cv2.filter2D(img_1,-1,kerner)

image_sum = abs(image_op)/2 + abs(image_hoj)/2
image_sum = image_sum.astype(np.uint8)


#open cv versen 
img_opencv_sob = cv2.Laplacian(img_1, -1)
img_opencv_sob_x = cv2.Sobel(img_1, -1, 1, 0, ksize=3)
img_opencv_sob_y = cv2.Sobel(img_1, -1, 0, 1, ksize=3)


#cv2 canny

img_canny = cv2.Canny(img_1,100, 200)
# Display the picture
cv2.imshow("Our window", img_1)
cv2.imshow("op window", image_op)
cv2.imshow("hoj window", image_hoj)
cv2.imshow("sum window", image_sum)
cv2.imshow("sobel x opencv", img_opencv_sob_x)
cv2.imshow("sobel y opencv", img_opencv_sob_y)
cv2.imshow("sobel opencv", img_opencv_sob)
cv2.imshow("canny opencv", img_canny)
cv2.waitKey(0)
