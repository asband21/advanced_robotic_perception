import cv2
import numpy as np

gr = 10

# Open picture
img_1 = cv2.imread("UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test/Test016/001.tif", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Our window", img_1)

#1
for i in range(2,200):
    img_2 = cv2.imread(f"UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test/Test016/{i:03}.tif", cv2.IMREAD_GRAYSCALE)
    img_dif = cv2.absdiff(img_1, img_2)
    img_ret, img_the = cv2.threshold(img_dif, gr, 255, cv2.THRESH_BINARY)
    cv2.imshow("Our window img 2", img_2)
    cv2.imshow("Our window img diff", img_dif)
    cv2.imshow("Our window img thres", img_the)
    #cv2.waitKey(0)

cv2.waitKey(0)
#2

ll = img_1.shape
#mat = np.zeros(ll[0],ll[1],200)
mat = np.zeros((ll[0], ll[1], 200), dtype=np.uint8)
for i in range(1,200):
    mat[:,:,i] = cv2.imread(f"UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test/Test016/{i:03}.tif", cv2.IMREAD_GRAYSCALE)

#img_gem = np.arange(img_3, axis=2, dtype=np.uint8)
img_gem = np.mean(mat, axis=2).astype(np.uint8)

cv2.imshow("Our window", img_gem)

cv2.waitKey(0)
#3 
for i in range(2,200):
    img_2 = cv2.imread(f"UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test/Test016/{i:03}.tif", cv2.IMREAD_GRAYSCALE)
    img_dif = cv2.absdiff(img_gem, img_2)
    img_dif_gm = cv2.absdiff(img_1, img_2)
    img_ret, img_the = cv2.threshold(img_dif, gr,255 , cv2.THRESH_BINARY)
    img_ret, img_the_gm = cv2.threshold(img_dif_gm, gr, 255, cv2.THRESH_BINARY)
    cv2.imshow("Our window img 2", img_2)
    cv2.imshow("Our window img diff", img_dif)
    cv2.imshow("Our window img thres", img_the)
    cv2.imshow("Our window img thres gammel", img_the_gm)
    cv2.waitKey(0)


cv2.waitKey(0)

